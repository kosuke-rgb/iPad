"""学習した世界モデルの評価。

A. 予測の正確さ: 1手順の遷移、および深さ1〜3のプログラムのロールアウト
B. 「考えてから行動する」: 現実での実験（実行）回数をどれだけ減らせるか

B の手順（1問ごと）:
  1. 現実で何も実行せずに仮説の候補を作る（LLM のサンプル・Expert・組み合わせ探索、最大300個）
  2. 全候補を世界モデルで頭の中で実行（想像）し、同時に分析用として現実の結果も記録する
  3. 記録から次の条件を計算する（同じ候補・同じ記録を使うので公平）
     - 現実のみ          : 事前の順番どおりに現実で試し、最初に観測と合ったものを採用
     - 想像→現実        : 想像で観測と合った候補から先に現実で試す
     - 想像のみ          : 想像で合った最初の候補を、現実で一度も試さずに採用
     - 完璧な想像（上限）: 手書きのインタプリタを世界モデルとして使った場合
     現実の実験回数に上限 R を設けたときの正答率も計算する

python -m evaluation.world_model_eval --n 60
"""
import argparse
import json
import random
import time
from collections import Counter

import torch

from core.critic.critic import soft_similarity
from core.experts.experts import ALL_EXPERTS
from core.hypothesis.generator import compose
from core.llm.backend import ComputeMeter, TinyGPTBackend
from core.world_model.dsl import PRIM_NAMES, safe_run
from core.world_model.features import features
from core.world_model.learned import LearnedWorldModel, transition
from core.world_model.tasks import make_benchmark, random_input, sample_program

BUDGETS = [1, 2, 5, 10, 30, 10_000]


def accuracy_part(wm, n=600, seed=5):
    rng = random.Random(seed)
    trans = [transition(rng) for _ in range(n)]
    preds = wm.step_many([(p, tuple(a)) for p, a, _ in trans])
    per = Counter(), Counter()
    for (p, _, b), r in zip(trans, preds):
        per[1][p] += 1
        per[0][p] += r == b
    step_acc = sum(per[0].values()) / n
    by_prim = {p: per[0][p] / per[1][p] for p in sorted(per[1])}
    roll = {}
    for depth, split in ((1, None), (2, "id"), (3, "ood_depth")):
        progs = []
        while len(progs) < 100:
            prog = (rng.choice(PRIM_NAMES),) if depth == 1 else sample_program(rng, split)
            if prog:
                progs.append(prog)
        ins = [random_input(rng) for _ in range(3)]
        out = wm.run_many(progs, ins)
        ok = [out[(p, tuple(x))] == safe_run(p, x) for p in progs for x in ins]
        roll[depth] = sum(ok) / len(ok)
    return {"step_acc": step_acc, "by_prim": by_prim, "rollout_acc": roll}


def candidate_pool(lm, examples, rng, cap=300):
    meter = ComputeMeter()
    samples = lm.sample(examples, 1, 0.0, meter) + sorted(lm.sample(examples, 32, 1.0, meter), key=lambda s: -s[1])
    pool = []
    for p, _ in samples:
        if p not in pool:
            pool.append(p)
    f = features(examples)
    for e in ALL_EXPERTS:
        if e.gate(f):
            pool += [p for p in e.propose()[:60] if p not in pool]
    frags = Counter()
    for p, _ in samples:
        for q in set(p):
            frags[q] += 1
    for q in PRIM_NAMES:
        frags[q] += 0.05
    for p in compose(frags, [], limit=cap * 2):
        if len(pool) >= cap:
            break
        if p not in pool:
            pool.append(p)
    return pool[:cap], meter.llm_flops


def think_part(lm, wm, tasks):
    rows = []
    for t in tasks:
        ex = t.train
        pool, llm_flops = candidate_pool(lm, ex, random.Random(0))
        f0 = wm.flops
        imag = wm.run_many(pool, [x for x, _ in ex])
        real_fit = [all(safe_run(p, x) == y for x, y in ex) for p in pool]
        imag_fit = [all(imag[(p, tuple(x))] == y for x, y in ex) for p in pool]
        imag_soft = [sum(soft_similarity(imag[(p, tuple(x))], y) for x, y in ex) / len(ex) for p in pool]
        correct = [all(safe_run(p, x) == y for x, y in t.test) for p in pool]
        rows.append({"split": t.split, "pool": len(pool), "truth_in_pool": any(correct),
                     "real_fit": real_fit, "imag_fit": imag_fit, "imag_soft": imag_soft, "correct": correct,
                     "wm_flops": wm.flops - f0, "llm_flops": llm_flops})
    return rows


def simulate(rows):
    """記録から各条件の正答率と、現実での実験回数を計算する。"""
    res = {}
    for name in ("real_only", "imagine_then_act", "imagination_only", "perfect_imagination"):
        per_budget = {}
        for R in BUDGETS:
            acc, real = [], []
            for r in rows:
                n = r["pool"]
                if name == "real_only":
                    order = list(range(n))
                elif name == "imagine_then_act":
                    order = sorted(range(n), key=lambda i: (not r["imag_fit"][i], -r["imag_soft"][i], i))
                if name in ("real_only", "imagine_then_act"):
                    chosen, used = (order[0] if order else None), 0
                    for k, i in enumerate(order[:R]):
                        used = k + 1
                        if r["real_fit"][i]:
                            chosen = i
                            break
                    acc.append(chosen is not None and r["correct"][chosen])
                    real.append(used)
                else:
                    fits = r["imag_fit"] if name == "imagination_only" else r["real_fit"]
                    chosen = next((i for i in range(n) if fits[i]), 0 if n else None)
                    acc.append(chosen is not None and r["correct"][chosen])
                    real.append(0)
            per_budget[R] = {"acc": sum(acc) / len(acc), "real": sum(real) / len(real)}
        res[name] = per_budget
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", default="checkpoints/world_model.pt")
    ap.add_argument("--ckpt", default="checkpoints/gpt_0.8M.pt")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", default="results/world_model/eval.json")
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    wm = LearnedWorldModel(args.wm)
    lm = TinyGPTBackend(args.ckpt)
    t0 = time.time()
    acc = accuracy_part(wm)
    print("accuracy", json.dumps({k: v for k, v in acc.items() if k != "by_prim"}), f"{time.time() - t0:.0f}s", flush=True)
    tasks = []
    for i, s in enumerate(("id", "ood_combo", "ood_depth")):
        tasks += make_benchmark(s, args.n, 90_000 + i)
    rows = think_part(lm, wm, tasks)
    print(f"think part {time.time() - t0:.0f}s", flush=True)
    out = {"accuracy": acc, "wm_params": wm.n_params,
           "wm_flops_per_task": sum(r["wm_flops"] for r in rows) / len(rows),
           "llm_flops_per_task": sum(r["llm_flops"] for r in rows) / len(rows),
           "truth_in_pool": sum(r["truth_in_pool"] for r in rows) / len(rows),
           "imag_fit_precision": sum(sum(a and b for a, b in zip(r["imag_fit"], r["real_fit"])) for r in rows)
                                 / max(1, sum(sum(r["imag_fit"]) for r in rows)),
           "imag_fit_recall": sum(sum(a and b for a, b in zip(r["imag_fit"], r["real_fit"])) for r in rows)
                              / max(1, sum(sum(r["real_fit"]) for r in rows)),
           "all": simulate(rows)}
    for s in ("id", "ood_combo", "ood_depth"):
        out[s] = simulate([r for r in rows if r["split"] == s])
    json.dump(out, open(args.out, "w"), ensure_ascii=False)
    for name, v in out["all"].items():
        print(name, {R: (round(x["acc"], 3), round(x["real"], 1)) for R, x in v.items()})
    print({k: out[k] for k in ("truth_in_pool", "imag_fit_precision", "imag_fit_recall", "wm_flops_per_task")})


if __name__ == "__main__":
    main()
