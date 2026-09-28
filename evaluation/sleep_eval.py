"""睡眠による記憶の定着の評価。

昼: その日の新しい問題（OOD-深さ 100問 + OOD-組み合わせ 50問）を提案モデルで解き、長期記憶に経験を貯める。
夜: 睡眠（記憶の再生で LLM の重みを更新）。
翌日は別の新しい問題。これを数日繰り返し、次を測る。
  - その日の問題の正答率・計算量
  - 直感で解けた割合（LLM の最初の1回答だけで検証済みの正解に到達した割合）
  - 固定テスト（ID / OOD-深さ）での LLM 単体の正答率 → 経験が「自分自身」を変えたか、昔の知識を忘れていないか

python -m evaluation.sleep_eval --condition sleep --seed 0
python -m evaluation.sleep_eval --aggregate
"""
import argparse
import copy
import glob
import json
import os
import time

import torch

from core.agent import BASE, PROPOSED, Agent
from core.llm.backend import TinyGPTBackend
from core.memory.consolidation import SleepConfig, sleep
from core.memory.long_term import LongTermMemory
from core.world_model.tasks import make_benchmark
from evaluation.metrics import sign_test

OUT = "results/sleep"
CONDITIONS = {
    "no_sleep": None,
    "sleep": SleepConfig(),
    "sleep_no_rehearsal": SleepConfig(rehearsal=0.0),
    "sleep_no_dreams": SleepConfig(dreams_per_episode=0),
}
LABELS = {
    "no_sleep": "睡眠なし（記憶の検索のみ）",
    "sleep": "睡眠あり（再生＋夢＋復習）",
    "sleep_no_rehearsal": "睡眠あり − 復習（新しい経験だけ）",
    "sleep_no_dreams": "睡眠あり − 夢（実際の観測だけ）",
}


def day_tasks(seed, day):
    base = 50_000 + 1000 * seed + 10 * day
    return make_benchmark("ood_depth", 100, base) + make_benchmark("ood_combo", 50, base + 1)


def solve_all(agent, tasks):
    rows = []
    for t in tasks:
        t0 = time.perf_counter()
        r = agent.solve(t.train, [x for x, _ in t.test])
        rows.append({"correct": r.program is not None and all(p == y for p, (_, y) in zip(r.predictions, t.test)),
                     "intuition": r.verified and r.stages == ["direct"],
                     "sim": r.meter.sim_calls, "gflops": r.meter.llm_flops / 1e9,
                     "ms": (time.perf_counter() - t0) * 1000, "split": t.split})
    return rows


def test(lm, tests, memory):
    out = {}
    for name, tasks in tests.items():
        a = solve_all(Agent(BASE, lm), tasks)
        out[f"llm_alone_{name}"] = sum(r["correct"] for r in a) / len(a)
    return out


def run(args):
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    lm = TinyGPTBackend(args.ckpt)
    memory = LongTermMemory()
    agent = Agent(PROPOSED, lm, memory=memory)
    tests = {"id": make_benchmark("id", 100, 70_000 + args.seed),
             "ood_depth": make_benchmark("ood_depth", 100, 71_000 + args.seed)}
    cfg = CONDITIONS[args.condition]
    log = {"condition": args.condition, "seed": args.seed, "days": []}
    log["before"] = test(lm, tests, memory)
    print(args.condition, "before", log["before"], flush=True)
    for day in range(args.days):
        rows = solve_all(agent, day_tasks(args.seed, day))
        entry = {"day": day, "rows": rows,
                 "acc": sum(r["correct"] for r in rows) / len(rows),
                 "intuition": sum(r["intuition"] for r in rows) / len(rows),
                 "sim": sum(r["sim"] for r in rows) / len(rows),
                 "ms": sum(r["ms"] for r in rows) / len(rows)}
        if cfg is not None:
            entry["sleep"] = sleep(lm.model, memory, cfg, seed=args.seed * 100 + day, block_size=lm.block_size)
        entry["after"] = test(lm, tests, memory)
        log["days"].append(entry)
        print(args.condition, f"day{day}", {k: v for k, v in entry.items() if k != "rows"}, flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(log, open(f"{OUT}/{args.condition}_seed{args.seed}.json", "w"))


def aggregate():
    runs = {}
    for f in sorted(glob.glob(f"{OUT}/*_seed*.json")):
        d = json.load(open(f))
        runs.setdefault(d["condition"], []).append(d)
    days = len(next(iter(runs.values()))[0]["days"])
    L = [f"{len(next(iter(runs.values())))}シードの平均。各日150問（毎日新しい問題）。\n",
         "### その日の問題の正答率（直感で解けた割合 / 平均 Simulator 回数）\n",
         "| 条件 | " + " | ".join(f"{d + 1}日目" for d in range(days)) + " |",
         "|---|" + "---|" * days]
    for c in CONDITIONS:
        if c not in runs:
            continue
        cells = []
        for d in range(days):
            es = [r["days"][d] for r in runs[c]]
            m = lambda k: sum(e[k] for e in es) / len(es)
            cells.append(f"{100 * m('acc'):.1f}% ({100 * m('intuition'):.0f}% / {m('sim'):.0f})")
        L.append(f"| {LABELS[c]} | " + " | ".join(cells) + " |")
    L += ["\n### 固定テストでの LLM 単体の正答率（その夜の睡眠の後）\n",
          "| 条件 | テスト | 初日の前 | " + " | ".join(f"{d + 1}日目の夜の後" for d in range(days)) + " |",
          "|---|---|---|" + "---|" * days]
    for c in CONDITIONS:
        if c not in runs:
            continue
        for t, tn in (("id", "ID（昔の知識）"), ("ood_depth", "OOD-深さ")):
            k = f"llm_alone_{t}"
            before = sum(r["before"][k] for r in runs[c]) / len(runs[c])
            cells = [f"{100 * sum(r['days'][d]['after'][k] for r in runs[c]) / len(runs[c]):.1f}%" for d in range(days)]
            L.append(f"| {LABELS[c]} | {tn} | {100 * before:.1f}% | " + " | ".join(cells) + " |")
    if "no_sleep" in runs:
        L += ["\n### 最終日の問題での対応比較（睡眠なしに対する勝ち / 負け）\n", "| 条件 | 勝ち / 負け | 符号検定 p |", "|---|---|---|"]
        last = days - 1
        for c in CONDITIONS:
            if c == "no_sleep" or c not in runs:
                continue
            w = l_ = 0
            for a_run, b_run in zip(sorted(runs[c], key=lambda r: r["seed"]), sorted(runs["no_sleep"], key=lambda r: r["seed"])):
                for a, b in zip(a_run["days"][last]["rows"], b_run["days"][last]["rows"]):
                    w += a["correct"] and not b["correct"]
                    l_ += b["correct"] and not a["correct"]
            L.append(f"| {LABELS[c]} | +{w} / −{l_} | {sign_test(w, l_):.4f} |")
    table = "\n".join(L)
    json.dump({"table": table}, open(f"{OUT}/summary.json", "w"), ensure_ascii=False)
    print(table)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/gpt_0.8M.pt")
    ap.add_argument("--condition", default="sleep", choices=list(CONDITIONS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--days", type=int, default=4)
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    aggregate() if args.aggregate else run(args)
