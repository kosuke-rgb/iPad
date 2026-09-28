"""LLM の強さを変えたとき、認知モジュールの寄与がどう変わるかを測る（15章 Performance / Compute）。

python -m evaluation.scaling_eval --ckpt checkpoints/gpt_2.7M.pt --label 2.7M --seed 0
python -m evaluation.scaling_eval --aggregate
"""
import argparse
import glob
import json
import os
import random

import torch

from core.agent import BASE, PROPOSED, AgentConfig
from core.llm import tokenizer as T
from core.llm.backend import TinyGPTBackend
from core.llm.train import make_batch
from core.memory.long_term import StaticKnowledgeBase
from core.world_model.tasks import make_benchmark
from evaluation import benchmarks
from evaluation.metrics import sign_test
from evaluation.run import evaluate

OUT = "results/scaling"
MODULES = ["Router", "Memory", "Experts", "Compose", "Critic", "LLM samples"]


def configs(kind):
    base = [
        BASE,
        BASE.variant("E: Self-consistency x16", n_samples=16, self_consistency=True),
        BASE.variant("S: Sample & verify x16", n_samples=16, use_simulator=True, use_critic=True),
        PROPOSED,
        PROPOSED.variant("Proposed - LLM samples", n_samples=0),
    ]
    if kind == "core":
        return base
    return base + [
        PROPOSED.variant("Proposed - Router", use_router=False),
        PROPOSED.variant("Proposed - Memory", use_memory=False),
        PROPOSED.variant("Proposed - Experts", use_experts=False),
        PROPOSED.variant("Proposed - Compose", use_compose=False),
        PROPOSED.variant("Proposed - Critic", use_critic=False),
    ]


@torch.no_grad()
def val_loss(lm, n_batches=20):
    """学習と同じ分布の新しい問題での、プログラム部分の損失（LLM 単体の強さの指標）。"""
    rng = random.Random(424242)
    tot = 0.0
    for _ in range(n_batches):
        x, y = make_batch(rng, 64, lm.block_size)
        tot += lm.model(x, y)[1].item()
    return tot / n_batches


def run(args):
    torch.set_num_threads(1)
    kb = StaticKnowledgeBase(make_benchmark("train", 1000, seed=7))
    data = benchmarks.load(args.n, seed=1000 + 100 * args.seed)
    out = {"label": args.label, "seed": args.seed, "results": {}}
    if args.ckpt == "none":
        cfgs, lm = [AgentConfig(name="N: Enumeration (no LLM)", use_llm=False, enumerate=True)], None
    else:
        lm = TinyGPTBackend(args.ckpt)
        ck = torch.load(args.ckpt, map_location="cpu")
        out.update(n_params=lm.n_params, train_steps=ck.get("train_steps"),
                   train_seconds=ck.get("train_seconds"), val_loss=val_loss(lm))
        cfgs = configs(args.configs)
    for cfg in cfgs:
        rows = evaluate(cfg, lm, kb, data, seed=args.seed)
        out["results"][cfg.name] = {s: [{k: r[k] for k in ("correct", "gflops", "sim", "ms")} for r in rs]
                                    for s, rs in rows.items()}
        accs = {s: sum(r["correct"] for r in rs) / len(rs) for s, rs in rows.items()}
        print(args.label, args.seed, cfg.name, accs, flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(f"{OUT}/{args.label}_seed{args.seed}.json", "w"))


def _acc(rows):
    return sum(r["correct"] for r in rows) / len(rows)


def aggregate():
    files = sorted(glob.glob(f"{OUT}/*_seed*.json"))
    by_label = {}
    for f in files:
        d = json.load(open(f))
        by_label.setdefault(d["label"], []).append(d)
    splits = [s for s, _ in benchmarks.SPLITS]
    ood = ["ood_combo", "ood_depth", "ood_related"]

    def pooled(runs, name, ss):
        return [r for run in runs for s in ss for r in run["results"][name][s]]

    def order(label):
        runs = by_label[label]
        return runs[0].get("val_loss", 99) * -1 if label != "none" else -1e9

    labels = sorted(by_label, key=order)
    L = []
    L.append("### LLM の強さと、構成ごとの正答率（全シード平均）\n")
    L.append("| LLM | 非埋め込みパラメータ | 学習ステップ | 検証損失 | 構成 | ID | Few-shot | OOD-組み合わせ | OOD-深さ | OOD-関連 | LLM MFLOPs/問 | Simulator/問 | ms/問 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    summary = {}
    for lab in labels:
        runs = by_label[lab]
        r0 = runs[0]
        for name in r0["results"]:
            cells = [f"{100 * _acc(pooled(runs, name, [s])):.1f}%" for s in splits]
            allr = pooled(runs, name, splits)
            g = sum(r["gflops"] for r in allr) / len(allr) * 1000
            sim = sum(r["sim"] for r in allr) / len(allr)
            ms = sum(r["ms"] for r in allr) / len(allr)
            head = (f"{lab} | {r0.get('n_params', 0) / 1e6:.2f}M | {r0.get('train_steps') or '—'} | "
                    f"{r0['val_loss']:.3f}" if lab != "none" else "— | — | — | —")
            L.append(f"| {head} | {name} | " + " | ".join(cells) + f" | {g:.0f} | {sim:.0f} | {ms:.0f} |")
            summary.setdefault(lab, {})[name] = {"all": _acc(allr), "ood": _acc(pooled(runs, name, ood)),
                                                 "gflops": g / 1000, "sim": sim, "ms": ms}
    L.append("\n### モジュールの寄与（提案モデル − そのモジュールを外した版。全分割をまとめた正答率の差と、問題ごとの符号検定）\n")
    mods = [m for m in MODULES if any(f"Proposed - {m}" in by_label[l][0]["results"] for l in labels if l != "none")]
    L.append("| LLM | 検証損失 | " + " | ".join(mods) + " |")
    L.append("|---|---|" + "---|" * len(mods))
    contrib = {}
    for lab in labels:
        if lab == "none":
            continue
        runs = by_label[lab]
        cells = []
        for m in mods:
            name = f"Proposed - {m}"
            if name not in runs[0]["results"]:
                cells.append("—")
                continue
            a, b = pooled(runs, "Proposed", splits), pooled(runs, name, splits)
            w = sum(x["correct"] and not y["correct"] for x, y in zip(a, b))
            l_ = sum(y["correct"] and not x["correct"] for x, y in zip(a, b))
            d = 100 * (_acc(a) - _acc(b))
            p = sign_test(w, l_)
            cells.append(f"{d:+.1f} pt (p={p:.3f})")
            contrib.setdefault(lab, {})[m] = {"delta": d, "p": p}
        L.append(f"| {lab} | {runs[0]['val_loss']:.3f} | " + " | ".join(cells) + " |")
    table = "\n".join(L)
    json.dump({"table": table, "summary": summary, "contrib": contrib}, open(f"{OUT}/summary.json", "w"),
              ensure_ascii=False)
    print(table)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--label")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--configs", default="full")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    aggregate() if args.aggregate else run(args)
