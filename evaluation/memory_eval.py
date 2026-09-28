"""長期記憶の設計比較（複数シード）。

python -m evaluation.memory_eval --seed 0   # シードごとに別プロセスで実行できる
python -m evaluation.memory_eval --aggregate
"""
import argparse
import glob
import json

import torch

from core.agent import memory_configs
from core.llm.backend import TinyGPTBackend
from core.memory.long_term import StaticKnowledgeBase
from core.world_model.tasks import make_benchmark
from evaluation import benchmarks
from evaluation.metrics import wilson
from evaluation.run import evaluate, summarize


def run_seed(args):
    torch.set_num_threads(1)
    lm = TinyGPTBackend(args.ckpt)
    kb = StaticKnowledgeBase(make_benchmark("train", 1000, seed=7))
    data = benchmarks.load(args.n, seed=1000 + 100 * args.seed)
    out = {}
    for cfg in memory_configs():
        rows = evaluate(cfg, lm, kb, data, seed=args.seed)
        out[cfg.name] = {s: {"summary": summarize(r), "rows": r} for s, r in rows.items()}
        print(args.seed, cfg.name, {s: out[cfg.name][s]["summary"]["acc"] for s in rows}, flush=True)
    json.dump(out, open(f"results/memory_seed{args.seed}.json", "w"), ensure_ascii=False)


def aggregate():
    runs = [json.load(open(p)) for p in sorted(glob.glob("results/memory_seed*.json"))]
    names = list(runs[0])
    lines = [f"{len(runs)}シード × 各分割100問。括弧内は全シードをまとめた95%信頼区間。\n",
             "| 構成 | " + " | ".join(n for _, n in benchmarks.SPLITS) + " | Simulator 回数 | 実時間 ms |",
             "|---|" + "---|" * (len(benchmarks.SPLITS) + 2)]
    agg = {}
    for name in names:
        cells, sims, mss = [], [], []
        for s, _ in benchmarks.SPLITS:
            rows = [r for run in runs for r in run[name][s]["rows"]]
            k, n = sum(r["correct"] for r in rows), len(rows)
            lo, hi = wilson(k, n)
            per = [run[name][s]["summary"]["acc"] for run in runs]
            cells.append(f"{100 * k / n:.1f}% ({100 * lo:.0f}–{100 * hi:.0f})")
            sims += [r["sim"] for r in rows]
            mss += [r["ms"] for r in rows]
            agg.setdefault(name, {})[s] = {"acc": k / n, "per_seed": per}
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {sum(sims) / len(sims):.0f} | {sum(mss) / len(mss):.0f} |")
    table = "\n".join(lines)
    json.dump({"table": table, "acc": agg}, open("results/memory.json", "w"), ensure_ascii=False)
    print(table)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/tiny_gpt.pt")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    aggregate() if args.aggregate else run_seed(args)
