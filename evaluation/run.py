"""Baseline・提案モデル・アブレーションを同一条件で比較する。

python -m evaluation.run --ckpt checkpoints/tiny_gpt.pt --n 100
結果は results/*.json と docs/results.md に出力される。
"""
import argparse
import json
import random
import time

import torch

from core.agent import Agent, ablation_configs, baseline_configs
from core.llm.backend import ComputeMeter, TinyGPTBackend
from core.memory.long_term import LongTermMemory, StaticKnowledgeBase
from core.world_model.dsl import safe_run, to_str
from core.world_model.tasks import make_benchmark
from evaluation import benchmarks
from evaluation.metrics import wilson


def evaluate(cfg, lm, kb, data, seed=0):
    torch.manual_seed(seed)
    agent = Agent(cfg, lm if cfg.use_llm else None, kb=kb, memory=LongTermMemory())
    out = {}
    for split, _ in benchmarks.SPLITS:  # 同じ順序で流す（記憶は分割をまたいで蓄積される）
        rows = []
        for t in data[split]:
            t0 = time.perf_counter()
            r = agent.solve(t.train, [x for x, _ in t.test])
            dt = time.perf_counter() - t0
            correct = r.program is not None and all(p == y for p, (_, y) in zip(r.predictions, t.test))
            fits = r.program is not None and all(safe_run(r.program, x) == y for x, y in t.train)
            rows.append({"correct": correct, "fits_visible": fits, "ambiguous": r.ambiguous,
                         "verified": r.verified, "gflops": r.meter.llm_flops / 1e9,
                         "sim": r.meter.sim_calls, "ms": dt * 1000, "stages": r.stages,
                         "source": r.source, "program": to_str(r.program) if r.program else None,
                         "truth": to_str(t.program)})
        out[split] = rows
    return out


def summarize(rows):
    n = len(rows)
    k = sum(r["correct"] for r in rows)
    lo, hi = wilson(k, n)
    return {"acc": k / n, "ci": (lo, hi), "halluc": sum(not r["fits_visible"] for r in rows) / n,
            "gflops": sum(r["gflops"] for r in rows) / n, "sim": sum(r["sim"] for r in rows) / n,
            "ms": sum(r["ms"] for r in rows) / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/tiny_gpt.pt")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--which", default="baselines,ablations")
    ap.add_argument("--out", default="results/main.json")
    args = ap.parse_args()
    torch.set_num_threads(1)

    lm = TinyGPTBackend(args.ckpt)
    kb = StaticKnowledgeBase(make_benchmark("train", 1000, seed=7))
    data = benchmarks.load(args.n)
    cfgs = []
    if "baselines" in args.which:
        cfgs += baseline_configs()
    if "ablations" in args.which:
        cfgs += [c for c in ablation_configs() if c.name not in {x.name for x in cfgs}]
    results = {}
    for cfg in cfgs:
        t0 = time.time()
        rows = evaluate(cfg, lm, kb, data)
        results[cfg.name] = {s: {"summary": summarize(r), "rows": r} for s, r in rows.items()}
        accs = " ".join(f"{s}={results[cfg.name][s]['summary']['acc']:.2f}" for s in rows)
        print(f"{cfg.name:55s} {accs}  ({time.time() - t0:.0f}s)", flush=True)
    with open(args.out, "w") as f:
        json.dump({"n_params": lm.n_params, "results": results}, f, ensure_ascii=False)
    print(f"保存しました: {args.out}")


if __name__ == "__main__":
    main()
