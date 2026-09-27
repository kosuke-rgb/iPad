"""自律的発見ループの評価。

未知のブラックボックス（学習時に見ていない規則）を多数用意し、
合計の問い合わせ回数に上限をつけた状態で、いくつ正しく特定できるかを測る。

比較:
- Full    : 情報利得で実験を設計 + 好奇心で対象を選ぶ + 長期記憶（発見の再利用）
- -Active : 実験の入力をランダムに選ぶ
- -Curiosity: 調べる対象をランダムに選ぶ
- -Memory : 発見を記憶せず、毎回ゼロから考える
- Passive : 入力も対象もランダム
"""
import argparse
import json
import random

import torch

from core.discovery.loop import DiscoveryAgent
from core.llm.backend import TinyGPTBackend
from core.memory.long_term import LongTermMemory
from core.world_model.tasks import make_benchmark

CONFIGS = [
    ("Full (active + curiosity + memory)", True, True, True),
    ("- Active experiment design", False, True, True),
    ("- Curiosity (random target)", True, False, True),
    ("- Memory", True, True, False),
    ("Passive (random inputs & targets)", False, False, True),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/tiny_gpt.pt")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--queries_per_fn", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="results/discovery.json")
    args = ap.parse_args()
    torch.set_num_threads(1)
    lm = TinyGPTBackend(args.ckpt)
    out = {}
    for name, active, curiosity, mem in CONFIGS:
        runs = []
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            half = args.n // 2
            world = [t.program for t in make_benchmark("ood_depth", half, 500 + seed)] + \
                    [t.program for t in make_benchmark("ood_combo", args.n - half, 600 + seed)]
            random.Random(seed).shuffle(world)
            agent = DiscoveryAgent(lm, LongTermMemory() if mem else None, active=active,
                                   curiosity=curiosity, rng=random.Random(seed))
            r = agent.run(world, total_queries=args.n * args.queries_per_fn)
            runs.append(r)
        ident = [r["identified"] / r["total"] for r in runs]
        out[name] = {"identified_mean": sum(ident) / len(ident), "identified_per_seed": ident,
                     "gflops": sum(r["llm_gflops"] for r in runs) / len(runs),
                     "sim_calls": sum(r["sim_calls"] for r in runs) / len(runs), "runs": runs}
        print(f"{name:40s} identified={out[name]['identified_mean']:.3f} {ident}", flush=True)
    with open(args.out, "w") as f:
        json.dump(out, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
