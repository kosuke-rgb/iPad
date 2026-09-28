"""フェーズ1: 診断。

1. 探索予算のスイープ（仮説の実行回数の上限 10/20/50/200/2000）
   提案モデル（LLM 0.1M / 0.8M / 2.7M、長期記憶あり・なし）と総当たり探索
2. 正解の出どころの内訳（どのモジュールの仮説が正解になったか）
3. 既存アブレーションのやり直し（300問/分割 × 3シード、対応のある検定＋Holm 補正）
4. 発見ループ（好奇心など）のやり直し（ブラックボックス100個 × 3シード）

すべての条件は、シードごとに同じ問題セットで実行し、問題ごとの結果を保存する。
出力: results/phase1/（既存の結果は上書きしない）

python -m evaluation.phase1 run --workers 2
python -m evaluation.phase1 report
"""
import argparse
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch

OUT = "results/phase1"
N_PER_SPLIT = 300
SEEDS = [0, 1, 2]
BUDGETS = [10, 20, 50, 200, 2000]
MODELS = {"0.1M": "checkpoints/gpt_0.1M.pt", "0.8M": "checkpoints/gpt_0.8M.pt", "2.7M": "checkpoints/gpt_2.7M.pt"}
ABLATION_MODEL = "0.8M"
SOURCES = ["llm_direct", "llm_sample", "compose", "memory", "expert", "repair", "enumerate", "other"]


def source_code(r):
    s = r["source"] or ""
    if s == "llm":
        return "llm_direct" if r["stages"] == ["direct"] else "llm_sample"
    if s.startswith("expert"):
        return "expert"
    return s if s in SOURCES else "other"


# ------------------------------------------------------------------ jobs
def _cfg(name, budget):
    from core.agent import PROPOSED, AgentConfig
    variants = {
        "Proposed": {},
        "Proposed - Memory": {"use_memory": False},
        "Proposed - Router": {"use_router": False},
        "Proposed - Experts": {"use_experts": False},
        "Proposed - Compose": {"use_compose": False},
        "Proposed - Repair": {"use_repair": False},
        "Proposed - Critic": {"use_critic": False},
        "Proposed - LLM samples": {"n_samples": 0},
    }
    if name == "Enumeration":
        return AgentConfig(name=name, use_llm=False, enumerate=True, sim_budget=budget)
    return PROPOSED.variant(name, sim_budget=budget, **variants[name])


ABLATIONS = ["Proposed", "Proposed - Router", "Proposed - Memory", "Proposed - Experts", "Proposed - Compose",
             "Proposed - Repair", "Proposed - Critic", "Proposed - LLM samples"]
DISCOVERY = [("Full", True, True, True), ("- Active", False, True, True), ("- Curiosity", True, False, True),
             ("- Memory", True, True, False), ("Passive", False, False, True)]


def all_jobs():
    jobs = []
    for seed in SEEDS:
        for b in BUDGETS:
            jobs.append(("agent", "none", "Enumeration", b, seed))
        for m in MODELS:
            for b in BUDGETS:
                for c in ("Proposed", "Proposed - Memory"):
                    jobs.append(("agent", m, c, b, seed))
        for c in ABLATIONS:
            jobs.append(("agent", ABLATION_MODEL, c, 2000, seed))
        for name, *_ in DISCOVERY:
            jobs.append(("discovery", ABLATION_MODEL, name, 0, seed))
    seen, out = set(), []
    for j in jobs:
        if j not in seen:
            seen.add(j)
            out.append(j)
    # 軽いジョブから先に（途中でも結果を見られるように）
    weight = {"none": 0, "0.1M": 1, "0.8M": 2, "2.7M": 3}
    return sorted(out, key=lambda j: (j[0] == "discovery", weight[j[1]], j[3] == 2000 and j[2] != "Proposed", j[4]))


def job_path(job):
    kind, m, c, b, seed = job
    safe = c.replace(" ", "").replace("-", "_minus_")
    return f"{OUT}/{kind}/{m}__{safe}__b{b}__s{seed}.json"


_CACHE = {}


def _lm(m):
    from core.llm.backend import TinyGPTBackend
    if m == "none":
        return None
    if m not in _CACHE:
        _CACHE[m] = TinyGPTBackend(MODELS[m])
    return _CACHE[m]


def _data(seed):
    from evaluation import benchmarks
    key = ("data", seed)
    if key not in _CACHE:
        _CACHE[key] = benchmarks.load(N_PER_SPLIT, seed=1000 + 100 * seed)
    return _CACHE[key]


def run_job(job):
    torch.set_num_threads(1)
    path = job_path(job)
    if os.path.exists(path):
        return job, "skip", 0.0
    t0 = time.time()
    kind, m, c, b, seed = job
    if kind == "agent":
        from core.memory.long_term import StaticKnowledgeBase
        from core.world_model.tasks import make_benchmark
        from evaluation.run import evaluate
        if "kb" not in _CACHE:
            _CACHE["kb"] = StaticKnowledgeBase(make_benchmark("train", 1000, seed=7))
        rows = evaluate(_cfg(c, b), _lm(m), _CACHE["kb"], _data(seed), seed=seed)
        out = {s: [[int(r["correct"]), int(r["verified"]), int(r["ambiguous"]), source_code(r),
                    len(r["stages"]), r["sim"], round(r["ms"], 2), round(r["gflops"], 5)] for r in rs]
               for s, rs in rows.items()}
    else:
        from core.discovery.loop import DiscoveryAgent
        from core.memory.long_term import LongTermMemory
        from core.world_model.tasks import make_benchmark
        _, active, curiosity, mem = next(d for d in DISCOVERY if d[0] == c)
        torch.manual_seed(seed)
        world = [t.program for t in make_benchmark("ood_depth", 50, 7500 + seed)] + \
                [t.program for t in make_benchmark("ood_combo", 50, 7600 + seed)]
        random.Random(seed).shuffle(world)
        agent = DiscoveryAgent(_lm(m), LongTermMemory() if mem else None, active=active,
                               curiosity=curiosity, rng=random.Random(seed))
        r = agent.run(world, total_queries=len(world) * 4)
        out = {"per_function": r["per_function"], "queries_used": r["queries_used"],
               "llm_gflops": r["llm_gflops"], "sim_calls": r["sim_calls"]}
    out["_job"] = list(job)
    out["_seconds"] = time.time() - t0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(out, f)
    os.replace(path + ".tmp", path)
    return job, "done", time.time() - t0


def run(workers):
    jobs = [j for j in all_jobs() if not os.path.exists(job_path(j))]
    print(f"{len(jobs)} jobs", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_job, j) for j in jobs]
        for f in as_completed(futs):
            job, status, sec = f.result()
            print(status, job, f"{sec:.0f}s", flush=True)
    print("PHASE1_RUN_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report", "jobs"])
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    if args.cmd == "run":
        run(args.workers)
    elif args.cmd == "jobs":
        js = all_jobs()
        print(len(js), js[:5], js[-3:])
    else:
        from evaluation.phase1_report import report
        report()
