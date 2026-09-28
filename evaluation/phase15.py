"""フェーズ1.5: 暫定結果の再現と記憶の再設計。

0. 新しい既定設定（Default v2: Experts OFF, Repair OFF）で、予算 10/50/200/2000 のベースラインを取り直す
1. 学習した世界モデルの再現（3シード × 300問/分割）
2. 睡眠の対照実験（3シード）: 睡眠なし / 昼にその場で学習（2種）/ 睡眠 / 世界モデルの夢で睡眠
3. 長期記憶の再設計: 類似度のしきい値を検証用の問題で決め、予算 50 以下では検索しない。旧設計・新設計・記憶なしを予算別に比較
4. 破滅的忘却への対策（3シード）: 有限の復習バッファを基準に、復習の増量 / 夢による復習 / EWC を1つずつ ON

出力: results/phase1_5/（既存の結果は上書きしない）。ジョブごとに1ファイルで、途中から再開できる。

python -m evaluation.phase15 run --workers 4
python -m evaluation.phase15 report
"""
import argparse
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch

OUT = "results/phase1_5"
SEEDS = [0, 1, 2]
VAL_SEED = 9
N_PER_SPLIT = 300
BUDGETS = [10, 50, 200, 2000]
LLM = "checkpoints/gpt_0.8M.pt"
WM = "checkpoints/world_model.pt"
TAUS = [0.7, 0.8, 0.9, 1.0]
EWC_LAMBDAS = [1e1, 1e2, 1e3, 1e4, 1e5]
DAYS = 4
AWAKE_STEPS_PER_EPISODE = 4  # 1日150問 × 検証率 約0.9 × 4 ≒ 睡眠1晩の600ステップ
BUFFER_SIZE = 500

SLEEP_CONDITIONS = {
    # 2. 睡眠の対照実験
    "a_no_sleep": "睡眠なし",
    "b_awake_experience": "昼にその場で学習（その経験の観測だけ）",
    "b2_awake_mixture": "昼にその場で学習（睡眠と同じ材料・比率）",
    "c_sleep": "睡眠（再生＋夢＋復習）",
    "d_sleep_wm_dreams": "睡眠（夢を学習した世界モデルで生成）",
    # 4. 破滅的忘却への対策（基準: 有限の復習バッファ）
    "f_base": "睡眠・有限の復習バッファ（基準）",
    "f_more_rehearsal": "基準 ＋ 復習の増量（比率0.75・2倍のステップ）",
    "f_dream_rehearsal": "基準 ＋ 古い規則の夢を復習に混ぜる",
    "f_ewc": "基準 ＋ EWC",
}


# ======================================================================= helpers
_CACHE = {}


def _lm():
    from core.llm.backend import TinyGPTBackend
    return TinyGPTBackend(LLM)  # 睡眠で重みが変わるので、ジョブごとに新しく読み込む


def _data(seed):
    from evaluation import benchmarks
    if ("data", seed) not in _CACHE:
        _CACHE[("data", seed)] = benchmarks.load(N_PER_SPLIT, seed=1000 + 100 * seed)
    return _CACHE[("data", seed)]


def _kb():
    from core.memory.long_term import StaticKnowledgeBase
    from core.world_model.tasks import make_benchmark
    if "kb" not in _CACHE:
        _CACHE["kb"] = StaticKnowledgeBase(make_benchmark("train", 1000, seed=7))
    return _CACHE["kb"]


def _write(path, out, t0):
    out["_seconds"] = time.time() - t0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(out, f)
    os.replace(path + ".tmp", path)


def tau_star():
    p = f"{OUT}/memory_tau.json"
    return json.load(open(p))["tau"] if os.path.exists(p) else None


def ewc_star():
    p = f"{OUT}/ewc_lambda.json"
    return json.load(open(p))["lambda"] if os.path.exists(p) else None


# ======================================================================= agent jobs
def agent_cfg(name, budget):
    from core.agent import DEFAULT_V2
    if name == "default":
        return DEFAULT_V2.variant("Default v2", sim_budget=budget)
    if name == "no_memory":
        return DEFAULT_V2.variant("Default v2 - Memory", sim_budget=budget, use_memory=False)
    if name == "new_memory":
        return DEFAULT_V2.variant("Default v2 + Memory v3", sim_budget=budget,
                                  memory_min_similarity=tau_star(), memory_min_budget=51)
    if name.startswith("tau"):
        return DEFAULT_V2.variant(name, sim_budget=budget, memory_min_similarity=float(name[3:]))
    raise ValueError(name)


def run_agent(job):
    _, name, budget, seed = job
    from evaluation.phase1 import source_code
    from evaluation.run import evaluate
    lm = _CACHE.setdefault("lm_static", _lm())
    data = _data(seed)
    rows = evaluate(agent_cfg(name, budget), lm, _kb(), data, seed=seed)
    return {s: [[int(r["correct"]), int(r["verified"]), int(r["ambiguous"]), source_code(r),
                 len(r["stages"]), r["sim"], round(r["ms"], 2), round(r["gflops"], 5)] for r in rs]
            for s, rs in rows.items()}


# ======================================================================= world model
def run_wm(job):
    _, seed = job
    from core.llm.backend import TinyGPTBackend
    from core.world_model.learned import LearnedWorldModel
    from core.world_model.tasks import make_benchmark
    from evaluation.world_model_eval import accuracy_part, per_task_outcomes, think_part
    torch.manual_seed(seed)
    wm = LearnedWorldModel(WM)
    lm = TinyGPTBackend(LLM)
    acc = accuracy_part(wm, seed=100 + seed)
    tasks = []
    for i, s in enumerate(("id", "ood_combo", "ood_depth")):
        tasks += make_benchmark(s, N_PER_SPLIT, 95_000 + 1000 * seed + i)
    rows = think_part(lm, wm, tasks)
    return {"accuracy": acc, "per_task": per_task_outcomes(rows),
            "wm_flops": [r["wm_flops"] for r in rows], "llm_flops": [r["llm_flops"] for r in rows],
            "pool": [r["pool"] for r in rows], "split": [r["split"] for r in rows],
            "imag_fit": sum(sum(r["imag_fit"]) for r in rows),
            "imag_and_real_fit": sum(sum(a and b for a, b in zip(r["imag_fit"], r["real_fit"])) for r in rows),
            "real_fit": sum(sum(r["real_fit"]) for r in rows)}


# ======================================================================= sleep
def sleep_config(cond, seed, lam=None):
    from core.memory.consolidation import SleepConfig, fisher_diag, make_rehearsal_buffer
    if cond in ("a_no_sleep",):
        return None
    if cond in ("b_awake_experience", "b2_awake_mixture", "c_sleep"):
        return SleepConfig()
    if cond == "d_sleep_wm_dreams":
        from core.world_model.learned import LearnedWorldModel
        return SleepConfig(dream_world_model=LearnedWorldModel(WM))
    buf = make_rehearsal_buffer(BUFFER_SIZE, seed=31_000 + seed)
    if cond == "f_base":
        return SleepConfig(rehearsal_buffer=buf)
    if cond == "f_more_rehearsal":
        return SleepConfig(rehearsal_buffer=buf, rehearsal=0.75, steps=1200)
    if cond == "f_dream_rehearsal":
        return SleepConfig(rehearsal_buffer=buf, dream_rehearsal=0.5)
    if cond == "f_ewc":
        return SleepConfig(rehearsal_buffer=buf, ewc_lambda=lam if lam is not None else ewc_star())
    raise ValueError(cond)


def _tests(seed, n):
    from core.world_model.tasks import make_benchmark
    return {"id": make_benchmark("id", n, 70_000 + seed), "ood_depth": make_benchmark("ood_depth", n, 71_000 + seed)}


def _llm_alone(lm, tasks):
    from core.agent import BASE, Agent
    a = Agent(BASE, lm)
    out = []
    for t in tasks:
        r = a.solve(t.train, [x for x, _ in t.test])
        out.append(int(r.program is not None and all(p == y for p, (_, y) in zip(r.predictions, t.test))))
    return out


def run_sleep(cond, seed, days=DAYS, n_test=N_PER_SPLIT, lam=None):
    from core.agent import DEFAULT_V2, Agent
    from core.memory.consolidation import AwakeLearner, fisher_diag, sleep
    from core.memory.consolidation import encode_pair as _enc
    from core.memory.long_term import LongTermMemory
    from evaluation.sleep_eval import day_tasks
    torch.manual_seed(seed)
    lm = _lm()
    memory = LongTermMemory()
    agent = Agent(DEFAULT_V2, lm, memory=memory)
    cfg = sleep_config(cond, seed, lam)
    if cfg is not None and cfg.ewc_lambda:
        pairs = [p for p in (_enc(ex, prog, lm.block_size) for ex, prog in cfg.rehearsal_buffer) if p]
        cfg.ewc_state = fisher_diag(lm.model, pairs, seed=seed)
    awake = None
    if cond.startswith("b"):
        mode = "experience" if cond == "b_awake_experience" else "mixture"
        awake = AwakeLearner(lm.model, cfg, AWAKE_STEPS_PER_EPISODE, mode, seed=seed, block_size=lm.block_size)
    tests = _tests(seed, n_test)
    log = {"condition": cond, "seed": seed, "before": {k: _llm_alone(lm, v) for k, v in tests.items()}, "days": []}
    for day in range(days):
        rows = []
        steps_before = awake.steps if awake else 0
        for t in day_tasks(seed, day):
            t0 = time.perf_counter()
            r = agent.solve(t.train, [x for x, _ in t.test])
            rows.append([int(r.program is not None and all(p == y for p, (_, y) in zip(r.predictions, t.test))),
                         int(r.verified and r.stages == ["direct"]), r.meter.sim_calls,
                         round((time.perf_counter() - t0) * 1000, 2), round(r.meter.llm_flops / 1e9, 5)])
            if awake:
                awake.after_episode(memory)
        entry = {"day": day, "rows": rows}
        if awake:
            entry["train_steps"] = awake.steps - steps_before
        elif cfg is not None:
            entry["sleep"] = sleep(lm.model, memory, cfg, seed=seed * 100 + day, block_size=lm.block_size)
            entry["train_steps"] = cfg.steps
        else:
            entry["train_steps"] = 0
        entry["after"] = {k: _llm_alone(lm, v) for k, v in tests.items()}
        log["days"].append(entry)
    return log


# ======================================================================= jobs
def jobs_stage1():
    js = [("agent", f"tau{t}", b, VAL_SEED) for t in TAUS for b in (200, 2000)]
    js += [("ewc_cal", lam, VAL_SEED) for lam in EWC_LAMBDAS]
    js += [("sleep_cal_base", VAL_SEED)]
    return js


def jobs_stage2():
    js = []
    for s in SEEDS:
        for b in BUDGETS:
            js += [("agent", "default", b, s), ("agent", "no_memory", b, s), ("agent", "new_memory", b, s)]
    js += [("wm", s) for s in SEEDS]
    js += [("sleep", c, s) for c in SLEEP_CONDITIONS for s in SEEDS]
    # 重いジョブから先に（並列の最後に長いジョブだけが残らないように）
    heavy = {"wm": 0, "sleep": 1, "agent": 2}
    return sorted(js, key=lambda j: heavy[j[0]])


def job_path(job):
    kind = job[0]
    if kind == "agent":
        return f"{OUT}/agent/{job[1]}__b{job[2]}__s{job[3]}.json"
    if kind == "wm":
        return f"{OUT}/wm/seed{job[1]}.json"
    if kind == "sleep":
        return f"{OUT}/sleep/{job[1]}__s{job[2]}.json"
    if kind == "ewc_cal":
        return f"{OUT}/calibration/ewc_{job[1]:g}.json"
    if kind == "sleep_cal_base":
        return f"{OUT}/calibration/ewc_0.json"
    raise ValueError(job)


def run_job(job):
    torch.set_num_threads(1)
    path = job_path(job)
    if os.path.exists(path):
        return job, "skip", 0.0
    t0 = time.time()
    kind = job[0]
    if kind == "agent":
        out = run_agent(job)
    elif kind == "wm":
        out = run_wm(job)
    elif kind == "sleep":
        out = run_sleep(job[1], job[2])
    elif kind == "ewc_cal":
        out = run_sleep("f_ewc", job[2], days=1, n_test=150, lam=job[1])
    else:
        out = run_sleep("f_base", job[1], days=1, n_test=150)
    out["_job"] = list(job)
    _write(path, out, t0)
    return job, "done", time.time() - t0


def _pool(jobs, workers):
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_job, j) for j in jobs]
        for f in as_completed(futs):
            job, status, sec = f.result()
            print(status, job, f"{sec:.0f}s", flush=True)


def decide_stage1():
    """検証用の問題で、記憶のしきい値と EWC の強さを決める（評価用のシードは使わない）。"""
    if not os.path.exists(f"{OUT}/memory_tau.json"):
        scores = {}
        for t in TAUS:
            rs = []
            for b in (200, 2000):
                d = json.load(open(job_path(("agent", f"tau{t}", b, VAL_SEED))))
                rs += [r[0] for s in d if not s.startswith("_") for r in d[s]]
            scores[t] = sum(rs) / len(rs)
        best = max(TAUS, key=lambda t: (scores[t], t))
        json.dump({"tau": best, "validation_acc": scores, "rule": "予算200・2000の検証用問題での正答率が最大"},
                  open(f"{OUT}/memory_tau.json", "w"), ensure_ascii=False)
    if not os.path.exists(f"{OUT}/ewc_lambda.json"):
        scores = {}
        for lam in [0] + EWC_LAMBDAS:
            d = json.load(open(f"{OUT}/calibration/ewc_{lam:g}.json"))
            after = d["days"][0]["after"]
            scores[lam] = (sum(after["id"]) + sum(after["ood_depth"])) / (len(after["id"]) + len(after["ood_depth"]))
        best = max(EWC_LAMBDAS, key=lambda l: (scores[l], -l))
        json.dump({"lambda": best, "validation_score": {str(k): v for k, v in scores.items()},
                   "rule": "1晩の睡眠後の LLM 単体の（ID ＋ OOD-深さ）正答率が最大"},
                  open(f"{OUT}/ewc_lambda.json", "w"), ensure_ascii=False)
    print("tau*", tau_star(), "lambda*", ewc_star(), flush=True)


def run(workers):
    _pool([j for j in jobs_stage1() if not os.path.exists(job_path(j))], workers)
    decide_stage1()
    js = [j for j in jobs_stage2() if not os.path.exists(job_path(j))]
    print(f"stage2: {len(js)} jobs", flush=True)
    _pool(js, workers)
    print("PHASE15_RUN_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report", "jobs"])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.cmd == "run":
        run(args.workers)
    elif args.cmd == "jobs":
        print(len(jobs_stage1()), len(jobs_stage2()))
    else:
        from evaluation.phase15_report import report
        report()
