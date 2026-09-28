"""フェーズ1の集計。docs/phase1_diagnosis.md の表を作る。"""
import glob
import json

from evaluation.benchmarks import SPLITS
from evaluation.metrics import sign_test, wilson
from evaluation.phase1 import ABLATION_MODEL, ABLATIONS, BUDGETS, DISCOVERY, MODELS, OUT, SEEDS, SOURCES

SPLIT_NAMES = dict(SPLITS)
SHORT = {"id": "ID", "few_shot": "Few-shot", "ood_combo": "OOD-組", "ood_depth": "OOD-深", "ood_related": "OOD-関"}
SRC_NAMES = {"llm_direct": "LLM 直接（1回答）", "llm_sample": "LLM サンプル", "compose": "組み合わせ探索",
             "memory": "長期記憶", "expert": "Experts", "repair": "修復", "enumerate": "総当たり", "other": "その他"}


def load():
    agent, disc = {}, {}
    for f in glob.glob(f"{OUT}/agent/*.json"):
        d = json.load(open(f))
        _, m, c, b, s = d["_job"]
        agent[(m, c, b, s)] = d
    for f in glob.glob(f"{OUT}/discovery/*.json"):
        d = json.load(open(f))
        _, m, c, b, s = d["_job"]
        disc[(c, s)] = d
    return agent, disc


def rows(agent, m, c, b, splits=None):
    """全シードの問題ごとの結果（シード順・分割順に並べる。対応比較のため順序を固定）。"""
    out = []
    for s in SEEDS:
        d = agent.get((m, c, b, s))
        if d is None:
            return None
        for sp, _ in SPLITS:
            if splits is None or sp in splits:
                out += d[sp]
    return out


def acc_ci(rs):
    k, n = sum(r[0] for r in rs), len(rs)
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.1f}% ({100 * lo:.0f}–{100 * hi:.0f})"


def paired(a, b):
    w = sum(x[0] and not y[0] for x, y in zip(a, b))
    l_ = sum(y[0] and not x[0] for x, y in zip(a, b))
    return w, l_, sign_test(w, l_)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def mean(rs, k):
    return sum(r[k] for r in rs) / len(rs)


def report():
    agent, disc = load()
    L = ["# フェーズ1: 診断", "",
         f"条件: 1分割 300問 × 5分割 × {len(SEEDS)}シード = 各条件 4,500問（全条件で同じ問題）。"
         "差の検定は正確 McNemar 検定（不一致ペアの符号検定）。複数比較は Holm 法で補正し、補正後 p < 0.05 のものだけ「差あり」とする。",
         "注: OOD-組み合わせ は学習から除外した組み合わせ自体が少ないため、300問中の規則の種類は85通り程度（例は問題ごとに異なる）。", ""]

    # ---------------- 1. budget sweep
    L += ["## 1. 探索予算のスイープ", "", "### 1a. 全分割の正答率（95%信頼区間）", "",
          "| 構成 | " + " | ".join(f"予算 {b}" for b in BUDGETS) + " |", "|---|" + "---|" * len(BUDGETS)]
    lines_cfg = [("none", "Enumeration", "総当たり探索（LLM なし）")]
    for m in MODELS:
        lines_cfg += [(m, "Proposed", f"提案 LLM {m}"), (m, "Proposed - Memory", f"提案 LLM {m} − 記憶")]
    for m, c, label in lines_cfg:
        cells = []
        for b in BUDGETS:
            rs = rows(agent, m, c, b)
            cells.append(acc_ci(rs) if rs else "—")
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    for sp in ("ood_depth", "few_shot"):
        L += ["", f"### 1b. {SPLIT_NAMES[sp]} の正答率", "",
              "| 構成 | " + " | ".join(f"予算 {b}" for b in BUDGETS) + " |", "|---|" + "---|" * len(BUDGETS)]
        for m, c, label in lines_cfg:
            cells = []
            for b in BUDGETS:
                rs = rows(agent, m, c, b, {sp})
                cells.append(acc_ci(rs) if rs else "—")
            L.append(f"| {label} | " + " | ".join(cells) + " |")
    L += ["", "### 1c. 計算量（1問あたりの平均: LLM MFLOPs / 仮説の実行回数 / 実行時間 ms）", "",
          "| 構成 | " + " | ".join(f"予算 {b}" for b in BUDGETS) + " |", "|---|" + "---|" * len(BUDGETS)]
    for m, c, label in lines_cfg:
        cells = []
        for b in BUDGETS:
            rs = rows(agent, m, c, b)
            cells.append(f"{1000 * mean(rs, 7):.0f} / {mean(rs, 5):.0f} / {mean(rs, 6):.0f}" if rs else "—")
        L.append(f"| {label} | " + " | ".join(cells) + " |")

    # ---------------- 1d. paired tests
    comps = []
    for b in BUDGETS:
        comps += [(b, "LLM 2.7M vs 0.1M（提案）", ("2.7M", "Proposed"), ("0.1M", "Proposed")),
                  (b, "LLM 0.8M vs 0.1M（提案）", ("0.8M", "Proposed"), ("0.1M", "Proposed")),
                  (b, "LLM 2.7M vs 0.8M（提案）", ("2.7M", "Proposed"), ("0.8M", "Proposed"))]
        for m in MODELS:
            comps.append((b, f"記憶あり vs なし（LLM {m}）", (m, "Proposed"), (m, "Proposed - Memory")))
        comps.append((b, "提案 0.8M vs 総当たり", ("0.8M", "Proposed"), ("none", "Enumeration")))
    res = []
    for b, name, A, B in comps:
        a, bb = rows(agent, *A, b), rows(agent, *B, b)
        if not a or not bb:
            continue
        w, l_, p = paired(a, bb)
        res.append((b, name, 100 * (mean(a, 0) - mean(bb, 0)), w, l_, p))
    adj = holm([r[5] for r in res])
    L += ["", "### 1d. 対応比較（全分割 4,500問。Holm 補正は下表の全比較に対して）", "",
          "| 予算 | 比較 | 差 | 勝ち / 負け | p | Holm 補正後 p | 判定 |", "|---|---|---|---|---|---|---|"]
    for (b, name, d, w, l_, p), pa in zip(res, adj):
        L.append(f"| {b} | {name} | {d:+.1f} pt | +{w} / −{l_} | {p:.2g} | {pa:.2g} | {'**差あり**' if pa < 0.05 else '差なし'} |")

    # ---------------- 2. sources
    L += ["", "## 2. 正解の出どころの内訳（解けた問題のうち、各モジュールが出した仮説が正解だった割合）", ""]
    for m, b in ((ABLATION_MODEL, 2000), (ABLATION_MODEL, 50), ("0.1M", 2000)):
        L += [f"### 提案モデル（LLM {m}、予算 {b}）", "",
              "| 分割 | 解けた問題 | " + " | ".join(SRC_NAMES[s] for s in SOURCES if s != "enumerate") + " |",
              "|---|---|" + "---|" * (len(SOURCES) - 1)]
        for sp, _ in SPLITS:
            rs = rows(agent, m, "Proposed", b, {sp})
            if not rs:
                continue
            solved = [r for r in rs if r[0]]
            cells = [f"{100 * sum(r[3] == s for r in solved) / max(1, len(solved)):.0f}%" for s in SOURCES if s != "enumerate"]
            L.append(f"| {SPLIT_NAMES[sp]} | {len(solved)} / {len(rs)} | " + " | ".join(cells) + " |")
        L.append("")

    # ---------------- 3. ablation
    base = rows(agent, ABLATION_MODEL, "Proposed", 2000)
    L += [f"## 3. アブレーションのやり直し（LLM {ABLATION_MODEL}、予算 2000）", "",
          "| 構成 | " + " | ".join(SHORT[s] for s, _ in SPLITS) + " | 全体 | LLM MFLOPs | 実行回数 | ms |",
          "|---|" + "---|" * (len(SPLITS) + 4)]
    abl = []
    for c in ABLATIONS:
        if rows(agent, ABLATION_MODEL, c, 2000) is None:
            continue
        cells = [acc_ci(rows(agent, ABLATION_MODEL, c, 2000, {s})) for s, _ in SPLITS]
        allr = rows(agent, ABLATION_MODEL, c, 2000)
        L.append(f"| {c} | " + " | ".join(cells) + f" | {acc_ci(allr)} | {1000 * mean(allr, 7):.0f} | "
                 f"{mean(allr, 5):.0f} | {mean(allr, 6):.0f} |")
        if c != "Proposed" and base:
            abl.append((c, allr))
    if abl:
        tests = [paired(base, r) for _, r in abl]
        adj = holm([t[2] for t in tests])
        L += ["", "| 外したモジュール | 提案との差（全体） | 提案の勝ち / 負け | p | Holm 補正後 p | 判定 |", "|---|---|---|---|---|---|"]
        for (c, r), (w, l_, p), pa in zip(abl, tests, adj):
            L.append(f"| {c.replace('Proposed - ', '')} | {100 * (mean(base, 0) - mean(r, 0)):+.1f} pt | +{w} / −{l_} | "
                     f"{p:.2g} | {pa:.2g} | {'**差あり**' if pa < 0.05 else '差なし'} |")

    # ---------------- 4. discovery
    if disc:
        L += ["", "## 4. 発見ループのやり直し（ブラックボックス100個 × 3シード = 300個、問い合わせは平均4回/個）", "",
              "| 構成 | 特定できた割合（95%信頼区間） | LLM GFLOPs/シード | 仮説の実行回数/シード |", "|---|---|---|---|"]
        per = {}
        for name, *_ in DISCOVERY:
            ds = [disc.get((name, s)) for s in SEEDS]
            if any(d is None for d in ds):
                continue
            per[name] = [x for d in ds for x in d["per_function"]]
            k, n = sum(per[name]), len(per[name])
            lo, hi = wilson(k, n)
            L.append(f"| {name} | {100 * k / n:.1f}% ({100 * lo:.0f}–{100 * hi:.0f}) | "
                     f"{sum(d['llm_gflops'] for d in ds) / len(ds):.1f} | {sum(d['sim_calls'] for d in ds) / len(ds):.0f} |")
        if "Full" in per:
            others = [n for n in per if n != "Full"]
            tests = []
            for n in others:
                w = sum(a and not b for a, b in zip(per["Full"], per[n]))
                l_ = sum(b and not a for a, b in zip(per["Full"], per[n]))
                tests.append((w, l_, sign_test(w, l_)))
            adj = holm([t[2] for t in tests])
            L += ["", "| 比較 | Full の勝ち / 負け | p | Holm 補正後 p | 判定 |", "|---|---|---|---|---|"]
            for n, (w, l_, p), pa in zip(others, tests, adj):
                L.append(f"| Full vs {n} | +{w} / −{l_} | {p:.2g} | {pa:.2g} | {'**差あり**' if pa < 0.05 else '差なし'} |")
    text = "\n".join(L)
    open(f"{OUT}/report.md", "w").write(text)
    print(text)
