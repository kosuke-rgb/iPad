"""フェーズ1.5の集計。results/phase1_5/report.md を作る。"""
import glob
import json
import os

from evaluation.benchmarks import SPLITS
from evaluation.metrics import sign_test, wilson
from evaluation.phase15 import BUDGETS, OUT, SEEDS, SLEEP_CONDITIONS, job_path

SHORT = {"id": "ID", "few_shot": "Few-shot", "ood_combo": "OOD-組", "ood_depth": "OOD-深", "ood_related": "OOD-関"}


def ci(xs):
    k, n = sum(xs), len(xs)
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.1f}% ({100 * lo:.0f}–{100 * hi:.0f})"


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def paired(a, b):
    w = sum(x and not y for x, y in zip(a, b))
    l_ = sum(y and not x for x, y in zip(a, b))
    return w, l_, sign_test(w, l_)


def test_table(L, tests, head="比較"):
    """tests: [(ラベル, 差pt, w, l, p)] → Holm 補正つきの表。"""
    adj = holm([t[4] for t in tests])
    L += [f"| {head} | 差 | 勝ち / 負け | p | Holm 補正後 p | 判定 |", "|---|---|---|---|---|---|"]
    for (lab, d, w, l_, p), pa in zip(tests, adj):
        L.append(f"| {lab} | {d:+.1f} pt | +{w} / −{l_} | {p:.2g} | {pa:.2g} | {'**差あり**' if pa < 0.05 else '差なし'} |")


def _load(path):
    return json.load(open(path)) if os.path.exists(path) else None


# ------------------------------------------------------------------ agent
def agent_rows(name, b, splits=None, col=0):
    out = []
    for s in SEEDS:
        d = _load(job_path(("agent", name, b, s)))
        if d is None:
            return None
        for sp, _ in SPLITS:
            if splits is None or sp in splits:
                out += [r[col] for r in d[sp]]
    return out


def phase1_rows(b):
    out = []
    for s in SEEDS:
        d = _load(f"results/phase1/agent/0.8M__Proposed__b{b}__s{s}.json")
        if d is None:
            return None
        for sp, _ in SPLITS:
            out += [r[0] for r in d[sp]]
    return out


def section_baseline(L):
    L += ["## 0. 新しい既定設定（Default v2: Experts OFF・Repair OFF）のベースライン【確定: 3シード × 4,500問】", "",
          "| 予算 | " + " | ".join(SHORT[s] for s, _ in SPLITS) + " | 全体 | LLM MFLOPs | 実行回数 | ms |",
          "|---|" + "---|" * (len(SPLITS) + 4)]
    tests = []
    for b in BUDGETS:
        a = agent_rows("default", b)
        if a is None:
            continue
        cells = [ci(agent_rows("default", b, {s})) for s, _ in SPLITS]
        L.append(f"| {b} | " + " | ".join(cells) + f" | {ci(a)} | {1000 * mean(agent_rows('default', b, col=7)):.0f} | "
                 f"{mean(agent_rows('default', b, col=5)):.0f} | {mean(agent_rows('default', b, col=6)):.0f} |")
        old = phase1_rows(b)
        if old:
            w, l_, p = paired(a, old)
            tests.append((f"予算 {b}: Default v2 vs 旧既定（フェーズ1の Proposed）", 100 * (mean(a) - mean(old)), w, l_, p))
    if tests:
        L.append("")
        test_table(L, tests)
    L.append("")


def section_memory(L):
    tau = _load(f"{OUT}/memory_tau.json")
    L += ["## 3. 長期記憶の再設計【確定: 3シード × 4,500問】", ""]
    if tau:
        L += ["しきい値は検証用の問題（評価用とは別のシード、予算200・2000）で決めた。", "",
              "| しきい値 | 検証用の正答率 |", "|---|---|"]
        for t, v in tau["validation_acc"].items():
            L.append(f"| {t}{' ← 採用' if float(t) == tau['tau'] else ''} | {100 * v:.2f}% |")
        L += ["", f"新設計（Memory v3）: 類似度 ≥ {tau['tau']} のときだけ検索する。予算50以下では検索しない。"
                  "睡眠で再生する材料としての保存は常に行う。", ""]
    L += ["| 予算 | 旧設計（v2、しきい値0.8） | 新設計（v3） | 記憶なし | OOD-深（旧 / 新 / なし） |", "|---|---|---|---|---|"]
    tests = []
    for b in BUDGETS:
        o, n, z = agent_rows("default", b), agent_rows("new_memory", b), agent_rows("no_memory", b)
        if not (o and n and z):
            continue
        dp = [agent_rows(x, b, {"ood_depth"}) for x in ("default", "new_memory", "no_memory")]
        L.append(f"| {b} | {ci(o)} | {ci(n)} | {ci(z)} | " + " / ".join(f"{100 * mean(x):.1f}%" for x in dp) + " |")
        for lab, x, y in (("新 vs 旧", n, o), ("新 vs なし", n, z), ("旧 vs なし", o, z)):
            w, l_, p = paired(x, y)
            tests.append((f"予算 {b}: {lab}", 100 * (mean(x) - mean(y)), w, l_, p))
    if tests:
        L.append("")
        test_table(L, tests)
    L.append("")


# ------------------------------------------------------------------ world model
def section_wm(L):
    runs = [_load(job_path(("wm", s))) for s in SEEDS]
    if any(r is None for r in runs):
        return
    L += ["## 1a. 学習した世界モデルの再現【確定: 3シード × 900問（ID / OOD-組み合わせ / OOD-深さ 各300問）】", ""]
    step = mean([r["accuracy"]["step_acc"] for r in runs])
    roll = {d: mean([r["accuracy"]["rollout_acc"][d] for r in runs]) for d in ("1", "2", "3")}
    prec = sum(r["imag_and_real_fit"] for r in runs) / sum(r["imag_fit"] for r in runs)
    rec = sum(r["imag_and_real_fit"] for r in runs) / sum(r["real_fit"] for r in runs)
    wmf = mean([x for r in runs for x in r["wm_flops"]]) / 1e9
    llf = mean([x for r in runs for x in r["llm_flops"]]) / 1e9
    L += [f"予測の正確さ: 1手順 {100 * step:.1f}%、ロールアウト 深さ1 {100 * roll['1']:.1f}% / 深さ2 {100 * roll['2']:.1f}% / "
          f"深さ3 {100 * roll['3']:.1f}%。想像で合った候補の適合率 {100 * prec:.2f}%、再現率 {100 * rec:.1f}%。"
          f"計算量: 世界モデル {wmf:.1f} GFLOPs/問、候補づくりの LLM {llf:.2f} GFLOPs/問。", "",
          "| 現実の実験の上限 | 現実のみ | 想像→現実 | 想像のみ（0回） | 完璧な想像（上限） | 現実の実験回数（現実のみ / 想像→現実） |",
          "|---|---|---|---|---|---|"]
    tests = []
    for R in ("1", "2", "5", "10", "30", "10000"):
        col = {c: [x for r in runs for x in r["per_task"][c][R]] for c in r["per_task"]}
        acc = {c: [int(x[0]) for x in v] for c, v in col.items()}
        used = {c: mean([x[1] for x in v]) for c, v in col.items()}
        L.append(f"| {'無制限' if R == '10000' else R} | {ci(acc['real_only'])} | {ci(acc['imagine_then_act'])} | "
                 f"{ci(acc['imagination_only'])} | {ci(acc['perfect_imagination'])} | "
                 f"{used['real_only']:.1f} / {used['imagine_then_act']:.1f} |")
        w, l_, p = paired(acc["imagine_then_act"], acc["real_only"])
        tests.append((f"上限 {R}: 想像→現実 vs 現実のみ", 100 * (mean(acc['imagine_then_act']) - mean(acc['real_only'])), w, l_, p))
    acc1 = {c: [int(x[0]) for r in runs for x in r["per_task"][c]["1"]] for c in ("imagination_only", "real_only")}
    w, l_, p = paired(acc1["imagination_only"], acc1["real_only"])
    tests.append(("想像のみ（0回） vs 現実のみ（上限1）", 100 * (mean(acc1['imagination_only']) - mean(acc1['real_only'])), w, l_, p))
    L.append("")
    test_table(L, tests)
    L.append("")


# ------------------------------------------------------------------ sleep
def sleep_runs(cond):
    runs = [_load(job_path(("sleep", cond, s))) for s in SEEDS]
    return None if any(r is None for r in runs) else runs


def day_rows(runs, days, col=0):
    return [row[col] for r in runs for d in days for row in r["days"][d]["rows"]]


def test_rows(runs, key, after_day):
    return [x for r in runs for x in (r["before"][key] if after_day < 0 else r["days"][after_day]["after"][key])]


def section_sleep(L, conds, title, ref, extra_pairs):
    runs = {c: sleep_runs(c) for c in conds}
    runs = {c: v for c, v in runs.items() if v}
    if not runs:
        return
    days = len(next(iter(runs.values()))[0]["days"])
    L += [title, "",
          "その日の新しい問題の正答率（3シード合計 450問/日）と、各夜の後の LLM 単体の固定テスト（ID・OOD-深さ 各900問）。", "",
          "| 条件 | " + " | ".join(f"{d + 1}日目" for d in range(days)) + " | 直感で解けた割合（4日目） | 実行回数（4日目） | 学習ステップ/日 | ID: 前 → 最終夜の後 | OOD-深: 前 → 最終夜の後 |",
          "|---|" + "---|" * (days + 5)]
    for c in conds:
        if c not in runs:
            continue
        R = runs[c]
        cells = [ci(day_rows(R, [d])) for d in range(days)]
        L.append(f"| {SLEEP_CONDITIONS[c]} | " + " | ".join(cells) +
                 f" | {100 * mean(day_rows(R, [days - 1], 1)):.1f}% | {mean(day_rows(R, [days - 1], 2)):.0f} | "
                 f"{mean([d['train_steps'] for r in R for d in r['days']]):.0f} | "
                 f"{100 * mean(test_rows(R, 'id', -1)):.1f}% → {ci(test_rows(R, 'id', days - 1))} | "
                 f"{100 * mean(test_rows(R, 'ood_depth', -1)):.1f}% → {ci(test_rows(R, 'ood_depth', days - 1))} |")
    later = list(range(1, days))
    tests = []
    for a, b in [(c, ref) for c in conds if c != ref and c in runs] + extra_pairs:
        if a not in runs or b not in runs:
            continue
        x, y = day_rows(runs[a], later), day_rows(runs[b], later)
        w, l_, p = paired(x, y)
        tests.append((f"新しい問題（2〜{days}日目）: {SLEEP_CONDITIONS[a]} vs {SLEEP_CONDITIONS[b]}", 100 * (mean(x) - mean(y)), w, l_, p))
        for key, nm in (("id", "昔の問題 ID"), ("ood_depth", "LLM 単体 OOD-深")):
            x, y = test_rows(runs[a], key, days - 1), test_rows(runs[b], key, days - 1)
            w, l_, p = paired(x, y)
            tests.append((f"{nm}（最終夜の後）: {SLEEP_CONDITIONS[a]} vs {SLEEP_CONDITIONS[b]}", 100 * (mean(x) - mean(y)), w, l_, p))
    if tests:
        L.append("")
        test_table(L, tests)
    L.append("")
    L += ["日ごとの LLM 単体（ID / OOD-深さ、各夜の後）:", "",
          "| 条件 | 前 | " + " | ".join(f"{d + 1}日目の夜の後" for d in range(days)) + " |", "|---|---|" + "---|" * days]
    for c in conds:
        if c not in runs:
            continue
        R = runs[c]
        cells = [f"{100 * mean(test_rows(R, 'id', d)):.1f}% / {100 * mean(test_rows(R, 'ood_depth', d)):.1f}%" for d in range(days)]
        L.append(f"| {SLEEP_CONDITIONS[c]} | {100 * mean(test_rows(R, 'id', -1)):.1f}% / {100 * mean(test_rows(R, 'ood_depth', -1)):.1f}% | "
                 + " | ".join(cells) + " |")
    L.append("")


def report():
    L = ["# フェーズ1.5: 暫定結果の再現と記憶の再設計", "",
         "共通: 全条件で同じ問題セット。差は正確 McNemar 検定（不一致ペアの符号検定）、表ごとに Holm 法で補正し、補正後 p < 0.05 のみ「差あり」。"
         "LLM は 0.8M（15,000ステップ）。", ""]
    section_baseline(L)
    section_wm(L)
    section_sleep(L, ["a_no_sleep", "b_awake_experience", "b2_awake_mixture", "c_sleep", "d_sleep_wm_dreams"],
                  "## 2. 睡眠の対照実験（再現を兼ねる）【確定: 3シード】", "a_no_sleep",
                  [("c_sleep", "b_awake_experience"), ("c_sleep", "b2_awake_mixture"), ("d_sleep_wm_dreams", "c_sleep")])
    section_memory(L)
    lam = _load(f"{OUT}/ewc_lambda.json")
    if lam:
        L += [f"EWC の強さ λ は検証用のシードで決めた（{lam['rule']}）: " +
              ", ".join(f"λ={k}: {100 * v:.1f}%" for k, v in lam["validation_score"].items()) + f" → λ = {lam['lambda']:g}", ""]
    section_sleep(L, ["f_base", "f_more_rehearsal", "f_dream_rehearsal", "f_ewc"],
                  "## 4. 破滅的忘却への対策【確定: 3シード】（基準: 有限の復習バッファ500例）", "f_base", [])
    text = "\n".join(L)
    open(f"{OUT}/report.md", "w").write(text)
    print(text)
