"""results/*.json から docs/04_results.md の表を生成する。

python -m evaluation.report
"""
import json
import os

from evaluation.benchmarks import SPLITS


def _pct(x):
    return f"{100 * x:.0f}%"


def main_tables(path):
    d = json.load(open(path))
    res = d["results"]
    lines = [f"LLM: Tiny GPT（非埋め込みパラメータ {d['n_params'] / 1e6:.2f}M）。各分割100問、1問あたりの Simulator 予算は2,000回。\n"]
    lines.append("### 正答率（括弧内は95%信頼区間）\n")
    lines.append("| 構成 | " + " | ".join(n for _, n in SPLITS) + " |")
    lines.append("|---|" + "---|" * len(SPLITS))
    for name, r in res.items():
        cells = []
        for s, _ in SPLITS:
            m = r[s]["summary"]
            cells.append(f"{_pct(m['acc'])} ({_pct(m['ci'][0])}–{_pct(m['ci'][1])})")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("\n### 計算量（1問あたりの平均。全分割の平均）\n")
    lines.append("| 構成 | LLM MFLOPs | Simulator 回数 | 実時間 ms | 幻覚率※ | OOD 正答率 | OOD 正答 / LLM GFLOP |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, r in res.items():
        ms = [r[s]["summary"] for s, _ in SPLITS]
        f = sum(m["gflops"] for m in ms) / len(ms)
        sim = sum(m["sim"] for m in ms) / len(ms)
        t = sum(m["ms"] for m in ms) / len(ms)
        h = sum(m["halluc"] for m in ms) / len(ms)
        ood = (r["ood_combo"]["summary"]["acc"] + r["ood_depth"]["summary"]["acc"]) / 2
        eff = f"{ood / f:.1f}" if f > 0 else "—（LLM 不使用）"
        lines.append(f"| {name} | {f * 1000:.0f} | {sim:.0f} | {t:.1f} | {_pct(h)} | {_pct(ood)} | {eff} |")
    lines.append("\n※ 幻覚率 = 最終回答が、見えている例すら説明できていない割合。")
    return "\n".join(lines)


def discovery_table(path):
    d = json.load(open(path))
    lines = ["| 構成 | 特定できた割合（平均） | シードごと | LLM GFLOPs | Simulator 回数 |", "|---|---|---|---|---|"]
    for name, r in d.items():
        per = ", ".join(_pct(x) for x in r["identified_per_seed"])
        lines.append(f"| {name} | {_pct(r['identified_mean'])} | {per} | {r['gflops']:.2f} | {r['sim_calls']:.0f} |")
    return "\n".join(lines)


def quant_table(path):
    d = json.load(open(path))
    lines = ["| | サイズ KB | 16本サンプル ms | tokens/s | A: ID | A: OOD-深さ | 提案: ID | 提案: OOD-組み合わせ | 提案: OOD-深さ |",
             "|---|---|---|---|---|---|---|---|---|"]
    for k, e in d.items():
        a, p = e["A: Dense LLM"], e["Proposed"]
        lines.append(f"| {k} | {e['size_kb']:.0f} | {e['sample16_ms']:.1f} | {e['tokens_per_s']:.0f} | "
                     f"{_pct(a['id'])} | {_pct(a['ood_depth'])} | {_pct(p['id'])} | {_pct(p['ood_combo'])} | {_pct(p['ood_depth'])} |")
    return "\n".join(lines)


if __name__ == "__main__":
    parts = {}
    if os.path.exists("results/main.json"):
        parts["main"] = main_tables("results/main.json")
    if os.path.exists("results/discovery.json"):
        parts["discovery"] = discovery_table("results/discovery.json")
    if os.path.exists("results/quantization.json"):
        parts["quant"] = quant_table("results/quantization.json")
    json.dump(parts, open("results/tables.json", "w"), ensure_ascii=False)
    for v in parts.values():
        print(v, "\n")
