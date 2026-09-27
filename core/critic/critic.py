"""Critic: 自分の出力を疑うモジュール。

- 観測との矛盾（Simulator による実行結果の照合）
- 部分的な一致度（どこまで合っているか）
- 冗長・過剰な仮定（同じ操作の重複、打ち消し合う操作、より短い説明の存在）
- 曖昧さ（観測を説明できる仮説が複数あり、未知の入力で予測が割れる）
"""
from __future__ import annotations

from core.world_model.dsl import PRIMS, safe_run

IDEMPOTENT = {"sort", "sortd", "dedup", "evens", "odds"} | {n for n in PRIMS if n[:2] in ("gt", "lt", "mo")}
ORDER_OVERRIDES = {"sort", "sortd"}


def soft_similarity(out, target) -> float:
    if out is None:
        return 0.0
    if out == target:
        return 1.0
    if not out and not target:
        return 1.0
    len_score = 1 - abs(len(out) - len(target)) / max(len(out), len(target), 1)
    pos = sum(a == b for a, b in zip(out, target)) / max(len(target), 1)
    common = len(set(out) & set(target)) / max(len(set(out) | set(target)), 1)
    return 0.3 * len_score + 0.4 * pos + 0.3 * common


class Critic:
    def soft_score(self, outs, examples) -> float:
        return sum(soft_similarity(o, y) for o, (_, y) in zip(outs, examples)) / len(examples)

    def flags(self, program) -> list:
        issues = []
        for a, b in zip(program, program[1:]):
            if a == b and a in IDEMPOTENT:
                issues.append(f"冗長: {a} の重複")
            if a == "rev" and b == "rev":
                issues.append("打ち消し: rev|rev")
            if a in ORDER_OVERRIDES | {"rev", "rot1"} and b in ORDER_OVERRIDES:
                issues.append(f"無意味: {a} の後に {b}")
        return issues

    def rank_key(self, h):
        """Occam の剃刀: 短く、問題点がなく、もっともらしい説明を優先。"""
        lp = h.logprob if h.logprob is not None else -50.0
        return (len(h.program), len(self.flags(h.program)), -lp)

    def choose(self, verified: list, test_inputs) -> tuple:
        """(採用する仮説, 曖昧かどうか)"""
        ranked = sorted(verified, key=self.rank_key)
        best = ranked[0]
        rivals = [h for h in ranked[1:] if len(h.program) == len(best.program)]
        mine = [safe_run(best.program, x) for x in test_inputs]
        ambiguous = any([safe_run(h.program, x) for x in test_inputs] != mine for h in rivals)
        return best, ambiguous
