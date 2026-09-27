"""Novelty Engine: 仮説を「新規性・もっともらしさ・有用性・検証可能性」で評価する。

奇抜なだけの仮説は高く評価しない。総合点は
  novelty × plausibility × utility × testability
の積に近い形にして、どれか一つが低いと全体も低くなるようにする。
"""
from __future__ import annotations

import math

from core.world_model.tasks import signature


def behavioural_similarity(sig_a, sig_b) -> float:
    return sum(a == b for a, b in zip(sig_a, sig_b)) / len(sig_a)


class NoveltyEngine:
    def __init__(self, known_programs=()):
        self._known = {p: signature(p) for p in known_programs}

    def add_known(self, program):
        if program not in self._known:
            self._known[program] = signature(program)

    def novelty(self, program) -> float:
        """既存知識との振る舞いの近さ（1 - 最大類似度）。"""
        if not self._known:
            return 1.0
        s = signature(program)
        return 1.0 - max(behavioural_similarity(s, k) for k in self._known.values())

    def evaluate(self, program, logprob=None, fit=0.0, n_distinguishing=0, n_candidates=1) -> dict:
        nov = self.novelty(program)
        plaus = 1.0 if logprob is None else math.exp(max(logprob, -20) / 4)
        testability = 1.0 if n_candidates <= 1 else min(1.0, n_distinguishing / max(n_candidates - 1, 1))
        score = (0.2 + 0.8 * nov) * (0.2 + 0.8 * plaus) * (0.2 + 0.8 * fit) * (0.5 + 0.5 * testability)
        return {"novelty": nov, "plausibility": plaus, "utility": fit,
                "testability": testability, "score": score}
