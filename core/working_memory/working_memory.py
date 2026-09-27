"""Working Memory: いま考えている問題の状態を、容量制限つきで保持する。

保持するもの: 目標・観測・仮説（状態とスコア）・反証・シミュレーション結果・次の行動。
容量を超えたら「重要度 = 説明率 + もっともらしさ + 新規性」の低い仮説から捨てる。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class Hypothesis:
    program: tuple
    source: str               # llm / memory / expert / compose / rag / repair
    logprob: float | None = None
    fit: float | None = None  # Simulator で測った説明率
    novelty: float | None = None
    status: str = "open"      # open / verified / refuted

    def importance(self) -> float:
        lp = 0.0 if self.logprob is None else max(-10.0, self.logprob) / 10.0
        return (self.fit or 0.0) * 2 + lp + 0.5 * (self.novelty or 0.0)


@dataclass
class WorkingMemory:
    goal: str
    observations: list
    capacity: int = 64
    hypotheses: dict = field(default_factory=dict)  # program -> Hypothesis
    refuted: int = 0
    fragments: Counter = field(default_factory=Counter)
    log: list = field(default_factory=list)
    next_action: str = ""

    def add(self, h: Hypothesis) -> bool:
        """新しい仮説なら追加して True。"""
        if h.program in self.hypotheses:
            old = self.hypotheses[h.program]
            if old.logprob is None and h.logprob is not None:
                old.logprob = h.logprob
            return False
        self.hypotheses[h.program] = h
        return True

    def mark(self, program, fit: float):
        h = self.hypotheses[program]
        h.fit = fit
        h.status = "verified" if fit == 1.0 else "refuted"
        for p in set(program):
            self.fragments[p] += 1.0 + fit
        if fit < 1.0:
            self.refuted += 1
        self._compress()

    def verified(self) -> list:
        return [h for h in self.hypotheses.values() if h.status == "verified"]

    def fragment_stats(self) -> Counter:
        """検証した仮説に現れたプリミティブの頻度（組み合わせ探索の語彙になる）。
        圧縮で仮説本体を捨てても、この統計は残る。"""
        return Counter(self.fragments)

    def best_partial(self, k: int) -> list:
        cand = [h for h in self.hypotheses.values() if h.status == "refuted" and (h.fit or 0) > 0]
        return sorted(cand, key=lambda h: -h.importance())[:k]

    def _compress(self):
        if len(self.hypotheses) <= self.capacity:
            return
        # 検証済みは必ず残し、反証済みを重要度の低い順に捨てる（統計は fragment_stats に残る前提で軽量化）
        refuted = sorted((h for h in self.hypotheses.values() if h.status == "refuted"),
                         key=lambda h: h.importance())
        for h in refuted[: len(self.hypotheses) - self.capacity]:
            del self.hypotheses[h.program]

    def note(self, msg: str):
        self.log.append(msg)
