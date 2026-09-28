"""Simulator: 仮説（プログラム）を実際に実行して観測と照合する。

「考えた結果をそのまま正解とみなさない」ための中核。
呼び出し回数を ComputeMeter で数え、予算を超えたら実行を拒否する。
"""
from __future__ import annotations

from core.llm.backend import ComputeMeter
from core.world_model.dsl import Program, safe_run


class BudgetExceeded(Exception):
    pass


class Simulator:
    def __init__(self, meter: ComputeMeter, budget: int):
        self.meter = meter
        self.budget = budget
        self._cache: dict = {}

    @property
    def remaining(self) -> int:
        return self.budget - self.meter.sim_calls

    def execute(self, program: Program, examples) -> list:
        """各入力に対する出力（実行失敗は None）。同じ仮説の再実行は数えない。"""
        key = (program, tuple(tuple(x) for x, _ in examples))
        if key in self._cache:
            return self._cache[key]
        if self.meter.sim_calls >= self.budget:
            raise BudgetExceeded
        self.meter.sim_calls += 1
        outs = [safe_run(program, x) for x, _ in examples]
        self._cache[key] = outs
        return outs

    def fits(self, program: Program, examples) -> float:
        """観測をどれだけ説明できるか（0〜1）。1.0 なら全例と一致。"""
        outs = self.execute(program, examples)
        return sum(o == y for o, (_, y) in zip(outs, examples)) / len(examples)
