"""Sparse Experts: 特定の種類の問題だけを得意とする専門モジュール。

各 Expert は「ゲート」（特徴量から自分が役立ちそうか判定）と、
狭い部分空間での探索を持つ。Router はゲートが開いた Expert だけを起動する。
Expert の候補は Simulator で検証されるので、その分の計算量も記録される。
"""
from __future__ import annotations

import itertools

from core.world_model.dsl import PRIMS

MAPS = [n for n, p in PRIMS.items() if p.kind == "map"]
FILTERS = [n for n, p in PRIMS.items() if p.kind == "filter"]
PERMS = ["rev", "sort", "sortd", "rot1"]


class Expert:
    name = "expert"

    def gate(self, f: dict) -> bool:
        raise NotImplementedError

    def propose(self) -> list:
        raise NotImplementedError


def _upto2(units):
    return [(a,) for a in units] + list(itertools.product(units, repeat=2))


class ArithmeticExpert(Expert):
    """要素ごとの算術変換（長さが変わらない問題）。"""
    name = "arithmetic"

    def gate(self, f):
        return f["same_len"] and not f["perm"]

    def propose(self):
        return _upto2(MAPS)


class OrderingExpert(Expert):
    """並べ替え（要素の多重集合が変わらない問題）。"""
    name = "ordering"

    def gate(self, f):
        return f["perm"]

    def propose(self):
        return _upto2(PERMS)


class SelectionExpert(Expert):
    """選択・切り出し（出力が入力の部分列になる問題）。"""
    name = "selection"

    def gate(self, f):
        return f["subseq"] and not f["same_len"]

    def propose(self):
        return _upto2(FILTERS)


class AccumulationExpert(Expert):
    """累積和を含む変換。"""
    name = "accumulation"

    def gate(self, f):
        return f["same_len"] and f["out_bigger"] and not f["perm"]

    def propose(self):
        return [("cumsum",)] + [(m, "cumsum") for m in MAPS] + [("cumsum", m) for m in MAPS]


ALL_EXPERTS = [ArithmeticExpert(), OrderingExpert(), SelectionExpert(), AccumulationExpert()]
