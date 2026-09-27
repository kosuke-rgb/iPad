"""Hypothesis Generator: 既存知識から「足りない部分」を推定し、複数の仮説を作る。

1. LLM から多様な仮説をサンプリングする
2. 仮説に現れた部品（プリミティブ）と、長期記憶のマクロを語彙として、
   新しい組み合わせを「もっともらしい順」に列挙する（学習時に見ていない深さ・組み合わせに届く）
3. 惜しい仮説（部分的に観測を説明する）を1箇所だけ変えて修復する
"""
from __future__ import annotations

import heapq
import itertools
import math

from core.world_model.dsl import PRIM_NAMES

MAX_DEPTH = 4


def compose(fragments, macros, max_units=3, limit=4000):
    """部品の頻度を確率とみなし、対数確率の高い順に組み合わせを返す。"""
    units = [(f,) for f in fragments] + [tuple(m) for m in macros]
    if not units:
        return []
    weights = {}
    total = sum(fragments.values()) or 1.0
    for f, c in fragments.items():
        weights[(f,)] = c / total
    for m in macros:
        weights[tuple(m)] = max(weights.get(tuple(m), 0.0), 0.15)
    logw = {u: math.log(max(w, 1e-6)) for u, w in weights.items()}
    # プリミティブ1つごとのペナルティ（短い説明を優先）。部品単位で数えると、
    # マクロを使った長いプログラムが不当に安く見え、探索予算を浪費する（実験で確認）。
    penalty = math.log(0.3)
    heap = []
    for n in range(1, max_units + 1):
        for combo in itertools.product(units, repeat=n):
            prog = tuple(itertools.chain.from_iterable(combo))
            if len(prog) > MAX_DEPTH:
                continue
            score = sum(logw[u] for u in combo) + penalty * len(prog)
            heapq.heappush(heap, (-score, prog))
    seen, out = set(), []
    while heap and len(out) < limit:
        _, p = heapq.heappop(heap)
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def mutations(program):
    """1箇所だけ変えた仮説（置換・挿入・削除）。"""
    out = []
    for i in range(len(program)):
        for n in PRIM_NAMES:
            if n != program[i]:
                out.append(program[:i] + (n,) + program[i + 1:])
    if len(program) < MAX_DEPTH:
        for i in range(len(program) + 1):
            for n in PRIM_NAMES:
                out.append(program[:i] + (n,) + program[i:])
    if len(program) > 1:
        for i in range(len(program)):
            out.append(program[:i] + program[i + 1:])
    return out
