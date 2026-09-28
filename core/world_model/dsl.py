"""World Model の「世界」: 整数リストを変換する小さな DSL。

プログラムはプリミティブ（基本操作）を左から順に適用するパイプライン。
例: ("add1", "sort") は「全要素に1を足してから昇順に並べる」。
任意コード実行は行わず、この DSL の解釈だけで仮説を検証するので安全。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

MAX_LEN = 16
MAX_ABS = 10_000

Program = tuple  # tuple[str, ...]


@dataclass(frozen=True)
class Prim:
    name: str
    kind: str  # "map" | "filter" | "struct"
    fn: Callable[[list], list]


def _dedup(xs):
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _cumsum(xs):
    out, s = [], 0
    for x in xs:
        s += x
        out.append(s)
    return out


def _build():
    p = []
    for k in range(1, 6):
        p.append(Prim(f"add{k}", "map", lambda xs, k=k: [x + k for x in xs]))
    for k in (2, 3):
        p.append(Prim(f"mul{k}", "map", lambda xs, k=k: [x * k for x in xs]))
    for k in (2, 3):
        p.append(Prim(f"mod{k}", "map", lambda xs, k=k: [x % k for x in xs]))
    for k in (2, 4, 6):
        p.append(Prim(f"gt{k}", "filter", lambda xs, k=k: [x for x in xs if x > k]))
    for k in (3, 5, 7):
        p.append(Prim(f"lt{k}", "filter", lambda xs, k=k: [x for x in xs if x < k]))
    p.append(Prim("evens", "filter", lambda xs: [x for x in xs if x % 2 == 0]))
    p.append(Prim("odds", "filter", lambda xs: [x for x in xs if x % 2 == 1]))
    for k in (2, 3, 4):
        p.append(Prim(f"take{k}", "filter", lambda xs, k=k: xs[:k]))
    for k in (1, 2):
        p.append(Prim(f"drop{k}", "filter", lambda xs, k=k: xs[k:]))
    p.append(Prim("dedup", "filter", _dedup))
    p.append(Prim("rev", "struct", lambda xs: xs[::-1]))
    p.append(Prim("sort", "struct", sorted))
    p.append(Prim("sortd", "struct", lambda xs: sorted(xs, reverse=True)))
    p.append(Prim("rot1", "struct", lambda xs: xs[1:] + xs[:1]))
    p.append(Prim("cumsum", "struct", _cumsum))
    return {q.name: q for q in p}


PRIMS: dict[str, Prim] = _build()
PRIM_NAMES: list[str] = list(PRIMS)


class ExecError(Exception):
    pass


def run(program: Program, xs: list) -> list:
    out = list(xs)
    for name in program:
        out = PRIMS[name].fn(out)
        if len(out) > MAX_LEN or any(abs(v) > MAX_ABS for v in out):
            raise ExecError("値が範囲外")
    return out


def safe_run(program: Program, xs: list):
    try:
        return run(program, xs)
    except (ExecError, KeyError):
        return None


def to_str(program: Program) -> str:
    return "|".join(program)


def from_str(s: str) -> Program | None:
    parts = tuple(t for t in s.strip().split("|") if t)
    if not parts or any(t not in PRIMS for t in parts):
        return None
    return parts
