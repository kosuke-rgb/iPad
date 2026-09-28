"""タスク生成。学習分布（ID）と分布外（OOD）を明確に分ける。

- ID       : 深さ1〜2 のプログラム（ただし一部のプリミティブ対は学習から除外）
- OOD-combo: 学習で一度も見せていないプリミティブ対からなる深さ2
- OOD-depth: 深さ3（学習では深さ3を一切見せない）
- Few-shot : ID と同じ分布だが、見える例が2個しかない
"""
from __future__ import annotations

import itertools
import random
import zlib
from dataclasses import dataclass, field

from .dsl import PRIM_NAMES, Program, safe_run

PROBE_SEED = 12345


def random_input(rng: random.Random) -> list:
    return [rng.randint(0, 9) for _ in range(rng.randint(3, 6))]


def _probe_inputs(n=40):
    rng = random.Random(PROBE_SEED)
    return [random_input(rng) for _ in range(n)]


PROBES = _probe_inputs()


def signature(program: Program, inputs=PROBES):
    """振る舞いの指紋。これが同じなら（ほぼ）同じ関数とみなす。"""
    return tuple(tuple(o) if (o := safe_run(program, x)) is not None else None for x in inputs)


def is_heldout_pair(a: str, b: str) -> bool:
    """学習から除外する順序付きプリミティブ対（約20%）。決定的に決まる。"""
    return zlib.crc32(f"{a}|{b}".encode()) % 5 == 0


class Signatures:
    """深さごとの振る舞いの指紋を前計算（冗長なタスクを除くため）。"""

    def __init__(self):
        self.depth1 = {signature((a,)) for a in PRIM_NAMES}
        self.train_depth2 = set()
        self.all_upto2 = set(self.depth1)
        for a, b in itertools.product(PRIM_NAMES, repeat=2):
            s = signature((a, b))
            self.all_upto2.add(s)
            if not is_heldout_pair(a, b):
                self.train_depth2.add(s)
        identity = signature(())
        self.depth1.add(identity)
        self.all_upto2.add(identity)


_SIGS: Signatures | None = None


def sigs() -> Signatures:
    global _SIGS
    if _SIGS is None:
        _SIGS = Signatures()
    return _SIGS


@dataclass
class Task:
    program: Program
    train: list  # [(input, output), ...] 見える例
    test: list   # [(input, output), ...] 評価用（エージェントには入力だけ渡す）
    split: str = ""
    meta: dict = field(default_factory=dict)


def _examples(rng, program, n):
    out = []
    for _ in range(n * 20):
        x = random_input(rng)
        y = safe_run(program, x)
        if y is None:
            continue
        out.append((x, y))
        if len(out) == n:
            return out
    return None


def _informative(train) -> bool:
    nonempty = sum(1 for _, y in train if y)
    changed = any(x != y for x, y in train)
    return nonempty >= len(train) - 1 and changed


def sample_program(rng: random.Random, split: str) -> Program | None:
    s = sigs()
    if split == "train":
        if rng.random() < 0.3:
            p = (rng.choice(PRIM_NAMES),)
            return p if signature(p) != signature(()) else None
        a, b = rng.choice(PRIM_NAMES), rng.choice(PRIM_NAMES)
        if is_heldout_pair(a, b):
            return None
        sig = signature((a, b))
        return (a, b) if sig not in s.depth1 else None
    if split in ("id", "few_shot"):
        a, b = rng.choice(PRIM_NAMES), rng.choice(PRIM_NAMES)
        if is_heldout_pair(a, b) or signature((a, b)) in s.depth1:
            return None
        return (a, b)
    if split == "ood_combo":
        a, b = rng.choice(PRIM_NAMES), rng.choice(PRIM_NAMES)
        sig = signature((a, b))
        if not is_heldout_pair(a, b) or sig in s.depth1 or sig in s.train_depth2:
            return None
        return (a, b)
    if split == "ood_depth":
        p = tuple(rng.choice(PRIM_NAMES) for _ in range(3))
        return p if signature(p) not in s.all_upto2 else None
    raise ValueError(split)


def make_task(rng: random.Random, split: str) -> Task:
    n_train = 2 if split == "few_shot" else 3
    while True:
        p = sample_program(rng, split)
        if p is None:
            continue
        ex = _examples(rng, p, n_train + 3)
        if ex is None or not _informative(ex[:n_train]):
            continue
        return Task(program=p, train=ex[:n_train], test=ex[n_train:], split=split)


def make_benchmark(split: str, n: int, seed: int) -> list:
    rng = random.Random(seed)
    return [make_task(rng, split) for _ in range(n)]


def make_related_benchmark(base_programs, n: int, seed: int) -> list:
    """OOD-関連: 以前に見た深さ2の規則に、1手順を前後いずれかに足した深さ3の問題。

    「経験が役に立ちうる」状況で記憶の効果を測るための分割。
    記憶に有利になるよう作ってあるので、他の分割とは分けて報告する。
    """
    rng = random.Random(seed)
    s = sigs()
    out = []
    while len(out) < n:
        base = tuple(rng.choice(base_programs))
        extra = rng.choice(PRIM_NAMES)
        p = (extra,) + base if rng.random() < 0.5 else base + (extra,)
        if signature(p) in s.all_upto2:
            continue
        ex = _examples(rng, p, 6)
        if ex is None or not _informative(ex[:3]):
            continue
        out.append(Task(program=p, train=ex[:3], test=ex[3:], split="ood_related",
                        meta={"base": base}))
    return out
