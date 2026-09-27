"""プログラム合成タスク用のトークナイザ。

プロンプト形式: "3 1 4>4 2 5;0 7>1 8=" の後にプログラム "add1|sort\n" が続く。
数字は1桁ずつ、プリミティブ名は1トークン。
"""
import re

from core.world_model.dsl import PRIM_NAMES

SYMBOLS = [" ", ">", ";", "=", "|", "-", "\n"] + [str(d) for d in range(10)]
PAD = "<pad>"
VOCAB = [PAD] + SYMBOLS + PRIM_NAMES
STOI = {t: i for i, t in enumerate(VOCAB)}
_PATTERN = re.compile("|".join(re.escape(t) for t in sorted(SYMBOLS + PRIM_NAMES, key=len, reverse=True)))

PIPE, EOS, PAD_ID = STOI["|"], STOI["\n"], STOI[PAD]
PRIM_IDS = [STOI[p] for p in PRIM_NAMES]


def encode(s: str) -> list:
    toks = _PATTERN.findall(s)
    if "".join(toks) != s:
        raise ValueError(f"トークン化できない文字列: {s!r}")
    return [STOI[t] for t in toks]


def decode(ids) -> str:
    return "".join(VOCAB[i] for i in ids if i != PAD_ID)


def fmt_list(xs) -> str:
    return " ".join(str(v) for v in xs)


def fmt_examples(examples) -> str:
    return ";".join(f"{fmt_list(x)}>{fmt_list(y)}" for x, y in examples) + "="
