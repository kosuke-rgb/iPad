"""入出力例から計算する安価な特徴量。Router と記憶の検索に使う（LLM を呼ばない）。"""


def _is_subseq(small, big):
    it = iter(big)
    return all(any(v == w for w in it) for v in small)


def features(examples) -> dict:
    ex = [(x, y) for x, y in examples]
    return {
        "same_len": all(len(x) == len(y) for x, y in ex),
        "perm": all(sorted(x) == sorted(y) for x, y in ex),
        "subseq": all(_is_subseq(y, x) for x, y in ex),
        "shorter": all(len(y) <= len(x) for x, y in ex) and any(len(y) < len(x) for x, y in ex),
        "longer": any(len(y) > len(x) for x, y in ex),
        "out_sorted": all(y == sorted(y) for _, y in ex),
        "out_sorted_desc": all(y == sorted(y, reverse=True) for _, y in ex),
        "out_small": all(all(0 <= v <= 2 for v in y) for _, y in ex),
        "out_bigger": sum(sum(y) for _, y in ex) > sum(sum(x) for x, _ in ex),
        "out_empty_any": any(len(y) == 0 for _, y in ex),
    }


def vector(f: dict) -> tuple:
    return tuple(int(f[k]) for k in sorted(f))


def similarity(a: tuple, b: tuple) -> float:
    return sum(i == j for i, j in zip(a, b)) / len(a)
