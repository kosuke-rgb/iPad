"""評価用データセット。シードを固定して毎回同じ問題を使う。

分割はこの順にエージェントへ流す（長期記憶は分割をまたいで蓄積される）。
"""
from core.world_model.tasks import make_benchmark, make_related_benchmark

SPLITS = [
    ("id", "ID（学習と同じ分布）"),
    ("few_shot", "Few-shot（例が2個だけ）"),
    ("ood_combo", "OOD-組み合わせ（未学習のプリミティブ対）"),
    ("ood_depth", "OOD-深さ（未学習の3段階プログラム）"),
    ("ood_related", "OOD-関連（以前に解いた規則＋1手順）"),
]


def load(n_per_split=100, seed=1000):
    data = {s: make_benchmark(s, n_per_split, seed + i)
            for i, (s, _) in enumerate(SPLITS) if s != "ood_related"}
    data["ood_related"] = make_related_benchmark(
        [t.program for t in data["id"]], n_per_split, seed + len(SPLITS))
    return data
