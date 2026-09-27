"""評価用データセット。シードを固定して毎回同じ問題を使う。"""
from core.world_model.tasks import make_benchmark

SPLITS = [
    ("id", "ID（学習と同じ分布）"),
    ("few_shot", "Few-shot（例が2個だけ）"),
    ("ood_combo", "OOD-組み合わせ（未学習のプリミティブ対）"),
    ("ood_depth", "OOD-深さ（未学習の3段階プログラム）"),
]


def load(n_per_split=100, seed=1000):
    return {s: make_benchmark(s, n_per_split, seed + i) for i, (s, _) in enumerate(SPLITS)}
