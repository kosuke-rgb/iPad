"""Long-term Memory: AI 自身が経験したことを蓄積する。

1エピソード = Observation → Hypothesis → Experiment → Result → Conclusion。
単なる文書検索（RAG）と違い、自分で検証した結論と、その結論に至るまでの経験を持つ。
さらに、検証済みの結論から繰り返し現れる部分手順を「マクロ」として抽出し、
次の問題で一つの部品として使えるようにする（ライブラリ学習）。
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass

from core.world_model.features import similarity


@dataclass
class Episode:
    observation: list          # 見えた入出力例
    features: tuple
    hypotheses_tried: int
    experiments: int           # シミュレーション回数（または能動的な問い合わせ回数）
    result: str                # "verified" / "unverified" / "failed"
    conclusion: tuple | None   # 採用したプログラム
    source: str = ""           # どのモジュールが結論を出したか


class LongTermMemory:
    def __init__(self):
        self.episodes: list[Episode] = []

    def store(self, ep: Episode):
        self.episodes.append(ep)

    def retrieve(self, fvec: tuple, k: int = 8, verified_only: bool = False) -> list:
        """特徴が似た過去エピソードの結論を、似ている順・よく使われた順に返す。"""
        freq = Counter(e.conclusion for e in self.episodes if e.conclusion)
        scored = {}
        for e in self.episodes:
            if not e.conclusion or (verified_only and e.result != "verified"):
                continue
            s = similarity(fvec, e.features) + 0.01 * freq[e.conclusion]
            scored[e.conclusion] = max(scored.get(e.conclusion, 0.0), s)
        return [p for p, _ in sorted(scored.items(), key=lambda kv: -kv[1])[:k]]

    def macros(self, k: int = 8) -> list:
        """検証済みの結論から、長さ2以上の部分手順を頻度順に返す。"""
        c = Counter()
        for e in self.episodes:
            if e.result == "verified" and e.conclusion and len(e.conclusion) >= 2:
                p = e.conclusion
                for i in range(len(p)):
                    for j in range(i + 2, len(p) + 1):
                        c[p[i:j]] += 1
        return [m for m, _ in c.most_common(k)]

    def known_programs(self) -> set:
        return {e.conclusion for e in self.episodes if e.conclusion}

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump([asdict(e) for e in self.episodes], f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "LongTermMemory":
        m = cls()
        with open(path) as f:
            for d in json.load(f):
                d["features"] = tuple(d["features"])
                d["conclusion"] = tuple(d["conclusion"]) if d["conclusion"] else None
                m.store(Episode(**d))
        return m


class StaticKnowledgeBase:
    """Baseline B 用の RAG。学習データ（問題と正解プログラム）を特徴量で検索するだけで、
    自分の経験は増えない。"""

    def __init__(self, tasks):
        from core.world_model.features import features, vector
        self.entries = [(vector(features(t.train)), t.program) for t in tasks]

    def retrieve(self, fvec: tuple, k: int = 8) -> list:
        ranked = sorted(self.entries, key=lambda e: -similarity(fvec, e[0]))
        out = []
        for _, p in ranked:
            if p not in out:
                out.append(p)
            if len(out) == k:
                break
        return out
