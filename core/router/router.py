"""Router / Executive: 毎回すべての処理を実行しないための制御。

- 問題の特徴量（LLM を使わない安価な計算）から、起動する Expert を選ぶ。
- 処理を安い順に並べ、検証済みの仮説が得られた時点で打ち切る（動的計算）。
  簡単な問題 → LLM の直接回答だけで終了
  専門的な問題 → 記憶 / Expert
  難しい・未知の問題 → サンプリング → 組み合わせ探索 → 修復
"""
from __future__ import annotations

from core.experts.experts import ALL_EXPERTS

STAGE_ORDER = ["direct", "memory", "rag", "experts", "sample", "compose", "repair"]


class Router:
    def __init__(self, enabled: bool):
        self.enabled = enabled

    def select_experts(self, f: dict) -> list:
        if not self.enabled:
            return list(ALL_EXPERTS)  # ゲートなし = 全 Expert を毎回計算（密な計算）
        return [e for e in ALL_EXPERTS if e.gate(f)]

    def plan(self, available: list) -> list:
        return [s for s in STAGE_ORDER if s in available]

    def should_stop(self, wm) -> bool:
        """検証済みの仮説があれば、それ以上の計算は行わない。"""
        return self.enabled and bool(wm.verified())
