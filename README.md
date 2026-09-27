# Brain-Inspired Small LLM — PC Research → Smartphone

小型 LLM に「記憶・Router・専門家・仮説生成・Critic・世界モデル／シミュレーション・自律的発見」を組み合わせる研究プロジェクト。
すべてのモジュールを ON/OFF できるようにし、**本当に性能が上がったか、どれだけの計算で上がったか**を実験で判断する。

現在は **Stage 1（PC Research）の第1ラウンド**。結果は [docs/04_results.md](docs/04_results.md)。

## ドキュメント

| | |
|---|---|
| [docs/01_research.md](docs/01_research.md) | Step 1: 関連研究の調査と、それぞれの設計への反映 |
| [docs/02_architecture.md](docs/02_architecture.md) | Step 2: アーキテクチャと実験領域 |
| [docs/03_hypotheses.md](docs/03_hypotheses.md) | Step 3: 実験仮説と棄却条件 |
| [docs/04_results.md](docs/04_results.md) | Step 4〜13: Baseline 比較・アブレーション・発見ループ・量子化 |

## 構成

```
core/
  llm/            小型 GPT（0.8M）、トークナイザ、学習、差し替え可能なバックエンド＋計算量計測
  world_model/    DSL（世界の法則）、タスク生成（ID / Few-shot / OOD）、特徴量
  simulator/      仮説の実行と照合（回数を計測・予算管理）
  working_memory/ 容量制限つきの思考状態
  memory/         長期記憶（経験＋マクロ）、RAG 用の静的知識ベース
  router/         動的計算（処理の順序と打ち切り、Expert の選択）
  experts/        ゲートつき専門家
  hypothesis/     仮説生成（サンプリング・組み合わせ・修復）、Novelty Engine
  critic/         矛盾・冗長・曖昧さの検出、Occam 選択
  discovery/      自律的発見ループ（対象選択・実験計画・知識更新）
  agent.py        全モジュールの統合と Baseline / アブレーションの設定
evaluation/       評価の実行、発見ループの評価、レポート生成
mobile/           量子化（Stage 2 の予備実験）
results/          実験結果（JSON）
```

## 使い方

```bash
pip install -r requirements.txt
python -m core.llm.train --max_iters 5000   # 約40分（CPU）
python -m evaluation.run --n 100
python -m evaluation.discovery_eval
python -m mobile.quantize
python -m evaluation.report
```

## 第1ラウンドの要点

- 提案モデルは、同じ小型 LLM の Baseline A〜E を全分割で上回った（OOD-深さ 40% 対 最大 9%）。
- 効いているのは主に「生成 → 実行して検証」と「動的計算（Router）」。
- LLM を使わない総当たり探索とは、多くの分割で互角だった。勝ったのは、探索空間が大きい OOD-深さ だけ（40% 対 19%）。実時間では総当たりより約12倍遅い。
- 長期記憶・好奇心・能動的な実験計画・修復は、まだ効果を示せていない → 再設計の対象。
