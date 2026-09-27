# Step 2 — アーキテクチャ設計（Stage 1: PC Research）

## 実験領域

小型 LLM ＋認知モジュールの効果を、自動で・誤魔化しなく測るために、最初の領域として
**「少数の入出力例から隠れた規則（整数リストを変換するプログラム）を見つける」**タスクを使う。

```
例:  3 1 4 > 4 2 5      ← 見える例（2〜3個）
     0 7   > 1 8
     ...
答え: add1          ← 隠れた規則。見えていない入力で正しく予測できたら正解
```

- DSL は 28 種類のプリミティブ（算術・選択・並べ替え・累積）。深さ3で 21,952 通り、深さ4で約61万通り。
- 学習には深さ1〜2だけを使い、プリミティブ対の約20%を学習から除外する。
- こうすることで **ID / Few-shot / OOD-組み合わせ / OOD-深さ** を明確に分けて測れる。

## モジュール構成

```
USER INPUT（入出力例）
  → Perception         core/world_model/features.py  安価な特徴量（LLM を使わない）
  → Router/Executive   core/router/router.py         どの処理をどの順で行い、いつ止めるか
  → Language Core      core/llm/                     小型 GPT（0.8M, 差し替え可能）
  → Memory             core/memory/long_term.py      自分で検証した経験 ＋ マクロ抽出
  → Sparse Experts     core/experts/experts.py       ゲートが開いた専門家だけ起動
  → Working Memory     core/working_memory/          容量制限つきの思考状態
  → Hypothesis Gen.    core/hypothesis/generator.py  サンプリング・組み合わせ・修復
  → Novelty Engine     core/hypothesis/novelty.py    新規性×もっともらしさ×有用性×検証可能性
  → World Model        core/world_model/dsl.py       世界の法則（安全なインタプリタ）
  → Simulator          core/simulator/simulator.py   仮説を実行して照合（回数を計測）
  → Critic             core/critic/critic.py         矛盾・冗長・曖昧さの検出と Occam 選択
  → Memory Update      Agent._remember
  → Final Response
Autonomous Discovery   core/discovery/loop.py        対象の選択・実験計画・知識更新を自分で行う
```

## Router の処理順（動的計算）

安い処理から順に試し、Simulator で検証済みの仮説が得られた時点で打ち切る。

| 段階 | 内容 | 主なコスト |
|---|---|---|
| direct | LLM の貪欲デコード（1回答） | LLM 1回 |
| memory | 似た過去経験の結論を再利用 | 検証数回 |
| experts | ゲートが開いた Expert の部分空間 | 検証 数十〜数百回 |
| sample | LLM から16本サンプル | LLM 1回（16系列） |
| compose | 仮説の部品とマクロの組み合わせを、もっともらしい順に列挙 | 検証 最大数千回 |
| repair | 惜しい仮説を1箇所だけ変える | 検証 数百回 |

## 計算量の測り方

- **LLM**: FLOPs ≈ 2 × 非埋め込みパラメータ数 × 処理トークン数（KV キャッシュを仮定）
- **Simulator**: 仮説1つを見える例すべてで実行して1回。予算の上限は1問あたり2,000回
- **実時間**: 1問あたりのミリ秒（CPU 1スレッド）

LLM の FLOPs と Simulator の回数は単位が違うので、足し合わせずに両方を報告する。

## スマホ移植を見据えた設計

- LLM は `LanguageModel` インターフェースの裏に隠し、量子化モデルや別アーキテクチャに差し替えられる（`mobile/quantize.py` で int8 化を試行済み）。
- 各モジュールは独立しており、効果のないモジュールは設定フラグ1つで外せる。
- Simulator・Critic・Experts・Memory は純 Python で軽く、NPU を必要としない。
