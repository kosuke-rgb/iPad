# Step 1 — 関連研究・既存技術の調査

> 注: 本調査は2026年時点の知識に基づく要約です。論文名は検索用のキーワードとして挙げています。
> 実装に取り込む前に、原論文で詳細を確認してください。

各分野について「何が分かっているか」と「本プロジェクトのどのモジュールに、どう反映したか」を対応づけます。

## 1. Mixture-of-Experts / Sparse Computation

- **要点**: トークンごとに一部の Expert だけを計算するので、総パラメータ数を増やしても実計算量はほぼ一定に保てる（Shazeer+ 2017 *Sparsely-Gated MoE*, Switch Transformer, GShard, Mixtral, DeepSeekMoE）。
- **課題**: ルーティングが偏る（負荷の偏り）、小規模ではあまり効かない、メモリ（全 Expert の重み）はむしろ増える。スマホではメモリ帯域が律速なので、計算量の削減がそのまま速度向上になるとは限らない。
- **本プロジェクトへの反映**: `core/experts` はモデル内部の MoE ではなく、**モジュール単位の疎な起動**（Router がゲートの開いた Expert だけを呼ぶ）として実装した。まず「疎な起動で計算量が減るか・精度が落ちないか」を測り、効果があればモデル内部の MoE を検討する。

## 2. Sparse Transformer / 効率的アテンション

- **要点**: Sparse Transformer, Longformer, BigBird（疎なアテンションパターン）、FlashAttention（IO を意識した厳密な計算）、Mamba・RWKV などの状態空間モデルや線形 RNN（長い系列で計算量 O(n)）。
- **反映**: Stage 1 では標準のアテンションを使う（研究速度を優先）。系列長が伸びる Stage 2 以降で、状態空間モデルや疎なアテンションに置き換える候補とする。`LanguageModel` インターフェースで本体を差し替えられるようにした。

## 3. Neural Memory / 長期記憶

- **要点**: Memory Networks, Neural Turing Machine / DNC（微分可能な外部メモリ）、Memorizing Transformers（kNN メモリ）、MemGPT（OS のように階層的に記憶を管理）、Titans（テスト時に更新される記憶）など。
- **エージェントの記憶**: Generative Agents（観測・反省・計画を記憶ストリームに保存）、Reflexion（失敗からの言語的な反省を記憶）、Voyager（検証済みのスキルをライブラリとして蓄積）。
- **ライブラリ学習**: DreamCoder（解けた問題から共通部分を抽象化して次の探索を速くする）。
- **反映**: `core/memory` は RAG と区別し、**自分で検証した結論とその過程**（Observation → Hypothesis → Experiment → Result → Conclusion）を保存する。さらに Voyager / DreamCoder の考え方で、検証済みの結論から**マクロ（再利用可能な部分手順）**を抽出する。

## 4. Retrieval-Augmented Generation

- **要点**: RAG（Lewis+ 2020）, REALM, RETRO（検索でパラメータ数を大幅に減らせる）, Atlas。知識をパラメータの外に置けるので、小型モデルと相性がよい。
- **限界**: 検索結果の正しさを検証しない。自分の経験から学ばない。
- **反映**: Baseline B として実装（学習データを特徴量で検索するだけ）。提案モデルの長期記憶と比べることで、「経験の蓄積」が単なる検索を上回るかを測る。

## 5. World Models

- **要点**: World Models（Ha & Schmidhuber）, Dreamer 系列（学習した潜在空間の中で想像して方策を学ぶ）, MuZero（学習したモデルでの計画）, JEPA（LeCun の予測表現）。
- **反映**: Stage 1 では、世界の法則が既知の**明示的な世界モデル**（`core/world_model/dsl.py` のインタプリタ）を使う。これは「学習した世界モデル」ではない点に注意。学習した世界モデルへの置き換えは後の課題とする。

## 6. Agentic AI / ツール利用

- **要点**: ReAct（推論と行動の交互実行）, Toolformer（ツール呼び出しの自己学習）, Program-of-Thoughts / PAL（計算をコード実行に任せる）, Reflexion, Voyager。
- **反映**: `core/simulator` は PAL の考え方で、仮説を実行して検証する。**LLM が「合っている」と言っても採用しない**。

## 7. Test-Time Compute（推論時の計算）

- **要点**: Chain-of-Thought, Self-Consistency（複数サンプルの多数決）, Tree of Thoughts, Best-of-N と検証器（Cobbe+ 2021）, 推論時計算のスケーリング（Snell+ 2024, o1 / R1 系の長い推論）。重要な知見として、**難しい問題ほど推論時の計算が効き、簡単な問題に計算を使うのは無駄**。また、**検証器があると計算量を増やしたときの伸びが大きい**。
- **反映**: Baseline E（Self-Consistency）と S（サンプル＋検証）を実装。Router は「簡単な問題は安く、難しい問題にだけ計算を使う」動的計算を担う。

## 8. Neuro-symbolic AI / プログラム合成

- **要点**: DeepCoder（ニューラルネットで探索を誘導）, RobustFill, DreamCoder, AlphaCode（大量サンプル＋テストで絞り込み）, ARC-AGI（少数例からの抽象推論。テスト時学習＋プログラム探索の組み合わせが強い）。
- **反映**: 評価タスク自体を「少数の入出力例から規則（プログラム）を見つける」問題にした。これは ARC と同じく、未知問題への汎化・発見を自動で採点できる。`core/hypothesis` は DeepCoder の考え方で、LLM の出力を探索の手がかりに使う（組み合わせ探索）。

## 9. Meta-learning

- **要点**: MAML, Reptile, Prototypical Networks, In-context learning（大規模モデルはプロンプト内の例から学ぶ）, テスト時学習（ARC で有効）。
- **反映**: Stage 1 の LLM は、例 → プログラムを**文脈内で**推論する（in-context 推論）。テスト時学習（問題ごとの微調整）は今後の比較対象とする。

## 10. Active Learning / 実験計画

- **要点**: 不確実性サンプリング、Query-by-Committee（仮説集団の意見が割れる点を問う）、ベイズ実験計画（期待情報利得の最大化）、BALD。
- **反映**: `core/discovery` の実験計画は Query-by-Committee ＋情報利得: **生き残った仮説の予測が最も割れる入力**を次の実験に選ぶ。

## 11. Intrinsic Motivation / Curiosity-driven learning

- **要点**: Schmidhuber の圧縮進捗、ICM（予測誤差を好奇心とする）、RND、Learning Progress（学習の進みが速い対象を選ぶ）、Go-Explore、POET（自分で課題を作る）。
- **注意点**: 予測誤差をそのまま好奇心にすると、原理的に予測できないノイズに引き寄せられる（noisy-TV 問題）。
- **反映**: 発見ループでは「仮説集合の不確実性（エントロピー）」が最も高い対象を次に調べる。ノイズがない環境なので noisy-TV 問題は起きないが、実世界に拡張するときは learning progress への切り替えが必要。

## 12. Automated Scientific Discovery

- **要点**: BACON（データから物理法則を再発見）, AI Feynman / PySR（記号回帰）, FunSearch・AlphaEvolve（LLM ＋自動評価器による進化的探索で新しい構成を発見）, The AI Scientist（研究の全工程の自動化の試み）, Robot scientist Adam/Eve。
- **最重要の教訓**: 発見系の成功例は、ほぼすべて**信頼できる自動評価器（検証器）**に依存している。LLM の「思いつき」だけで成果が出た例はない。
- **反映**: 仮説の生成（LLM）と評価（Simulator + Critic）を分離した。Novelty Engine は「新しいだけ」ではなく、もっともらしさ・有用性・検証可能性との積で評価する。

## まとめ: 設計判断

1. **検証器を中心に置く**。多くの分野に共通する知見として、効果が大きいのは「生成 → 検証」のループであり、検証できないモジュールの効果は疑わしい。
2. **最初の評価タスクは自動採点できる領域にする**。言語タスクで「発見」を評価するのは難しく、評価そのものが論点になってしまう。そこでプログラム帰納から始める。
3. **計算量を必ず併記する**。推論時計算の研究は、計算量を揃えないと比較が無意味になることを示している。
4. **「脳に似ている」は根拠にしない**。各モジュールは同じコードで ON/OFF し、アブレーションで効果を示す。
