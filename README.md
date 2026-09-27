# iPad — ゼロから作る小さな LLM

PyTorch だけで書いた、GPT 型（デコーダのみの Transformer）の小さな言語モデルです。
LLM の仕組み（トークン化 → 埋め込み → 自己注意 → 次トークン予測 → 生成）を、短いコードで一通り体験できます。

## ファイル構成

| ファイル | 内容 |
| --- | --- |
| `model.py` | GPT 本体（自己注意、MLP、Transformer ブロック、生成関数） |
| `tokenizer.py` | 文字単位のトークナイザ（日本語もそのまま扱える） |
| `train.py` | テキストファイルから学習し、`ckpt.pt` に保存 |
| `generate.py` | 学習済みモデルで続きの文章を生成 |
| `data/input.txt` | 動作確認用のサンプル文章（約 1,900 文字） |

## 使い方

```bash
pip install -r requirements.txt

# 学習（CPU で数分。GPU / Apple Silicon の Mac があれば自動で使います）
python train.py --data data/input.txt --max_iters 2000

# 生成
python generate.py --prompt "むかしむかし、" --max_new_tokens 200
```

iPad 単体では PyTorch が動かないので、Google Colab などのクラウド環境か PC / Mac で実行してください。

## 賢くするには

サンプル文章は小さすぎるため、モデルはほぼ丸暗記します（train loss は下がるが val loss は上がる＝過学習）。
それらしい文章を「新しく」書けるようにするには:

1. **データを増やす** — 数 MB 以上のテキスト（青空文庫の作品、Wikipedia のダンプなど）を `data/input.txt` に置く
2. **モデルを大きくする** — 例: `--n_layer 6 --n_head 6 --n_embd 384 --block_size 256`（GPU 推奨）
3. **トークナイザを改良する** — 文字単位の代わりに BPE / SentencePiece を使うと効率が上がる
4. **会話できるようにする** — 質問と回答のペアで追加学習（インストラクションチューニング）する

## パラメータ

`python train.py --help` で一覧を確認できます。主なもの:

- `--block_size` コンテキスト長（一度に見る文字数）
- `--n_layer` / `--n_head` / `--n_embd` モデルの深さ・ヘッド数・幅
- `--max_iters` 学習ステップ数、`--lr` 学習率
- 生成時の `--temperature`（大きいほど多様）と `--top_k`（候補の絞り込み）
