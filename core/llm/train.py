"""小型 LLM（例 → プログラム）を合成データで学習する。

python -m core.llm.train --max_iters 4000 --out checkpoints/tiny_gpt.pt
学習データは深さ1〜2・学習用プリミティブ対のみ（OOD は一切含まない）。
"""
import argparse
import os
import random
import time

import torch

from core.llm import tokenizer as T
from core.llm.tiny_gpt import GPT, GPTConfig
from core.world_model.dsl import to_str
from core.world_model.tasks import make_task


def encode_pair(examples, program, block_size):
    """(例, プログラム) を学習用の (入力, 目標) に変換。長すぎれば None。"""
    prompt = T.encode(T.fmt_examples(examples))
    target = T.encode(to_str(program) + "\n")
    seq = prompt + target
    if len(seq) > block_size + 1:
        return None
    return seq[:-1], [-100] * (len(prompt) - 1) + target  # プログラム部分だけ損失を計算


def collate(pairs):
    """パディングはバッチ内の最長系列まで（block_size まで埋めると計算の約3/4が無駄になる）。"""
    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]
    L = max(len(x) for x in xs)
    xs = [x + [T.PAD_ID] * (L - len(x)) for x in xs]
    ys = [y + [-100] * (L - len(y)) for y in ys]
    return torch.tensor(xs), torch.tensor(ys)


def make_batch(rng, batch_size, block_size):
    pairs = []
    while len(pairs) < batch_size:
        t = make_task(rng, "train")
        e = encode_pair(t.train, t.program, block_size)
        if e is not None:
            pairs.append(e)
    return collate(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="checkpoints/tiny_gpt.pt")
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_embd", type=int, default=128)
    ap.add_argument("--block_size", type=int, default=192)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_iters", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--save_every", type=int, default=0, help="途中のチェックポイントを保存する間隔")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    cfg = GPTConfig(vocab_size=len(T.VOCAB), block_size=args.block_size,
                    n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)
    model = GPT(cfg)
    print(f"パラメータ数(埋め込み除く): {model.num_params() / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    warmup = 200
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / warmup) * max(0.05, 1 - i / args.max_iters))
    t0 = time.time()
    for it in range(1, args.max_iters + 1):
        x, y = make_batch(rng, args.batch_size, args.block_size)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % 100 == 0 or it == 1:
            print(f"step {it:5d} | loss {loss.item():.3f} | {time.time() - t0:.0f}s", flush=True)
        if args.save_every and it % args.save_every == 0 and it < args.max_iters:
            save(model, cfg, args.out.replace(".pt", f"_step{it}.pt"), it, time.time() - t0)
    save(model, cfg, args.out, args.max_iters, time.time() - t0)


def save(model, cfg, path, steps, seconds):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg.__dict__, "vocab": T.VOCAB,
                "train_steps": steps, "train_seconds": seconds}, path)
    print(f"保存しました: {path}", flush=True)


if __name__ == "__main__":
    main()
