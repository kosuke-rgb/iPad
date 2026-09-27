"""テキストファイルから小さな GPT を学習する。

例: python train.py --data data/input.txt --max_iters 2000
"""
import argparse
import time

import torch

from model import GPT, GPTConfig
from tokenizer import CharTokenizer


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # Apple Silicon の Mac
        return "mps"
    return "cpu"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/input.txt")
    p.add_argument("--out", default="ckpt.pt")
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_iters", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval_interval", type=int, default=200)
    p.add_argument("--eval_iters", type=int, default=20)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default=pick_device())
    args = p.parse_args()

    torch.manual_seed(args.seed)
    with open(args.data, encoding="utf-8") as f:
        text = f.read()
    tok = CharTokenizer.from_text(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    splits = {"train": data[:n], "val": data[n:]}
    block_size = min(args.block_size, len(splits["val"]) - 1)
    if block_size < 8:
        raise SystemExit("学習データが少なすぎます。もっと長いテキストを用意してください。")
    print(f"文字数: {len(data):,} / 語彙サイズ: {tok.vocab_size} / device: {args.device}")

    def get_batch(split):
        d = splits[split]
        ix = torch.randint(len(d) - block_size, (args.batch_size,))
        x = torch.stack([d[i:i + block_size] for i in ix])
        y = torch.stack([d[i + 1:i + 1 + block_size] for i in ix])
        return x.to(args.device), y.to(args.device)

    config = GPTConfig(
        vocab_size=tok.vocab_size, block_size=block_size, n_layer=args.n_layer,
        n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout,
    )
    model = GPT(config).to(args.device)
    print(f"パラメータ数: {model.num_params() / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_iters, eta_min=args.lr / 10)

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split in splits:
            losses = torch.zeros(args.eval_iters)
            for k in range(args.eval_iters):
                _, loss = model(*get_batch(split))
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    t0 = time.time()
    for it in range(args.max_iters + 1):
        if it % args.eval_interval == 0 or it == args.max_iters:
            l = estimate_loss()
            print(f"step {it:5d} | train loss {l['train']:.3f} | val loss {l['val']:.3f} | {time.time() - t0:.0f}s")
        if it == args.max_iters:
            break
        _, loss = model(*get_batch("train"))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

    torch.save({
        "model": model.state_dict(),
        "config": config.__dict__,
        "tokenizer": tok.to_json(),
    }, args.out)
    print(f"保存しました: {args.out}")


if __name__ == "__main__":
    main()
