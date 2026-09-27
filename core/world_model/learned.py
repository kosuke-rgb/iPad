"""学習した世界モデル: 観察から「操作の結果」を予測するニューラルネット。

手書きのインタプリタ（dsl.py）は「現実の世界」として扱い、エージェントはその内部を知らない。
エージェントは (操作, いまの状態) → 次の状態 という観察（遷移）から世界の法則を学ぶ。
複数手順のプログラムは、1手順ずつ予測を重ねて頭の中でシミュレーション（ロールアウト）する。

python -m core.world_model.learned --max_iters 6000 --out checkpoints/world_model.pt
"""
from __future__ import annotations

import argparse
import os
import random
import time

import torch

from core.llm import tokenizer as T
from core.llm.tiny_gpt import GPT, GPTConfig
from core.world_model.dsl import PRIM_NAMES, safe_run
from core.world_model.tasks import random_input

OUT_TOKENS = [T.STOI[c] for c in "0123456789 -\n"]
MAX_OUT = 48


def fmt_query(prim, xs):
    return f"{prim} {T.fmt_list(xs)}>"


def parse(s):
    s = s.strip()
    if not s:
        return []
    try:
        return [int(v) for v in s.split(" ")]
    except ValueError:
        return None


def random_state(rng):
    """観察される状態: ランダムな入力に、0〜2手順の操作をしたもの（実際の探索中に現れる状態に近い）。"""
    while True:
        xs = random_input(rng)
        for _ in range(rng.randint(0, 2)):
            ys = safe_run((rng.choice(PRIM_NAMES),), xs)
            if ys is None:
                break
            xs = ys
        else:
            return xs


def transition(rng):
    while True:
        xs, p = random_state(rng), rng.choice(PRIM_NAMES)
        ys = safe_run((p,), xs)
        if ys is not None:
            return p, xs, ys


def make_batch(rng, batch_size):
    xs, ys = [], []
    for _ in range(batch_size):
        p, a, b = transition(rng)
        q = T.encode(fmt_query(p, a))
        t = T.encode(T.fmt_list(b) + "\n")
        seq = q + t
        xs.append(seq[:-1])
        ys.append([-100] * (len(q) - 1) + t)
    L = max(len(x) for x in xs)
    return (torch.tensor([x + [T.PAD_ID] * (L - len(x)) for x in xs]),
            torch.tensor([y + [-100] * (L - len(y)) for y in ys]))


class LearnedWorldModel:
    def __init__(self, path, device="cpu"):
        ck = torch.load(path, map_location=device)
        self.model = GPT(GPTConfig(**ck["config"]))
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.n_params = self.model.num_params()
        self.cache: dict = {}
        self.flops = 0.0
        self.queries = 0
        self._mask = torch.full((len(T.VOCAB),), -float("inf"))
        self._mask[OUT_TOKENS] = 0

    @torch.no_grad()
    def _predict(self, queries):
        """[(prim, state)] を一括で予測（貪欲デコード）。"""
        prompts = [T.encode(fmt_query(p, tuple(x))) for p, x in queries]
        n = len(prompts)
        L = max(len(p) for p in prompts)
        # 左詰めだと位置がずれるので、右寄せではなく「プロンプト長ごと」にまとめて処理する
        out = [None] * n
        by_len = {}
        for i, p in enumerate(prompts):
            by_len.setdefault(len(p), []).append(i)
        for plen, idxs in by_len.items():
            idx = torch.tensor([prompts[i] for i in idxs])
            done = torch.zeros(len(idxs), dtype=torch.bool)
            gen = [[] for _ in idxs]
            for _ in range(MAX_OUT):
                logits = self.model(idx)[0][:, -1, :] + self._mask
                nxt = logits.argmax(-1)
                self.flops += 2.0 * self.n_params * int((~done).sum())
                for j in range(len(idxs)):
                    if not done[j]:
                        gen[j].append(int(nxt[j]))
                done |= nxt == T.EOS
                if bool(done.all()) or idx.size(1) >= self.model.config.block_size:
                    break
                idx = torch.cat([idx, nxt[:, None]], dim=1)
            self.flops += 2.0 * self.n_params * plen * len(idxs)
            for j, i in enumerate(idxs):
                out[i] = parse(T.decode(gen[j])) if gen[j] and gen[j][-1] == T.EOS else None
        return out

    def step_many(self, queries):
        todo = list({q for q in queries if q not in self.cache})
        self.queries += len(todo)
        for k in range(0, len(todo), 512):
            chunk = todo[k:k + 512]
            for q, r in zip(chunk, self._predict(chunk)):
                self.cache[q] = r
        return [self.cache[q] for q in queries]

    def run_many(self, programs, inputs):
        """各 (プログラム, 入力) を頭の中で実行した結果。予測が壊れたら None。"""
        states = {(p, tuple(x)): tuple(x) for p in programs for x in inputs}
        alive = dict(states)
        depth = max((len(p) for p in programs), default=0)
        for d in range(depth):
            keys = [k for k in alive if len(k[0]) > d and alive[k] is not None]
            qs = [(k[0][d], alive[k]) for k in keys]
            for k, r in zip(keys, self.step_many(qs)):
                alive[k] = tuple(r) if r is not None else None
        return {k: (list(v) if v is not None else None) for k, v in alive.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="checkpoints/world_model.pt")
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_embd", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--max_iters", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    rng = random.Random(0)
    cfg = GPTConfig(vocab_size=len(T.VOCAB), block_size=96, n_layer=args.n_layer,
                    n_head=args.n_head, n_embd=args.n_embd)
    model = GPT(cfg)
    print(f"パラメータ数(埋め込み除く): {model.num_params() / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / 200) * max(0.05, 1 - i / args.max_iters))
    t0 = time.time()
    for it in range(1, args.max_iters + 1):
        x, y = make_batch(rng, args.batch_size)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % 200 == 0 or it == 1:
            print(f"step {it:5d} | loss {loss.item():.4f} | {time.time() - t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg.__dict__, "vocab": T.VOCAB,
                "train_steps": args.max_iters}, args.out)
    print(f"保存しました: {args.out}", flush=True)


if __name__ == "__main__":
    main()
