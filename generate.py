"""学習済みモデルで文章を生成する。

例: python generate.py --prompt "昔々" --max_new_tokens 200
"""
import argparse

import torch

from model import GPT, GPTConfig
from tokenizer import CharTokenizer
from train import pick_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="ckpt.pt")
    p.add_argument("--prompt", default="\n")
    p.add_argument("--max_new_tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=pick_device())
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    ckpt = torch.load(args.ckpt, map_location=args.device)
    tok = CharTokenizer.from_json(ckpt["tokenizer"])
    model = GPT(GPTConfig(**ckpt["config"])).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    idx = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=args.device)
    out = model.generate(idx, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
