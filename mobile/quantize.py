"""Stage 2 の最初の実験: int8 動的量子化で性能がどれだけ保たれるかを測る。

python -m mobile.quantize --ckpt checkpoints/tiny_gpt.pt
モデルサイズ・推論速度・（Baseline A と提案モデルの）正答率を、量子化前後で比べる。
"""
import argparse
import io
import json
import time

import torch

from core.agent import BASE, PROPOSED
from core.llm.backend import ComputeMeter, TinyGPTBackend
from core.llm.tiny_gpt import GPT, GPTConfig
from core.memory.long_term import StaticKnowledgeBase
from core.world_model.tasks import make_benchmark
from evaluation import benchmarks
from evaluation.run import evaluate, summarize


def model_bytes(model) -> int:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell()


def quantized_model(ckpt):
    ck = torch.load(ckpt, map_location="cpu")
    m = GPT(GPTConfig(**ck["config"]))
    m.load_state_dict(ck["model"])
    m.eval()
    return torch.ao.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)


def latency(lm, tasks, n=16):
    meter = ComputeMeter()
    t0 = time.perf_counter()
    for t in tasks:
        lm.sample(t.train, n, 1.0, meter)
    return (time.perf_counter() - t0) / len(tasks) * 1000, meter.llm_tokens / (time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/tiny_gpt.pt")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="results/quantization.json")
    args = ap.parse_args()
    torch.set_num_threads(1)
    fp32 = TinyGPTBackend(args.ckpt)
    int8 = TinyGPTBackend(args.ckpt, model=quantized_model(args.ckpt))
    data = benchmarks.load(args.n)
    kb = StaticKnowledgeBase(make_benchmark("train", 1000, seed=7))
    lat_tasks = data["id"][:30]
    report = {}
    for name, lm in (("fp32", fp32), ("int8", int8)):
        ms, tps = latency(lm, lat_tasks)
        entry = {"size_kb": model_bytes(lm.model) / 1024, "sample16_ms": ms, "tokens_per_s": tps}
        for cfg in (BASE, PROPOSED):
            res = evaluate(cfg, lm, kb, data)
            entry[cfg.name] = {s: summarize(r)["acc"] for s, r in res.items()}
        report[name] = entry
        print(name, json.dumps(entry, ensure_ascii=False), flush=True)
    with open(args.out, "w") as f:
        json.dump(report, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
