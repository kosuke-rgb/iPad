"""LLM バックエンド（差し替え可能）と計算量の計測。

認知モジュールは LanguageModel インターフェースだけに依存するので、
Tiny GPT を別の小型モデル（HF の事前学習モデルや量子化モデル）に置き換えられる。

計算量: FLOPs ≈ 2 × 非埋め込みパラメータ数 × 処理トークン数（KV キャッシュを仮定し、
同じプロンプトから n 本サンプルする場合はプロンプトを1回だけ数える）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from core.llm import tokenizer as T
from core.llm.tiny_gpt import GPT, GPTConfig
from core.world_model.dsl import Program, from_str, to_str

MAX_DEPTH = 4


@dataclass
class ComputeMeter:
    llm_tokens: int = 0
    llm_flops: float = 0.0
    sim_calls: int = 0
    llm_calls: int = 0

    def add(self, other: "ComputeMeter"):
        self.llm_tokens += other.llm_tokens
        self.llm_flops += other.llm_flops
        self.sim_calls += other.sim_calls
        self.llm_calls += other.llm_calls


class LanguageModel:
    """認知モジュールから見た LLM のインターフェース。"""

    n_params: int = 0

    def sample(self, examples, n: int, temperature: float, meter: ComputeMeter) -> list:
        """[(program, logprob), ...] を返す。temperature=0 なら貪欲に1本。"""
        raise NotImplementedError

    def score(self, examples, programs: list, meter: ComputeMeter) -> list:
        """各プログラムの対数尤度（もっともらしさ）を返す。"""
        raise NotImplementedError


class TinyGPTBackend(LanguageModel):
    def __init__(self, ckpt_path: str, device: str = "cpu", model: GPT | None = None):
        ck = torch.load(ckpt_path, map_location=device)
        assert ck["vocab"] == T.VOCAB, "語彙が一致しません（DSL を変えたら再学習が必要）"
        self.model = model if model is not None else GPT(GPTConfig(**ck["config"]))
        if model is None:
            self.model.load_state_dict(ck["model"])
        self.model.to(device).eval()
        self.device = device
        self.block_size = self.model.config.block_size
        # 量子化モデルでも同じ基準で数えるため、設定から元のパラメータ数を求める
        self.n_params = GPT(GPTConfig(**ck["config"])).num_params()
        self._prim_mask = torch.full((len(T.VOCAB),), -math.inf)
        self._prim_mask[T.PRIM_IDS] = 0
        self._sep_mask = torch.full((len(T.VOCAB),), -math.inf)
        self._sep_mask[[T.PIPE, T.EOS]] = 0
        self._eos_mask = torch.full((len(T.VOCAB),), -math.inf)
        self._eos_mask[T.EOS] = 0

    def _prompt(self, examples):
        ids = T.encode(T.fmt_examples(examples))
        return ids[-(self.block_size - 2 * MAX_DEPTH):]

    def _count(self, meter, tokens):
        meter.llm_tokens += tokens
        meter.llm_flops += 2.0 * self.n_params * tokens
        meter.llm_calls += 1

    def _mask(self, step, n_prims):
        if step % 2 == 0:
            return self._prim_mask
        return self._eos_mask if n_prims >= MAX_DEPTH else self._sep_mask

    @torch.no_grad()
    def sample(self, examples, n, temperature, meter):
        prompt = self._prompt(examples)
        n = 1 if temperature <= 0 else n
        idx = torch.tensor([prompt] * n, device=self.device)
        logp = torch.zeros(n)
        done = torch.zeros(n, dtype=torch.bool)
        out = [[] for _ in range(n)]
        gen = 0
        for step in range(2 * MAX_DEPTH):
            logits = self.model(idx)[0][:, -1, :].float().cpu()
            logits = logits + self._mask(step, step // 2 + 1 if step % 2 else 0)
            lp = F.log_softmax(logits / max(temperature, 1e-6) if temperature > 0 else logits, dim=-1)
            nxt = lp.argmax(-1) if temperature <= 0 else torch.multinomial(lp.exp(), 1).squeeze(1)
            base_lp = F.log_softmax(logits, dim=-1).gather(1, nxt[:, None]).squeeze(1)
            logp += torch.where(done, torch.zeros_like(base_lp), base_lp)
            gen += int((~done).sum())
            for i in range(n):
                if not done[i]:
                    out[i].append(int(nxt[i]))
            done |= nxt == T.EOS
            if bool(done.all()):
                break
            idx = torch.cat([idx, nxt[:, None].to(self.device)], dim=1)
        self._count(meter, len(prompt) + gen)
        res = []
        for ids, l in zip(out, logp.tolist()):
            p = from_str(T.decode(ids))
            if p is not None:
                res.append((p, l))
        return res

    @torch.no_grad()
    def score(self, examples, programs, meter):
        if not programs:
            return []
        prompt = self._prompt(examples)
        seqs = [prompt + T.encode(to_str(p) + "\n") for p in programs]
        L = max(len(s) for s in seqs)
        idx = torch.tensor([s + [T.PAD_ID] * (L - len(s)) for s in seqs], device=self.device)
        logits = self.model(idx)[0].float().cpu()
        scores = []
        for i, (s, p) in enumerate(zip(seqs, programs)):
            total = 0.0
            for j in range(len(prompt), len(s)):
                step = j - len(prompt)
                lg = logits[i, j - 1] + self._mask(step, step // 2 + 1 if step % 2 else 0)
                total += float(F.log_softmax(lg, -1)[s[j]])
            scores.append(total)
        self._count(meter, len(prompt) + sum(len(s) - len(prompt) for s in seqs))
        return scores
