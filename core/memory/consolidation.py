"""睡眠による記憶の定着（Complementary Learning Systems に着想）。

脳では、昼の経験はまず海馬に速く記録され、睡眠中の再生（replay）によって
ゆっくりと大脳皮質（ここでは LLM の重み）に移されると考えられている。
一度に新しいことだけを学ぶと古い知識が壊れる（破滅的忘却）ため、
新しい経験と古い知識を交互に再生する（interleaved replay）。

ここでの対応:
  海馬          = LongTermMemory（検証済みのエピソード）
  大脳皮質      = LLM の重み
  再生          = エピソードの (例 → 結論) を再学習
  夢            = 同じ結論（規則）を、新しい入力で頭の中で実行して作った例
  古い知識の復習 = 元の学習分布の問題
正解ラベルは使わない。学習に使うのはエージェント自身が検証した結論だけ（誤りも含みうる）。
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from core.llm.train import collate, encode_pair, make_task
from core.world_model.dsl import safe_run
from core.world_model.tasks import random_input


@dataclass
class SleepConfig:
    steps: int = 600
    batch_size: int = 32
    lr: float = 3e-4
    rehearsal: float = 0.5        # バッチのうち古い知識の復習が占める割合（0 なら新しい経験だけ）
    dreams_per_episode: int = 8   # 1エピソードから作る夢の数（0 なら実際の観測だけを再生）
    # --- フェーズ1.5 で追加（既定値は従来と同じ動作） ---
    dream_world_model: object = None   # 夢の出力を学習した世界モデルで作る（None なら手書きのインタプリタ）
    rehearsal_buffer: list = None      # 復習の材料を、保存した有限個の古い例に限る（None なら元の学習分布から無限に作る）
    dream_rehearsal: float = 0.0       # 復習のうち、バッファの古い規則から新しい例を夢で作る割合
    ewc_lambda: float = 0.0            # EWC の強さ（0 なら使わない）
    ewc_state: dict = None             # {"fisher": {...}, "anchor": {...}}


def dream(program, rng, n_examples=3, tries=30, world_model=None):
    """結論（規則）を新しい入力で頭の中で実行し、新しい例を作る。"""
    if world_model is not None:
        xs = [random_input(rng) for _ in range(n_examples + 2)]
        out = world_model.run_many([program], xs)
        ex = [(x, out[(program, tuple(x))]) for x in xs if out[(program, tuple(x))] is not None]
        return ex[:n_examples] if len(ex) >= n_examples else None
    ex = []
    for _ in range(tries):
        x = random_input(rng)
        y = safe_run(program, x)
        if y is not None:
            ex.append((x, y))
        if len(ex) == n_examples:
            return ex
    return None


def make_rehearsal_buffer(n, seed, block_size=192):
    """起きる前から持っている「古い知識」を、有限個の例として保存したもの。"""
    rng = random.Random(seed)
    buf = []
    while len(buf) < n:
        t = make_task(rng, "train")
        if encode_pair(t.train, t.program, block_size):
            buf.append((t.train, t.program))
    return buf


def _rehearsal_pair(rng, cfg, block_size):
    if cfg.rehearsal_buffer is None:
        while True:
            t = make_task(rng, "train")
            p = encode_pair(t.train, t.program, block_size)
            if p:
                return p
    ex, prog = rng.choice(cfg.rehearsal_buffer)
    if cfg.dream_rehearsal and rng.random() < cfg.dream_rehearsal:
        d = dream(prog, rng)
        p = d and encode_pair(d, prog, block_size)
        if p:
            return p
    return encode_pair(ex, prog, block_size)


def fisher_diag(model, pairs, batch_size=32, n_batches=20, seed=0):
    """EWC 用: 古い知識での損失の勾配の2乗（フィッシャー情報の対角近似）。"""
    rng = random.Random(seed)
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    model.eval()
    for _ in range(n_batches):
        x, y = collate([rng.choice(pairs) for _ in range(batch_size)])
        model.zero_grad(set_to_none=True)
        model(x, y)[1].backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.detach() ** 2 / n_batches
    model.zero_grad(set_to_none=True)
    return {"fisher": fisher, "anchor": {n: p.detach().clone() for n, p in model.named_parameters()}}


def _ewc_penalty(model, state):
    return sum((state["fisher"][n] * (p - state["anchor"][n]) ** 2).sum() for n, p in model.named_parameters())


def build_replay(memory, rng, cfg: SleepConfig, block_size):
    pairs = []
    for e in memory.episodes:
        if e.result != "verified" or not e.conclusion:
            continue
        obs = [(list(x), list(y)) for x, y in e.observation]
        p = encode_pair(obs, e.conclusion, block_size)
        if p:
            pairs.append(p)
        for _ in range(cfg.dreams_per_episode):
            d = dream(e.conclusion, rng, world_model=cfg.dream_world_model)
            p = d and encode_pair(d, e.conclusion, block_size)
            if p:
                pairs.append(p)
    return pairs


def sleep(model, memory, cfg: SleepConfig, seed=0, block_size=192) -> dict:
    """LLM の重みをその場で更新する。"""
    rng = random.Random(seed)
    replay = build_replay(memory, rng, cfg, block_size)
    if not replay:
        return {"replayed": 0, "steps": 0}
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0)
    n_old = int(round(cfg.batch_size * cfg.rehearsal))
    losses = []
    for _ in range(cfg.steps):
        batch = [rng.choice(replay) for _ in range(cfg.batch_size - n_old)]
        while len(batch) < cfg.batch_size:
            p = _rehearsal_pair(rng, cfg, block_size)
            if p:
                batch.append(p)
        x, y = collate(batch)
        _, loss = model(x, y)
        total = loss + (cfg.ewc_lambda * _ewc_penalty(model, cfg.ewc_state) if cfg.ewc_lambda else 0.0)
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    model.eval()
    return {"replayed": len(replay), "steps": cfg.steps,
            "loss_start": sum(losses[:20]) / min(20, len(losses)), "loss_end": sum(losses[-20:]) / min(20, len(losses))}


class AwakeLearner:
    """睡眠の構造を使わず、昼に経験した直後にその場で重みを更新する（対照実験用）。

    mode="experience": その経験の実際の観測だけで学習（夢も復習もなし）
    mode="mixture"   : 睡眠と同じ材料（これまでの全経験＋夢＋復習）を、同じ比率で、昼のうちに少しずつ学習
    """

    def __init__(self, model, cfg: SleepConfig, steps_per_episode: int, mode: str, seed=0, block_size=192):
        self.model, self.cfg, self.k, self.mode = model, cfg, steps_per_episode, mode
        self.rng = random.Random(seed)
        self.block_size = block_size
        self.opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0)
        self.steps = 0

    def after_episode(self, memory):
        e = memory.episodes[-1] if memory.episodes else None
        if e is None or e.result != "verified" or not e.conclusion:
            return
        if self.mode == "experience":
            obs = [(list(x), list(y)) for x, y in e.observation]
            p = encode_pair(obs, e.conclusion, self.block_size)
            if not p:
                return
            batches = [[p] * self.cfg.batch_size for _ in range(self.k)]
        else:
            pool = build_replay(_Last(memory), self.rng, self.cfg, self.block_size)
            self._pool = getattr(self, "_pool", []) + pool
            n_old = int(round(self.cfg.batch_size * self.cfg.rehearsal))
            batches = []
            for _ in range(self.k):
                b = [self.rng.choice(self._pool) for _ in range(self.cfg.batch_size - n_old)]
                while len(b) < self.cfg.batch_size:
                    q = _rehearsal_pair(self.rng, self.cfg, self.block_size)
                    if q:
                        b.append(q)
                batches.append(b)
        self.model.train()
        for b in batches:
            x, y = collate(b)
            _, loss = self.model(x, y)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            self.steps += 1
        self.model.eval()


class _Last:
    """build_replay に最新の1エピソードだけを渡すための薄い入れ物。"""

    def __init__(self, memory):
        self.episodes = memory.episodes[-1:]
