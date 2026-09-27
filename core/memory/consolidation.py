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


def dream(program, rng, n_examples=3, tries=30):
    """結論（規則）を新しい入力で頭の中で実行し、新しい例を作る。"""
    ex = []
    for _ in range(tries):
        x = random_input(rng)
        y = safe_run(program, x)
        if y is not None:
            ex.append((x, y))
        if len(ex) == n_examples:
            return ex
    return None


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
            d = dream(e.conclusion, rng)
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
            t = make_task(rng, "train")
            p = encode_pair(t.train, t.program, block_size)
            if p:
                batch.append(p)
        x, y = collate(batch)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    model.eval()
    return {"replayed": len(replay), "steps": cfg.steps,
            "loss_start": sum(losses[:20]) / min(20, len(losses)), "loss_end": sum(losses[-20:]) / min(20, len(losses))}
