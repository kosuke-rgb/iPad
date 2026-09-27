"""Brain-inspired エージェント本体。

各認知モジュールは AgentConfig のフラグで ON/OFF でき、
Baseline A〜E と提案モデル、および「提案モデル − 1モジュール」のアブレーションを
同じコードで同一条件のもと比較できる。
"""
from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import dataclass, field, replace

from core.critic.critic import Critic
from core.hypothesis.generator import compose, mutations
from core.llm.backend import ComputeMeter, LanguageModel
from core.memory.long_term import Episode, LongTermMemory, StaticKnowledgeBase
from core.router.router import Router
from core.simulator.simulator import BudgetExceeded, Simulator
from core.working_memory.working_memory import Hypothesis, WorkingMemory
from core.world_model.dsl import PRIM_NAMES, safe_run
from core.world_model.features import features, vector


@dataclass
class AgentConfig:
    name: str
    use_llm: bool = True
    n_samples: int = 0           # 追加サンプル数（0 なら貪欲な1回答のみ）
    temperature: float = 1.0
    self_consistency: bool = False
    use_simulator: bool = False
    use_rag: bool = False
    use_memory: bool = False
    memory_version: int = 2      # 1: 初版（固定重みのマクロ）, 2: 証拠に基づくマクロ＋類似経験のみ
    use_macros: bool = False     # マクロは有意な効果が確認できなかったので既定で OFF（docs/05）
    use_experts: bool = False
    use_router: bool = False
    use_compose: bool = False
    use_repair: bool = False
    use_critic: bool = False
    enumerate: bool = False      # LLM なしの総当たり探索
    sim_budget: int = 2000

    def variant(self, name, **kw):
        return replace(self, name=name, **kw)


@dataclass
class Result:
    program: tuple | None
    predictions: list
    meter: ComputeMeter
    stages: list = field(default_factory=list)
    source: str = ""
    verified: bool = False
    ambiguous: bool = False
    n_hypotheses: int = 0


class Agent:
    def __init__(self, cfg: AgentConfig, lm: LanguageModel | None,
                 kb: StaticKnowledgeBase | None = None, memory: LongTermMemory | None = None):
        self.cfg = cfg
        self.lm = lm
        self.kb = kb
        self.memory = memory if memory is not None else LongTermMemory()
        self.router = Router(cfg.use_router)
        self.critic = Critic()

    # ------------------------------------------------------------------ stages
    def _stages(self):
        c = self.cfg
        s = []
        if c.use_llm:
            s.append("direct")
        if c.use_memory:
            s.append("memory")
        if c.use_rag:
            s.append("rag")
        if c.use_experts:
            s.append("experts")
        if c.use_llm and c.n_samples:
            s.append("sample")
        if c.use_compose:
            s.append("compose")
        if c.use_repair:
            s.append("repair")
        return self.router.plan(s)

    def _propose(self, stage, examples, fvec, feats, wm, meter):
        c = self.cfg
        if stage == "direct":
            return [Hypothesis(p, "llm", lp) for p, lp in self.lm.sample(examples, 1, 0.0, meter)]
        if stage == "sample":
            return [Hypothesis(p, "llm", lp) for p, lp in self.lm.sample(examples, c.n_samples, c.temperature, meter)]
        if stage == "memory":
            min_sim = 0.8 if c.memory_version >= 2 else 0.0
            return [Hypothesis(p, "memory") for p in
                    self.memory.retrieve(fvec, 8, verified_only=c.use_simulator, min_similarity=min_sim)]
        if stage == "rag":
            return [Hypothesis(p, "rag") for p in self.kb.retrieve(fvec, 8)]
        if stage == "experts":
            return [Hypothesis(p, "expert:" + e.name) for e in self.router.select_experts(feats) for p in e.propose()]
        if stage == "compose":
            frags = wm.fragment_stats()
            for p in PRIM_NAMES:  # 未観測の部品にも小さな確率を残す（探索の完全性）
                frags[p] += 0.05
            macros = self._macros(fvec)
            return [Hypothesis(p, "compose") for p in compose(frags, macros)]
        if stage == "repair":
            out = []
            for h in wm.best_partial(3):
                out += [Hypothesis(p, "repair") for p in mutations(h.program)]
            return out
        raise ValueError(stage)

    def _macros(self, fvec):
        c = self.cfg
        if not (c.use_memory and c.use_macros):
            return []
        if c.memory_version == 1:
            return self.memory.macros()
        return self.memory.weighted_macros(fvec, min_support=2.0, min_similarity=0.5)

    # ------------------------------------------------------------------ solve
    def solve(self, examples, test_inputs) -> Result:
        c = self.cfg
        meter = ComputeMeter()
        if c.enumerate:
            return self._enumerate(examples, test_inputs, meter)
        sim = Simulator(meter, c.sim_budget) if c.use_simulator else None
        wm = WorkingMemory(goal="観測を説明するプログラムを見つける", observations=examples)
        feats = features(examples)
        fvec = vector(feats)
        stages_run = []
        try:
            for stage in self._stages():
                if self.router.should_stop(wm):
                    break
                stages_run.append(stage)
                wm.next_action = stage
                for h in self._propose(stage, examples, fvec, feats, wm, meter):
                    if not wm.add(h) or sim is None:
                        continue
                    outs = sim.execute(h.program, examples)
                    fit = sum(o == y for o, (_, y) in zip(outs, examples)) / len(examples)
                    wm.mark(h.program, fit)
                    if fit < 1.0:
                        h.fit = fit + 0.5 * self.critic.soft_score(outs, examples) / 2
                    elif stage in ("compose", "repair") and self.router.enabled:
                        break  # 大きな探索は最初の検証済み仮説で打ち切る
        except BudgetExceeded:
            wm.note("シミュレーション予算を使い切った")
        res = self._decide(wm, examples, test_inputs, meter, sim)
        res.stages = stages_run
        res.n_hypotheses = len(wm.hypotheses)
        self._remember(examples, fvec, res, meter)
        return res

    def _decide(self, wm, examples, test_inputs, meter, sim) -> Result:
        c = self.cfg
        hyps = list(wm.hypotheses.values())
        if sim is not None:
            verified = wm.verified()
            if verified:
                if c.use_critic:
                    self._score_missing(verified, examples, meter)
                    best, amb = self.critic.choose(verified, test_inputs)
                else:
                    best, amb = verified[0], False
                return Result(best.program, [safe_run(best.program, x) for x in test_inputs],
                              meter, source=best.source, verified=True, ambiguous=amb)
            partial = [h for h in hyps if h.fit]
            best = max(partial, key=lambda h: h.fit) if partial else (hyps[0] if hyps else None)
        elif c.self_consistency:
            votes = Counter()
            first = {}
            for h in hyps:
                key = tuple(tuple(o) if (o := safe_run(h.program, x)) is not None else None for x in test_inputs)
                votes[key] += 1
                first.setdefault(key, h)
            best = first[votes.most_common(1)[0][0]] if votes else None
        else:
            self._score_missing(hyps, examples, meter)
            best = max(hyps, key=lambda h: h.logprob if h.logprob is not None else -1e9) if hyps else None
        if best is None:
            return Result(None, [None] * len(test_inputs), meter)
        return Result(best.program, [safe_run(best.program, x) for x in test_inputs], meter, source=best.source)

    def _score_missing(self, hyps, examples, meter):
        if self.lm is None:
            return
        todo = [h for h in hyps if h.logprob is None]
        for h, s in zip(todo, self.lm.score(examples, [h.program for h in todo], meter)):
            h.logprob = s

    def _remember(self, examples, fvec, res, meter):
        if not self.cfg.use_memory:
            return
        self.memory.store(Episode(
            observation=[list(map(list, e)) for e in examples], features=fvec,
            hypotheses_tried=res.n_hypotheses, experiments=meter.sim_calls,
            result="verified" if res.verified else ("unverified" if res.program else "failed"),
            conclusion=res.program, source=res.source))

    def _enumerate(self, examples, test_inputs, meter):
        sim = Simulator(meter, self.cfg.sim_budget)
        try:
            for d in range(1, 5):
                for p in itertools.product(PRIM_NAMES, repeat=d):
                    if sim.fits(p, examples) == 1.0:
                        return Result(p, [safe_run(p, x) for x in test_inputs], meter,
                                      source="enumerate", verified=True, stages=["enumerate"])
        except BudgetExceeded:
            pass
        return Result(None, [None] * len(test_inputs), meter, stages=["enumerate"])


# ---------------------------------------------------------------------- configs
BASE = AgentConfig(name="A: Dense LLM")
PROPOSED = AgentConfig(
    name="Proposed", n_samples=16, use_simulator=True, use_memory=True, use_experts=True,
    use_router=True, use_compose=True, use_repair=True, use_critic=True)


def baseline_configs():
    return [
        BASE,
        BASE.variant("B: LLM + RAG", use_rag=True),
        BASE.variant("C: LLM + Long-term Memory", use_memory=True),
        BASE.variant("D: LLM + Sparse Experts", use_experts=True, use_simulator=True, use_router=True),
        BASE.variant("E: LLM + Dynamic Reasoning (self-consistency x16)", n_samples=16, self_consistency=True),
        BASE.variant("S: LLM + Simulator (sample & verify x16)", n_samples=16, use_simulator=True, use_critic=True),
        AgentConfig(name="N: Enumeration (no LLM)", use_llm=False, enumerate=True),
        PROPOSED,
    ]


def memory_configs():
    """長期記憶の設計比較（docs/05_memory_redesign.md）。"""
    return [
        PROPOSED.variant("Memory: none", use_memory=False),
        PROPOSED.variant("Memory v1 (fixed-weight macros)", memory_version=1, use_macros=True),
        PROPOSED.variant("Memory v2 retrieval only (no macros)", use_macros=False),
        PROPOSED.variant("Memory v2 (evidence-weighted macros)", use_macros=True),
    ]


def ablation_configs():
    return [
        PROPOSED,
        PROPOSED.variant("Proposed - Memory", use_memory=False),
        PROPOSED.variant("Proposed - Router", use_router=False),
        PROPOSED.variant("Proposed - Experts", use_experts=False),
        PROPOSED.variant("Proposed - Compose", use_compose=False),
        PROPOSED.variant("Proposed - Repair", use_repair=False),
        PROPOSED.variant("Proposed - Critic", use_critic=False),
        PROPOSED.variant("Proposed - LLM samples", n_samples=0),
    ]
