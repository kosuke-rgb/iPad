"""自律的発見ループ。

人間から問題を与えられるのではなく、エージェント自身が
「どのブラックボックスを調べるか」「どんな実験（入力）を試すか」を決める。

  Current Knowledge → Identify Unknowns → Generate Hypotheses → Rank Hypotheses
  → Experiment（ブラックボックスへの問い合わせ）→ Observe → Critic
  → Update Memory → Generate New Questions（次に調べる対象を好奇心で選ぶ）→ Repeat
"""
from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field

from core.critic.critic import Critic
from core.hypothesis.generator import compose, mutations
from core.hypothesis.novelty import NoveltyEngine
from core.llm.backend import ComputeMeter, LanguageModel
from core.memory.long_term import Episode, LongTermMemory
from core.simulator.simulator import BudgetExceeded, Simulator
from core.working_memory.working_memory import Hypothesis, WorkingMemory
from core.world_model.dsl import PRIM_NAMES, safe_run
from core.world_model.features import features, vector
from core.world_model.tasks import random_input, signature


class Oracle:
    """中身の見えない現象（隠れたプログラム）。問い合わせると出力だけ返す。"""

    def __init__(self, program):
        self._program = program
        self.queries = 0

    def __call__(self, x):
        self.queries += 1
        return safe_run(self._program, x)

    def is_identified_by(self, program) -> bool:
        return program is not None and signature(program) == signature(self._program)


@dataclass
class Investigation:
    oracle: Oracle
    observations: list
    wm: WorkingMemory
    concluded: tuple | None = None
    candidates: list = field(default_factory=list)  # 観測と矛盾しない仮説
    generated: bool = False


def _prior(program) -> float:
    return 0.3 ** len(program)


def _entropy(groups: Counter) -> float:
    tot = sum(groups.values())
    return -sum(v / tot * math.log(v / tot) for v in groups.values() if v)


class DiscoveryAgent:
    def __init__(self, lm: LanguageModel | None, memory: LongTermMemory | None, active: bool,
                 curiosity: bool, sim_budget_per_round: int = 1500, max_candidates: int = 64,
                 rng: random.Random | None = None):
        self.lm = lm
        self.memory = memory
        self.active = active
        self.curiosity = curiosity
        self.budget = sim_budget_per_round
        self.max_candidates = max_candidates
        self.rng = rng or random.Random(0)
        self.critic = Critic()
        self.meter = ComputeMeter()
        self.novelty = NoveltyEngine(memory.known_programs() if memory else ())
        self.log: list = []

    # -------------------------------------------------------- hypotheses
    def _generate(self, inv: Investigation):
        """観測と矛盾しない仮説集合を作る。"""
        obs = inv.observations
        sim = Simulator(self.meter, self.meter.sim_calls + self.budget)
        wm = inv.wm
        proposals = []
        if self.lm is not None:
            proposals += [Hypothesis(p, "llm", lp) for p, lp in self.lm.sample(obs, 32, 1.0, self.meter)]
        if self.memory is not None:
            proposals += [Hypothesis(p, "memory") for p in self.memory.retrieve(vector(features(obs)), 8, True)]
        consistent = []
        try:
            for h in proposals:
                wm.add(h)
                if sim.fits(h.program, obs) == 1.0:
                    consistent.append(h.program)
                    wm.mark(h.program, 1.0)
                else:
                    wm.mark(h.program, 0.0)
            frags = wm.fragment_stats()
            for p in PRIM_NAMES:
                frags[p] += 0.05
            macros = self.memory.macros() if self.memory is not None else []
            extra = compose(frags, macros, limit=self.budget)
            for h in wm.best_partial(2):
                extra += mutations(h.program)
            for p in extra:
                if len(consistent) >= self.max_candidates:
                    break
                if p not in consistent and sim.fits(p, obs) == 1.0:
                    consistent.append(p)
        except BudgetExceeded:
            pass
        inv.candidates = list(dict.fromkeys(inv.candidates + consistent))
        inv.generated = True

    def _prune(self, inv: Investigation):
        """Critic: 新しい観測と矛盾する仮説を捨てる。"""
        x, y = inv.observations[-1]
        inv.candidates = [p for p in inv.candidates if safe_run(p, x) == y]

    CONFIRMATION_UNCERTAINTY = 0.8
    NO_HYPOTHESIS_UNCERTAINTY = 0.7  # 仮説が1つもない対象（∞にすると、そこへ問い合わせが吸い込まれ続ける）

    def _uncertainty(self, inv: Investigation, pool) -> float:
        if not inv.candidates:
            return self.NO_HYPOTHESIS_UNCERTAINTY
        best = 0.0
        for x in pool:
            g = Counter()
            for p in inv.candidates:
                o = safe_run(p, x)
                g[tuple(o) if o is not None else None] += _prior(p)
            best = max(best, _entropy(g))
        if best == 0.0 and len(inv.observations) < 3:
            return self.CONFIRMATION_UNCERTAINTY  # 仮説は1つに絞れたが、確認の実験がまだ
        return best

    # -------------------------------------------------------- experiment design
    def _design_experiment(self, inv: Investigation):
        pool = [random_input(self.rng) for _ in range(40)]
        if not self.active or len(inv.candidates) < 2:
            return pool[0]
        def gain(x):
            g = Counter()
            for p in inv.candidates:
                o = safe_run(p, x)
                g[tuple(o) if o is not None else None] += _prior(p)
            return _entropy(g)
        return max(pool, key=gain)

    # -------------------------------------------------------- main loop
    def run(self, programs, total_queries: int, initial_obs: int = 2) -> dict:
        invs = []
        for prog in programs:
            o = Oracle(prog)
            obs = [(x, o(x)) for x in (random_input(self.rng) for _ in range(initial_obs))]
            invs.append(Investigation(o, obs, WorkingMemory("現象の規則を特定する", obs)))
        used = sum(i.oracle.queries for i in invs)
        while used < total_queries:
            open_invs = [i for i in invs if i.concluded is None]
            if not open_invs:
                break
            for i in open_invs:
                if not i.generated or not i.candidates:
                    self._generate(i)
                    if self._settled(i):
                        self._conclude(i)
            open_invs = [i for i in open_invs if i.concluded is None]
            if not open_invs:
                break
            # Generate New Questions: 最も不確かな対象を次に調べる（好奇心）
            if self.curiosity:
                pool = [random_input(self.rng) for _ in range(10)]
                target = max(open_invs, key=lambda i: self._uncertainty(i, pool))
            else:
                target = self.rng.choice(open_invs)
            x = self._design_experiment(target)
            target.observations.append((x, target.oracle(x)))
            used += 1
            self._prune(target)
            if not target.candidates:
                target.generated = False  # すべて反証された → 仮説を作り直す
                continue
            if self._settled(target):
                self._conclude(target)
        for i in invs:
            if i.concluded is None and i.candidates:
                self._conclude(i, final=True)
        return self._report(invs, used)

    @staticmethod
    def _settled(inv: Investigation) -> bool:
        """残った仮説の振る舞いが1つに絞れ、観測も3つ以上あれば結論を出す。"""
        return (bool(inv.candidates) and len(inv.observations) >= 3
                and len({signature(p) for p in inv.candidates}) == 1)

    def _conclude(self, inv: Investigation, final=False):
        best = min(inv.candidates, key=lambda p: (len(p), len(self.critic.flags(p))))
        inv.concluded = best
        nov = self.novelty.evaluate(best, fit=1.0)
        self.log.append({"program": "|".join(best), "queries": inv.oracle.queries,
                         "novelty": round(nov["novelty"], 3), "forced": final})
        if self.memory is not None:
            self.memory.store(Episode(
                observation=[list(map(list, e)) for e in inv.observations],
                features=vector(features(inv.observations)), hypotheses_tried=len(inv.wm.hypotheses),
                experiments=inv.oracle.queries, result="verified", conclusion=best, source="discovery"))
            self.novelty.add_known(best)

    def _report(self, invs, used) -> dict:
        correct = [i.oracle.is_identified_by(i.concluded) for i in invs]
        return {"identified": sum(correct), "total": len(invs), "queries_used": used,
                "llm_gflops": self.meter.llm_flops / 1e9, "sim_calls": self.meter.sim_calls,
                "log": self.log}
