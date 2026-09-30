"""Exact finite-horizon I-POMDP agents for the L3-vs-L2 Tiger study.

Every intentional agent, real or modeled, chooses uniformly among the exact
maximizers of its H-step receding decision problem, solved by exhaustive
enumeration (FinitePolicyReference) against its own exactly solved lower-level
models. L0 is uniform random. This is the I-POMCP design's exact counterpart:
same frames, priors, physics, fixed receding lookahead and exact filter; only
the finite MCTS policy providers are replaced by exact solutions.
"""

import random
from functools import lru_cache

from examples.experiments.tiger_l3_vs_l2_experiment import NESTED_I, NESTED_J, tiger
from examples.tiger.model.tiger_model import TIGER_LEFT, TIGER_RIGHT
from ipomdp.finite_belief import FiniteBelief, InteractiveState, MentalModel
from ipomdp.frame import AgentFrame
from solvers.exact.finite_policy_reference import FinitePolicyReference
from solvers.planner import Planner
from solvers.policy import sample_policy
from solvers.random_planner import RandomPlanner
from solvers.solver_bank import SolverBank
from solvers.solver_types import SolverKey

TIE_TOLERANCE = 1e-9


class ExactIPOMDPPlanner(Planner):
    def __init__(self, key, physics, bank, reference, horizon):
        self.key, self.pomdp_model, self.solver_bank = key, physics, bank
        self.reference, self.horizon = reference, horizon
        self.belief = self.initial_belief = None
        self.last_q = {}

    def set_initial_belief(self, belief, count):
        self.belief = self.initial_belief = belief

    def model(self, belief=None):
        return MentalModel(
            AgentFrame(self.key.agent_id, self.key.level, self.pomdp_model),
            self.belief if belief is None else belief,
        )

    def q_values(self, model):
        return self.reference.q_values(model, self.horizon)

    def policy_for(self, model, modeled=False):
        q = self.q_values(model)
        best = max(q.values())
        chosen = [a for a, v in q.items() if v >= best - TIE_TOLERANCE * max(1.0, abs(best))]
        if not modeled:
            self.last_q = q
        return {a: 1.0 / len(chosen) for a in chosen}

    def get_action(self, belief=None):
        return sample_policy(self.policy_for(self.model(belief)))

    def update_root(self, action, observation):
        # Exact policies are deterministic functions of immutable models, so
        # caches stay valid across steps; only the real belief advances.
        self.belief = self.solver_bank.filter.update(
            self.model(), action, observation, terminal=False
        ).belief

    def get_detailed_stats(self):
        return {"action_values": {str(a): q for a, q in self.last_q.items()}}


def build_exact_agent(agent_id, level, horizon, gamma=0.95, cache=200000, physics=None):
    """Mirror I_POMDP_Bootstrapper._create with exact policy providers.

    Pass the I-POMCP agent's physics object to make models comparable: frames
    compare physics by identity, so a shadow sharing it holds equal models.
    """
    physics = physics or tiger()
    bank = SolverBank(seed=0)  # exact policies consume no search randomness
    bank.set_physical_prior(physics, (TIGER_LEFT, TIGER_RIGHT))
    reference = FinitePolicyReference(physics, bank, gamma, max_branches=10_000_000)
    # Nested exact solves revisit many models; enlarge the default caches.
    reference._q = lru_cache(maxsize=cache)(reference._q.__wrapped__)
    reference.branches = lru_cache(maxsize=cache)(reference.branches.__wrapped__)
    nested = NESTED_I if agent_id == "i" else NESTED_J

    def create(who, lvl):
        key = SolverKey(who, lvl)
        if bank.has_solver(key):
            return bank.get_solver(key)
        if lvl == 0:
            solver = RandomPlanner(physics.get_all_actions(who))
            bank.register_solver(key, solver)
            return solver
        other = "j" if who == "i" else "i"
        for lower in range(lvl):
            create(who, lower)
            create(other, lower)
        weights = nested.get(lvl) or dict.fromkeys(range(lvl), 1 / lvl)
        total = sum(weights.values())
        rows = []
        for lower, mass in weights.items():
            if mass <= 0:
                continue
            solver = bank.get_solver(SolverKey(other, lower))
            opponent = MentalModel(
                AgentFrame(other, lower, physics), solver.initial_belief if lower else None
            )
            for state in bank.initial_states(physics, 2):
                rows.append((InteractiveState(state, opponent), mass / total / 2))
        solver = ExactIPOMDPPlanner(key, physics, bank, reference, horizon)
        solver.set_initial_belief(FiniteBelief(tuple(rows)), 2)
        bank.register_solver(key, solver)
        return solver

    return create(agent_id, level)


if __name__ == "__main__":
    import sys
    import time

    horizon = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    random.seed(0)
    for agent, level in (("j", 2), ("i", 3)):
        start = time.perf_counter()
        planner = build_exact_agent(agent, level, horizon)
        policy = planner.policy_for(planner.model())
        print(agent, level, horizon, policy, planner.last_q, f"{time.perf_counter() - start:.1f}s")
