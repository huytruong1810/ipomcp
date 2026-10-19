"""Exact finite-horizon I-POMDP agents, recursively exact down to uniform L0.

Every intentional agent, real or modeled, chooses uniformly among the exact
maximizers of its H-step receding decision problem. The problem is solved by
exhaustive enumeration (``BestResponseReference``) against its own exactly
solved lower-level models. This is the exact counterpart of an I-POMCP
hierarchy with the same frames, priors, physics, fixed receding lookahead and
belief update: only the finite search policies are replaced by exact ones.

Cost grows exponentially with horizon and nesting depth. Use it as a reference
at small horizons and levels, never as a planner.
"""

import random

from ipomdp.belief.finite_belief import FiniteBelief, InteractiveState, MentalModel
from ipomdp.belief.frame import AgentFrame
from ipomdp.planning.planner import Planner
from ipomdp.planning.policy import sample_policy
from ipomdp.planning.random_planner import RandomPlanner
from ipomdp.planning.solver_bank import SolverBank
from ipomdp.planning.solver_types import SolverKey
from ipomdp.reference.best_response import BestResponseReference

# Relative tolerance for treating exact Q values as tied. Exact enumeration in
# floating point can differ in the last bits between algebraically equal paths.
TIE_TOLERANCE = 1e-9


class ExactIPOMDPPlanner(Planner):
    def __init__(self, key, physics, bank, reference, horizon, tie_rng=None):
        self.key, self.pomdp_model, self.solver_bank = key, physics, bank
        self.reference, self.horizon = reference, horizon
        # The real agent's draws among tied maximizers; models only need policies.
        self.tie_rng = tie_rng
        self.belief = self.initial_belief = None
        self.last_q = {}

    def set_initial_belief(self, belief):
        self.belief = self.initial_belief = belief

    def model(self, belief=None):
        frame = AgentFrame(self.key.agent_id, self.key.level, self.pomdp_model)
        return MentalModel(frame, self.belief if belief is None else belief)

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
        if self.tie_rng is None:
            raise ValueError("An exact planner needs a tie_rng to act")
        return sample_policy(self.policy_for(self.model(belief)), self.tie_rng)

    def update_root(self, action, observation):
        # Exact policies are pure functions of immutable models, so caches stay
        # valid across steps; only the real belief advances.
        self.belief = self.solver_bank.filter.update(
            self.model(), action, observation, terminal=False
        ).belief

    def get_detailed_stats(self):
        return {"action_values": {str(a): q for a, q in self.last_q.items()}}


def build_exact_agent(
    agent_id, opponent_id, level, physics, nested, horizon, gamma, *, seed=0, max_branches=100_000
):
    """Build an exact agent and its exact nested models in a fresh bank.

    Args:
        nested: {level: {opponent level: weight}} for every intentional level.
        horizon: Receding lookahead of every exact agent in the hierarchy.
        seed: Seeds the real agent's tie-breaking stream only; exact policies
            consume no search randomness, so modeled policies do not depend on it.
        max_branches: Budget of each real-step belief update; above it the update
            raises ``InferenceBudgetExceeded`` (the end of the exact window).
    """
    bank = SolverBank(seed=seed, max_branches=max_branches)
    reference = BestResponseReference(
        physics, bank, gamma, max_branches=10_000_000, cache_size=200_000
    )
    prior = bank.physical_prior(physics)

    def create(who, lvl):
        key = SolverKey(who, lvl)
        if bank.has_solver(key):
            return bank.get_solver(key)
        if lvl == 0:
            solver = RandomPlanner(physics.get_all_actions(who), random.Random(0))
            bank.register_solver(key, solver)
            return solver
        other = opponent_id if who == agent_id else agent_id
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
            rows += [(InteractiveState(state, opponent), p * mass / total) for state, p in prior]
        tie_rng = random.Random(bank.stream_seed("exact/ties", who, lvl))
        solver = ExactIPOMDPPlanner(key, physics, bank, reference, horizon, tie_rng)
        solver.set_initial_belief(FiniteBelief(tuple(rows)))
        bank.register_solver(key, solver)
        return solver

    return create(agent_id, level)
