"""Exact finite-horizon I-POMDP agents, recursively exact down to uniform L0.

Every intentional agent, real or modeled, chooses uniformly among the exact
maximizers of its H-step receding decision problem. The problem is solved by
exhaustive enumeration (``FinitePolicyReference``) against its own exactly
solved lower-level models. This is the exact counterpart of an I-POMCP
hierarchy with the same frames, priors, physics, fixed receding lookahead and
belief update: only the finite search policies are replaced by exact ones.

Cost grows exponentially with horizon and nesting depth. Use it as a reference
at small horizons and levels, never as a planner.
"""

from ipomdp.belief.finite_belief import FiniteBelief, InteractiveState, MentalModel
from ipomdp.belief.frame import AgentFrame
from ipomdp.planning.planner import Planner
from ipomdp.planning.policy import sample_policy
from ipomdp.planning.random_planner import RandomPlanner
from ipomdp.planning.solver_bank import SolverBank
from ipomdp.planning.solver_types import SolverKey
from ipomdp.reference.finite_policy_reference import FinitePolicyReference

# Relative tolerance for treating exact Q values as tied. Exact enumeration in
# floating point can differ in the last bits between algebraically equal paths.
TIE_TOLERANCE = 1e-9


class ExactIPOMDPPlanner(Planner):
    def __init__(self, key, physics, bank, reference, horizon):
        self.key, self.pomdp_model, self.solver_bank = key, physics, bank
        self.reference, self.horizon = reference, horizon
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
        return sample_policy(self.policy_for(self.model(belief)))

    def update_root(self, action, observation):
        # Exact policies are pure functions of immutable models, so caches stay
        # valid across steps; only the real belief advances.
        self.belief = self.solver_bank.filter.update(
            self.model(), action, observation, terminal=False
        ).belief

    def get_detailed_stats(self):
        return {"action_values": {str(a): q for a, q in self.last_q.items()}}


def build_exact_agent(agent_id, opponent_id, level, physics, prior_states, nested, horizon, gamma):
    """Build an exact agent and its exact nested models in a fresh bank.

    Args:
        prior_states: Equally weighted physical prior, shared by every level.
        nested: {level: {opponent level: weight}} for every intentional level.
        horizon: Receding lookahead of every exact agent in the hierarchy.
    """
    bank = SolverBank(seed=0)  # exact policies consume no search randomness
    bank.set_physical_prior(physics, prior_states)
    reference = FinitePolicyReference(
        physics, bank, gamma, max_branches=10_000_000, cache_size=200_000
    )
    states = bank.physical_prior(physics)

    def create(who, lvl):
        key = SolverKey(who, lvl)
        if bank.has_solver(key):
            return bank.get_solver(key)
        if lvl == 0:
            solver = RandomPlanner(physics.get_all_actions(who))
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
            rows += [
                (InteractiveState(state, opponent), mass / total / len(states)) for state in states
            ]
        solver = ExactIPOMDPPlanner(key, physics, bank, reference, horizon)
        solver.set_initial_belief(FiniteBelief(tuple(rows)))
        bank.register_solver(key, solver)
        return solver

    return create(agent_id, level)
