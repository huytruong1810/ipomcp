"""Per-experiment solvers, immutable policy results and recursive inference.

One bank fixes physics/configuration per (agent, level), and owns its random seed.
Cached policy results are values, not evolving trees. Deterministic private solves
make cache eviction harmless to the modeled policy. Private beliefs are never
stored in a registered planner's live root on behalf of another agent.
"""

import random
from functools import lru_cache

from ipomdp.finite_filter import FiniteInteractiveFilter
from solvers.policy import digest, stable_value
from solvers.solver_types import SolverKey


class SolverBank:
    def __init__(self, seed=None):
        self._solvers = {}
        self._physical_priors = {}
        self.seed = random.getrandbits(64) if seed is None else seed
        self.filter = FiniteInteractiveFilter(self.policy, cache_size=512)
        self._cached_policy = lru_cache(maxsize=512)(self._evaluate_policy)
        self._belief_digest = lru_cache(maxsize=4096)(self._encode_belief)

    def initial_states(self, physics, count):
        """One shared empirical physical prior per domain within this bank.

        Independent finite priors can make a simulated observation unsupported by
        the opponent solely through sampling. Sharing the same unconditional prior
        measure avoids that incoherence; it never conditions a private belief on
        the outer true state. Different banks still use independent prior samples.
        Changing its size after constructing models would invalidate those models.
        """
        key = digest(
            [type(physics).__module__, type(physics).__qualname__, stable_value(vars(physics))]
        )
        if key not in self._physical_priors:
            self._physical_priors[key] = tuple(physics.get_initial_state() for _ in range(count))
        states = self._physical_priors[key]
        if len(states) != count:
            raise ValueError(
                "All models in one bank must share the same initial physical sample count"
            )
        return states

    def register_solver(self, key, solver):
        if key in self._solvers:
            raise ValueError(f"Solver already registered: {key}")
        self._solvers[key] = solver

    def has_solver(self, key):
        return key in self._solvers

    def __contains__(self, key):
        return key in self._solvers

    def get_solver(self, key):
        return self._solvers[key]

    def get_solver_for_frame(self, frame):
        return self.get_solver(SolverKey(frame.agent_id, frame.level))

    def _encode_belief(self, belief):
        rows = []
        for atom, mass in belief.mass:
            opponent = atom.opponent
            physics = opponent.frame.pomdp_model
            rows.append(
                [
                    stable_value(atom.state),
                    mass.hex(),
                    stable_value(opponent.frame.agent_id),
                    opponent.frame.level,
                    type(physics).__module__,
                    type(physics).__qualname__,
                    stable_value(vars(physics)),
                    self._belief_digest(opponent.belief) if opponent.belief else None,
                ]
            )
        import json

        return digest(sorted(rows, key=json.dumps))

    def search_seed(self, model, settings):
        return digest(
            [
                self.seed,
                stable_value(model.frame.agent_id),
                model.frame.level,
                self._belief_digest(model.belief),
                stable_value(settings),
            ]
        )

    def _evaluate_policy(self, model):
        solver = self.get_solver_for_frame(model.frame)
        return tuple(solver.policy_for(model, modeled=True).items())

    def policy(self, model):
        return dict(self._cached_policy(model))

    def clear_caches(self):
        self._cached_policy.cache_clear()
        self.filter.clear_caches()
        self._belief_digest.cache_clear()
