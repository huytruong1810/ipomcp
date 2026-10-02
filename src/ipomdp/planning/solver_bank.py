"""The solver bank: one agent's registry of nested models and their policies.

Responsibilities
----------------
A real agent and every model nested in its belief share one bank:

- **Registry.** One solver per (agent, level), so a modeled opponent's policy is
  computed by the solver declared for that level.
- **Physical prior.** The domain's exact prior, shared by every nesting level.
- **Policy cache.** A modeled policy is a pure function of the modeled agent's
  belief, because search randomness is seeded from the bank seed and the
  belief's canonical digest. Cached results are immutable values, so cache
  eviction changes cost only, never behaviour.
- **Interactive filter.** The belief update (exact or particle), parameterised
  by ``policy`` so that nested opponents act by their own modeled policies.
- **Expected rewards.** E[R | b, a] over the full joint belief, used by the
  planner at every tree history and at its exact last step.

Separate real agents use separate banks, so one agent's private models never
leak into the other's.
"""

import gc
import math
from functools import lru_cache

from ipomdp.belief import canonical
from ipomdp.belief.finite_filter import FiniteInteractiveFilter, checked_distribution
from ipomdp.belief.particle_filter import ParticleInteractiveFilter
from ipomdp.planning.solver_types import SolverKey


class SolverBank:
    def __init__(self, seed, *, particles=None, max_support=None):
        """``seed`` determines all randomness of this bank's models.

        Belief representation, at every nesting level:
        - default: exact enumeration (``FiniteInteractiveFilter``);
        - ``particles=N``: the interactive particle filter with N particles;
        - ``max_support=K``: exact updates compressed to at most K atoms.
        """
        if type(seed) is not int:
            raise TypeError("A solver bank needs an explicit integer seed")
        if particles is not None and max_support is not None:
            raise ValueError("Choose particles or a bounded exact support, not both")
        self._solvers = {}
        self._physical_priors = {}
        self.seed = seed
        self.filter = (
            ParticleInteractiveFilter(self.policy, particles, seed)
            if particles is not None
            else FiniteInteractiveFilter(self.policy, cache_size=4096, max_support=max_support)
        )
        self._cached_policy = lru_cache(maxsize=32768)(self._evaluate_policy)
        # Canonical order makes random.choices intervals independent of how an
        # equal belief was built; see ipomdp.belief.canonical.
        self.ordered_mass = lru_cache(maxsize=16384)(canonical.canonical_mass)
        self.expected_rewards = lru_cache(maxsize=1024)(self._expected_rewards)

    # ------------------------------------------------------------------ #
    # Physical prior
    # ------------------------------------------------------------------ #

    def physical_prior(self, physics):
        """The domain's exact prior, shared by every nesting level of this bank.

        Every model of a domain uses the same unconditional prior, so a nested
        model never rules out an observation merely because its prior differs;
        the prior never conditions on the true state.
        """
        key = canonical.digest(canonical.physics_identity(physics))
        if key not in self._physical_priors:
            self._physical_priors[key] = physics.initial_distribution()
        return self._physical_priors[key]

    # ------------------------------------------------------------------ #
    # Registry
    # ------------------------------------------------------------------ #

    def register_solver(self, key, solver):
        if key in self._solvers:
            raise ValueError(f"Solver already registered: {key}")
        self._solvers[key] = solver

    def has_solver(self, key):
        return key in self._solvers

    def get_solver(self, key):
        return self._solvers[key]

    def get_solver_for_frame(self, frame):
        solver = self.get_solver(SolverKey(frame.agent_id, frame.level))
        if frame.level and solver.pomdp_model is not frame.pomdp_model:
            raise ValueError("Frame physics differs from the registered solver")
        return solver

    # ------------------------------------------------------------------ #
    # Policies and integrals
    # ------------------------------------------------------------------ #

    def stream_seed(self, *labels):
        """Seed for a named random stream of this bank (for example an agent's actions)."""
        return canonical.digest([self.seed, *map(canonical.stable_value, labels)])

    def search_seed(self, model, settings):
        """Search randomness for one solve: equal problems get equal seeds."""
        return canonical.digest(
            [
                self.seed,
                canonical.stable_value(model.frame.agent_id),
                model.frame.level,
                canonical.belief_digest(model.belief),
                canonical.stable_value(settings),
            ]
        )

    def policy(self, model):
        """The modeled policy of an intentional mental model (cached, immutable)."""
        return dict(self._cached_policy(model))

    def _evaluate_policy(self, model):
        solver = self.get_solver_for_frame(model.frame)
        return tuple(solver.policy_for(model, modeled=True).items())

    def _expected_rewards(self, model):
        """E[R | b, a] for every action, integrated over the full joint belief.

        Sums over physical states, opponent models, the opponents' modeled
        action distributions and finite transitions. The planner uses it at
        every tree history in place of the sampled immediate reward: action
        selection never inspects the sampled hidden state, so the expected
        backup target is unchanged while the variance of rare penalties
        disappears. At the last decision these are the complete action values.
        """
        frame, physics = model.frame, model.frame.pomdp_model
        all_actions = physics.get_all_actions(frame.agent_id)
        terms = {a: [] for a in all_actions}
        for atom, mass in model.belief.mass:
            if physics.is_terminal(atom.state):
                continue
            other_id = atom.opponent.frame.agent_id
            for other_action, probability in self.filter.action_distribution(atom.opponent):
                for action in all_actions:
                    joint = {frame.agent_id: action, other_id: other_action}
                    for following, transition_mass in checked_distribution(
                        physics.transition_distribution(atom.state, joint), "Transition"
                    ):
                        terms[action].append(
                            mass
                            * probability
                            * transition_mass
                            * physics.get_reward(atom.state, joint, following, frame.agent_id)
                        )
        return tuple((action, math.fsum(terms[action])) for action in all_actions)

    def clear_caches(self):
        """Drop cached policies, posteriors and integrals (memory only, never results)."""
        self._cached_policy.cache_clear()
        self.filter.clear_caches()
        self.ordered_mass.cache_clear()
        canonical.belief_digest.cache_clear()
        self.expected_rewards.cache_clear()
        gc.collect()
