"""The solver bank: one agent's registry of nested models and their policies.

Responsibilities
----------------
A real agent and every model nested in its belief share one bank:

- **Registry.** One solver per (agent, level), so a modeled opponent's policy is
  computed by the solver declared for that level.
- **Physical prior.** One prior per domain, shared by every nesting level.
- **Policy cache.** A modeled policy is a pure function of the modeled agent's
  belief, because search randomness is seeded from the bank seed and the
  belief's canonical digest. Cached results are immutable values, so cache
  eviction changes cost only, never behaviour.
- **Interactive filter.** The belief update, parameterised by ``policy`` so that
  nested opponents act by their own modeled policies.
- **Expected rewards.** E[R | b, a] over the full joint belief, used by the
  planner at every tree history and at its exact last step.

Separate real agents use separate banks, so one agent's private models never
leak into the other's.
"""

import gc
import math
import random
from functools import lru_cache

from ipomdp.belief import canonical
from ipomdp.belief.finite_filter import FiniteInteractiveFilter, checked_distribution
from ipomdp.planning.solver_types import SolverKey


class SolverBank:
    def __init__(self, seed=None, max_support=None):
        """``max_support`` = None keeps exact enumeration; an integer K bounds
        every posterior at every nesting level by stratified resampling."""
        self._solvers = {}
        self._physical_priors = {}
        self.seed = random.getrandbits(64) if seed is None else seed
        self.filter = FiniteInteractiveFilter(self.policy, cache_size=4096, max_support=max_support)
        self._cached_policy = lru_cache(maxsize=32768)(self._evaluate_policy)
        # Canonical order makes random.choices intervals independent of how an
        # equal belief was built; see ipomdp.belief.canonical.
        self.ordered_mass = lru_cache(maxsize=16384)(canonical.canonical_mass)
        self.expected_rewards = lru_cache(maxsize=1024)(self._expected_rewards)

    # ------------------------------------------------------------------ #
    # Physical prior
    # ------------------------------------------------------------------ #

    def physical_prior(self, physics, samples=None):
        """The one physical prior shared by every model of this domain in the bank.

        It is either declared with ``set_physical_prior`` or, on first request,
        drawn as ``samples`` states from ``get_initial_state``. Sharing one
        unconditional prior across nesting levels prevents a nested model from
        ruling out an observation only because its own sample missed a state;
        it never conditions a private belief on the true state. A request whose
        sample size disagrees with the established prior is an error, because
        models built from different priors would not be comparable.
        """
        key = self._prior_key(physics)
        if key not in self._physical_priors:
            if samples is None:
                raise ValueError("No declared physical prior; state the number of prior samples")
            self._physical_priors[key] = tuple(physics.get_initial_state() for _ in range(samples))
        states = self._physical_priors[key]
        if samples is not None and len(states) != samples:
            raise ValueError("All models in one bank must share one physical prior")
        return states

    def set_physical_prior(self, physics, states):
        """Declare an exact, equally weighted physical prior before any model exists.

        A sampled prior only approximates a declared distribution such as a
        50/50 tiger. Equal-weight atoms are merged exactly, so a balanced tuple
        represents that distribution without sampling error. Replacing an
        established prior is refused.
        """
        states = tuple(states)
        if not states:
            raise ValueError("A physical prior needs at least one state")
        key = self._prior_key(physics)
        if key in self._physical_priors:
            raise ValueError("This bank already fixed a physical prior for these physics")
        self._physical_priors[key] = states

    @staticmethod
    def _prior_key(physics):
        return canonical.digest(canonical.physics_identity(physics))

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
            for other_action, probability in self.filter.action_distribution(
                atom.opponent, atom.state
            ):
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
