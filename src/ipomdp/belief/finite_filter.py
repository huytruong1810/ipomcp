"""Enumerated recursive Bayesian conditioning for two-agent finite supports.

This kernel is exact up to floating-point arithmetic relative to its supplied
prior, transition kernel and policy provider. It is a bounded correctness reference
for sampled implementations, NOT an assertion that high-level enumeration is cheap.

For each (s, theta_j), enumerate a_j, s' and o_j with mass
    b_i(s, theta_j) pi_j(a_j | theta_j) T(s' | s, a_i, a_j)
    O_j(o_j | s', a_i, a_j) O_i(o_i | s', a_i, a_j).
Replace theta_j's belief by U_j(b_j, a_j, o_j), retaining the complete joint
hypothesis. U_j marginalizes unknown actions using j's SUBJECTIVE models. It never
receives the outer physical state or action. O_j is included once because private
observations are enumerated here; a sampling implementation must not double-weight it.

The sensors are conditionally independent given post-state and joint action, as
in the three current domains. Correlated sensors require a different joint kernel.
Policy and dynamics providers must remain fixed and pure for this object's lifetime.
An externally mutable MCTS node cannot be a policy or belief cache key.

Optional bounded support (``max_support``) makes the kernel an approximation:
any posterior with more distinct joint hypotheses than the bound is replaced by
a stratified systematic resample (see ``resample`` for strata and error bound).
It applies at every nesting level, because nested posteriors come from this same
update. Resampling is seeded from the canonical digest of the exact posterior,
so the update stays a deterministic, order-independent, cacheable function.
The default (None) keeps exact enumeration.
"""

import hashlib
import math
import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache

from ipomdp.belief import canonical
from ipomdp.belief.finite_belief import FiniteBelief, InteractiveState, MentalModel
from ipomdp.model.pomdp import Action, Observation

Policy = Callable[[MentalModel], Mapping[Action, float]]


class InferenceBudgetExceeded(RuntimeError):
    """Finite enumeration exceeded an explicit resource budget; no posterior is returned."""


class UnsupportedObservation(ValueError):
    """Zero evidence relative to the supplied finite model, not global impossibility.

    A sampled prior can have lost support. Even an exact subjective model can be
    misspecified. Neither case licenses substituting an initial or previous belief.
    Callers must preserve the failing action/observation and report the diagnosis.
    """


@dataclass(frozen=True, slots=True)
class Posterior:
    belief: FiniteBelief
    evidence: float


@lru_cache(maxsize=1024)
def _check_distribution_cached(items: tuple, description: str):
    if not items or any(not math.isfinite(p) or p < 0 for _, p in items):
        raise ValueError(f"{description} must have finite nonnegative probabilities")
    if not math.isclose(math.fsum(p for _, p in items), 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{description} probabilities must sum to one")
    if len({value for value, _ in items}) != len(items):
        raise ValueError(f"{description} must not repeat outcomes")
    return tuple((value, p) for value, p in items if p > 0)


def checked_distribution(items, description):
    """Validate a probability kernel without silently normalizing model errors."""
    return _check_distribution_cached(tuple(items), description)


class InteractiveFilter:
    """What every interactive belief update shares: opponents act by declared policies.

    Subclasses implement ``_update(model, action, observation, terminal)``,
    returning a ``Posterior``, and ``initial_belief(rows)``. ``update`` is the
    memoized ``_update``: an update must be a pure function of its arguments so
    that cached results, and the modeled policies computed from them, do not
    depend on query order. Cache eviction changes cost only.
    """

    def __init__(self, policy: Policy, cache_size: int = 4096):
        if type(cache_size) is not int or cache_size < 0:
            raise ValueError("cache_size must be a nonnegative integer")
        self._policy = policy
        self.update = lru_cache(maxsize=cache_size)(self._update)
        self._intentional_distribution = lru_cache(maxsize=cache_size)(self._validate_policy)

    def action_distribution(self, opponent):
        """The opponent's action distribution, a function of its mental model only.

        L0 is uniform over the full action set, like ``RandomPlanner``.
        Intentional models act by the supplied policy at their own belief. No
        physical state is an argument, so the prediction cannot depend on
        anything the opponent does not know.
        """
        frame = opponent.frame
        if frame.level == 0:
            actions = frame.pomdp_model.get_all_actions(frame.agent_id)
            if not actions or len(set(actions)) != len(actions):
                raise ValueError("L0 needs a nonempty distinct action set")
            return tuple((action, 1.0 / len(actions)) for action in actions)
        return self._intentional_distribution(opponent)

    def clear_caches(self):
        self.update.cache_clear()
        self._intentional_distribution.cache_clear()

    def _validate_policy(self, opponent):
        """Validate once per immutable model."""
        frame = opponent.frame
        distribution = checked_distribution(self._policy(opponent).items(), "Policy")
        actions = set(frame.pomdp_model.get_all_actions(frame.agent_id))
        if any(action not in actions for action, _ in distribution):
            raise ValueError("Policy assigns mass to an unknown action")
        return distribution

    def initial_belief(self, rows):
        """The belief an agent starts from, given the exact prior rows."""
        raise NotImplementedError

    def _update(self, model, action, observation, terminal=None):
        raise NotImplementedError


class FiniteInteractiveFilter(InteractiveFilter):
    """Exact enumeration of the I-POMDP belief update (see the module docstring)."""

    def __init__(
        self,
        policy: Policy,
        cache_size: int = 4096,
        max_branches: int = 100000,
        max_support: int | None = None,
    ):
        super().__init__(policy, cache_size)
        if type(max_branches) is not int or max_branches <= 0:
            raise ValueError("max_branches must be a positive integer")
        if max_support is not None and (type(max_support) is not int or max_support < 1):
            raise ValueError("max_support must be None or a positive integer")
        self.max_branches = max_branches
        self.max_support = max_support

    def initial_belief(self, rows):
        belief = FiniteBelief(tuple(rows))
        if self.max_support is not None and len(belief.mass) > self.max_support:
            belief = resample(belief, self.max_support)
        return belief

    def _update(
        self,
        model: MentalModel,
        action: Action,
        observation: Observation,
        terminal: bool | None = None,
    ) -> Posterior:
        if model.frame.level == 0:
            raise ValueError("Uniform-random L0 has no belief to update")
        frame = model.frame
        physics = frame.pomdp_model
        rows = []

        def append(row):
            if len(rows) >= self.max_branches:
                raise InferenceBudgetExceeded(
                    f"{frame!r}: enumerated branch budget {self.max_branches} exceeded"
                )
            rows.append(row)

        for atom, prior_mass in model.belief.mass:
            opponent = atom.opponent
            other = opponent.frame.agent_id
            for other_action, action_mass in self.action_distribution(opponent):
                joint = {frame.agent_id: action, other: other_action}
                transitions = checked_distribution(
                    physics.transition_distribution(atom.state, joint), "Transition"
                )
                for following, transition_mass in transitions:
                    ended = physics.is_terminal(following)
                    if terminal is not None and ended != terminal:
                        continue
                    own_obs = checked_distribution(
                        physics.observation_distribution(following, joint, frame.agent_id),
                        "Observation",
                    )
                    likelihood = 0.0
                    for obs_val, obs_prob in own_obs:
                        if obs_val == observation:
                            likelihood = obs_prob
                            break
                    if likelihood == 0:
                        continue
                    weight = prior_mass * action_mass * transition_mass * likelihood
                    if opponent.frame.level == 0:
                        append((InteractiveState(following, opponent), weight))
                        continue
                    # Private sensors describe the actual generative event; subjective
                    # filtering inside update uses the opponent frame's own dynamics.
                    private_obs = checked_distribution(
                        physics.observation_distribution(following, joint, other),
                        "Private observation",
                    )
                    for private, private_mass in private_obs:
                        try:
                            belief = self.update(
                                opponent, other_action, private, terminal=ended
                            ).belief
                        except UnsupportedObservation as exc:
                            raise UnsupportedObservation(
                                f"Nested {opponent.frame!r} cannot condition on "
                                f"action={other_action!r}, observation={private!r}; "
                                "its subjective support contradicts a positive outer event"
                            ) from exc
                        advanced = MentalModel(opponent.frame, belief)
                        append((InteractiveState(following, advanced), weight * private_mass))
        evidence = math.fsum(weight for _, weight in rows)
        if evidence <= 0:
            raise UnsupportedObservation(
                f"{frame!r}: zero finite-support evidence for "
                f"action={action!r}, observation={observation!r}"
            )
        belief = FiniteBelief(tuple(rows))
        if self.max_support is not None and len(belief.mass) > self.max_support:
            belief = resample(belief, self.max_support)
        return Posterior(belief, evidence)


def resample(belief: FiniteBelief, count: int) -> FiniteBelief:
    """Stratified systematic resampling to at most ``count`` atoms, deterministic.

    Why stratify. Strata are (opponent level, physical state), and each stratum
    keeps its exact total mass. The opponent-level posterior and the physical
    belief therefore stay exact. No level can be dropped by sampling, in
    particular the uniform L0 hypothesis, which supports every observation:
    plain resampling at small ``count`` dropped L0 and later produced
    zero-evidence failures. Only nested private models within a stratum are
    sampled.

    Allocation. With S strata and K = ``count`` >= S, stratum k with total mass
    t_k receives floor(1 + (K - S) t_k) >= 1 draws, plus one more by largest
    remainder (ties broken by stratum key) until the draws sum to K.

    Error bound. Systematic resampling gives each atom floor or ceil of
    n_k w / t_k hits, so its retained mass t_k hits / n_k differs from its exact
    mass w by less than t_k / n_k. Since n_k > (K - S) t_k, the per-atom mass
    error is below 1 / (K - S) whenever K > S. When K == S every stratum keeps
    one atom and the bound is only t_k. This is a per-update, per-atom bound.
    It does not bound error accumulated over time or across nesting levels,
    because every nested posterior is resampled as well.

    Determinism. Equal beliefs must give equal outputs regardless of the order
    in which their atoms were inserted. The update cache is keyed by belief
    equality, so an order-dependent result would make posteriors depend on
    query order. So strata, the atom order within each stratum and the seed
    all come from the canonical encoding in ``ipomdp.belief.canonical``.
    """
    strata = {}
    for atom, weight in belief.mass:
        key = (atom.opponent.frame.level, canonical.state_key(atom.state))
        strata.setdefault(key, []).append((atom, weight))
    if count < len(strata):
        raise ValueError("Support bound must be at least the number of level/state strata")
    if len(belief.mass) <= count:
        return belief
    keys = sorted(strata)
    for k in keys:
        strata[k].sort(key=canonical.atom_sort_key)
    totals = {k: math.fsum(w for _, w in strata[k]) for k in keys}
    quota = {k: 1 + (count - len(keys)) * totals[k] for k in keys}
    draws = {k: int(quota[k]) for k in keys}
    for k in sorted(keys, key=lambda k: (int(quota[k]) - quota[k], k))[
        : count - sum(draws.values())
    ]:
        draws[k] += 1
    seed = hashlib.sha256((canonical.belief_digest(belief) + str(count)).encode()).digest()
    rng = random.Random(seed)
    kept = []
    for k in keys:
        rows, n = strata[k], draws[k]
        if len(rows) <= n:
            kept.extend(rows)
            continue
        offset = rng.random() / n
        cumulative, index = 0.0, 0
        for atom, weight in rows:
            cumulative += weight / totals[k]
            hits = 0
            while index < n and offset + index / n < cumulative:
                hits += 1
                index += 1
            if hits:
                kept.append((atom, totals[k] * hits / n))
        if index < n:
            # Cumulative rounding can leave final draws; FiniteBelief merges duplicates.
            kept.append((rows[-1][0], totals[k] * (n - index) / n))
    return FiniteBelief(tuple(kept))
