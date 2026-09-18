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
"""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache

from core.pomdp_model import Action, Observation
from ipomdp.finite_belief import FiniteBelief, InteractiveState, MentalModel

Policy = Callable[[MentalModel], Mapping[Action, float]]


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


def checked_distribution(items, description):
    """Validate a probability kernel without silently normalizing model errors."""
    items = tuple(items)
    if not items or any(not math.isfinite(p) or p < 0 for _, p in items):
        raise ValueError(f"{description} must have finite nonnegative probabilities")
    if not math.isclose(math.fsum(p for _, p in items), 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{description} probabilities must sum to one")
    if len({value for value, _ in items}) != len(items):
        raise ValueError(f"{description} must not repeat outcomes")
    return tuple((value, p) for value, p in items if p > 0)


class FiniteInteractiveFilter:
    """Pure recursion with bounded, instance-owned memoization.

    Cache eviction changes cost only. Cached outputs are immutable; clearing the
    cache cannot change the target posterior or policy. The deliberate absence of
    random draws makes small-case results independent of query order and seeds.
    """

    def __init__(self, policy: Policy, cache_size: int = 4096):
        if type(cache_size) is not int or cache_size < 0:
            raise ValueError("cache_size must be a nonnegative integer")
        self._policy = policy
        self.update = lru_cache(maxsize=cache_size)(self._update)

    def action_distribution(self, opponent, physical_state):
        """L0 is uniformly random; intentional policies use only subjective beliefs.

        L0 legal actions may depend on physical state only when the domain guarantees
        that the availability is observable (e.g. one's own arrow inventory).
        The intentional callback receives no outer state, avoiding information leaks.
        """
        frame = opponent.frame
        if frame.level == 0:
            actions = frame.pomdp_model.get_legal_actions(physical_state, frame.agent_id)
            if not actions or len(set(actions)) != len(actions):
                raise ValueError("L0 needs a nonempty distinct legal action set")
            return tuple((action, 1.0 / len(actions)) for action in actions)
        distribution = checked_distribution(self._policy(opponent).items(), "Policy")
        legal_sets = {
            frozenset(frame.pomdp_model.get_legal_actions(atom.state, frame.agent_id))
            for atom, _ in opponent.belief.mass
        }
        if len(legal_sets) != 1:
            raise ValueError("Action availability differs across a subjective belief's support")
        legal = next(iter(legal_sets))
        if any(action not in legal for action, _ in distribution):
            raise ValueError("Policy assigns mass to an unavailable action")
        return distribution

    def _update(self, model: MentalModel, action: Action, observation: Observation) -> Posterior:
        if model.frame.level == 0:
            raise ValueError("Uniform-random L0 has no belief to update")
        frame = model.frame
        physics = frame.pomdp_model
        rows = []
        for atom, prior_mass in model.belief.mass:
            if action not in physics.get_legal_actions(atom.state, frame.agent_id):
                raise ValueError("Own action must be legal throughout the subjective belief")
            opponent = atom.opponent
            other = opponent.frame.agent_id
            for other_action, action_mass in self.action_distribution(opponent, atom.state):
                joint = {frame.agent_id: action, other: other_action}
                transitions = checked_distribution(
                    physics.transition_distribution(atom.state, joint), "Transition"
                )
                for following, transition_mass in transitions:
                    own_obs = checked_distribution(
                        (
                            (o, physics.get_observation_prob(o, following, joint, frame.agent_id))
                            for o in physics.get_all_observations(frame.agent_id)
                        ),
                        "Observation",
                    )
                    likelihood = dict(own_obs).get(observation, 0.0)
                    if likelihood == 0:
                        continue
                    weight = prior_mass * action_mass * transition_mass * likelihood
                    if opponent.frame.level == 0:
                        rows.append((InteractiveState(following, opponent), weight))
                        continue
                    # Private sensors describe the actual generative event; subjective
                    # filtering inside update uses the opponent frame's own dynamics.
                    private_obs = checked_distribution(
                        (
                            (o, physics.get_observation_prob(o, following, joint, other))
                            for o in physics.get_all_observations(other)
                        ),
                        "Private observation",
                    )
                    for private, private_mass in private_obs:
                        try:
                            belief = self.update(opponent, other_action, private).belief
                        except UnsupportedObservation as exc:
                            raise UnsupportedObservation(
                                f"Nested {opponent.frame!r} cannot condition on "
                                f"action={other_action!r}, observation={private!r}; "
                                "its subjective support contradicts a positive outer event"
                            ) from exc
                        advanced = MentalModel(opponent.frame, belief)
                        rows.append((InteractiveState(following, advanced), weight * private_mass))
        evidence = math.fsum(weight for _, weight in rows)
        if evidence <= 0:
            raise UnsupportedObservation(
                f"{frame!r}: zero finite-support evidence for "
                f"action={action!r}, observation={observation!r}"
            )
        return Posterior(FiniteBelief(tuple(rows)), evidence)
