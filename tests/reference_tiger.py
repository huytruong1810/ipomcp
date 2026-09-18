"""Exact finite-state reference for Tiger against fixed subintentional policies.

This test oracle enumerates states, opponent actions, and observations. It is not
an oracle for intentional, recursively updating opponents. Keeping this small
reference independent of the particle filter prevents a test from merely copying
the implementation's resampling or tree bookkeeping mistakes.
"""

from functools import lru_cache

from examples.tiger.model.tiger_model import (
    LISTEN,
    OPEN_LEFT,
    OPEN_RIGHT,
    TIGER_LEFT,
    TIGER_RIGHT,
    TigerModel,
)

STATES = (TIGER_LEFT, TIGER_RIGHT)
ACTIONS = (LISTEN, OPEN_LEFT, OPEN_RIGHT)


def transition_probability(previous, following, action_i, action_j, persistent=False):
    if persistent or action_i == action_j == LISTEN:
        return float(previous == following)
    return 0.5


def posterior(prior, action, observation, policies, model=None):
    """Normalize joint (state,type) mass after marginalizing the hidden action.

    prior maps (physical state, type) to mass; policies maps type to a fixed
    categorical action policy. Types are constant across physical resets.
    """
    model = model or TigerModel()
    unnormalized = {}
    for (previous, kind), mass in prior.items():
        for other_action, probability in policies[kind].items():
            joint = {"i": action, "j": other_action}
            for following in STATES:
                weight = (
                    mass
                    * probability
                    * transition_probability(
                        previous, following, action, other_action, model.persistent
                    )
                    * model.get_observation_prob(observation, following, joint, "i")
                )
                key = (following, kind)
                unnormalized[key] = unnormalized.get(key, 0.0) + weight
    total = sum(unnormalized.values())
    if total == 0:
        raise ValueError("Impossible observation under reference model")
    return {key: weight / total for key, weight in unnormalized.items()}


def finite_horizon_values(probability_left, depth, gamma=0.95, opponent_policy=None):
    """Exact Bellman enumeration for one fixed state-independent opponent policy."""
    model = TigerModel()
    policy = opponent_policy or {action: 1 / 3 for action in ACTIONS}

    @lru_cache(None)
    def values(p_left, remaining):
        if remaining == 0:
            return (0.0,) * len(ACTIONS)
        result = []
        for action in ACTIONS:
            reward = 0.0
            observations = {}
            for previous, mass in [(TIGER_LEFT, p_left), (TIGER_RIGHT, 1 - p_left)]:
                for other, p_action in policy.items():
                    joint = {"i": action, "j": other}
                    for following in STATES:
                        predictive = (
                            mass
                            * p_action
                            * transition_probability(previous, following, action, other)
                        )
                        reward += predictive * model.get_reward(previous, joint, following, "i")
                        for obs in model.get_all_observations("i"):
                            weight = predictive * model.get_observation_prob(
                                obs, following, joint, "i"
                            )
                            if weight:
                                total, left = observations.get(obs, (0.0, 0.0))
                                observations[obs] = (
                                    total + weight,
                                    left + weight * (following == TIGER_LEFT),
                                )
            continuation = sum(
                mass * max(values(left / mass, remaining - 1))
                for mass, left in observations.values()
            )
            result.append(reward + gamma * continuation)
        return tuple(result)

    return dict(zip(ACTIONS, values(probability_left, depth)))
