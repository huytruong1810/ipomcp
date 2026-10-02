"""The greedy decision rule shared by real decisions and opponent prediction.

Every planner and every modeled opponent distributes probability uniformly over
the actions with maximal estimated value. Using one rule for both keeps the
opponent model consistent with how the modeled agent would actually act. A
policy computed from a finite search is an approximation of the optimal policy,
not the optimal policy itself.
"""

import math


def greedy_policy(values):
    if not values or any(not math.isfinite(q) for q in values.values()):
        raise ValueError("A policy requires finite action values")
    best = max(values.values())
    actions = [a for a, q in values.items() if q == best]
    return {a: 1.0 / len(actions) for a in actions}


def sample_policy(policy, rng):
    return rng.choices(list(policy), weights=list(policy.values()), k=1)[0]
