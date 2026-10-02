"""One greedy policy contract and reproducible, isolated search randomness.

MCTS and sampled lookahead approximate Q; action execution and opponent prediction
both distribute mass uniformly over maximizing estimates. A bounded solve is not
an optimal policy. In particular, different real/model budgets remain an explicit
approximation. Nested solves restore the caller's RNG stream even on failure.
"""

import math
import random
from contextlib import contextmanager


def greedy_policy(values):
    if not values or any(not math.isfinite(q) for q in values.values()):
        raise ValueError("A policy requires finite action values")
    best = max(values.values())
    actions = [a for a, q in values.items() if q == best]
    return {a: 1.0 / len(actions) for a in actions}


def sample_policy(policy):
    return random.choices(list(policy), weights=list(policy.values()), k=1)[0]


@contextmanager
def search_randomness(seed):
    """Process-local serial searches only; simultaneous threads are not supported."""
    previous = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(previous)
