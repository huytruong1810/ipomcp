"""The greedy decision rule shared by real decisions and opponent prediction.

Every planner and every modeled opponent distributes probability uniformly over
the actions with maximal estimated value. Using one rule for both keeps the
opponent model consistent with how the modeled agent would actually act. A
policy computed from a finite search is an approximation of the optimal policy,
not the optimal policy itself.

Ties: with ``tolerance`` 0 only exactly equal values tie. Exact values (an
exact reference, or a search whose chance scores are exact) are compared with
``TIE_TOLERANCE``, relative, because algebraically equal values computed along
different paths can differ in the last bits; the exact I-POMDP then plays all
its tied maximizers, and so does a modeled agent whose values are exact
(THEORY.md, "Tie convention"; RESULTS 12.87).
"""

import math

# Relative tolerance for treating exact Q values as tied.
TIE_TOLERANCE = 1e-9


def greedy_policy(values, tolerance=0.0):
    if not values or any(not math.isfinite(q) for q in values.values()):
        raise ValueError("A policy requires finite action values")
    best = max(values.values())
    floor = best - tolerance * max(1.0, abs(best))
    actions = [a for a, q in values.items() if q >= floor]
    return {a: 1.0 / len(actions) for a in actions}


def sample_policy(policy, rng):
    return rng.choices(list(policy), weights=list(policy.values()), k=1)[0]
