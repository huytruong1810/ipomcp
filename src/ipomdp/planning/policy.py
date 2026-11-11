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


def plausible_sets(bounds, tie=TIE_TOLERANCE):
    """(certain, plausible) actions from certified intervals {a: (lo, hi)} (audit spec 2.2).

    With M^- = max lo and M^+ = max hi, an action is plausible if hi >= M^- - tau^+
    (it may be within tau of the optimum) and certain if lo >= M^+ - tau^- (it is
    within tau of the optimum whatever the true values). The relative tolerance
    is made absolute conservatively: tau^+ = tie * max(1, |M^-|, |M^+|) and
    tau^- = tie * max(1, min(|M^-|, |M^+|)), or tie when [M^-, M^+] contains 0.
    Then certain is a subset of the exact optimal set, which is a subset of plausible.
    """
    if not bounds:
        raise ValueError("Plausible sets need at least one interval")
    m_lo = max(lo for lo, _ in bounds.values())
    m_hi = max(hi for _, hi in bounds.values())
    tau_plus = tie * max(1.0, abs(m_lo), abs(m_hi))
    tau_minus = tie if m_lo <= 0 <= m_hi else tie * max(1.0, min(abs(m_lo), abs(m_hi)))
    plausible = frozenset(a for a, (_, hi) in bounds.items() if hi >= m_lo - tau_plus)
    certain = frozenset(a for a, (lo, _) in bounds.items() if lo >= m_hi - tau_minus)
    return certain & plausible, plausible


def plausible_policy(bounds, tie=TIE_TOLERANCE):
    """(Unif(plausible), resolved, delta_hat): the prediction of Theorem 1b and its total-variation bound."""
    certain, plausible = plausible_sets(bounds, tie)
    policy = {a: 1.0 / len(plausible) for a in bounds if a in plausible}  # action order, as greedy_policy
    delta = 1.0 - max(1, len(certain)) / len(plausible)
    return policy, certain == plausible, delta
