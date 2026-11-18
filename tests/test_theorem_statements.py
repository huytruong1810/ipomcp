"""Numerical checks of theorem statements that no other test exercises (the theorem ledger's missing rows).

A statement is checked on many random instances against its own inequality, so a misstated constant (a missing
factor 1/2, a wrong norm) shows up as a counterexample. This is evidence for the statement as written, not a proof.
"""

import random

import pytest


@pytest.mark.parametrize("seed", range(50))
def test_the_l1_lemma_holds_with_its_constant_and_the_constant_is_attained(seed):
    """Supplement, lem:l1: f(b) = max_sigma alpha_sigma . b with c_lo <= alpha <= c_hi is (c_hi - c_lo)/2-Lipschitz in l1.

    The factor 1/2 is tight: two point masses on states where one vector is c_hi and c_lo reach it.
    """
    rng = random.Random(seed)
    states, vectors = rng.randint(2, 6), rng.randint(1, 6)
    c_lo, c_hi = sorted(rng.uniform(-50, 50) for _ in range(2))
    alphas = [[rng.uniform(c_lo, c_hi) for _ in range(states)] for _ in range(vectors)]

    def f(b):
        return max(sum(a * p for a, p in zip(alpha, b)) for alpha in alphas)

    def belief():
        raw = [rng.random() ** 3 for _ in range(states)]
        return [w / sum(raw) for w in raw]

    for _ in range(200):
        b, c = belief(), belief()
        distance = sum(abs(x - y) for x, y in zip(b, c))
        assert abs(f(b) - f(c)) <= 0.5 * (c_hi - c_lo) * distance + 1e-9
    # Tightness: one vector equal to c_hi on state 0 and c_lo on state 1, point masses on those states.
    tight = [c_hi, c_lo] + [c_lo] * (states - 2)
    alphas[:] = [tight]
    b, c = [1.0] + [0.0] * (states - 1), [0.0, 1.0] + [0.0] * (states - 2)
    assert abs(f(b) - f(c)) == pytest.approx(0.5 * (c_hi - c_lo) * 2.0)
