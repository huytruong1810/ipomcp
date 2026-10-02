"""Systematic resampling: N draws from a weighted set with the least added variance.

One uniform offset u in [0, 1/N) and the N evenly spaced points u + k/N are
matched against the cumulative weights, in the order given. Each atom of weight
w is drawn floor(N w) or ceil(N w) times, so the per-atom mass error is below
1/N and small discrete beliefs are reproduced almost exactly (a 50/50 belief
with N = 64 gives exactly 32/32; independent draws gave 28/36 in a probe).
Callers pass atoms in canonical order, so the draws depend only on what the
belief is, never on how it was built.
"""

import math


def systematic_draws(weighted_atoms, count, rng):
    """``count`` draws from ((atom, weight), ...) by systematic resampling."""
    total = math.fsum(w for _, w in weighted_atoms)
    offset = rng.random() / count
    draws, cumulative, index = [], 0.0, 0
    for atom, weight in weighted_atoms:
        cumulative += weight / total
        while index < count and offset + index / count < cumulative:
            draws.append(atom)
            index += 1
    # Floating-point shortfall at the end of the cumulative sum goes to the last atom.
    draws.extend(weighted_atoms[-1][0] for _ in range(count - index))
    return draws
