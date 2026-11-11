"""Systematic and stratified resampling: N draws from a weighted set with low added variance.

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


def stratified_draws(weighted_atoms, count, rng):
    """``count`` draws from ((atom, weight), ...) by stratified resampling: one point per 1/N stratum."""
    total = math.fsum(w for _, w in weighted_atoms)
    points = [(index + rng.random()) / count for index in range(count)]
    draws, cumulative, index = [], 0.0, 0
    for atom, weight in weighted_atoms:
        cumulative += weight / total
        while index < count and points[index] < cumulative:
            draws.append(atom)
            index += 1
    # Floating-point shortfall at the end of the cumulative sum goes to the last atom.
    draws.extend(weighted_atoms[-1][0] for _ in range(count - index))
    return draws


def head_tail(weighted_atoms, count, tail, rng):
    """Head-tail compression (audit spec 2.3): ((atom, mass), ...) with at most ``count`` atoms, and its loss.

    The ``count - tail`` heaviest atoms keep their exact masses (ties broken by
    the order given, which callers make canonical). The rest, of total mass
    epsilon (the loss), is replaced by ``tail`` stratified draws from it in the
    order given, each of mass epsilon / tail; with ``tail`` 0 the head is
    renormalized. The L1 error is at most 2 epsilon for every draw, the draws
    are unbiased when ``tail`` >= 1, and with ``tail`` 0 the error is exactly
    2 epsilon, the least of any belief with at most ``count`` atoms.
    """
    if type(count) is not int or count < 1 or type(tail) is not int or not 0 <= tail < count:
        raise ValueError("Need 0 <= tail < count")
    total = math.fsum(w for _, w in weighted_atoms)
    normalized = [(atom, w / total) for atom, w in weighted_atoms]
    if len(normalized) <= count:
        return tuple(normalized), 0.0
    order = sorted(range(len(normalized)), key=lambda i: -normalized[i][1])  # stable: ties keep the given order
    head_index = set(order[: count - tail])
    head = [normalized[i] for i in range(len(normalized)) if i in head_index]
    rest = [normalized[i] for i in range(len(normalized)) if i not in head_index]
    loss = math.fsum(w for _, w in rest)
    if tail == 0:
        kept = math.fsum(w for _, w in head)
        return tuple((atom, w / kept) for atom, w in head), loss
    draws = stratified_draws(rest, tail, rng)
    return tuple(head) + tuple((atom, loss / tail) for atom in draws), loss
