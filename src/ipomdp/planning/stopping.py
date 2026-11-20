"""Deterministic resolution for modeled policies: one tree grown to the cap.

Why
---
A modeled opponent's policy is a greedy decision on searched values, and a
flat budget pays for the hardest decision at every belief the outer search
queries. Most queried beliefs are clear; a few are near ties that need
thousands of simulations. Spending adaptively per belief is the lever.

Rule
----
The decision comes from one tree, seeded exactly as a flat search of the cap
and grown by doubling from ``pilot`` simulations; at the cap it *is* that flat
search. After each doubling, the tree's certified root intervals decide
whether to stop (``ModeledResolution``). The test needs no pilot trees and no
distributional assumption: the intervals contain the exact values at every
checkpoint.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ModeledResolution:
    """Deterministic stopping for modeled solves (THEORY.md, "Modeled levels and bi-level soundness"); the cap is ``modeled_simulations``.

    One tree, grown by doubling from ``pilot`` to the cap. After each doubling
    its certified root intervals (widened by the solve's deviation bound) give
    the plausible and certain action sets (``policy.plausible_sets``); the
    solve stops when they agree (resolved), when the relative width of the
    leading interval is at most ``rel_width``, or at the cap. No pilot trees,
    no distributional assumption, no selection of a tested pair: the plausible
    set contains the exact optimal set at every checkpoint.

    Attributes:
        pilot: First checkpoint (at least one simulation per action).
        rel_width: Also stop when (M^+ - M^-) / W_H <= rel_width (0: never).
        tie: Tie tolerance tau of the exact solution (relative, as ``policy.TIE_TOLERANCE``).
    """

    pilot: int = 50
    rel_width: float = 0.0
    tie: float = 1e-9

    def __post_init__(self):
        if type(self.pilot) is not int or self.pilot < 1:
            raise ValueError("pilot must be a positive integer")
        if not math.isfinite(self.rel_width) or self.rel_width < 0:
            raise ValueError("rel_width must be finite and nonnegative")
        if not math.isfinite(self.tie) or self.tie < 0:
            raise ValueError("tie must be finite and nonnegative")

    def checkpoints(self, cap):
        """Tree sizes at which resolution is checked: pilot, 2 pilot, 4 pilot, ..., the cap."""
        if cap < self.pilot:
            raise ValueError("The pilot cannot exceed the modeled cap")
        sizes, n = [], self.pilot
        while n < cap:
            sizes.append(n)
            n *= 2
        return (*sizes, cap)
