"""Sequential stopping for modeled policies: one growing tree, a calibrated test.

Why
---
A modeled opponent's policy is a greedy decision on searched values, and a
flat budget pays for the hardest decision at every belief the outer search
queries. Most queried beliefs are clear; a few are near ties that need
thousands of simulations (RESULTS.md Sections 12.5-12.6: the spread of a value
estimate, not the allocation, sets that floor). Spending adaptively per
belief is the lever.

Why one growing tree, not replicate searches
--------------------------------------------
The rule this module replaces averaged independent replicate searches of 200
simulations and stopped once their mean values separated (12.8). It is not
consistent: averaging converges to the value of a 200-simulation tree, which
is biased when such a tree is shallow for its horizon. For j-L2 at L3 H4 it
stayed 16-20/30 wrong on 0.19-gap beliefs even with a 25,600 cap, while one
search of 6,400 was 0/30 (12.10). Measured on those beliefs, one tree's scored
gap converges with its size, and its spread falls about as n^-0.8, faster
than averaging k replicates (k^-0.5): depth buys both bias and variance.

Rule
----
The decision always comes from one main tree, seeded exactly as a flat search
of the cap and grown by doubling from ``pilot`` simulations; at the cap it *is*
that flat search, so the rule is consistent. A single tree with Bellman
backups has no honest standard error, so the test is calibrated by
``trees - 1`` independent pilot trees of ``pilot`` simulations (seeded by
replicate index). With the main tree's own checkpoint at ``pilot`` they give
sigma, the standard deviation over ``trees`` trees of the scored difference
between the main tree's two leading actions. At checkpoint n the main tree
stops when its own leading gap exceeds

    t(1 - alpha / (2 K), trees - 1) * sigma * sqrt(pilot / n),

with K the number of checkpoints up to the cap (Bonferroni over the tests).
sqrt(pilot / n) assumes a tree's scored values have variance falling at least
as 1 / n, the Monte Carlo rate; measured, it falls about as n^-1.6, so the
test is conservative at larger n. Pilots cost (trees - 1) * pilot simulations
once per solve.
"""

import math
from dataclasses import dataclass

from scipy.stats import t as student


@dataclass(frozen=True)
class ModeledStopping:
    """Sequential stopping for modeled policies; the cap is ``modeled_simulations``.

    Attributes:
        pilot: Simulations of each pilot tree, and the main tree's first
            checkpoint (at least one per action).
        trees: Trees in the calibration, the main tree included (>= 2, for a
            standard deviation).
        alpha: Probability, over all checkpoints, of stopping when the leading
            gap is noise of the size the pilots measured.
    """

    pilot: int = 200
    trees: int = 4
    alpha: float = 0.01

    def __post_init__(self):
        if type(self.pilot) is not int or self.pilot < 1:
            raise ValueError("pilot must be a positive integer")
        if type(self.trees) is not int or self.trees < 2:
            raise ValueError("trees must be an integer >= 2")
        if not math.isfinite(self.alpha) or not 0 < self.alpha < 1:
            raise ValueError("alpha must lie in (0, 1)")

    def checkpoints(self, cap):
        """Main-tree sizes at which the test runs: pilot, 2 pilot, 4 pilot, ..., the cap."""
        if cap < self.pilot:
            raise ValueError("The pilot cannot exceed the modeled cap")
        sizes = []
        n = self.pilot
        while n < cap:
            sizes.append(n)
            n *= 2
        return (*sizes, cap)

    def separated(self, values, calibration, simulations, tests):
        """True when ``values``' leading gap is significant against the pilots' spread.

        Args:
            values: The main tree's scored action values at ``simulations``.
            calibration: Scored action values of every calibration tree at
                ``pilot`` simulations, the main tree's first checkpoint included.
            simulations: The main tree's current size.
            tests: Number of checkpoints up to the cap (Bonferroni).
        """
        if len(values) < 2:
            return True
        first, second = sorted(values, key=values.get, reverse=True)[:2]
        differences = [v[first] - v[second] for v in calibration]
        mean = math.fsum(differences) / len(differences)
        spread = math.sqrt(math.fsum((d - mean) ** 2 for d in differences) / (len(differences) - 1))
        critical = student.ppf(1 - self.alpha / (2 * tests), self.trees - 1)
        return bool(
            values[first] - values[second] > critical * spread * math.sqrt(self.pilot / simulations)
        )
