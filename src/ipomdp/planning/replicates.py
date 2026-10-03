"""Sequential replicate searches for modeled policies.

Why
---
A modeled opponent's policy is a greedy decision on searched values, and a
flat budget pays for the hardest decision at every belief the outer search
queries. Measured on L2 Tiger H4 (RESULTS.md Sections 12.5-12.6): the j-L1
decisions that stay wrong up to 1,600 simulations are near ties (gaps of
0.2-0.3), UCB already splits visits evenly between the two contenders, and the
spread of a value estimate is about 6.5 per simulation. Nothing in action
selection can beat that floor; only spending more where the decision is close
can. Most queried beliefs are clear and need a few hundred simulations.

Rule
----
Run independent replicate searches of ``batch`` simulations (root
parallelization, Chaslot et al. 2008) with seeds keyed by replicate index.
Replicates give an honest standard error, which a single tree with Bellman
backups does not. After ``min_replicates``, stop as soon as the two best
actions' mean values differ by more than ``z`` standard errors of their
difference (fixed-confidence best-arm identification, cf. LUCB, Kalyanakrishnan
et al. 2012), or when the cap (``modeled_simulations``) is spent. The policy is
greedy on the replicate-mean values.

Offline A/B on the 235 distinct j-L1 beliefs queried at L2 H4, 5 seeds each
(``results/sequential_replicates_ab_20261003``): with a 6,400 cap it is correct
on 1,175/1,175 at 699 simulations on average, against 1,175/1,175 at 1,600 for
a single search. At an equal 1,600 cap it is worse on near ties (17/20 vs
20/20): small trees lose depth, so the gain is adaptivity, not replication.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ModeledStopping:
    """Sequential stopping for modeled policies; the cap is ``modeled_simulations``.

    Attributes:
        batch: Simulations per replicate search (at least one per action).
        min_replicates: Replicates before the first stopping test (>= 2, for a
            standard error).
        z: Separation, in standard errors of the difference between the two
            best mean values, at which the decision is taken.
    """

    batch: int = 200
    min_replicates: int = 3
    z: float = 3.0

    def __post_init__(self):
        if type(self.batch) is not int or self.batch < 1:
            raise ValueError("batch must be a positive integer")
        if type(self.min_replicates) is not int or self.min_replicates < 2:
            raise ValueError("min_replicates must be an integer >= 2")
        if not math.isfinite(self.z) or self.z <= 0:
            raise ValueError("z must be finite and positive")

    def separated(self, samples):
        """True when the two best replicate-mean values are ``z`` standard errors apart."""
        if len(samples) < self.min_replicates:
            return False
        stats = sorted((_mean_and_variance(samples, a) for a in samples[0]), reverse=True)
        if len(stats) < 2:
            return True
        (first, first_var), (second, second_var) = stats[:2]
        return first - second > self.z * math.sqrt(first_var + second_var)


def mean_values(samples):
    """Replicate-mean value of every action."""
    return {action: math.fsum(s[action] for s in samples) / len(samples) for action in samples[0]}


def _mean_and_variance(samples, action):
    """(mean, variance of the mean) of one action's value over replicates."""
    values = [s[action] for s in samples]
    mean = math.fsum(values) / len(values)
    variance = math.fsum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, variance / len(values)
