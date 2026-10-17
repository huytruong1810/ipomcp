"""Compressed interactive filter: exact updates, supports bounded by N atoms.

Why not sample the update
-------------------------
The published interactive particle filter (``belief.particle_filter``) samples
every term of the update per particle: the opponent's action, the successor
state and the opponent's private observation. Two errors follow, measured on
L2 Tiger against exact beliefs (docs/RESULTS.md):

- **Quantization of small beliefs.** A nested belief with two physical states
  is held as N equal-weight samples, so its probabilities move in steps of
  1/N. A modeled opponent acts greedily, so a step across its decision
  threshold flips its predicted action; at N = 64 this alone raised one exact
  Q value from 3.3 to 6.3.
- **Accumulation.** Every in-tree and real update draws afresh, so error grows
  with history even where exact supports are small.

Update
------
Given a belief with at most N atoms:

1. enumerate the exact I-POMDP update of that belief (``FiniteInteractiveFilter``:
   every opponent action, successor state and private opponent observation, with
   nested opponents updated by this same filter, recursively);
2. if the posterior has more than N distinct atoms, replace it by N stratified
   draws in canonical order (``belief.resampling``; systematic draws aliased
   on structured uniform beliefs, docs/RESULTS.md 12.57), with identical draws merged.

Small beliefs therefore stay exact, and the only approximation is step 2,
whose per-atom mass error is below 2/N. ``evidence`` is the exact
P(o_i | compressed prior, a_i). Initial beliefs are the exact prior rows,
compressed the same way.

Cost: one update enumerates at most N x |A_j| x |S'| x |O_j| branches per
level, against N propagations for the sampled filter. Supports never exceed N at
any level, so the planner's per-history cost stays bounded as histories grow.
Measured on L2 Tiger mid-episode decisions, N = 64 was both faster than the
sampled filter at N = 64 and as accurate as exact beliefs (docs/RESULTS.md).

Compression draws from a stream seeded by the filter seed and the posterior's
canonical digest, so the update is a pure function, cacheable like the exact one.

Failure mode (stated, not repaired): with deterministic sensors (Wumpus), a
compressed prior can miss the true world, and the update then raises
``UnsupportedObservation``.
"""

import random

from ipomdp.belief import canonical
from ipomdp.belief.finite_belief import FiniteBelief
from ipomdp.belief.finite_filter import FiniteInteractiveFilter, Posterior
from ipomdp.belief.resampling import stratified_draws


class CompressedInteractiveFilter(FiniteInteractiveFilter):
    def __init__(
        self, policy, particles: int, seed: int, cache_size: int = 65536, max_branches=None
    ):
        if type(particles) is not int or particles < 1:
            raise ValueError("particles must be a positive integer")
        if type(seed) is not int:
            raise TypeError("A compressed filter needs an explicit integer seed")
        # An update enumerates at most N x |A_j| x |S'| x |O_j| branches; the
        # default guard is far above that for every current domain.
        # Cache: at 4,096 entries an L3 episode evicted and recomputed 22% of its
        # updates, and each recomputed posterior is a new object equal to cached
        # ones downstream, so later lookups compared whole beliefs recursively
        # (PLAN, cheaper nested solves, S3).
        super().__init__(policy, cache_size, max_branches or 10_000 * particles)
        self.particles, self.seed = particles, seed

    def initial_belief(self, rows):
        return self.compress(FiniteBelief(tuple(rows)), "initial")

    def _update(self, model, action, observation, terminal=None):
        posterior = super()._update(model, action, observation, terminal)
        return Posterior(self.compress(posterior.belief, "update"), posterior.evidence)

    def compress(self, belief, label="search"):
        """``belief`` itself if it has at most N atoms, else N stratified draws (merged)."""
        if len(belief.mass) <= self.particles:
            return belief
        ordered, belief_digest = canonical.canonical_mass_and_digest(belief)
        rng = random.Random(canonical.digest([self.seed, label, belief_digest]))
        draws = stratified_draws(ordered, self.particles, rng)
        return FiniteBelief(tuple((atom, 1.0) for atom in draws))
