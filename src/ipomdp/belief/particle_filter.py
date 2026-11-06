"""Interactive particle filter (IPF): the sampled I-POMDP belief update.

Reference: P. Doshi and P. J. Gmytrasiewicz, "Monte Carlo Sampling Methods for
Approximating Interactive POMDPs", JAIR 34 (2009).

``ParticleInteractiveFilter`` samples every term of the update, including the
other agent's private observation, and holds every nested level as particles.
``PublishedParticleFilter`` (below) is the filter exactly as published: the
bottom intentional level is updated exactly and the other agent's observations
are enumerated (see its docstring and RESULTS 12.116).
The planner's default bounded belief is ``belief.compressed_filter``, which
enumerates each update instead of sampling it and is far more accurate at the
same N (see that module).

Why
---
The exact update enumerates every opponent action, successor state and private
opponent observation at every nesting level. Exact supports therefore grow
geometrically with time and nesting (2,428 atoms at step 16 for L2 Tiger), and
domains with large priors (Wumpus: 94,500 states) cannot be represented at all.
The IPF replaces enumeration by sampling, so a belief at every level is a
weighted set of at most N particles and the cost of one update is bounded by N
propagations per level.

Update (agent i, own action a_i, own observation o_i)
-----------------------------------------------------
1. Draw N particles (s, m_j) from the current belief by systematic resampling
   (``belief.resampling``).
2. For each: draw a_j from the opponent's declared policy at m_j, draw
   s' ~ T(s, a_i, a_j), and weight by O_i(o_i | s', a_i, a_j).
3. If the opponent is intentional and the episode continues, draw its private
   observation o_j ~ O_j(s', a) and advance m_j by the same particle update,
   recursively: nested beliefs also hold at most N particles.
4. The posterior is the weighted set, with identical particles merged.
   ``evidence`` is the mean weight, an unbiased estimate of P(o_i | b, a_i).

Every update draws from its own ``random.Random`` seeded by the filter seed and
the canonical digests of its arguments. The update is therefore a pure
function, cacheable like the exact one, and modeled opponent policies remain
pure functions of their (particle) beliefs.

Failure mode, and the smoothed variant
--------------------------------------
With ``smoothing`` 0 (the published filter), if no particle explains o_i the
update raises ``UnsupportedObservation``; the same holds for a nested model
that cannot explain its own sampled observation. With noisy sensors (Tiger,
UAV) every observation has positive likelihood under every particle, so this
cannot happen. With deterministic sensors (Wumpus) the particle set can miss
the true world, and such failures surface explicitly (RESULTS 12.59).

``smoothing`` ε > 0 is the standard likelihood-smoothing repair: every
particle is weighted by (1 - ε) O_i(o_i | s', a) + ε / |Ω_i|, at every nesting
level, so no declared observation has zero likelihood and the update never
raises for lack of support. ``evidence`` then estimates P(o_i | b, a_i) under the
smoothed sensor. ε = 0 is the published filter, draw for draw: weights are
deterministic, so the random stream is the same.

Initial beliefs are N draws from the exact prior rows (equal weights merged).
"""

import math
import random

from ipomdp.belief import canonical
from ipomdp.belief.finite_belief import FiniteBelief, InteractiveState, MentalModel
from ipomdp.belief.finite_filter import (
    FiniteInteractiveFilter,
    InteractiveFilter,
    Posterior,
    UnsupportedObservation,
)
from ipomdp.belief.resampling import systematic_draws

# The ε of the solver bank's "smoothed" filter, declared before the G3 runs
# (docs/PLAN.md, publication gaps): a 1% uniform sensor-noise floor.
IPF_SMOOTHING = 0.01


class ParticleInteractiveFilter(InteractiveFilter):
    def __init__(
        self, policy, particles: int, seed: int, cache_size: int = 4096, smoothing: float = 0.0
    ):
        super().__init__(policy, cache_size)
        if type(particles) is not int or particles < 1:
            raise ValueError("particles must be a positive integer")
        if type(seed) is not int:
            raise TypeError("A particle filter needs an explicit integer seed")
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        self.particles, self.seed, self.smoothing = particles, seed, smoothing

    def initial_belief(self, rows):
        exact = FiniteBelief(tuple(rows))
        rng = self._rng("initial", canonical.belief_digest(exact))
        drawn = systematic_draws(canonical.canonical_mass(exact), self.particles, rng)
        return FiniteBelief(tuple((atom, 1.0) for atom in drawn))

    def _update(self, model, action, observation, terminal=None):
        if model.frame.level == 0:
            raise ValueError("Uniform-random L0 has no belief to update")
        frame, physics = model.frame, model.frame.pomdp_model
        rng = self._rng(
            "update",
            canonical.belief_digest(model.belief),
            canonical.stable_value(model.frame.agent_id),
            canonical.stable_value(action),
            canonical.stable_value(observation),
            terminal,
        )
        observations = physics.get_all_observations(frame.agent_id)
        # Smoothing spreads ε over the agent's declared observations only, so an
        # observation outside them still has no support and still raises.
        floor = self.smoothing / len(observations) if observation in observations else 0.0
        rows, total_weight = [], 0.0
        for atom in systematic_draws(canonical.canonical_mass(model.belief), self.particles, rng):
            opponent = atom.opponent
            other = opponent.frame.agent_id
            actions, policy = zip(*self.action_distribution(opponent))
            joint = {frame.agent_id: action, other: rng.choices(actions, weights=policy, k=1)[0]}
            following = physics.sample_transition(atom.state, joint, rng)
            ended = physics.is_terminal(following)
            if terminal is not None and ended != terminal:
                continue
            weight = physics.observation_probability(observation, following, joint, frame.agent_id)
            if self.smoothing:
                weight = (1.0 - self.smoothing) * weight + floor
            if weight == 0:
                continue
            if opponent.frame.level > 0 and not ended:
                private = physics.sample_observation(following, joint, other, rng)
                try:
                    nested = self.update(opponent, joint[other], private, terminal=False).belief
                except UnsupportedObservation as exc:
                    raise UnsupportedObservation(
                        f"Nested {opponent.frame!r} cannot explain its own sampled "
                        f"action={joint[other]!r}, observation={private!r}"
                    ) from exc
                opponent = MentalModel(opponent.frame, nested)
            rows.append((InteractiveState(following, opponent), weight))
            total_weight += weight
        if not rows:
            raise UnsupportedObservation(
                f"{frame!r}: no particle explains action={action!r}, observation={observation!r}"
            )
        return Posterior(FiniteBelief(tuple(rows)), total_weight / self.particles)

    def _rng(self, *labels):
        return random.Random(canonical.digest([self.seed, *labels]))


class PublishedParticleFilter(ParticleInteractiveFilter):
    """The interactive particle filter exactly as Doshi & Gmytrasiewicz (2009) publish it (their Figs. 4 and 5).

    It differs from ``ParticleInteractiveFilter`` (this module's sampled variant) in three published details:

    - **The bottom intentional level is exact.** Their level 0 is a POMDP that treats the other agent's actions
      as noise (LEVEL0BELIEFUPDATE, Fig. 5); here that is a level-1 model facing a uniform level 0. Its belief is
      a distribution over states, updated exactly inside every particle (``FiniteInteractiveFilter``), never
      held as particles.
    - **The other agent's observation is enumerated, not sampled** (Fig. 4, lines 6 and 15): each propagated
      particle (s, m_j) spawns one particle per o_j, carrying m_j updated by o_j and weighted by
      O_j(o_j | s', a) O_i(o_i | s', a). The |Omega_j| N weighted particles are then resampled.
    - **Resampling is multinomial,** N draws with replacement after weighting (line 19), and a belief is a
      multiset of N particles: each particle is propagated with its own draws of a_j and s'.

    The other agent's action is drawn from its policy (uniform over its optimal set, as ``greedy_policy``
    gives), and s' from T. A branch whose nested update has no support (a deterministic percept that the other
    agent's own belief rules out) is dropped, the reading under which the published normalization is defined.
    Smoothing, when set, applies to each agent's own observation weight at every level, as in the sampled
    variant. ``evidence`` is the mean weight, an unbiased estimate of P(o_i | b, a_i).

    Initial beliefs: a level-1 prior stays exact; a higher prior is N multinomial draws from the exact rows.
    """

    def __init__(self, policy, particles: int, seed: int, cache_size: int = 4096, smoothing: float = 0.0):
        super().__init__(policy, particles, seed, cache_size, smoothing)
        self.exact = FiniteInteractiveFilter(policy, cache_size, max_branches=10_000_000)

    def initial_belief(self, rows):
        exact = FiniteBelief(tuple(rows))
        if all(atom.opponent.frame.level == 0 for atom, _ in exact.mass):
            return exact
        rng = self._rng("initial-published", canonical.belief_digest(exact))
        ordered = canonical.canonical_mass(exact)
        atoms, weights = zip(*ordered)
        return FiniteBelief(tuple((atom, 1.0) for atom in rng.choices(atoms, weights=weights, k=self.particles)))

    def _particles(self, belief, rng):
        """The belief as a list of N particles: its counts if it is an N-particle multiset, else N draws."""
        ordered = canonical.canonical_mass(belief)
        total = math.fsum(w for _, w in ordered)
        counts = [w / total * self.particles for _, w in ordered]
        rounded = [round(c) for c in counts]
        if sum(rounded) == self.particles and all(abs(c - r) < 1e-6 for c, r in zip(counts, rounded)):
            return [atom for (atom, _), r in zip(ordered, rounded) for _ in range(r)]
        atoms, weights = zip(*ordered)
        return rng.choices(atoms, weights=weights, k=self.particles)

    def _update(self, model, action, observation, terminal=None):
        if model.frame.level == 0:
            raise ValueError("Uniform-random L0 has no belief to update")
        if model.frame.level == 1:
            # LEVEL0BELIEFUPDATE in the paper's numbering: exact, with the other agent's actions as noise.
            return self.exact.update(model, action, observation, terminal=terminal)
        frame, physics = model.frame, model.frame.pomdp_model
        rng = self._rng(
            "update-published",
            canonical.belief_digest(model.belief),
            canonical.stable_value(model.frame.agent_id),
            canonical.stable_value(action),
            canonical.stable_value(observation),
            terminal,
        )
        own = physics.get_all_observations(frame.agent_id)
        floor = self.smoothing / len(own) if observation in own else 0.0
        rows = []
        for atom in self._particles(model.belief, rng):
            opponent = atom.opponent
            other = opponent.frame.agent_id
            actions, policy = zip(*self.action_distribution(opponent))
            joint = {frame.agent_id: action, other: rng.choices(actions, weights=policy, k=1)[0]}
            following = physics.sample_transition(atom.state, joint, rng)
            ended = physics.is_terminal(following)
            if terminal is not None and ended != terminal:
                continue
            weight = physics.observation_probability(observation, following, joint, frame.agent_id)
            if self.smoothing:
                weight = (1.0 - self.smoothing) * weight + floor
            if weight == 0:
                continue
            if opponent.frame.level == 0 or ended:
                rows.append((InteractiveState(following, opponent), weight))
                continue
            for private, p_private in physics.observation_distribution(following, joint, other):
                if p_private <= 0:
                    continue
                try:
                    nested = self.update(opponent, joint[other], private, terminal=False).belief
                except UnsupportedObservation:
                    continue
                rows.append((InteractiveState(following, MentalModel(opponent.frame, nested)), weight * p_private))
        if not rows:
            raise UnsupportedObservation(
                f"{frame!r}: no particle explains action={action!r}, observation={observation!r}"
            )
        evidence = math.fsum(w for _, w in rows) / self.particles
        atoms, weights = zip(*rows)
        drawn = rng.choices(atoms, weights=weights, k=self.particles)
        return Posterior(FiniteBelief(tuple((atom, 1.0) for atom in drawn)), evidence)
