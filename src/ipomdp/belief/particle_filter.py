"""Interactive particle filter (IPF): the sampled I-POMDP belief update.

Reference: P. Doshi and P. J. Gmytrasiewicz, "Monte Carlo Sampling Methods for
Approximating Interactive POMDPs", JAIR 34 (2009).

This is the filter as published, kept as the belief of the IPF + RTS baseline.
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

import random

from ipomdp.belief import canonical
from ipomdp.belief.finite_belief import FiniteBelief, InteractiveState, MentalModel
from ipomdp.belief.finite_filter import InteractiveFilter, Posterior, UnsupportedObservation
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
