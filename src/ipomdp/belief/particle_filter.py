"""Interactive particle filter (IPF): the sampled I-POMDP belief update.

Reference: P. Doshi and P. J. Gmytrasiewicz, "Monte Carlo Sampling Methods for
Approximating Interactive POMDPs", JAIR 34 (2009).

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
   (one uniform offset, N evenly spaced points over the cumulative weights, in
   canonical atom order). Each atom of weight w is drawn floor(N w) or
   ceil(N w) times, so small discrete beliefs are reproduced almost exactly
   (a 50/50 prior with N = 64 gives exactly 32/32; independent draws gave
   28/36 in a probe).
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

Failure mode (stated, not repaired)
-----------------------------------
If no particle explains o_i, the update raises ``UnsupportedObservation``; the
same holds for a nested model that cannot explain its own sampled observation.
With noisy sensors (Tiger, UAV) every observation has positive likelihood under
every particle, so this cannot happen. With deterministic sensors (Wumpus) the
particle set can miss the true world; that requires reinvigoration, which is
not implemented, so such failures surface explicitly.

Initial beliefs are N draws from the exact prior rows (equal weights merged).
"""

import math
import random

from ipomdp.belief import canonical
from ipomdp.belief.finite_belief import FiniteBelief, InteractiveState, MentalModel
from ipomdp.belief.finite_filter import InteractiveFilter, Posterior, UnsupportedObservation


class ParticleInteractiveFilter(InteractiveFilter):
    def __init__(self, policy, particles: int, seed: int, cache_size: int = 4096):
        super().__init__(policy, cache_size)
        if type(particles) is not int or particles < 1:
            raise ValueError("particles must be a positive integer")
        if type(seed) is not int:
            raise TypeError("A particle filter needs an explicit integer seed")
        self.particles, self.seed = particles, seed

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
