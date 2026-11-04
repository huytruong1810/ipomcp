"""A wide initial belief: agent i is uncertain about which of M level-1 models of j it faces (T4).

The reachable-decision problems (``reachable_decisions``) start i from point-mass
nesting: one model of j per level, each at the domain's prior. Here i's level-2
prior spreads over M level-1 models of j that differ only in j's initial
physical belief, p_m = (m - 1/2) / M on a midpoint grid in (0, 1), with weight
w0 on a uniformly random (level-0) j:

    b_i(s, theta_j) = b(s) * w(theta_j),   w(j level 0) = w0,   w(j level 1 at p_m) = (1 - w0) / M.

j's prior is independent of the state, so i learns about the state from j only
through j's actions. With M = 1 and w0 = 0 the prior is ``nested_levels(2)``'s
point mass (p = 1/2), atom for atom (tests/test_wide_prior.py).

The grid is defined for two-state domains whose physical prior is a pair
((s0, b0), (s1, b1)) (Tiger): j's belief at p_m puts p_m on s0. Every p_m is
strictly inside (0, 1), so no model of j rules out an observation.
"""

import random

from ipomdp.belief import canonical
from ipomdp.belief.finite_belief import FiniteBelief, InteractiveState, MentalModel
from ipomdp.belief.frame import AgentFrame


def grid(m_count):
    """The midpoint grid (m - 1/2) / M, m = 1 .. M."""
    if type(m_count) is not int or m_count < 1:
        raise ValueError("M must be a positive integer")
    return [(m + 0.5) / m_count for m in range(m_count)]


def opponent_models(physics, m_count, w0, initial_belief=FiniteBelief):
    """((model of j, weight), ...): j level 0 with weight ``w0``, then M level-1 models of j.

    ``initial_belief`` maps rows to a belief in the planner's representation (a bank filter's
    ``initial_belief``: exact rows, compressed, or particles); level 1 j models i as level 0.
    """
    if not 0.0 <= w0 < 1.0:
        raise ValueError("w0 must be in [0, 1)")
    prior = tuple(physics.initial_distribution())
    if len(prior) != 2:
        raise ValueError("The wide-prior grid is defined for two-state domains")
    (s0, _), (s1, _) = prior
    i_level0 = MentalModel(AgentFrame("i", 0, physics))
    models = []
    if w0 > 0:
        models.append((MentalModel(AgentFrame("j", 0, physics)), w0))
    for p in grid(m_count):
        belief = initial_belief(((InteractiveState(s0, i_level0), p),
                                 (InteractiveState(s1, i_level0), 1.0 - p)))
        models.append((MentalModel(AgentFrame("j", 1, physics), belief), (1.0 - w0) / m_count))
    return models


def prior_rows(physics, m_count, w0, initial_belief=FiniteBelief, physical_prior=None):
    """i's level-2 prior rows ((InteractiveState, mass), ...), state independent of j's model."""
    states = physical_prior if physical_prior is not None else physics.initial_distribution()
    models = opponent_models(physics, m_count, w0, initial_belief)
    return tuple((InteractiveState(s, model), p_s * w) for s, p_s in states for model, w in models)


def prior_digest(physics, m_count, w0):
    """Digest of the exact (uncompressed) prior: what a problem file pins."""
    return canonical.belief_digest(FiniteBelief(prior_rows(physics, m_count, w0)))


def true_opponent(physics, m_count, w0, seed, initial_belief=FiniteBelief):
    """j's true model in problem ``seed``: drawn from the prior's marginal over models of j.

    The stream is separate from the history's world and script streams, so the state and i's actions
    are drawn exactly as with point-mass nesting.
    """
    models = opponent_models(physics, m_count, w0, initial_belief)
    rng = random.Random(f"reachable/opponent/{seed}")
    return rng.choices([m for m, _ in models], weights=[w for _, w in models])[0]
