"""Declared exact L1 policy for matched fixed-depth L2 Tiger experiments.

This is an opponent model, not an oracle supplied to the protagonist. The
provider receives only j's subjective finite belief and has no true-state
argument. It replans at one fixed finite horizon after every private update,
modeling i as uniform L0. It intentionally differs from finite-budget modeled
MCTS; results using this policy must not be labeled production-opponent tests.
"""

import math

from examples.tiger.model.tiger_model import TIGER_LEFT, TIGER_RIGHT
from solvers.exact.pomdp_exact_vi import ExactPOMDPSolver


class FixedTigerL1Policy:
    """Stationary mapping from j's private belief to an exact finite-depth policy.

    Uniform ties use absolute tolerance 1e-10, matching the L2 reference's
    declared opponent convention. The actor may change its action with belief;
    stationary means the computation horizon does not count down with i.
    """

    def __init__(self, model, horizon, gamma, agent_id="j", opponent_id="i"):
        if type(horizon) is not int or horizon < 1:
            raise ValueError("Fixed opponent horizon must be a positive integer")
        self.pomdp_model = model
        self.agent_id, self.opponent_id = agent_id, opponent_id
        self.horizon = horizon
        self.reference = ExactPOMDPSolver(
            model, horizon=horizon, gamma=gamma, agent_id=agent_id, opponent_id=opponent_id
        )

    def policy_for(self, model, modeled=False):
        """Integrate the supplied private belief; never condition it on reality."""
        frame = model.frame
        if (
            frame.level != 1
            or frame.agent_id != self.agent_id
            or frame.pomdp_model is not self.pomdp_model
        ):
            raise ValueError("Fixed Tiger policy requires its declared L1 frame and physics")
        for atom, _ in model.belief.mass:
            other = atom.opponent.frame
            if (
                atom.state not in (TIGER_LEFT, TIGER_RIGHT)
                or other.level != 0
                or other.agent_id != self.opponent_id
                or other.pomdp_model is not self.pomdp_model
            ):
                raise ValueError(
                    "Fixed Tiger L1 policy requires Tiger states and its uniform L0 opponent"
                )
        p = math.fsum(mass for atom, mass in model.belief.mass if atom.state == TIGER_LEFT)
        q = self.reference.q_values(p, self.horizon)
        best = max(q.values())
        winners = [a for a, value in q.items() if abs(value - best) <= 1e-10]
        return dict.fromkeys(winners, 1 / len(winners))
