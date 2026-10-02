"""Planner configuration: the only tunable quantities of the I-POMCP planner.

Why so few fields
-----------------
The planner implements one estimator, chosen because it is the only one shown
to converge to the exact finite-horizon solution (L1 against alpha-vector value
iteration at horizons 1-5; L2 against exhaustive references). Its components are
fixed, not configurable:

- empirical chance-weighted Bellman backups (``node.POMCPNode.bellman_backup``);
- exact one-step leaf values at the planning horizon, integrated over the full
  private-history belief before maximizing;
- exact expected immediate rewards at every tree history (a control variate);
- horizon-bounded UCB exploration scaled by the domain's declared reward bounds.

Earlier alternatives (sampled-return backups, empirical-range UCB, root or
recursive chance rescoring, a two-step exact tail) were removed because they
either did not converge or did not generalise beyond Tiger. A switch that
selects a non-convergent estimator is a way to produce wrong results, not a
feature.

What remains are computational budgets and two scalars of the decision problem.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PlannerConfig:
    """Budgets and decision-problem constants for one I-POMCP planner.

    Attributes:
        simulations: Root simulations for a real decision of this agent.
        depth: Planning horizon in decisions (fixed receding lookahead). The
            planner is never told the episode length.
        modeled_simulations: Root simulations used when this planner is queried
            as a *modeled* opponent inside another agent's belief. It is the
            budget of "simulating other minds", and is reported separately
            because it dominates opponent-model error.
        gamma: Discount factor of the planning objective.
        exploration: Dimensionless UCB constant multiplying the horizon-scaled
            reward width (see ``exploration.HorizonBoundUCB``).
    """

    simulations: int
    depth: int
    modeled_simulations: int
    gamma: float = 0.95
    exploration: float = 1.0

    def __post_init__(self):
        for name in ("simulations", "depth", "modeled_simulations"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.gamma) or not 0 <= self.gamma <= 1:
            raise ValueError("gamma must lie in [0, 1]")
        if not math.isfinite(self.exploration) or self.exploration <= 0:
            raise ValueError("exploration must be finite and positive")
