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
- horizon-bounded UCB exploration scaled by the domain's declared reward bounds;
- post-search scoring of the finished tree with exact chance probabilities
  (``chance_scoring``), whose values decide.

Earlier alternatives (sampled-return backups, empirical-range UCB, root-only
chance rescoring, a two-step exact tail) were removed because they either did
not converge, traded corrected errors for new ones, or did not generalise
beyond Tiger. A switch that selects a non-convergent estimator is a way to
produce wrong results, not a feature.

What remains are computational budgets, optional rules for spending the
modeled and root budgets adaptively (``stopping.ModeledStopping``), and two
scalars of the decision problem.
"""

import math
from dataclasses import dataclass

from ipomdp.planning.stopping import ModeledStopping


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
        modeled_stopping: When set, a modeled policy is solved by one tree
            grown until its decision is separated, with ``modeled_simulations``
            as the cap (``stopping``). When None, one search of
            ``modeled_simulations``.
        root_stopping: The same rule for this agent's own decisions, with
            ``simulations`` as the cap: near-tie root escalation (RESULTS
            12.90). The root tree grows by doubling until its two leading
            actions separate or its scores are exact; at the cap it is the flat
            search of ``simulations``. When None, one search of ``simulations``.
    """

    simulations: int
    depth: int
    modeled_simulations: int
    gamma: float = 0.95
    exploration: float = 1.0
    modeled_stopping: ModeledStopping | None = None
    root_stopping: ModeledStopping | None = None

    def __post_init__(self):
        for name in ("simulations", "depth", "modeled_simulations"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.gamma) or not 0 <= self.gamma <= 1:
            raise ValueError("gamma must lie in [0, 1]")
        if not math.isfinite(self.exploration) or self.exploration <= 0:
            raise ValueError("exploration must be finite and positive")
        if self.modeled_stopping is not None:
            if type(self.modeled_stopping) is not ModeledStopping:
                raise TypeError("modeled_stopping must be a ModeledStopping")
            self.modeled_stopping.checkpoints(self.modeled_simulations)
        if self.root_stopping is not None:
            if type(self.root_stopping) is not ModeledStopping:
                raise TypeError("root_stopping must be a ModeledStopping")
            self.root_stopping.checkpoints(self.simulations)
