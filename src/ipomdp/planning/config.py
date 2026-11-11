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
        frontier_bounds: Bounds on what search left unknown, used by certified
            scoring. ``"scalar"`` (default) is the horizon-only range of
            ``chance_scoring._Span``; ``"ifib"`` is the belief-dependent
            interactive fast-informed bound of ``frontier_bounds``.
        scoring: ``"legacy"`` (default) keeps an action's whole search value
            when one of its observation branches was never sampled at three or
            more steps to go; ``"pw"`` weights every branch by its exact
            probability, puts a bounded stand-in on the unsampled ones and clips
            each value into its certified interval (it always computes bounds).
        unsampled_value: Point of an unsampled branch's interval used as its
            stand-in under ``scoring="pw"``: ``"lower"``, ``"mid"`` or ``"upper"``.
        exact_tail: Solve a history with two steps to go exactly when search
            first expands it (certified search, ``i_pomcp._simulate_certified``).
        prune: Keep belief-dependent bounds on every expanded history, drop an
            action whose upper bound falls below the history's lower bound, and
            stop the search once the root decision is resolved.
        selection: ``"ucb"`` (default), ``"puct"`` (prior from the I-FIB upper
            bounds) or ``"iemax"`` (largest upper bound first).
        puct_temperature: Temperature of the PUCT prior, in units of the
            return width.
        leaf_eval: Value of a new frontier: ``"rollout"`` (default) or
            ``"bound"``, a point of its I-FIB branch interval (no belief update).
        lazy_expansion: With bound leaves, compute a frontier's belief only at
            its second visit.

    Every certified-search field needs ``frontier_bounds != "scalar"``.

    The search seed ignores the stopping rules and the scoring-only fields
    (``SCORING_FIELDS``: they read a finished tree and never touch the search
    stream, so scoring variants score the same tree). Search-changing fields
    added later enter the seed only when they differ from their defaults
    (``NEUTRAL_DEFAULTS``), so every stored result is reproduced by a default
    configuration.
    """

    simulations: int
    depth: int
    modeled_simulations: int
    gamma: float = 0.95
    exploration: float = 1.0
    modeled_stopping: ModeledStopping | None = None
    root_stopping: ModeledStopping | None = None
    frontier_bounds: str = "scalar"
    scoring: str = "legacy"
    unsampled_value: str = "lower"
    exact_tail: bool = False
    prune: bool = False
    selection: str = "ucb"
    puct_temperature: float = 0.25
    leaf_eval: str = "rollout"
    lazy_expansion: bool = False

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
        if self.frontier_bounds not in FRONTIER_BOUNDS:
            raise ValueError(f"frontier_bounds must be one of {FRONTIER_BOUNDS}")
        if self.scoring not in SCORING:
            raise ValueError(f"scoring must be one of {SCORING}")
        if self.unsampled_value not in UNSAMPLED_VALUES:
            raise ValueError(f"unsampled_value must be one of {UNSAMPLED_VALUES}")
        for name in ("exact_tail", "prune", "lazy_expansion"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool")
        if self.selection not in SELECTIONS:
            raise ValueError(f"selection must be one of {SELECTIONS}")
        if self.leaf_eval not in LEAF_EVALS:
            raise ValueError(f"leaf_eval must be one of {LEAF_EVALS}")
        if not math.isfinite(self.puct_temperature) or self.puct_temperature <= 0:
            raise ValueError("puct_temperature must be finite and positive")
        if self.certified_search and self.frontier_bounds == "scalar":
            raise ValueError("Certified search needs belief-dependent frontier bounds")
        if self.lazy_expansion and self.leaf_eval != "bound":
            raise ValueError("Lazy expansion needs bound leaves")

    @property
    def certified_search(self):
        """True when search runs ``_simulate_certified`` instead of the legacy descent."""
        return (
            self.exact_tail
            or self.prune
            or self.selection != "ucb"
            or self.leaf_eval != "rollout"
            or self.lazy_expansion
        )


FRONTIER_BOUNDS = ("scalar", "ifib")
SCORING = ("legacy", "pw")
UNSAMPLED_VALUES = ("lower", "mid", "upper")
SELECTIONS = ("ucb", "puct", "iemax")
LEAF_EVALS = ("rollout", "bound")

# Fields that read the finished tree only; never part of the search seed.
SCORING_FIELDS = ("frontier_bounds", "scoring", "unsampled_value")

# Search-changing fields added later, dropped from the seed while they hold these values.
NEUTRAL_DEFAULTS = {
    "exact_tail": False,
    "prune": False,
    "selection": "ucb",
    "puct_temperature": 0.25,
    "leaf_eval": "rollout",
    "lazy_expansion": False,
}
