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

What remains are computational budgets, an optional rule for spending the
modeled budget adaptively (``stopping.ModeledResolution``), the options of
certified search, and two scalars of the decision problem.
"""

import math
from dataclasses import dataclass

from ipomdp.planning.stopping import ModeledResolution


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
        frontier_bounds: Bounds on what search left unknown, used by certified
            scoring. ``"scalar"`` (default) is the horizon-only range of
            ``chance_scoring._Span``; ``"ifib"`` is the belief-dependent
            interactive fast-informed bound of ``frontier_bounds``.
        scoring: ``"legacy"`` (default) keeps an action's whole search value
            when one of its observation branches was never sampled at three or
            more steps to go; ``"pw"`` weights every branch by its exact
            probability, puts a bounded stand-in on the unsampled ones and clips
            each value into its certified interval (it always computes bounds).
            The stand-in of an unsampled branch is the lower end of its interval.
        exact_tail: Solve a history with two steps to go exactly when search
            first expands it (certified search, ``i_pomcp._simulate_certified``).
        prune: Keep belief-dependent bounds on every expanded history, drop an
            action whose upper bound falls below the history's lower bound, and
            stop the search once the root decision is resolved.
        selection: ``"ucb"`` (default) or ``"iemax"`` (largest upper bound
            first; ties by the PUCT score, whose prior is a softmax of the
            I-FIB upper bounds, ``exploration.IEMax``).
        leaf_eval: Value of a new frontier: ``"rollout"`` (default) or
            ``"bound"``, a point of its I-FIB branch interval (no belief update).
        lazy_expansion: With bound leaves, compute a frontier's belief only at
            its second visit.

        modeled_resolution: When set, a modeled policy is solved by one tree
            grown by doubling until its certified intervals resolve which
            actions are optimal (``stopping.ModeledResolution``), with
            ``modeled_simulations`` as the cap.
        modeled_prediction: ``"argmax"`` (default): a modeled agent plays its
            greedy maximizers; ``"plausible"``: uniformly the actions its
            intervals cannot rule out (bi-level soundness, THEORY.md); needs ``modeled_resolution``.
        outcome: How certified search chooses the next observation below the
            selected action. ``"sample"`` (default): drawn with its probability;
            a new history is a frontier and is expanded at its second visit
            (``lazy_expansion``). ``"gap"``: the open branch with the largest
            probability-weighted gap between its bounds, the rule of
            bound-directed solvers (HSVI). Under the gap rule the first history
            on the path that is not yet expanded is expanded and the simulation
            ends (one expansion per simulation), ties between actions are
            broken without the random stream, and ``leaf_eval`` and
            ``lazy_expansion`` have no effect. It needs ``prune`` and
            ``selection="iemax"``.
        root_rule: Which root action the gap rule refines. ``"upper"`` (default,
            the only value under ``outcome="sample"``): the largest upper
            bound. ``"open"`` (the only value under ``outcome="gap"``): the
            largest upper bound among the actions that may still be optimal and
            whose own interval is open, so that a decision is never left
            unresolved because the action that keeps it open is not the largest
            upper bound.

    Every certified-search field needs ``frontier_bounds != "scalar"``.

    The search seed holds every field except the scoring-only ones
    (``SCORING_FIELDS``: they read a finished tree and never touch the search
    stream, so scoring variants score the same tree).
    """

    simulations: int
    depth: int
    modeled_simulations: int
    gamma: float = 0.95
    exploration: float = 1.0
    frontier_bounds: str = "scalar"
    scoring: str = "legacy"
    exact_tail: bool = False
    prune: bool = False
    selection: str = "ucb"
    leaf_eval: str = "rollout"
    lazy_expansion: bool = False
    modeled_resolution: ModeledResolution | None = None
    modeled_prediction: str = "argmax"
    outcome: str = "sample"
    root_rule: str = "upper"

    def __post_init__(self):
        for name in ("simulations", "depth", "modeled_simulations"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.gamma) or not 0 <= self.gamma <= 1:
            raise ValueError("gamma must lie in [0, 1]")
        if not math.isfinite(self.exploration) or self.exploration <= 0:
            raise ValueError("exploration must be finite and positive")
        if self.frontier_bounds not in FRONTIER_BOUNDS:
            raise ValueError(f"frontier_bounds must be one of {FRONTIER_BOUNDS}")
        if self.scoring not in SCORING:
            raise ValueError(f"scoring must be one of {SCORING}")
        for name in ("exact_tail", "prune", "lazy_expansion"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool")
        if self.selection not in SELECTIONS:
            raise ValueError(f"selection must be one of {SELECTIONS}")
        if self.leaf_eval not in LEAF_EVALS:
            raise ValueError(f"leaf_eval must be one of {LEAF_EVALS}")
        if self.certified_search and self.frontier_bounds == "scalar":
            raise ValueError("Certified search needs belief-dependent frontier bounds")
        if self.lazy_expansion and self.leaf_eval != "bound":
            raise ValueError("Lazy expansion needs bound leaves")
        if self.modeled_resolution is not None:
            if type(self.modeled_resolution) is not ModeledResolution:
                raise TypeError("modeled_resolution must be a ModeledResolution")
            self.modeled_resolution.checkpoints(self.modeled_simulations)
        if self.modeled_prediction not in PREDICTIONS:
            raise ValueError(f"modeled_prediction must be one of {PREDICTIONS}")
        if self.modeled_prediction == "plausible" and self.modeled_resolution is None:
            raise ValueError("Plausible prediction needs modeled_resolution")
        if self.outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {OUTCOMES}")
        if self.root_rule not in ROOT_RULES:
            raise ValueError(f"root_rule must be one of {ROOT_RULES}")
        if self.outcome == "sample" and self.root_rule != "upper":
            raise ValueError("The root rule 'open' belongs to a gap-directed outcome rule")
        if self.outcome != "sample":
            if self.selection != "iemax" or not self.prune:
                raise ValueError("A gap-directed outcome rule needs pruning and selection by upper bound")
            if self.root_rule == "upper":
                raise ValueError("A gap-directed outcome rule needs the root rule 'open'")

    @property
    def certified_search(self):
        """True when search runs ``_simulate_certified`` instead of the legacy descent."""
        return (
            self.exact_tail
            or self.prune
            or self.selection != "ucb"
            or self.leaf_eval != "rollout"
            or self.lazy_expansion
            or self.outcome != "sample"
        )


FRONTIER_BOUNDS = ("scalar", "ifib")
SCORING = ("legacy", "pw")
SELECTIONS = ("ucb", "iemax")
PREDICTIONS = ("argmax", "plausible")
LEAF_EVALS = ("rollout", "bound")
OUTCOMES = ("sample", "gap")
ROOT_RULES = ("upper", "open")

# Fields that read the finished tree only, or decide how a finished modeled solve is
# turned into a prediction; never part of the search seed (a resolution's tree is
# seeded as the flat search of its cap).
SCORING_FIELDS = ("frontier_bounds", "scoring", "modeled_resolution", "modeled_prediction")
