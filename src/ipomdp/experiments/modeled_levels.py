"""Per-level modeled solves: how each intentional level below an agent is solved.

Why per level
-------------
An agent at level L models its opponent at L - 1, who models it at L - 2, and
so on down to level 1. Each of those levels is queried at different beliefs and
may need a different budget. Each level's own ``PlannerConfig`` already carries
its budget; this module names the per-level choice once, so the oracle, the
panel and the qualification driver build the same hierarchy from it.

Labels
------
A *stack* is one ``ModeledSolve`` (or None for an exact level) per intentional
level below the agent, top-down. Its label, used on the command line and as
the panel's ``modeled`` cell value, joins the levels' labels with "/":

- ``400``: one flat search of 400 simulations;
- ``exact``: the exact best response to the levels below it
  (``reference.exact_agent``).

``1600/400`` at level 3 solves j-L2 with a flat 1,600 and the i-L1 it models
with a flat 400. A single label applies to every level, so ``400`` at level 3
means ``400/400`` and ``exact`` means an exact hierarchy.
"""

import random
import re
from dataclasses import dataclass, replace

from ipomdp.baselines.rts import RTSConfig, RTSPlanner
from ipomdp.planning.config import PlannerConfig
from ipomdp.planning.i_pomcp import IPOMCPPlanner
from ipomdp.planning.random_planner import RandomPlanner
from ipomdp.planning.solver_types import SolverKey
from ipomdp.reference.exact_agent import ExactIPOMDPPlanner

EXACT = "exact"
_SOLVE = re.compile(r"^(\d+)$")


@dataclass(frozen=True)
class ModeledSolve:
    """One finite modeled level: its budget.

    Attributes:
        budget: I-POMCP simulations, or RTS observation samples per node.
    """

    budget: int

    def __post_init__(self):
        if type(self.budget) is not int or self.budget < 1:
            raise ValueError("A modeled budget must be a positive integer")

    @property
    def label(self):
        return str(self.budget)


def parse_stack(label, levels):
    """The stack named by ``label`` for ``levels`` intentional levels, top-down.

    One level's label applies to all of them; otherwise the label must name
    every level.
    """
    if type(levels) is not int or levels < 1:
        raise ValueError("A stack needs at least one modeled level")
    parts = str(label).split("/")
    if len(parts) == 1:
        parts *= levels
    if len(parts) != levels:
        raise ValueError(f"Modeled stack {label!r} does not name {levels} levels")
    return tuple(_parse_solve(part) for part in parts)


def stack_label(stack):
    """Canonical label of a stack (single-level labels collapse when every level agrees)."""
    labels = [EXACT if solve is None else solve.label for solve in stack]
    return labels[0] if len(set(labels)) == 1 else "/".join(labels)


def _parse_solve(text):
    if text == EXACT:
        return None
    match = _SOLVE.match(text)
    if match is None:
        raise ValueError(f"Not a modeled solve: {text!r} (e.g. 400, exact)")
    return ModeledSolve(int(match.group(1)))


def modeled_keys(agent, level):
    """Solver keys of the intentional levels below ``agent`` at ``level``, top-down.

    An agent at level L models the other agent at L - 1, who models the first
    at L - 2, and so on down to level 1; both level-0 agents are uniform random.
    """
    other = "j" if agent == "i" else "i"
    return [
        SolverKey(other if (level - lvl) % 2 else agent, lvl) for lvl in range(level - 1, 0, -1)
    ]


def planner_config(planner_kind, budget, depth, gamma, exploration, backup="sparse"):
    """The configuration of a planner of this kind whose root and modeled budgets are ``budget``.

    ``backup`` (IPF + RTS): the ``RTSConfig`` backup, "sparse" unless the caller names another. It is an
    argument and nothing else: a result must not depend on the environment of the shell that ran it, so
    there is no environment-variable override. The variant is part of ``RTSConfig``, so every row that
    records its configuration records it.
    """
    if planner_kind == "mcts":
        return PlannerConfig(
            simulations=budget,
            depth=depth,
            modeled_simulations=budget,
            gamma=gamma,
            exploration=exploration,
        )
    return RTSConfig(depth=depth, observation_samples=budget, gamma=gamma, backup=backup)


def make_planner(planner_kind, key, physics, bank, config):
    planner_class = IPOMCPPlanner if planner_kind == "mcts" else RTSPlanner
    return planner_class(key, physics, bank, config)


def register_levels(
    bank,
    physics,
    agent,
    level,
    stack,
    depth,
    best_response,
    *,
    planner_kind="mcts",
    gamma=0.95,
    exploration=1.0,
    wrap=None,
    level_options=None,
):
    """Register both uniform L0 agents and every intentional level below ``agent``.

    ``stack`` gives one ``ModeledSolve`` or None (exact) per level, top-down
    (``modeled_keys``). A finite level is solved by a planner of
    ``planner_kind`` at ``depth`` whose modeled budget is its
    ``ModeledSolve``'s; an exact level is the exact best response at ``depth`` to
    whatever is registered below it. ``wrap(key, solver)`` may replace a
    solver before registration (for example to observe its queries).
    ``level_options(budget)`` (I-POMCP only) gives ``PlannerConfig`` fields that
    replace those of each finite level's configuration (``experiments.profile``).
    """
    keys = modeled_keys(agent, level)
    if len(stack) != len(keys):
        raise ValueError(f"Level {level} needs {len(keys)} modeled levels, got {len(stack)}")
    # L0 action streams are never drawn in a single root decision; seed them anyway.
    for name in ("i", "j"):
        rng = random.Random(bank.stream_seed("actions", f"{name}_L0"))
        bank.register_solver(SolverKey(name, 0), RandomPlanner(physics.get_all_actions(name), rng))
    for key, solve in zip(keys, stack):
        if solve is None:
            solver = ExactIPOMDPPlanner(key, physics, bank, best_response, depth)
        else:
            config = planner_config(planner_kind, solve.budget, depth, gamma, exploration)
            if level_options is not None:
                config = replace(config, **level_options(solve.budget))
            solver = make_planner(planner_kind, key, physics, bank, config)
        bank.register_solver(key, solver if wrap is None else wrap(key, solver))
