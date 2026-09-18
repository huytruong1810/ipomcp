"""
solvers — Planning algorithms for I-POMDP agents.

This package contains:

* :class:`Planner` (ABC) — The common interface shared by all planners.
* :class:`RandomPlanner` — Level-0 baseline (uniform random action selection).
* :class:`IPOMCPPlanner` — The main I-POMCP solver: Monte-Carlo Tree Search
  with interactive particle filtering for multi-agent planning.
* :class:`RTSPlanner` — Reachability Tree Sampling, an alternative planner
  that exhaustively evaluates a shallow action–observation tree.
* :class:`InteractiveGenerativeModel` — Joint-action selection, physical
  transition, and nested mental-state propagation (shared by both planners).
* :class:`POMCPNode` — A node in the MCTS search tree, storing visit counts,
  Q-values, and belief particles.
* :class:`SolverBank` — A registry mapping ``SolverKey`` → ``Planner`` so
  that higher-level agents can look up lower-level opponent solvers.
* :class:`SolverKey` / :class:`AgentFrame` — Lightweight identifiers for
  (agent, level) pairs and agent "frames" (identity + model).
"""

from solvers.exploration import ExplorationStrategy, NormalizedUCB, StandardUCB
from solvers.generative_model import InteractiveGenerativeModel
from solvers.i_pomcp import IPOMCPPlanner
from solvers.node import POMCPNode
from solvers.planner import Planner
from solvers.random_planner import RandomPlanner
from solvers.rts_planner import RTSPlanner
from solvers.solver_bank import SolverBank
from solvers.solver_types import AgentFrame, SolverKey

__all__ = [
    "Planner",
    "RandomPlanner",
    "POMCPNode",
    "SolverKey",
    "AgentFrame",
    "SolverBank",
    "IPOMCPPlanner",
    "RTSPlanner",
    "InteractiveGenerativeModel",
    "ExplorationStrategy",
    "StandardUCB",
    "NormalizedUCB",
]
