# Absolute Path: <project_root>/solvers/solver_bank.py

"""
solver_bank — Central registry for multi-level agent solvers.

DESIGN DECISION RECORD (Phase 5 Overhaul):
------------------------------------------
1. REMOVED STATE-BLIND ACTION CACHING:
   `get_random_action` previously cached the action space and returned a random 
   action. Because we introduced `get_legal_actions(state)` in Phase 1, global 
   caching causes Level-0 agents to select illegal actions (like walking into walls), 
   crashing the physics engine. We deleted this method entirely to force the Generative 
   Model to resolve actions using the current physical state.
"""

from typing import Dict

from solvers.solver_types import SolverKey, AgentFrame
from solvers.planner import Planner


class SolverBank:
    """Registry mapping ``SolverKey`` -> ``Planner`` for the entire agent hierarchy."""

    def __init__(self) -> None:
        self._solvers: Dict[SolverKey, Planner] = {}

    def register_solver(self, key: SolverKey, solver: Planner) -> None:
        self._solvers[key] = solver

    def has_solver(self, key: SolverKey) -> bool:
        """Returns True if the specified SolverKey is registered."""
        return key in self._solvers

    def __contains__(self, key: SolverKey) -> bool:
        return key in self._solvers

    def get_solver(self, key: SolverKey) -> Planner:
        return self._solvers[key]

    def get_solver_for_frame(self, frame: AgentFrame) -> Planner:
        return self.get_solver(SolverKey(frame.agent_id, frame.level))