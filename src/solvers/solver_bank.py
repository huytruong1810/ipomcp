"""Registry of planner implementations by agent identity and reasoning level.

A bank belongs to one experiment with one agreed model/configuration per key.
The key is not a private-belief identity. Private histories live in particle model
nodes; they must not be conflated merely because they share a solver in this bank.
"""

from typing import Dict

from solvers.planner import Planner
from solvers.solver_types import AgentFrame, SolverKey


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
