"""
solver_types — Lightweight identifiers for agents and reasoning levels.

Defines two small data classes used pervasively across the solver stack:

* :class:`SolverKey` — An immutable ``(agent_id, level)`` pair that serves
  as the primary key in :class:`SolverBank`.
* :class:`AgentFrame` — Extends ``SolverKey`` with a reference to the
  agent's ``POMDPModel``, capturing *who* the agent is, *what level* they
  reason at, and *which physics model* they use.
"""

from dataclasses import dataclass

from core.pomdp_model import AgentID
from ipomdp.frame import AgentFrame


@dataclass(frozen=True)
class SolverKey:
    """Immutable identifier for a specific (agent, reasoning-level) pair.

    Used as the dictionary key in :class:`SolverBank` to register and
    retrieve planners.

    Attributes:
        agent_id: The agent's string identifier (e.g. ``'i'`` or ``'j'``).
        level:    The reasoning level (0 = random, 1 = models opponents as
                  Level 0, 2 = models opponents as Level 1, …).
    """

    agent_id: AgentID
    level: int

    def __repr__(self) -> str:
        return f"{self.agent_id}_L{self.level}"


__all__ = ["SolverKey", "AgentFrame"]
