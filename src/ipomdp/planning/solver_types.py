"""Registry key of a solver: one (agent, reasoning level) pair.

A ``SolverKey`` identifies which solver in a bank answers for an agent at a
level. The full identity of a modeled agent (agent, level and physics) is the
``belief.frame.AgentFrame``; the key deliberately omits physics, and
``SolverBank.get_solver_for_frame`` checks that a frame's physics is the
registered solver's.
"""

from dataclasses import dataclass

from ipomdp.model.pomdp import AgentID


@dataclass(frozen=True)
class SolverKey:
    agent_id: AgentID
    level: int

    def __repr__(self) -> str:
        return f"{self.agent_id}_L{self.level}"
