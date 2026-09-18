"""Fixed agent identity, reasoning level and subjective physics for one experiment."""

from dataclasses import dataclass

from core.pomdp_model import AgentID, POMDPModel


@dataclass(frozen=True, slots=True)
class AgentFrame:
    """A descriptor that pairs an agent identifier and reasoning level with a physics model.

    ``AgentFrame`` is stored inside each immutable mental model to
    record which model of an opponent the particle assumes.

    Attributes:
        agent_id:    The opponent's identifier.
        level:       The reasoning level attributed to the opponent.
        pomdp_model: The POMDPModel instance the opponent is assumed to
                     use (typically the same shared environment model).
    """

    agent_id: AgentID
    level: int
    pomdp_model: POMDPModel

    def __repr__(self) -> str:
        return f"<Frame {self.agent_id} L{self.level}>"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AgentFrame):
            return False
        return (
            self.agent_id == other.agent_id
            and self.level == other.level
            and self.pomdp_model is other.pomdp_model
        )

    def __hash__(self) -> int:
        return hash((self.agent_id, self.level, id(self.pomdp_model)))
