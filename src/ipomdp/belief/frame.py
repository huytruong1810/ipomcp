"""Fixed agent identity, reasoning level and subjective physics for one experiment."""

from dataclasses import dataclass, field

from ipomdp.model.pomdp import AgentID, POMDPModel


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
    _hash: int = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        # Frames are few and long-lived but hashed inside every atom hash
        # (20M times in 10 L3 H4 queries), so the hash is computed once.
        object.__setattr__(self, "_hash", hash((self.agent_id, self.level, id(self.pomdp_model))))

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
        return self._hash

    # The pickled state is the three declared fields; the hash holds
    # id(pomdp_model), which changes on unpickle, so it is recomputed on load.
    def __getstate__(self):
        return [self.agent_id, self.level, self.pomdp_model]

    def __setstate__(self, state):
        for name, value in zip(("agent_id", "level", "pomdp_model"), state):
            object.__setattr__(self, name, value)
        self.__post_init__()
