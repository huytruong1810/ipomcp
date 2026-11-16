"""Fixed agent identity, reasoning level and subjective physics for one experiment."""

from dataclasses import dataclass, field

from ipomdp.model.pomdp import AgentID, POMDPModel


@dataclass(frozen=True, slots=True)
class AgentFrame:
    """An agent's identifier, reasoning level and physics model: everything about it but its belief.

    Every ``MentalModel`` holds one: the modeled opponent's inside a belief
    atom, and the planning agent's own at the root of its hierarchy.

    Attributes:
        agent_id:    The agent's identifier.
        level:       Its reasoning level (0 is the uniformly random agent).
        pomdp_model: The ``POMDPModel`` it is assumed to act in (typically the
                     one shared environment model).

    Two frames are equal when their agent and level are equal and their physics
    is the *same object*. Comparing physics by value would cost a structural
    comparison inside every atom hash; ``canonical.physics_identity`` gives the
    value-based identity where one is needed (digests and seeds).
    """

    agent_id: AgentID
    level: int
    pomdp_model: POMDPModel
    _hash: int = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        # Frames are few and long-lived but hashed inside every atom hash (tens
        # of millions of times per level-3 decision), so the hash is computed once.
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
