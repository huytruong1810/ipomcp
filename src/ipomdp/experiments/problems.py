"""Decision problems: an agent's prior over (state, opponent model), from a chain of physical beliefs.

A chain lists one physical belief per level, top-down from the deciding agent
(``chain[0]``), through the model it holds of its opponent (``chain[1]``),
down to level 1; level 0 is uniformly random and needs none. Each entry maps
the enclosing true state to a distribution over states: the top entry is
called with None; entry k + 1 is called with the state of the level-k atom
it sits in. Every atom (s, m) of a level-l belief therefore carries the model
of the opponent built from the rest of the chain at state s.

- Constant entries (``constant``) give every opponent model the same belief:
  Tiger's point-model problems (``tiger_chain``).
- State-dependent entries let an agent know private parts of the state: the
  UAV target knows its own sector, so its belief depends on the state of the
  atom it is modeled in (``uav_chain``).
"""

from dataclasses import dataclass

from ipomdp.belief.finite_belief import FiniteBelief, InteractiveState, MentalModel
from ipomdp.belief.frame import AgentFrame


@dataclass(frozen=True)
class Constant:
    """A physical belief that ignores the enclosing state."""

    distribution: tuple

    def __call__(self, state):
        return self.distribution


def constant(distribution):
    return Constant(tuple(distribution))


def chain_model(physics, agent, level, chain, state=None):
    """The mental model of ``agent`` at ``level`` whose beliefs follow ``chain`` (top-down)."""
    if level == 0:
        return MentalModel(AgentFrame(agent, 0, physics))
    if len(chain) < level:
        raise ValueError(f"A level-{level} chain needs {level} physical beliefs, got {len(chain)}")
    other, rest = "j" if agent == "i" else "i", chain[1:]
    below = level - 1
    shared = None
    if below == 0 or isinstance(rest[0], Constant):
        shared = chain_model(physics, other, below, rest)
    atoms = tuple(
        (
            InteractiveState(
                s, shared if shared is not None else chain_model(physics, other, below, rest, s)
            ),
            p,
        )
        for s, p in chain[0](state)
    )
    return MentalModel(AgentFrame(agent, level, physics), FiniteBelief(atoms))


def tiger_chain(belief_p, opponent_belief=0.5, nested_beliefs=()):
    """Tiger's point-model chain: i's P(tiger left), then j's, then the models below j."""
    from ipomdp.domains.tiger import TIGER_LEFT, TIGER_RIGHT

    return tuple(
        constant(((TIGER_LEFT, p), (TIGER_RIGHT, 1 - p)))
        for p in (belief_p, opponent_belief, *nested_beliefs)
    )


@dataclass(frozen=True)
class _OwnSector:
    """A UAV-domain belief: the agent's own sector from the enclosing state, the other's from a law.

    ``law`` is ((sector, weight), ...) over the other agent's sector; sectors equal
    to the agent's own are dropped (the state would be terminal) and the rest
    renormalized. At the top (no enclosing state) the agent is the UAV, whose
    own sector is its start.
    """

    agent: str
    law: tuple
    start: tuple

    def __call__(self, state):
        if state is None and self.agent != "i":
            raise ValueError("UAV-domain problems are decided by the UAV (agent i)")
        own = (self.start if state is None else state[0]) if self.agent == "i" else state[1]
        kept = [(cell, w) for cell, w in self.law if cell != own and w > 0]
        total = sum(w for _, w in kept)
        if total <= 0:
            raise ValueError("The belief puts no mass on a sector other than the agent's own")
        if self.agent == "i":
            return tuple(((own, cell), w / total) for cell, w in kept)
        return tuple(((cell, own), w / total) for cell, w in kept)


def uav_chain(physics, laws):
    """A UAV-domain chain, the UAV (``i``) deciding: ``laws`` top-down, each {sector: weight}.

    ``laws[0]`` is i's law over the target's sector, ``laws[1]`` the modeled
    target's law over the UAV's sector, and so on, alternating. Each agent knows
    its own sector exactly (the UAV from its start or the enclosing state, the
    target from the enclosing state).
    """
    return tuple(
        _OwnSector("i" if k % 2 == 0 else "j", tuple(sorted(law.items())), physics.uav_start)
        for k, law in enumerate(laws)
    )
