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

A ``Problem`` names one decision problem for the gate drivers (``panel``):
the ``evaluate_case`` arguments that define it and the columns it adds to its
row. ``tiger_problems`` and ``uav_problems`` build a level's problem grid.
"""

import itertools
from dataclasses import dataclass, field

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


@dataclass(frozen=True)
class Problem:
    """One decision problem: its name, its ``evaluate_case`` arguments and its extra row columns."""

    name: str
    case: dict
    labels: dict = field(default_factory=dict)


TIGER_BELIEFS = (0.05, 0.2, 0.5, 0.8, 0.95)
TIGER_OPPONENT_BELIEFS = (0.085, 0.5, 0.915)
TIGER_NESTED_BELIEFS = (0.5,)


def tiger_problems(
    level,
    beliefs=TIGER_BELIEFS,
    opponent_beliefs=TIGER_OPPONENT_BELIEFS,
    nested_beliefs=TIGER_NESTED_BELIEFS,
):
    """Tiger point-belief problems: i's beliefs x j's x those below j, crossed per nested level.

    Level 1 has no opponent model: its problems are i's beliefs alone.
    """
    if level == 1:
        return [Problem(f"p={p}", dict(belief_p=p)) for p in beliefs]
    return [
        Problem(
            "/".join(f"{x}" for x in (p, p_j, *chain)),
            dict(belief_p=p, opponent_belief=p_j, nested_beliefs=chain),
            dict(opponent_belief=p_j, nested_beliefs=list(chain)),
        )
        for p in beliefs
        for p_j in opponent_beliefs
        for chain in itertools.product(nested_beliefs, repeat=level - 2)
    ]


def uav_laws(physics, concentration=0.75):
    """Named laws over the other agent's sector, for UAV problem grids.

    ``uniform`` weighs every sector alike. Each other law puts ``concentration``
    of its mass uniformly on a named set of sectors and the rest uniformly on
    the whole grid, so no sector is ruled out: ``top-row``; ``near-start``,
    the UAV's start and the sectors next to it; ``far-corner``, the corner
    opposite the start. ``_OwnSector`` drops the agent's own sector and
    renormalizes.
    """
    cells = physics.cells()
    start = physics.uav_start
    far = (physics.rows - 1 - start[0], physics.cols - 1 - start[1])
    named = {
        "top-row": [c for c in cells if c[0] == 0],
        "near-start": [c for c in cells if abs(c[0] - start[0]) + abs(c[1] - start[1]) <= 1],
        "far-corner": [far],
    }
    laws = {"uniform": {c: 1 / len(cells) for c in cells}}
    for name, chosen in named.items():
        laws[name] = {
            c: (1 - concentration) / len(cells)
            + (concentration / len(chosen) if c in chosen else 0)
            for c in cells
        }
    return laws


def uav_problems(physics, level, laws=None):
    """UAV problems: every combination of named laws, one per level (i's first)."""
    laws = uav_laws(physics) if laws is None else laws
    problems = []
    for names in itertools.product(sorted(laws), repeat=level):
        name = "/".join(names)
        chain = uav_chain(physics, [laws[n] for n in names])
        problems.append(Problem(name, dict(chain=chain, problem=name)))
    return problems
