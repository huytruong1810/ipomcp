# Absolute Path: <project_root>/ipomdp/belief.py

"""
belief.py — Immutable particle representations and intentional frames for Interactive POMDPs.

Provides:
- `AgentFrame`: Descriptor pairing an agent identifier, reasoning level, and environment model.
- `InteractiveParticle`: Single particle hypothesis in an I-POMDP belief state, encoding physical
  state and recursive opponent intentional models.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, TYPE_CHECKING
from core.pomdp_model import State, AgentID, POMDPModel

if TYPE_CHECKING:
    from solvers.node import POMCPNode


@dataclass(slots=True)
class AgentFrame:
    """A descriptor that pairs an agent identifier and reasoning level with a physics model.

    ``AgentFrame`` is stored inside each :class:`InteractiveParticle` to
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


@dataclass(frozen=True, slots=True)
class InteractiveParticle:
    r"""Represents a single particle hypothesis in an Interactive POMDP belief state.

    Mathematical Formulation:
        An interactive state for agent $i$ at strategy level $l \ge 1$ is defined as:
        $$is_{i,l} = \langle s, \boldsymbol{\theta}_{-i} \rangle \in S \times \Theta_{-i}^{<l}$$
        where $s \in S$ is the physical environment state and
        $\boldsymbol{\theta}_{-i} = (\theta_j)_{j \neq i}$ is the tuple of intentional
        models for all opponent agents $j$.

        Each intentional model $\theta_j = \langle b_j, \widehat{\theta}_j \rangle$ is
        concretely represented by:
        1. `AgentFrame`: Opponent identity, assumed strategic level $k < l$, and physics model.
        2. `POMCPNode`: Root pointer of the opponent's internal MCTS search tree (representing
           their subjective nested belief $b_j$ and expected policy $\pi_j$). For Level-0 agents,
           the node pointer is `None` (pure stochastic baseline).

    Attributes:
        state: The physical environment state $s \in S$.
        models: Dictionary mapping opponent `AgentID` to a tuple of `(AgentFrame, Optional[POMCPNode])`.
    """
    state: State

    # Map of opponent AgentID -> (Their Strategy Level/Frame, Their MCTS Root Node)
    models: Dict[AgentID, Tuple[AgentFrame, Optional['POMCPNode']]]