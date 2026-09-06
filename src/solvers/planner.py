"""
planner — Abstract base class for all I-POMDP planning agents.

Every planner in the framework — from the trivial :class:`RandomPlanner`
to the full :class:`IPOMCPPlanner` — implements this interface.  The
:class:`GenericBatchRunner` and :class:`I_POMDP_Bootstrapper` interact
with planners exclusively through these methods, enabling plug-and-play
swapping of planning algorithms.
"""

import abc
from typing import Any, Optional

from core.pomdp_model import Action

# Forward reference to avoid circular import with ``solvers.node``.
POMCPNode = Any


class Planner(abc.ABC):
    """Abstract interface for all agents (Level 0, 1, 2, …).

    Concrete sub-classes must implement :meth:`get_action`.  The remaining
    methods have sensible default (no-op) implementations so that simple
    planners (e.g., ``RandomPlanner``) need not override them.
    """

    @abc.abstractmethod
    def get_action(self, belief: Optional[Any] = None) -> Action:
        """Select the next action to execute.

        For tree-based planners this triggers the full MCTS search; for
        Level-0 planners it simply draws a random action.

        Args:
            belief: An optional external belief object.  Tree-based planners
                    typically use their internal root node's particles instead.

        Returns:
            The chosen action.
        """

    def extend_search(self, node: POMCPNode, n_sims: int) -> None:
        """Deepen the search tree starting from *node*.

        Called by the :class:`InteractiveGenerativeModel` during JIT
        expansion: when a higher-level agent visits an opponent's tree
        node that has a low visit count, it triggers this method to
        "sharpen" the opponent's predicted policy at that node.

        The default implementation is a **no-op** (suitable for Level-0 /
        ``RandomPlanner``).

        Args:
            node:   The MCTS node to expand.
            n_sims: Number of additional simulations to run from *node*.
        """

    def update_root(self, action: Action, observation: Any, min_particles: int) -> None:
        """Advance the planner's root after a real-world step.

        Implements amortised online belief updating: the root pointer is
        moved to the child node corresponding to ``(action, observation)``,
        and particles are reinvigorated if the count drops below
        *min_particles*.

        The default implementation is a **no-op** (suitable for Level-0 /
        ``RandomPlanner``).

        Args:
            action:        The action that was actually executed.
            observation:   The observation that was actually received.
            min_particles: Minimum particle count; triggers reinvigoration
                           if the new root has fewer particles.
        """

    def get_detailed_stats(self) -> dict:
        """Return a snapshot of internal statistics for logging.

        Typical contents include root Q-values, visit counts, belief
        size, and a shallow tree serialisation.

        Returns:
            A JSON-serialisable dictionary (empty by default).
        """
        return {}

    def visualize(self, filename: str, step: int) -> None:
        """Export a visualisation of the current planning tree.

        Requires Graphviz.  The default implementation is a **no-op**.

        Args:
            filename: Output file path (without extension).
            step:     Current simulation step (for labelling).
        """
