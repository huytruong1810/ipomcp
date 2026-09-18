"""
random_planner — Level-0 baseline agent (uniform random policy).

A :class:`RandomPlanner` selects actions uniformly at random from the
available action space.  It serves two roles in the I-POMCP framework:

1. **Baseline agent** — Used as the opponent model for Level-1 agents
   (who assume their opponents act randomly).
2. **Reference policy** — Provides an empirical comparator for evaluating whether
   higher-level planning yields an advantage; it is not a mathematical lower bound.

Because Level-0 agents have no internal belief or search tree, all
inherited ``Planner`` methods (``extend_search``, ``update_root``, etc.)
remain no-ops.
"""

import random
from typing import Any, List, Optional

from core.pomdp_model import Action
from solvers.planner import Planner


class RandomPlanner(Planner):
    """Level-0 agent: selects actions uniformly at random.

    Attributes:
        possible_actions: The fixed list of actions to choose from.
    """

    def __init__(self, possible_actions: List[Action]) -> None:
        """Initialise with a known action space.

        Args:
            possible_actions: All legal actions for this agent.
        """
        if not possible_actions:
            raise ValueError("Random policy requires a nonempty action space")
        self.possible_actions: List[Action] = list(possible_actions)

    def get_action(self, belief: Optional[Any] = None) -> Action:
        """Return a uniformly random action (ignores *belief*)."""
        return random.choice(self.possible_actions)

    def extend_search(self, node: Any, n_sims: int) -> None:
        """No-op — Level-0 agents do not maintain a search tree."""
