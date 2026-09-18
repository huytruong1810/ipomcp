"""History nodes with bounded trajectory reservoirs and non-owning parent links.

Algorithm R gives each routed item equal inclusion probability. That property is
about the input stream; it does not prove that a biased stream is a Bayesian
posterior. Memory is O(capacity) per node, plus children and referenced nested
models. The number of nodes and referenced beliefs is not globally bounded by
capacity. Weak parent pointers let obsolete siblings be collected after rerooting.
"""

import random
import weakref
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from core.pomdp_model import Action, Observation

# Safely import InteractiveState for strict typing
if TYPE_CHECKING:
    from ipomdp.finite_belief import InteractiveState


class POMCPNode:
    """
    A single node in the POMCP / I-POMCP search tree.

    Corresponds to a *history* h = (a_1, o_1, ..., a_t, o_t). Children are
    indexed by `(action, observation)` pairs, forming an AND-OR tree.
    """

    def __init__(self, parent: Optional["POMCPNode"] = None, capacity: int = 500) -> None:
        """
        Initializes the MCTS node.

        Args:
            parent: Back-pointer to the parent node (`None` for the root).
            capacity: Maximum number of particles to hold before Reservoir Sampling kicks in.
        """
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("Node capacity must be a positive integer.")
        self.visit_count: int = 0
        self.action_counts: Dict[Action, int] = {}
        self.action_values: Dict[Action, float] = {}

        # AND-OR Tree: children[action][observation] -> POMCPNode
        self.children: Dict[Action, Dict[Observation, "POMCPNode"]] = {}
        self._parent_ref = weakref.ref(parent) if parent is not None else None

        # Bounded local belief, strictly typed
        self.belief_particles: List["InteractiveState"] = []
        self.capacity: int = capacity

        # Tracks how many particles have EVER passed through here (Used for Algorithm R math)
        self._total_particles_routed: int = 0

    @property
    def parent(self):
        """Non-owning navigation link; descendants never own discarded siblings."""
        return self._parent_ref() if self._parent_ref is not None else None

    @parent.setter
    def parent(self, parent):
        self._parent_ref = weakref.ref(parent) if parent is not None else None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_parent_ref"] = None
        return state

    def add_particle(self, particle: "InteractiveState") -> None:
        """
        Appends a particle to this node's local belief.
        Implements Algorithm R (Reservoir Sampling) to guarantee bounded memory footprints
        while maintaining an unbiased statistical sample of historical trajectories.
        """
        self._total_particles_routed += 1

        if len(self.belief_particles) < self.capacity:
            # Reservoir is not full yet; append directly.
            self.belief_particles.append(particle)
        else:
            # Reservoir is full. The new particle replaces an existing one with decreasing probability.
            replace_idx = random.randint(0, self._total_particles_routed - 1)
            if replace_idx < self.capacity:
                self.belief_particles[replace_idx] = particle

    def get_child(self, action: Action, observation: Observation) -> Optional["POMCPNode"]:
        """Safely fetches a child node if it exists, otherwise returns None."""
        return self.children.get(action, {}).get(observation)

    def create_child(self, action: Action, observation: Observation) -> "POMCPNode":
        """
        Instantiates and links a new child node in the AND-OR tree.
        Crucially passes down the `capacity` limit from the parent to ensure
        the entire tree respects the memory constraints.
        """
        if action not in self.children:
            self.children[action] = {}

        if observation not in self.children[action]:
            # Inherit the reservoir capacity configuration from the parent node
            self.children[action][observation] = POMCPNode(parent=self, capacity=self.capacity)

        return self.children[action][observation]

    def to_dict(self, max_depth: int = 3, current_depth: int = 0) -> Dict[str, Any]:
        """
        Recursively serializes the sub-tree rooted at this node for Artifact Logging.
        Primarily used for debugging and exporting to Graphviz.
        """
        node_dict: Dict[str, Any] = {
            "visit_count": self.visit_count,
            "action_values": {str(a): v for a, v in self.action_values.items()},
            "action_counts": {str(a): c for a, c in self.action_counts.items()},
            "n_particles": len(self.belief_particles),
            "total_routed": self._total_particles_routed,
        }

        if current_depth < max_depth:
            children_dict: Dict[str, Dict[str, Any]] = {}
            for action, obs_map in self.children.items():
                action_str = str(action)
                children_dict[action_str] = {}
                for obs, child_node in obs_map.items():
                    obs_str = str(obs)
                    children_dict[action_str][obs_str] = child_node.to_dict(
                        max_depth, current_depth + 1
                    )

            node_dict["children"] = children_dict

        return node_dict
