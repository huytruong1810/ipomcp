# Absolute Path: <project_root>/solvers/node.py

"""
node.py — MCTS search-tree node with strict static typing and memory bounds.

DESIGN DECISION RECORD (Big-Tech Refactor Phase 1):
---------------------------------------------------
1. STRICT TYPE COERCION:
   `belief_particles` is now explicitly typed to `List['InteractiveParticle']` rather
   than `List[Any]`. This prevents developers from accidentally appending raw physical
   states or generic dicts to the node's local belief, which would crash the MCTS fast-loop.

2. RESERVOIR SAMPLING:
   Maintains the Algorithm R implementation to guarantee $O(1)$ memory bound scaling
   per node while preserving an unbiased uniform probability distribution of historical trajectories.
"""

import random
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from core.pomdp_model import Action, Observation

# [BIG-TECH REFACTOR]: Safely import InteractiveParticle for strict typing
if TYPE_CHECKING:
    from ipomdp.belief import InteractiveParticle


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
        self.visit_count: int = 0
        self.action_counts: Dict[Action, int] = {}
        self.action_values: Dict[Action, float] = {}

        # AND-OR Tree: children[action][observation] -> POMCPNode
        self.children: Dict[Action, Dict[Observation, "POMCPNode"]] = {}
        self.parent: Optional["POMCPNode"] = parent

        # Bounded local belief, strictly typed
        self.belief_particles: List['InteractiveParticle'] = []
        self.capacity: int = capacity

        # Tracks how many particles have EVER passed through here (Used for Algorithm R math)
        self._total_particles_routed: int = 0

    def add_particle(self, particle: 'InteractiveParticle') -> None:
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
            "total_routed": self._total_particles_routed
        }

        if current_depth < max_depth:
            children_dict: Dict[str, Dict[str, Any]] = {}
            for action, obs_map in self.children.items():
                action_str = str(action)
                children_dict[action_str] = {}
                for obs, child_node in obs_map.items():
                    obs_str = str(obs)
                    children_dict[action_str][obs_str] = child_node.to_dict(max_depth, current_depth + 1)

            node_dict["children"] = children_dict

        return node_dict