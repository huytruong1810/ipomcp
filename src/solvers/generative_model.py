"""
generative_model.py — Joint action simulation and nested MCTS expansion.

Simulates simultaneous actions and state transitions across multi-agent hierarchies.
Performs Entropy-Gated Just-In-Time (JIT) mental model expansion for opponent sub-trees
during Monte Carlo tree search rollouts.
"""

import math
import random
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from core.config import JITConfig
from core.pomdp_model import Action, AgentID, POMDPModel
from ipomdp.belief import InteractiveParticle
from solvers.node import POMCPNode
from solvers.solver_types import AgentFrame

if TYPE_CHECKING:
    from solvers.solver_bank import SolverBank


class InteractiveGenerativeModel:
    """
    Handles the recursive simulation of joint actions and mental state updates
    during the Monte Carlo tree search rollouts.
    """

    def __init__(self, solver_bank: "SolverBank", config: Optional[JITConfig] = None):
        self.solver_bank = solver_bank
        self.config = config if config is not None else JITConfig()

        # Pre-compute inverse temperature for fast-math multiplication
        self._inv_temp = 1.0 / max(self.config.temperature, 1e-6)

    def _compute_policy_entropy(self, node: POMCPNode) -> float:
        if not node.action_values or len(node.action_values) < 2:
            return 1.0

        q_values = list(node.action_values.values())
        max_q = max(q_values)

        try:
            # FAST-MATH: Multiplication instead of division
            exp_qs = [math.exp((q - max_q) * self._inv_temp) for q in q_values]
            sum_exp = sum(exp_qs)

            # Avoid divide-by-zero during normalization
            if sum_exp == 0:
                return 1.0

            probs = [e / sum_exp for e in exp_qs]

            entropy = -sum(p * math.log(p) for p in probs if p > 0)
            max_entropy = math.log(len(probs))
            return entropy / max_entropy if max_entropy > 0 else 0.0

        except OverflowError:
            return 0.0

    def tree_step(
        self,
        particle: InteractiveParticle,
        action_i: Action,
        agent_id_i: AgentID,
        physics_model: POMDPModel,
    ) -> Tuple[InteractiveParticle, Dict[AgentID, Action], float, bool]:

        joint_action: Dict[AgentID, Action] = {agent_id_i: action_i}

        for other_id, (frame, node_ptr) in particle.models.items():
            if frame.level == 0:
                legal_actions_j = physics_model.get_legal_actions(particle.state, other_id)
                joint_action[other_id] = random.choice(legal_actions_j)

            else:
                if node_ptr is not None:
                    # Entropy-Gated JIT Expansion using injected Config
                    if (
                        node_ptr.visit_count < self.config.visit_threshold
                        or self._compute_policy_entropy(node_ptr) > self.config.entropy_threshold
                    ):
                        opp_solver = self.solver_bank.get_solver_for_frame(frame)
                        local_bounds = {"q_min": float("inf"), "q_max": -float("inf")}
                        opp_solver.extend_search(
                            node_ptr, n_sims=self.config.sims, bounds=local_bounds
                        )

                    action_j = self._sample_action_from_node(node_ptr)
                else:
                    action_j = None

                if action_j is None:
                    legal_actions_j = physics_model.get_legal_actions(particle.state, other_id)
                    action_j = random.choice(legal_actions_j)

                joint_action[other_id] = action_j

        s_next = physics_model.sample_transition(particle.state, joint_action)
        reward_i = physics_model.get_reward(particle.state, joint_action, s_next, agent_id_i)
        is_terminal = physics_model.is_terminal(s_next)

        next_models: Dict[AgentID, Tuple[AgentFrame, Optional[POMCPNode]]] = {}
        for other_id, (frame, node_ptr) in particle.models.items():
            if frame.level == 0 or node_ptr is None:
                next_models[other_id] = (frame, None)
            else:
                act_j = joint_action[other_id]
                o_j = physics_model.sample_observation(s_next, joint_action, other_id)
                child_j = node_ptr.get_child(act_j, o_j)
                if child_j is None:
                    child_j = node_ptr.create_child(act_j, o_j)
                # Replenish opponent child belief on every traversal via Algorithm R reservoir sampling
                if node_ptr.belief_particles:
                    p_sample = random.choice(node_ptr.belief_particles)
                    child_j.add_particle(InteractiveParticle(state=s_next, models=p_sample.models))
                next_models[other_id] = (frame, child_j)

        p_next = InteractiveParticle(state=s_next, models=next_models)
        return p_next, joint_action, reward_i, is_terminal

    def _sample_action_from_node(self, node: POMCPNode) -> Optional[Action]:
        if node.visit_count == 0:
            return None

        # Prefer scale-invariant Boltzmann exploration over normalized Q-values
        if node.action_values:
            actions = list(node.action_values.keys())
            q_values = [node.action_values[a] for a in actions]
            max_q = max(q_values)
            min_q = min(q_values)
            q_range = max_q - min_q

            if self.config.temperature <= 1e-3 or q_range < 1e-6:
                best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
                return random.choice(best_actions)

            tau = max(self.config.temperature, 1e-3)
            exp_scaled_qs = [math.exp(((q - max_q) / q_range) / tau) for q in q_values]
            sum_exp = sum(exp_scaled_qs)
            if sum_exp > 0:
                probs = [e / sum_exp for e in exp_scaled_qs]
                return random.choices(actions, weights=probs, k=1)[0]
            best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
            return random.choice(best_actions)

        # Fallback to action_counts if action_values is empty
        if node.action_counts:
            actions = list(node.action_counts.keys())
            max_c = max(node.action_counts.values())
            best = [a for a in actions if node.action_counts[a] == max_c]
            return random.choice(best)

        return None
