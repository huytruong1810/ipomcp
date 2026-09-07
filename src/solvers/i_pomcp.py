# Absolute Path: <project_root>/solvers/i_pomcp.py

import random
from typing import List, Optional, Dict, TYPE_CHECKING

from core.pomdp_model import Action, Observation, POMDPModel
from core.config import IPOMCPConfig
from core.logger import get_logger
from solvers.exploration import ExplorationStrategy, StandardUCB
from solvers.planner import Planner
from solvers.node import POMCPNode
from solvers.generative_model import InteractiveGenerativeModel
from solvers.solver_types import SolverKey
from solvers.solver_bank import SolverBank
from ipomdp.belief import InteractiveParticle

logger = get_logger("IPOMCPPlanner")


class IPOMCPPlanner(Planner):
    """
    Monte Carlo Tree Search planner for multi-agent I-POMDP environments.
    """

    def __init__(self,
                 solver_key: SolverKey,
                 pomdp_model: POMDPModel,
                 action_space: List[Action],
                 solver_bank: SolverBank,
                 config: Optional[IPOMCPConfig] = None,
                 exploration_strategy: Optional[ExplorationStrategy] = None):

        self.key = solver_key
        self.pomdp_model = pomdp_model
        self.actions = action_space
        self.solver_bank = solver_bank
        self.config = config if config is not None else IPOMCPConfig()

        if exploration_strategy is None:
            self.exploration_strategy = StandardUCB(exploration_const=self.config.mcts.exploration_fallback_const)
        else:
            self.exploration_strategy = exploration_strategy

        self.deprivation_events = 0

        self.gen_model = InteractiveGenerativeModel(solver_bank, config=self.config.jit)
        self.root = POMCPNode(capacity=self.config.mcts.node_capacity)

        self.initial_particles: List['InteractiveParticle'] = []

    def get_action(self, belief: Optional[List['InteractiveParticle']] = None) -> Action:
        if not self.initial_particles and self.root.belief_particles:
            self.initial_particles = list(self.root.belief_particles)

        self.extend_search(self.root, n_sims=self.config.mcts.n_sims)

        if not self.root.action_counts:
            state = self.root.belief_particles[0].state if self.root.belief_particles else self.pomdp_model.get_initial_state()
            legal_actions = self.pomdp_model.get_legal_actions(state, self.key.agent_id)
            return random.choice(legal_actions)

        best_action = max(self.root.action_counts, key=self.root.action_counts.get)
        return best_action

    def extend_search(self, node_ptr: POMCPNode, n_sims: int, bounds: Optional[Dict[str, float]] = None) -> None:
        if not node_ptr.belief_particles:
            return

        local_bounds = bounds if bounds is not None else {'q_min': float('inf'), 'q_max': -float('inf')}

        for _ in range(n_sims):
            particle = random.choice(node_ptr.belief_particles)
            self._simulate(particle, node_ptr, depth=0, bounds=local_bounds)

    def _simulate(self, particle: 'InteractiveParticle', node: POMCPNode, depth: int,
                  bounds: Dict[str, float]) -> float:
        if depth >= self.config.mcts.max_depth or self.pomdp_model.is_terminal(particle.state):
            node.add_particle(particle)
            node.visit_count += 1
            return 0.0

        legal_actions = self.pomdp_model.get_legal_actions(particle.state, self.key.agent_id)

        action = self.exploration_strategy.select_action(node, legal_actions,
                                                         q_min=bounds['q_min'],
                                                         q_max=bounds['q_max'])

        p_next, joint_action, reward_i, is_terminal = self.gen_model.tree_step(
            particle, action, self.key.agent_id, self.pomdp_model
        )

        observation = self.pomdp_model.sample_observation(p_next.state, joint_action, self.key.agent_id)

        is_new_node = False
        if action not in node.children or observation not in node.children[action]:
            child = node.create_child(action, observation)
            child.add_particle(p_next)
            is_new_node = True
        else:
            child = node.children[action][observation]

        if is_new_node:
            q = reward_i + self.config.mcts.gamma * self._rollout(p_next, depth + 1)
        else:
            q = reward_i + self.config.mcts.gamma * self._simulate(p_next, child, depth + 1, bounds)

        node.action_counts[action] = node.action_counts.get(action, 0) + 1
        current_q = node.action_values.get(action, 0.0)
        node.action_values[action] = current_q + (q - current_q) / node.action_counts[action]

        node.visit_count += 1
        node.add_particle(particle)

        if q < bounds['q_min']: bounds['q_min'] = q
        if q > bounds['q_max']: bounds['q_max'] = q

        return q

    def _rollout(self, particle: 'InteractiveParticle', depth: int) -> float:
        if depth >= self.config.mcts.max_depth or self.pomdp_model.is_terminal(particle.state):
            return 0.0

        legal_actions = self.pomdp_model.get_legal_actions(particle.state, self.key.agent_id)
        if not legal_actions:
            return 0.0

        action = random.choice(legal_actions)
        p_next, joint_action, reward_i, is_terminal = self.gen_model.tree_step(
            particle, action, self.key.agent_id, self.pomdp_model
        )

        return reward_i + self.config.mcts.gamma * self._rollout(p_next, depth + 1)

    def update_root(self, action: Action, observation: Observation, min_particles: int = 0) -> None:
        child = self.root.get_child(action, observation)

        if child and len(child.belief_particles) > 0:
            self.root = child
            current_count = len(self.root.belief_particles)

            if current_count < min_particles:
                survivors = list(self.root.belief_particles)
                while len(self.root.belief_particles) < min_particles:
                    p = random.choice(survivors)
                    self.root.add_particle(p)

            if self.config.reinvigoration.enabled:
                self._reinvigorate_mental_models()

        else:
            logger.warning(f"[{self.key}] Particle Deprivation (Obs: {observation}). Falling back to prior.")
            self.deprivation_events += 1

            self.root = POMCPNode(capacity=self.config.mcts.node_capacity)
            if self.initial_particles:
                for p in self.initial_particles:
                    self.root.add_particle(p)

    def _reinvigorate_mental_models(self) -> None:
        if not self.root.belief_particles:
            return

        for p in self.root.belief_particles:
            for other_id, (frame, node_ptr) in p.models.items():
                if frame.level > 0 and node_ptr is not None:
                    if node_ptr.visit_count < self.config.reinvigoration.visit_threshold:
                        opp_solver = self.solver_bank.get_solver_for_frame(frame)
                        local_bounds = {'q_min': float('inf'), 'q_max': -float('inf')}
                        opp_solver.extend_search(node_ptr, n_sims=self.config.reinvigoration.sims, bounds=local_bounds)

    def get_detailed_stats(self) -> dict:
        return {
            "key": str(self.key),
            "n_sims": self.config.mcts.n_sims,
            "deprivation_events": self.deprivation_events,
            "root_visit_count": self.root.visit_count,
            "action_values": {str(a): v for a, v in self.root.action_values.items()},
            "action_counts": {str(a): c for a, c in self.root.action_counts.items()},
            "belief_size": len(self.root.belief_particles)
        }