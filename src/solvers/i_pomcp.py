import random
from typing import List, Optional, Dict, Tuple, TYPE_CHECKING

from core.pomdp_model import Action, Observation, State, AgentID, POMDPModel
from core.config import IPOMCPConfig
from core.logger import get_logger
from solvers.exploration import ExplorationStrategy, StandardUCB
from solvers.planner import Planner
from solvers.node import POMCPNode
from solvers.generative_model import InteractiveGenerativeModel
from solvers.solver_types import SolverKey, AgentFrame
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
        self.prior_level_weights: Optional[Dict[int, float]] = None

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

    def _get_opponent_level_prior(self, other_id: AgentID) -> Dict[int, float]:
        """Returns the prior probability distribution over opponent levels."""
        if self.prior_level_weights is not None:
            return dict(self.prior_level_weights)
        if self.initial_particles:
            counts: Dict[int, int] = {}
            total = 0
            for p in self.initial_particles:
                if other_id in p.models:
                    lvl = p.models[other_id][0].level
                    counts[lvl] = counts.get(lvl, 0) + 1
                    total += 1
            if total > 0:
                return {lvl: c / total for lvl, c in sorted(counts.items())}
        if self.key.level > 0:
            return {k: 1.0 / self.key.level for k in range(self.key.level)}
        return {0: 1.0}

    def _get_particle_level_distribution(self, particles: List['InteractiveParticle'], other_id: AgentID) -> Dict[int, float]:
        """Computes the empirical distribution over opponent levels from a particle population."""
        if not particles:
            return {}
        counts: Dict[int, int] = {}
        total = 0
        for p in particles:
            if hasattr(p, 'models') and other_id in p.models:
                lvl = p.models[other_id][0].level
                counts[lvl] = counts.get(lvl, 0) + 1
                total += 1
        if total == 0:
            return {}
        return {lvl: c / total for lvl, c in sorted(counts.items())}

    def _create_fresh_opponent_node(self, other_id: AgentID, level: int) -> Optional[POMCPNode]:
        """Creates a fresh, canonical root node for an opponent of the specified level."""
        if level == 0:
            return None
        node = POMCPNode(capacity=self.config.mcts.node_capacity)
        key = SolverKey(other_id, level)
        if self.solver_bank.has_solver(key):
            opp_solver = self.solver_bank.get_solver(key)
            if hasattr(opp_solver, 'initial_particles') and opp_solver.initial_particles:
                for p in opp_solver.initial_particles:
                    node.add_particle(p)
            elif hasattr(opp_solver, 'root') and hasattr(opp_solver.root, 'belief_particles') and opp_solver.root.belief_particles:
                for p in opp_solver.root.belief_particles:
                    node.add_particle(p)
            elif hasattr(opp_solver, 'belief') and opp_solver.belief:
                for p in opp_solver.belief:
                    node.add_particle(p)
        return node

    def _sample_consistent_state(self, action: Action, observation: Observation,
                                 candidate_states: Optional[List[State]] = None) -> State:
        """Samples a physical state consistent with the given action and observation."""
        if hasattr(self.pomdp_model, 'sample_state_consistent_with_obs'):
            return self.pomdp_model.sample_state_consistent_with_obs(action, observation, agent_id=self.key.agent_id)

        candidates = list(candidate_states) if candidate_states else []
        for _ in range(50):
            s = random.choice(candidates) if candidates else self.pomdp_model.get_initial_state()
            joint_action = {self.key.agent_id: action}
            w = self.pomdp_model.get_observation_prob(observation, s, joint_action, self.key.agent_id)
            if w > 0.0 and random.random() < w:
                return s

        for _ in range(50):
            s = self.pomdp_model.get_initial_state()
            joint_action = {self.key.agent_id: action}
            w = self.pomdp_model.get_observation_prob(observation, s, joint_action, self.key.agent_id)
            if w > 0.0 and random.random() < w:
                return s

        return candidates[0] if candidates else self.pomdp_model.get_initial_state()

    def update_root(self, action: Action, observation: Observation, min_particles: int = 0) -> None:
        if not self.initial_particles and self.root.belief_particles:
            self.initial_particles = list(self.root.belief_particles)

        target_count = min(
            self.config.mcts.node_capacity,
            max(min_particles, len(self.initial_particles), self.config.reinvigoration.min_particles)
        )
        if target_count <= 0:
            target_count = self.config.reinvigoration.min_particles

        # Identify all opponents
        seed_particles = self.initial_particles if self.initial_particles else self.root.belief_particles
        other_agent_ids = sorted(list({k for p in seed_particles for k in p.models.keys()}))

        # Check for domain epoch reset (e.g. door opened in Tiger)
        is_reset = hasattr(self.pomdp_model, 'is_epoch_reset') and self.pomdp_model.is_epoch_reset(action, observation)

        if is_reset:
            new_root = POMCPNode(capacity=self.config.mcts.node_capacity)
            opp_weights: Dict[AgentID, Dict[int, float]] = {}
            candidate_roots: Dict[AgentID, Dict[int, Optional[POMCPNode]]] = {}

            for other_id in other_agent_ids:
                # Use empirical posterior accumulated in current root belief; fallback to prior
                prev_dist = self._get_particle_level_distribution(self.root.belief_particles, other_id)
                if not prev_dist:
                    prev_dist = self._get_opponent_level_prior(other_id)

                candidate_levels = sorted(list(self._get_opponent_level_prior(other_id).keys()))
                k_levels = max(1, len(candidate_levels))

                # Extinction floor: Laplace / uniform smoothing over candidate levels (never tethered to prior)
                eps = 0.01 if self.config.reinvigoration.preserve_levels else 0.0
                smoothed = {
                    lvl: (1.0 - eps) * prev_dist.get(lvl, 1.0 / k_levels) + eps * (1.0 / k_levels)
                    for lvl in candidate_levels
                }
                total_w = sum(smoothed.values())
                opp_weights[other_id] = {lvl: w / total_w for lvl, w in smoothed.items()}

                candidate_roots[other_id] = {
                    lvl: self._create_fresh_opponent_node(other_id, lvl)
                    for lvl in candidate_levels if lvl > 0
                }

            for _ in range(target_count):
                s = self._sample_consistent_state(action, observation)
                models = {}
                for other_id in other_agent_ids:
                    lvls = list(opp_weights[other_id].keys())
                    probs = list(opp_weights[other_id].values())
                    chosen_lvl = random.choices(lvls, weights=probs, k=1)[0]
                    frame = AgentFrame(other_id, chosen_lvl, self.pomdp_model)
                    node_ptr = candidate_roots[other_id].get(chosen_lvl) if chosen_lvl > 0 else None
                    models[other_id] = (frame, node_ptr)
                new_root.add_particle(InteractiveParticle(state=s, models=models))

            # Extinction protection guarantee
            if self.config.reinvigoration.preserve_levels:
                for other_id in other_agent_ids:
                    curr_dist = self._get_particle_level_distribution(new_root.belief_particles, other_id)
                    candidate_levels = list(opp_weights[other_id].keys())
                    for lvl in candidate_levels:
                        if curr_dist.get(lvl, 0.0) == 0:
                            idx = random.randint(0, len(new_root.belief_particles) - 1)
                            s = self._sample_consistent_state(action, observation)
                            models = dict(new_root.belief_particles[idx].models)
                            frame = AgentFrame(other_id, lvl, self.pomdp_model)
                            node_ptr = candidate_roots[other_id].get(lvl) if lvl > 0 else None
                            models[other_id] = (frame, node_ptr)
                            new_root.belief_particles[idx] = InteractiveParticle(state=s, models=models)

            self.root = new_root
            if self.config.reinvigoration.enabled:
                self._reinvigorate_mental_models()
            return

        # Normal step (no epoch reset)
        child = self.root.get_child(action, observation)
        survivors = list(child.belief_particles) if child and child.belief_particles else []
        current_count = len(survivors)

        candidate_roots: Dict[AgentID, Dict[int, Optional[POMCPNode]]] = {}
        post_weights: Dict[AgentID, Dict[int, float]] = {}

        for other_id in other_agent_ids:
            candidate_levels = sorted(list(self._get_opponent_level_prior(other_id).keys()))
            k_levels = max(1, len(candidate_levels))

            prev_dist = self._get_particle_level_distribution(self.root.belief_particles, other_id)
            if not prev_dist:
                prev_dist = self._get_opponent_level_prior(other_id)

            counts = {
                lvl: sum(1 for p in survivors if p.models.get(other_id, (None, None))[0].level == lvl)
                for lvl in candidate_levels
            }
            k = sum(counts.values())

            # Dirichlet-Multinomial Bayesian update
            N0 = 5
            raw_post = {
                lvl: (N0 * prev_dist.get(lvl, 1.0 / k_levels) + counts[lvl]) / (N0 + k)
                for lvl in candidate_levels
            }

            eps = 0.01 if self.config.reinvigoration.preserve_levels else 0.0
            smoothed = {
                lvl: (1.0 - eps) * raw_post[lvl] + eps * (1.0 / k_levels)
                for lvl in candidate_levels
            }
            total_w = sum(smoothed.values())
            post_weights[other_id] = {lvl: w / total_w for lvl, w in smoothed.items()}

            candidate_roots[other_id] = {}
            for lvl in candidate_levels:
                if lvl > 0:
                    surv_nodes = [
                        p.models[other_id][1] for p in survivors
                        if p.models.get(other_id, (None, None))[0].level == lvl and p.models[other_id][1] is not None
                    ]
                    if surv_nodes:
                        candidate_roots[other_id][lvl] = surv_nodes[0]
                    else:
                        candidate_roots[other_id][lvl] = self._create_fresh_opponent_node(other_id, lvl)

        if current_count == 0:
            logger.warning(f"[{self.key}] Particle Deprivation (Obs: {observation}). Resampling consistent particles.")
            self.deprivation_events += 1

        new_root = POMCPNode(capacity=self.config.mcts.node_capacity)
        alpha = self.config.reinvigoration.alpha
        n_surv_target = min(current_count, int(target_count * (1.0 - alpha)))
        replenished = [random.choice(survivors) for _ in range(n_surv_target)] if n_surv_target > 0 else []

        surv_states = [p.state for p in survivors] if survivors else []
        n_needed = target_count - len(replenished)

        for _ in range(n_needed):
            s = self._sample_consistent_state(action, observation, candidate_states=surv_states)
            models = {}
            for other_id in other_agent_ids:
                lvls = list(post_weights[other_id].keys())
                probs = list(post_weights[other_id].values())
                chosen_lvl = random.choices(lvls, weights=probs, k=1)[0]
                frame = AgentFrame(other_id, chosen_lvl, self.pomdp_model)
                node_ptr = candidate_roots[other_id].get(chosen_lvl) if chosen_lvl > 0 else None
                models[other_id] = (frame, node_ptr)
            replenished.append(InteractiveParticle(state=s, models=models))

        # Extinction protection guarantee
        if self.config.reinvigoration.preserve_levels:
            for other_id in other_agent_ids:
                rep_dist = self._get_particle_level_distribution(replenished, other_id)
                candidate_levels = list(post_weights[other_id].keys())
                for lvl in candidate_levels:
                    if rep_dist.get(lvl, 0.0) == 0:
                        idx = random.randint(0, len(replenished) - 1)
                        s = self._sample_consistent_state(action, observation, candidate_states=surv_states)
                        models = dict(replenished[idx].models)
                        frame = AgentFrame(other_id, lvl, self.pomdp_model)
                        node_ptr = candidate_roots[other_id].get(lvl) if lvl > 0 else None
                        models[other_id] = (frame, node_ptr)
                        replenished[idx] = InteractiveParticle(state=s, models=models)

        new_root.belief_particles = replenished
        new_root._total_particles_routed = len(replenished)
        self.root = new_root

        if self.config.reinvigoration.enabled:
            self._reinvigorate_mental_models()

    def _reinvigorate_mental_models(self) -> None:
        if not self.root.belief_particles:
            return

        seen_nodes = set()
        for p in self.root.belief_particles:
            for other_id, (frame, node_ptr) in p.models.items():
                if frame.level > 0 and node_ptr is not None:
                    node_id = id(node_ptr)
                    if node_id not in seen_nodes:
                        seen_nodes.add(node_id)
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