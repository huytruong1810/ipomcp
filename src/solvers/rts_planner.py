"""
rts_planner.py — Reachability Tree Sampling (RTS) Planner with Interactive Particle Filtering (I-PF).

Reference:
    "Monte Carlo Sampling Methods for Approximating Interactive POMDPs"
    Prashant Doshi & Piotr J. Gmytrasiewicz, Journal of Artificial Intelligence Research (JAIR), 2009.
    Sections 5–8 (Interactive Particle Filtering) & Section 11 (Sampling Reachability Trees).

Mathematical Formulation:
1. Interactive Particle Filter (I-PF):
   Maintains an interactive belief B_i^l = { <s^(c), theta_j^(c)> }_(c=1)^C over S x Theta_j^(<l).
   For action a_i and observation o_i:
   - For each particle c:
       a_j^(c) ~ pi_j(b_j^(c))
       s'^(c) ~ T(s^(c), a_i, a_j^(c))
       o_j^(c) ~ O_j(s'^(c), a_i, a_j^(c))
       b_j'^(c) = SE_j(b_j^(c), a_j^(c), o_j^(c))
       w^(c) = O_i(o_i | s'^(c), a_i, a_j^(c))
   - Resample C particles proportional to w^(c).

2. Reachability Tree Sampling (RTS):
   Recursively expands forward lookahead tree from B_i^l up to horizon H:
   Q(B, a_i) = R(B, a_i) + gamma * sum_{o_i} P(o_i | B, a_i) * max_{a_i'} Q(B'_{a_i, o_i}, a_i')
   where P(o_i | B, a_i) = (1/C) * sum_{c=1}^C w^(c)(o_i).
"""

import random
from typing import Dict, List, Optional, Tuple

from core.config import RTSConfig
from core.distribution import ParticleDistribution
from core.logger import get_logger
from core.pomdp_model import Action, AgentID, Observation, POMDPModel, State
from ipomdp.belief import InteractiveParticle
from solvers.node import POMCPNode
from solvers.planner import Planner
from solvers.solver_bank import SolverBank
from solvers.solver_types import AgentFrame, SolverKey

logger = get_logger("RTSPlanner")


class RTSPlanner(Planner):
    """
    Reachability Tree Sampling (RTS) Sampled Lookahead Planner with Interactive Particle Filtering (I-PF).
    Inspired by Doshi & Gmytrasiewicz (2009); finite particles and top-k observation pruning make this approximate.
    Nested model propagation remains subject to the limitations in docs/THEORY.md.
    """

    def __init__(
        self,
        solver_key: SolverKey,
        pomdp_model: POMDPModel,
        action_space: List[Action],
        solver_bank: SolverBank,
        config: Optional[RTSConfig] = None,
    ):

        self.key = solver_key
        self.pomdp_model = pomdp_model
        self.actions = list(action_space)
        self.solver_bank = solver_bank

        base_config = config if config is not None else RTSConfig()
        self.config = base_config
        self.gamma = base_config.gamma
        self.max_depth = base_config.max_depth
        self.obs_branching = base_config.obs_branching
        self.num_particles = base_config.num_particles

        self.belief: List[InteractiveParticle] = []
        self.last_action_values: Dict[Action, float] = {}

    def set_initial_belief(self, particles: List[InteractiveParticle]) -> None:
        self.belief = list(particles)

    def get_action_values(self) -> Dict[Action, float]:
        """Returns the most recent approximate action value estimates."""
        return dict(self.last_action_values)

    def get_detailed_stats(self) -> dict:
        """Returns snapshot of RTS planning stats for logging."""
        return {
            "solver_key": str(self.key),
            "belief_size": len(self.belief),
            "action_values": {str(a): float(v) for a, v in self.last_action_values.items()},
            "max_depth": self.max_depth,
            "obs_branching": self.obs_branching,
        }

    def get_action(self, belief: Optional[List[InteractiveParticle]] = None) -> Action:
        """
        Executes Reachability Tree Sampling (RTS) lookahead search to select the optimal action.
        """
        current_belief = belief if belief is not None else self.belief
        if not current_belief:
            return random.choice(self.actions)

        legal_actions = self.pomdp_model.get_legal_actions(
            current_belief[0].state, self.key.agent_id
        )
        if not legal_actions:
            legal_actions = self.actions

        best_action = None
        best_value = -float("inf")
        action_values = {}

        for action in legal_actions:
            q_val = self._evaluate_action_branch(current_belief, action, depth=0)
            action_values[action] = float(q_val)
            if q_val > best_value:
                best_value = q_val
                best_action = action

        self.last_action_values = action_values
        return best_action if best_action is not None else random.choice(legal_actions)

    def _sample_opponent_action(
        self, frame: AgentFrame, node_ptr: Optional[POMCPNode], state: State
    ) -> Action:
        """
        Evaluates the opponent's policy pi_j(b_j) for an interactive particle.
        """
        other_id = frame.agent_id
        if frame.level == 0 or node_ptr is None:
            legal_actions = self.pomdp_model.get_legal_actions(state, other_id)
            return random.choice(legal_actions)

        # Level >= 1 Opponent: evaluate policy from registered opponent solver
        opp_key = SolverKey(other_id, frame.level)
        if self.solver_bank.has_solver(opp_key):
            opp_solver = self.solver_bank.get_solver(opp_key)
            if isinstance(opp_solver, RTSPlanner):
                particles = getattr(node_ptr, "belief_particles", None)
                if particles:
                    return opp_solver.get_action(particles)
                return opp_solver.get_action()
            elif hasattr(opp_solver, "get_action"):
                if hasattr(node_ptr, "action_values") and node_ptr.action_values:
                    return max(node_ptr.action_values, key=node_ptr.action_values.get)
                if hasattr(opp_solver, "extend_search"):
                    n_sims = min(
                        30,
                        getattr(getattr(opp_solver, "config", None), "mcts", None).n_sims
                        if hasattr(opp_solver, "config")
                        else 30,
                    )
                    opp_solver.extend_search(node_ptr, n_sims=n_sims)
                    if node_ptr.action_counts:
                        return max(node_ptr.action_counts, key=node_ptr.action_counts.get)
                return opp_solver.get_action()

        legal_actions = self.pomdp_model.get_legal_actions(state, other_id)
        return random.choice(legal_actions)

    def _evaluate_action_branch(
        self, belief: List[InteractiveParticle], action: Action, depth: int
    ) -> float:
        """
        Recursive RTS Reachability Tree evaluation via Backward Induction (Section 11).
        """
        if depth >= self.max_depth or not belief:
            return 0.0

        C = len(belief)
        if C == 0:
            return 0.0

        agent_id = self.key.agent_id
        all_obs = self.pomdp_model.get_all_observations(agent_id)

        # Step 1: Forward predictive propagation per particle
        # Store: (p_next, joint_action, reward_i, is_terminal)
        transitioned_records: List[
            Tuple[InteractiveParticle, Dict[AgentID, Action], float, bool]
        ] = []
        total_reward = 0.0

        for p in belief:
            if self.pomdp_model.is_terminal(p.state):
                transitioned_records.append((p, {}, 0.0, True))
                continue
            joint_action: Dict[AgentID, Action] = {agent_id: action}
            next_models: Dict[AgentID, Tuple[AgentFrame, Optional[POMCPNode]]] = {}

            # Determine opponent actions under their respective models
            for other_id, (frame, node_ptr) in p.models.items():
                a_j = self._sample_opponent_action(frame, node_ptr, p.state)
                joint_action[other_id] = a_j

            # Physical transition
            s_next = self.pomdp_model.sample_transition(p.state, joint_action)
            r_i = self.pomdp_model.get_reward(p.state, joint_action, s_next, agent_id)
            is_terminal = self.pomdp_model.is_terminal(s_next)
            total_reward += r_i

            # Opponent belief propagation
            for other_id, (frame, node_ptr) in p.models.items():
                if frame.level == 0 or node_ptr is None:
                    next_models[other_id] = (frame, None)
                else:
                    # Opponent receives observation and updates belief
                    o_j = self.pomdp_model.sample_observation(s_next, joint_action, other_id)
                    new_node = POMCPNode()
                    if hasattr(node_ptr, "belief_particles") and node_ptr.belief_particles:
                        # Bayesian state estimation for opponent particles
                        opp_particles = []
                        opp_weights = []
                        for opp_p in node_ptr.belief_particles:
                            opp_state = opp_p.state
                            opp_models = opp_p.models
                            opp_s_next = self.pomdp_model.sample_transition(opp_state, joint_action)
                            opp_w = self.pomdp_model.get_observation_prob(
                                o_j, opp_s_next, joint_action, other_id
                            )
                            opp_particle = InteractiveParticle(state=opp_s_next, models=opp_models)
                            opp_particles.append(opp_particle)
                            opp_weights.append(opp_w)
                        if sum(opp_weights) > 0:
                            opp_dist = ParticleDistribution(opp_particles, opp_weights)
                            opp_dist.normalize()
                            for resampled_p in opp_dist.resample(len(node_ptr.belief_particles)):
                                new_node.add_particle(resampled_p)
                    next_models[other_id] = (frame, new_node)

            p_next = InteractiveParticle(state=s_next, models=next_models)
            transitioned_records.append((p_next, joint_action, r_i, is_terminal))

        avg_reward = total_reward / C

        # Step 2: Calculate particle-estimated observation likelihoods P(o_i | B, a_i)
        obs_weights: Dict[Observation, List[float]] = {o: [] for o in all_obs}
        obs_probs: Dict[Observation, float] = {}

        for p_next, joint_action, _, is_term in transitioned_records:
            for o in all_obs:
                if is_term:
                    w = 0.0
                else:
                    w = self.pomdp_model.get_observation_prob(
                        o, p_next.state, joint_action, agent_id
                    )
                obs_weights[o].append(w)

        for o in all_obs:
            p_o = sum(obs_weights[o]) / C
            if p_o > 0:
                obs_probs[o] = p_o

        if not obs_probs:
            return avg_reward

        # Branching selection: all positive observations or top-K
        # Keep the original mass: terminal trajectories have zero continuation.
        normalized_probs = obs_probs
        branch_limit = self.obs_branching if self.obs_branching is not None else len(all_obs)
        active_obs = sorted(
            normalized_probs.keys(), key=lambda o: normalized_probs[o], reverse=True
        )[:branch_limit]

        # Step 3: Backward Induction over reachable child beliefs
        future_expected_val = 0.0
        for o in active_obs:
            prob_o = normalized_probs[o]
            weights = obs_weights[o]

            if sum(weights) == 0:
                continue

            # Resample C particles with importance weights w(o) to form child belief B'(a, o)
            particles_list = [r[0] for r in transitioned_records]
            dist = ParticleDistribution(particles_list, weights)
            dist.normalize()
            child_belief = dist.resample(self.num_particles)

            # Evaluate max Q over child belief at next depth
            legal_next = self.pomdp_model.get_legal_actions(child_belief[0].state, agent_id)
            if not legal_next:
                legal_next = self.actions

            max_child_q = -float("inf")
            for next_a in legal_next:
                child_q = self._evaluate_action_branch(child_belief, next_a, depth + 1)
                if child_q > max_child_q:
                    max_child_q = child_q

            if max_child_q != -float("inf"):
                future_expected_val += prob_o * max_child_q

        return avg_reward + self.gamma * future_expected_val

    def update_root(self, action: Action, observation: Observation, min_particles: int = 0) -> None:
        """
        Executes online Interactive Particle Filtering (I-PF) update (Section 6).
        """
        self.belief = self._ipf_update(self.belief, action, observation)

    def _ipf_update(
        self, belief: List[InteractiveParticle], action: Action, observation: Observation
    ) -> List[InteractiveParticle]:
        if not belief:
            return []

        agent_id = self.key.agent_id
        next_particles = []
        weights = []

        for p in belief:
            joint_action: Dict[AgentID, Action] = {agent_id: action}
            next_models: Dict[AgentID, Tuple[AgentFrame, Optional[POMCPNode]]] = {}

            for other_id, (frame, node_ptr) in p.models.items():
                a_j = self._sample_opponent_action(frame, node_ptr, p.state)
                joint_action[other_id] = a_j

            s_next = self.pomdp_model.sample_transition(p.state, joint_action)
            w = self.pomdp_model.get_observation_prob(observation, s_next, joint_action, agent_id)

            for other_id, (frame, node_ptr) in p.models.items():
                if frame.level == 0 or node_ptr is None:
                    next_models[other_id] = (frame, None)
                else:
                    o_j = self.pomdp_model.sample_observation(s_next, joint_action, other_id)
                    new_node = POMCPNode()
                    if hasattr(node_ptr, "belief_particles") and node_ptr.belief_particles:
                        opp_particles = []
                        opp_weights = []
                        for opp_p in node_ptr.belief_particles:
                            opp_state = opp_p.state
                            opp_models = opp_p.models
                            opp_s_next = self.pomdp_model.sample_transition(opp_state, joint_action)
                            opp_w = self.pomdp_model.get_observation_prob(
                                o_j, opp_s_next, joint_action, other_id
                            )
                            opp_particle = InteractiveParticle(state=opp_s_next, models=opp_models)
                            opp_particles.append(opp_particle)
                            opp_weights.append(opp_w)
                        if sum(opp_weights) > 0:
                            opp_dist = ParticleDistribution(opp_particles, opp_weights)
                            opp_dist.normalize()
                            for resampled_p in opp_dist.resample(len(node_ptr.belief_particles)):
                                new_node.add_particle(resampled_p)
                    next_models[other_id] = (frame, new_node)

            p_next = InteractiveParticle(state=s_next, models=next_models)
            next_particles.append(p_next)
            weights.append(w)

        if sum(weights) == 0:
            logger.warning(
                f"[{self.key}] I-PF observation weight sum is zero. Preserving previous belief."
            )
            return belief

        dist = ParticleDistribution(next_particles, weights)
        dist.normalize()
        return dist.resample(self.num_particles)
