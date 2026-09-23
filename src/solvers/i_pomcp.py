"""Root-sampling MCTS over immutable joint interactive states.

The authoritative belief is a weighted FiniteBelief, independent of search visits
and child reservoirs. Real observations use the shared recursive Bayesian kernel.
Search trees are rebuilt for a solve, preventing cached policy predictions from
depending on unrelated earlier queries. Finite search and an empirical initial
prior remain approximations; finite conditioning is exact relative to that prior.
"""

import random
from dataclasses import asdict

from core.config import IPOMCPConfig
from ipomdp.finite_belief import InteractiveState, MentalModel
from ipomdp.frame import AgentFrame
from solvers.exploration import NormalizedUCB
from solvers.generative_model import InteractiveGenerativeModel
from solvers.node import POMCPNode
from solvers.planner import Planner
from solvers.policy import greedy_policy, sample_policy, search_randomness


class IPOMCPPlanner(Planner):
    def __init__(
        self,
        solver_key,
        pomdp_model,
        action_space,
        solver_bank,
        config=None,
        exploration_strategy=None,
    ):
        self.key, self.pomdp_model = solver_key, pomdp_model
        self.actions, self.solver_bank = list(action_space), solver_bank
        self.config = config or IPOMCPConfig()
        self.exploration_strategy = exploration_strategy or NormalizedUCB(
            self.config.mcts.exploration_const
        )
        self.gen_model = InteractiveGenerativeModel(solver_bank)
        self.root = POMCPNode(capacity=self.config.mcts.node_capacity)
        self.belief = None
        self.initial_belief = None
        self.initial_sample_count = 0

    def set_initial_belief(self, belief, sample_count):
        MentalModel(AgentFrame(self.key.agent_id, self.key.level, self.pomdp_model), belief)
        self.belief = self.initial_belief = belief
        self.initial_sample_count = sample_count

    def model(self, belief=None):
        return MentalModel(
            AgentFrame(self.key.agent_id, self.key.level, self.pomdp_model),
            self.belief if belief is None else belief,
        )

    def policy_for(self, model, modeled=False):
        if any(self.pomdp_model.is_terminal(atom.state) for atom, _ in model.belief.mass):
            raise ValueError("Condition on public continuation before requesting a policy")
        n_sims = self.config.opponent.n_sims if modeled else self.config.mcts.n_sims
        settings = {
            "planner": "MCTS",
            "config": asdict(self.config),
            "n_sims": n_sims,
            "exploration": vars(self.exploration_strategy),
        }
        with search_randomness(self.solver_bank.search_seed(model, settings)):
            node = POMCPNode(capacity=self.config.mcts.node_capacity)
            legal_sets = {
                frozenset(self.pomdp_model.get_legal_actions(p.state, self.key.agent_id))
                for p, _ in model.belief.mass
                if not self.pomdp_model.is_terminal(p.state)
            }
            if len(legal_sets) != 1:
                raise ValueError("Planning requires one nonterminal observable action set")
            if n_sims < len(next(iter(legal_sets))):
                raise ValueError("Search budget must evaluate every available action")
            particles, weights = zip(*model.belief.mass)
            bounds = {
                "q_min": float("inf"),
                "q_max": -float("inf"),
                "root_rewards": dict(self.solver_bank.expected_rewards(model)),
            }
            root_belief = None
            if hasattr(self.pomdp_model, "update_rollout_belief"):
                p_counts = {}
                for p, mass in model.belief.mass:
                    p_counts[p.state] = p_counts.get(p.state, 0.0) + mass
                total_mass = sum(p_counts.values())
                if total_mass > 0:
                    root_belief = {s: m / total_mass for s, m in p_counts.items()}

            # Draw in one batch; random.choices constructs its cumulative weights once.
            for particle in random.choices(particles, weights=weights, k=n_sims):
                self._simulate(particle, node, 0, bounds, belief=root_belief)
            policy = greedy_policy(node.action_values)
        if not modeled:
            self.root = node
        return policy

    def get_action(self, belief=None):
        return sample_policy(self.policy_for(self.model(belief)))

    def _simulate(self, particle, node, depth, bounds, belief=None):
        if depth >= self.config.mcts.max_depth or self.pomdp_model.is_terminal(particle.state):
            node.visit_count += 1
            return 0.0
        legal = (
            self.pomdp_model.get_candidate_actions(particle.state, self.key.agent_id, belief=belief)
            if depth > 0 and hasattr(self.pomdp_model, "get_candidate_actions")
            else self.pomdp_model.get_legal_actions(particle.state, self.key.agent_id)
        )
        action = self.exploration_strategy.select_action(
            node, legal, q_min=bounds["q_min"], q_max=bounds["q_max"]
        )
        if depth + 1 == self.config.mcts.max_depth:
            _, _, reward, _ = self.gen_model.sample_event(
                particle, action, self.key.agent_id, self.pomdp_model
            )
            q = reward
        else:
            following, joint, reward, terminal = self.gen_model.tree_step(
                particle, action, self.key.agent_id, self.pomdp_model
            )
            observation = self.pomdp_model.sample_observation(
                following.state, joint, self.key.agent_id
            )
            child = node.get_child(action, observation)
            new = child is None
            child = node.create_child(action, observation) if new else child
            child.add_particle(following)
            next_belief = (
                self.pomdp_model.update_rollout_belief(
                    belief, action, observation, self.key.agent_id
                )
                if hasattr(self.pomdp_model, "update_rollout_belief")
                else None
            )
            continuation = (
                0.0
                if terminal
                else (
                    self._rollout(following, depth + 1, belief=next_belief)
                    if new
                    else self._simulate(following, child, depth + 1, bounds, belief=next_belief)
                )
            )
            q = reward + self.config.mcts.gamma * continuation
        if depth == 0:
            q += bounds["root_rewards"][action] - reward
        node.action_counts[action] = node.action_counts.get(action, 0) + 1
        node.action_values[action] = (
            node.action_values.get(action, 0)
            + (q - node.action_values.get(action, 0)) / node.action_counts[action]
        )
        node.visit_count += 1
        bounds["q_min"], bounds["q_max"] = min(bounds["q_min"], q), max(bounds["q_max"], q)
        return q

    def _rollout(self, particle, depth, belief=None):
        if depth >= self.config.mcts.max_depth or self.pomdp_model.is_terminal(particle.state):
            return 0.0
        action = self.pomdp_model.get_rollout_action(
            particle.state, self.key.agent_id, belief=belief
        )
        if action not in self.pomdp_model.get_legal_actions(particle.state, self.key.agent_id):
            raise ValueError("Domain rollout policy returned an illegal action")
        following_state, joint, reward, terminal = self.gen_model.sample_event(
            particle, action, self.key.agent_id, self.pomdp_model
        )
        if terminal or depth + 1 >= self.config.mcts.max_depth:
            return reward
        obs = self.pomdp_model.sample_observation(following_state, joint, self.key.agent_id)
        next_belief = (
            self.pomdp_model.update_rollout_belief(belief, action, obs, self.key.agent_id)
            if hasattr(self.pomdp_model, "update_rollout_belief")
            else None
        )
        following = InteractiveState(following_state, particle.opponent)
        return reward + self.config.mcts.gamma * self._rollout(
            following, depth + 1, belief=next_belief
        )

    def update_root(self, action, observation):
        """Condition a continuing episode on the private sensor and public survival."""
        # Weighted support is retained in full; child reservoirs do not estimate it.
        self.belief = self.solver_bank.filter.update(
            self.model(), action, observation, terminal=False
        ).belief
        self.root = POMCPNode(capacity=self.config.mcts.node_capacity)
        self.solver_bank.clear_caches()

    def get_detailed_stats(self):
        return {
            "key": str(self.key),
            "n_sims": self.config.mcts.n_sims,
            "modeled_n_sims": self.config.opponent.n_sims,
            "initial_sample_count": self.initial_sample_count,
            "belief_size": len(self.belief.mass),
            "root_visit_count": self.root.visit_count,
            "action_values": {str(a): q for a, q in self.root.action_values.items()},
            "action_counts": {str(a): c for a, c in self.root.action_counts.items()},
        }

    def visualize(self, filename, step):
        from utils.visualizer import ForestVisualizer

        return ForestVisualizer(self.solver_bank).export_forest(self, filename, step)
