"""Sampled reachability lookahead using the same immutable model transition as MCTS.

Observation branch probabilities retain survival mass. Top-k omission remains an
explicit approximation. Online observations use the common recursive finite filter,
never the sampled lookahead reservoir or a preserved old belief on zero weight.
"""

import math
import random
from dataclasses import asdict, dataclass

from ipomdp.baselines.particles import ParticleDistribution
from ipomdp.belief.finite_belief import MentalModel
from ipomdp.belief.frame import AgentFrame
from ipomdp.planning.generative_model import InteractiveGenerativeModel
from ipomdp.planning.planner import Planner
from ipomdp.planning.policy import greedy_policy, sample_policy, search_randomness


@dataclass(frozen=True)
class RTSConfig:
    """Sampled reachability-tree lookahead budgets.

    Attributes:
        max_depth: Lookahead depth in decisions.
        obs_branching: Observation branches expanded per action (most likely first).
        num_particles: Weighted particles propagated per branch.
        gamma: Discount factor of the lookahead objective.
    """

    max_depth: int
    obs_branching: int
    num_particles: int
    gamma: float = 0.95

    def __post_init__(self):
        for name in ("max_depth", "obs_branching", "num_particles"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.gamma) or not 0 <= self.gamma <= 1:
            raise ValueError("gamma must lie in [0, 1]")


class RTSPlanner(Planner):
    def __init__(self, solver_key, pomdp_model, solver_bank, config: RTSConfig):
        self.key, self.pomdp_model = solver_key, pomdp_model
        self.actions = list(pomdp_model.get_all_actions(solver_key.agent_id))
        self.solver_bank = solver_bank
        self.config = config
        self.gamma, self.max_depth = self.config.gamma, self.config.max_depth
        self.obs_branching, self.num_particles = (
            self.config.obs_branching,
            self.config.num_particles,
        )
        self.belief = self.initial_belief = None
        self.last_action_values = {}
        self.gen_model = InteractiveGenerativeModel(solver_bank)

    def set_initial_belief(self, belief):
        self.model(belief)  # validates nesting
        self.belief = self.initial_belief = belief

    def model(self, belief=None):
        return MentalModel(
            AgentFrame(self.key.agent_id, self.key.level, self.pomdp_model),
            self.belief if belief is None else belief,
        )

    def policy_for(self, model, modeled=False):
        if any(self.pomdp_model.is_terminal(atom.state) for atom, _ in model.belief.mass):
            raise ValueError("Condition on public continuation before requesting a policy")
        with search_randomness(
            self.solver_bank.search_seed(model, {"planner": "RTS", "config": asdict(self.config)})
        ):
            atoms, weights = zip(*model.belief.mass)
            particles = random.choices(atoms, weights=weights, k=self.num_particles)
            legal_sets = {
                frozenset(self.pomdp_model.get_legal_actions(p.state, self.key.agent_id))
                for p, _ in model.belief.mass
                if not self.pomdp_model.is_terminal(p.state)
            }
            if len(legal_sets) != 1:
                raise ValueError("Planning requires one nonterminal observable action set")
            legal = next(iter(legal_sets))
            values = {
                a: self._evaluate_action_branch(particles, a, 0) for a in self.actions if a in legal
            }
        if not modeled:
            self.last_action_values = values
        return greedy_policy(values)

    def get_action(self, belief=None):
        return sample_policy(self.policy_for(self.model(belief)))

    def _evaluate_action_branch(self, belief, action, depth):
        if depth >= self.max_depth:
            return 0.0
        if not belief:
            raise ValueError("Lookahead needs a nonempty empirical belief")
        records, reward = [], 0.0
        for particle in belief:
            if self.pomdp_model.is_terminal(particle.state):
                records.append((particle, {}, 0.0, True))
            else:
                if depth + 1 == self.max_depth:
                    reward += self.gen_model.sample_event(
                        particle, action, self.key.agent_id, self.pomdp_model
                    )[2]
                    continue
                record = self.gen_model.tree_step(
                    particle, action, self.key.agent_id, self.pomdp_model
                )
                records.append(record)
                reward += record[2]
        reward /= len(belief)
        if depth + 1 == self.max_depth:
            return reward
        branches = []
        for observation in self.pomdp_model.get_all_observations(self.key.agent_id):
            weights = [
                0.0
                if terminal
                else self.pomdp_model.get_observation_prob(
                    observation, p.state, joint, self.key.agent_id
                )
                for p, joint, _, terminal in records
            ]
            probability = sum(weights) / len(belief)
            if probability:
                branches.append((probability, weights))
        continuation = 0.0
        for probability, weights in sorted(branches, key=lambda row: row[0], reverse=True)[
            : self.obs_branching
        ]:
            child = ParticleDistribution([r[0] for r in records], weights).resample(
                self.num_particles
            )
            legal_sets = {
                frozenset(self.pomdp_model.get_legal_actions(p.state, self.key.agent_id))
                for p in child
            }
            if len(legal_sets) != 1:
                raise ValueError("Observation branch mixes different known action inventories")
            legal = next(iter(legal_sets))
            continuation += probability * max(
                self._evaluate_action_branch(child, a, depth + 1)
                for a in self.actions
                if a in legal
            )
        return reward + self.gamma * continuation

    def update_root(self, action, observation):
        """Condition a continuing episode on the private sensor and public survival."""
        self.belief = self.solver_bank.filter.update(
            self.model(), action, observation, terminal=False
        ).belief
        self.last_action_values = {}

    def get_action_values(self):
        return dict(self.last_action_values)

    def get_detailed_stats(self):
        return {
            "solver_key": str(self.key),
            "belief_size": len(self.belief.mass),
            "action_values": {str(a): float(q) for a, q in self.last_action_values.items()},
            "max_depth": self.max_depth,
            "obs_branching": self.obs_branching,
        }
