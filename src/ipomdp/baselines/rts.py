"""Reachability-tree sampling (RTS) over interactive beliefs: the IPF + RTS baseline.

Reference: P. Doshi and P. J. Gmytrasiewicz, "Monte Carlo Sampling Methods for
Approximating Interactive POMDPs", JAIR 34 (2009), which pairs the interactive
particle filter with a sampled reachability tree for planning.

Lookahead
---------
From belief b with H decisions remaining, for every own action a:

    Q_H(b, a) = E[R | b, a] + gamma * (1 / K) * sum_{k=1..K} V_{H-1}(b'_k),
    o_k ~ P(o | b, a),   b'_k = update(b, a, o_k),   V_h(b) = max_a Q_h(b, a),

with Q_1(b, a) = E[R | b, a]. Observations are sampled by drawing a particle,
simulating one joint step with the opponent acting by its declared model, and
sampling the agent's observation; a step that ends the episode contributes no
continuation. The update is the bank's filter, so with ``SolverBank(particles=N)``
this is the IPF + RTS combination of the reference; with exact beliefs it is RTS
over exact beliefs.

Fair comparison with I-POMCP
----------------------------
Immediate rewards use the same exact integrator as I-POMCP
(``SolverBank.expected_rewards``), and opponents act by the same declared
models, so a comparison varies only how the future is explored: a fixed sampled
tree of (|A| K)^(H-1) belief updates here, an adaptive tree there. Cost is
reported as CPU time, because sampled observations and MCTS simulations are
different units of work.

Each solve draws from its own stream seeded like I-POMCP's, so a modeled RTS
policy is a pure function of the modeled belief.
"""

import math
import random
from dataclasses import asdict, dataclass

from ipomdp.belief.finite_belief import MentalModel
from ipomdp.belief.frame import AgentFrame
from ipomdp.planning.generative_model import InteractiveGenerativeModel
from ipomdp.planning.planner import Planner
from ipomdp.planning.policy import greedy_policy, sample_policy


@dataclass(frozen=True)
class RTSConfig:
    """Attributes:
    depth: Lookahead in decisions.
    observation_samples: Sampled observations K per (belief, action) node.
    gamma: Discount of the lookahead objective.
    """

    depth: int
    observation_samples: int
    gamma: float = 0.95

    def __post_init__(self):
        for name in ("depth", "observation_samples"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.gamma) or not 0 <= self.gamma <= 1:
            raise ValueError("gamma must lie in [0, 1]")


class RTSPlanner(Planner):
    def __init__(self, solver_key, pomdp_model, solver_bank, config: RTSConfig):
        self.key, self.pomdp_model, self.solver_bank = solver_key, pomdp_model, solver_bank
        self.config = config
        self.actions = list(pomdp_model.get_all_actions(solver_key.agent_id))
        self.gen_model = InteractiveGenerativeModel(solver_bank)
        self.rng = random.Random(solver_bank.stream_seed("actions", str(solver_key)))
        self.recommendation_values = {}
        self.belief = self.initial_belief = None

    def set_initial_belief(self, belief):
        self.model(belief)  # validates nesting
        self.belief = self.initial_belief = belief

    def model(self, belief=None):
        frame = AgentFrame(self.key.agent_id, self.key.level, self.pomdp_model)
        return MentalModel(frame, self.belief if belief is None else belief)

    def get_action(self, belief=None):
        return sample_policy(self.policy_for(self.model(belief)), self.rng)

    def update_root(self, action, observation):
        self.belief = self.solver_bank.filter.update(
            self.model(), action, observation, terminal=False
        ).belief
        self.recommendation_values = {}

    def policy_for(self, model, modeled=False):
        if any(self.pomdp_model.is_terminal(atom.state) for atom, _ in model.belief.mass):
            raise ValueError("Condition on public continuation before requesting a policy")
        settings = {"planner": "RTS", "config": asdict(self.config)}
        model = self.solver_bank.search_model(model)
        rng = random.Random(self.solver_bank.search_seed(model, settings))
        values = self._q_values(model, self.config.depth, rng)
        if not modeled:
            self.recommendation_values = values
        return greedy_policy(values)

    def _q_values(self, model, remaining, rng):
        rewards = dict(self.solver_bank.expected_rewards(model))
        if remaining == 1:
            return rewards
        return {
            action: rewards[action]
            + self.config.gamma * self._continuation(model, action, remaining, rng)
            for action in self.actions
        }

    def _continuation(self, model, action, remaining, rng):
        """(1/K) sum_k V_{remaining-1}(b'_k) over K sampled own observations."""
        particles, weights = zip(*self.solver_bank.ordered_mass(model.belief))
        total = 0.0
        for particle in rng.choices(particles, weights=weights, k=self.config.observation_samples):
            following, joint, _, terminal = self.gen_model.tree_step(
                particle, action, self.key.agent_id, self.pomdp_model, rng
            )
            if terminal:
                continue
            observation = self.pomdp_model.sample_observation(
                following.state, joint, self.key.agent_id, rng
            )
            successor = MentalModel(
                model.frame,
                self.solver_bank.search_filter.update(
                    model, action, observation, terminal=False
                ).belief,
            )
            total += max(self._q_values(successor, remaining - 1, rng).values())
        return total / self.config.observation_samples

    def get_detailed_stats(self):
        return {
            "key": str(self.key),
            "belief_size": len(self.belief.mass),
            "action_values": {str(a): q for a, q in self.recommendation_values.items()},
        }
