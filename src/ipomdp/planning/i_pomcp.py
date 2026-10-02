"""I-POMCP: root-sampling Monte Carlo tree search over interactive beliefs.

Algorithm (one decision)
------------------------
The agent's belief b is a finite joint distribution over physical states and
complete opponent models (``belief.finite_belief``). For each of N simulations:

1. Draw a root particle (s, m_j) from b.
2. Descend the tree of the agent's private histories. At history h with
   remaining horizon H, choose an action by horizon-bounded UCB
   (``exploration.HorizonBoundUCB``), then simulate one joint step: the
   opponent acts by its own modeled policy at m_j, the state transitions, both
   agents observe, and the opponent's private model advances by its own
   subjective update (``generative_model``).
3. At a new child, estimate its value by one rollout with the domain's
   information-respecting rollout policy.
4. Back up with the empirical chance-weighted Bellman operator
   (``node.POMCPNode.bellman_backup``).

Two quantities are integrated exactly instead of sampled, because doing so was
what made the estimator converge against exact references:

- Immediate reward at every tree history: E[R | b_h, a], where b_h is the
  agent's exact posterior at private history h. The planner carries b_h down
  the tree with the same belief update used for real observations. b_h never
  receives the sampled hidden state or the opponent's actual action, so the
  integration adds no information the agent does not have.
- The last decision (H = 1): Q_1(b_h, a) = E[R | b_h, a], maximized after
  integrating, which is the exact finite-horizon boundary condition.

Continuations below the root are still sampled. Search is therefore an
estimator of the exact finite-horizon I-POMDP value relative to the supplied
belief and the opponents' modeled policies; it is not an exact solver.

Reproducibility
---------------
Each solve draws from its own ``random.Random`` seeded by (bank seed, agent,
level, canonical belief digest, configuration). Equal problems therefore
receive equal searches regardless of query order, which makes modeled opponent
policies pure functions of their beliefs. Real actions are sampled from a
separate stream owned by the planner. Nothing touches the global ``random``.
"""

import random
from dataclasses import asdict

from ipomdp.belief.finite_belief import MentalModel
from ipomdp.belief.frame import AgentFrame
from ipomdp.planning.config import PlannerConfig
from ipomdp.planning.exploration import HorizonBoundUCB
from ipomdp.planning.generative_model import InteractiveGenerativeModel
from ipomdp.planning.node import POMCPNode
from ipomdp.planning.planner import Planner
from ipomdp.planning.policy import greedy_policy, sample_policy


class IPOMCPPlanner(Planner):
    def __init__(self, solver_key, pomdp_model, solver_bank, config: PlannerConfig):
        self.key, self.pomdp_model, self.solver_bank = solver_key, pomdp_model, solver_bank
        self.config = config
        self.actions = list(pomdp_model.get_all_actions(solver_key.agent_id))
        self.exploration = HorizonBoundUCB.for_agent(
            pomdp_model, solver_key.agent_id, config.exploration
        )
        self.gen_model = InteractiveGenerativeModel(solver_bank)
        # Stream for sampling this agent's real actions from its greedy policy.
        self.rng = random.Random(solver_bank.stream_seed("actions", str(solver_key)))
        self.root = POMCPNode()
        self.recommendation_values = {}
        self.belief = None
        self.initial_belief = None

    # ------------------------------------------------------------------ #
    # Beliefs
    # ------------------------------------------------------------------ #

    def set_initial_belief(self, belief):
        self.model(belief)  # validates nesting
        self.belief = self.initial_belief = belief

    def model(self, belief=None):
        frame = AgentFrame(self.key.agent_id, self.key.level, self.pomdp_model)
        return MentalModel(frame, self.belief if belief is None else belief)

    def update_root(self, action, observation):
        """Condition a continuing episode on the agent's own action and observation.

        The bank's caches are kept. Modeled policies and belief updates are pure
        functions of their arguments, so results computed at earlier steps stay
        valid; nested models whose beliefs recur (common after resets) are not
        solved again. Measured on L3 Tiger with 64 particles, this cut CPU per
        decision by about a quarter with identical decisions. Memory is bounded
        by the caches' LRU sizes.
        """
        self.belief = self.solver_bank.filter.update(
            self.model(), action, observation, terminal=False
        ).belief
        self.root = POMCPNode()
        self.recommendation_values = {}

    # ------------------------------------------------------------------ #
    # Decisions
    # ------------------------------------------------------------------ #

    def get_action(self, belief=None):
        return sample_policy(self.policy_for(self.model(belief)), self.rng)

    def policy_for(self, model, modeled=False):
        """Greedy policy at ``model``'s belief; ``modeled`` selects the opponent budget."""
        if any(self.pomdp_model.is_terminal(atom.state) for atom, _ in model.belief.mass):
            raise ValueError("Condition on public continuation before requesting a policy")
        simulations = self.config.modeled_simulations if modeled else self.config.simulations
        if simulations < len(self.actions):
            raise ValueError("Search budget must evaluate every action")
        settings = {"planner": "I-POMCP", "config": asdict(self.config), "simulations": simulations}
        rng = random.Random(self.solver_bank.search_seed(model, settings))
        node = POMCPNode()
        particles, weights = zip(*self.solver_bank.ordered_mass(model.belief))
        memory = self._physical_marginal(model)
        # One batch draw: choices() builds its cumulative weights once.
        for particle in rng.choices(particles, weights=weights, k=simulations):
            self._simulate(particle, node, 0, memory, model, rng)
        values = dict(node.action_values)
        policy = greedy_policy(values)
        if not modeled:
            self.root = node
            self.recommendation_values = values
        return policy

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #

    def _simulate(self, particle, node, depth, memory, history_model, rng):
        """One descent; returns the sampled-continuation Q estimate at ``node``."""
        if self.pomdp_model.is_terminal(particle.state):
            node.visit_count += 1
            return 0.0
        remaining = self.config.depth - depth
        if remaining == 1:
            # Boundary history: solved by integration, not by sampled visits, so
            # it stores values without inventing per-action simulation counts.
            if not node.action_values:
                node.action_values.update(self._leaf_values(history_model))
            node.visit_count += 1
            return max(node.action_values.values())
        action = self.exploration.select_action(
            node, self.actions, remaining, self.config.gamma, rng
        )
        following, joint, _, terminal = self.gen_model.tree_step(
            particle, action, self.key.agent_id, self.pomdp_model, rng
        )
        if terminal:
            continuation = 0.0
        else:
            observation = self.pomdp_model.sample_observation(
                following.state, joint, self.key.agent_id, rng
            )
            next_memory = self.pomdp_model.update_rollout_memory(
                memory, action, observation, self.key.agent_id
            )
            next_model = self._history_successor(history_model, action, observation)
            child = node.get_child(action, observation)
            if child is None:
                child = node.create_child(action, observation)
                continuation = self._rollout(following, depth + 1, next_memory, next_model, rng)
                child.rollout_value = continuation
            else:
                continuation = self._simulate(
                    following, child, depth + 1, next_memory, next_model, rng
                )
            counts = node.continuation_counts.setdefault(action, {})
            counts[observation] = counts.get(observation, 0) + 1
        immediate = dict(self.solver_bank.expected_rewards(history_model))[action]
        node.action_counts[action] = node.action_counts.get(action, 0) + 1
        q = node.bellman_backup(action, immediate, self.config.gamma)
        node.visit_count += 1
        return q

    def _rollout(self, particle, depth, memory, history_model, rng):
        """Sampled return of the domain rollout policy, with the exact last step.

        Rollout rewards stay sampled: integrating them would require the exact
        posterior at every rollout history, which is the cost the tree already
        pays. Opponent private models still advance at every rollout step,
        because freezing them would change the opponent's dynamics.
        """
        if self.pomdp_model.is_terminal(particle.state):
            return 0.0
        if self.config.depth - depth == 1:
            return max(self._leaf_values(history_model).values())
        action = self.pomdp_model.rollout_action(self.key.agent_id, memory, rng)
        following, joint, reward, terminal = self.gen_model.tree_step(
            particle, action, self.key.agent_id, self.pomdp_model, rng
        )
        if terminal:
            return reward
        observation = self.pomdp_model.sample_observation(
            following.state, joint, self.key.agent_id, rng
        )
        next_memory = self.pomdp_model.update_rollout_memory(
            memory, action, observation, self.key.agent_id
        )
        next_model = self._history_successor(history_model, action, observation)
        return reward + self.config.gamma * self._rollout(
            following, depth + 1, next_memory, next_model, rng
        )

    # ------------------------------------------------------------------ #
    # Exact pieces
    # ------------------------------------------------------------------ #

    def _history_successor(self, model, action, observation):
        """The agent's exact posterior after its own action and observation."""
        posterior = self.solver_bank.filter.update(model, action, observation, terminal=False)
        return MentalModel(model.frame, posterior.belief)

    def _leaf_values(self, model):
        """Q_1(b, a) = E[R | b, a] over the full joint belief; maximize afterwards."""
        return dict(self.solver_bank.expected_rewards(model))

    @staticmethod
    def _physical_marginal(model):
        """Physical marginal of the belief: the initial memory of the rollout policy."""
        marginal = {}
        for atom, mass in model.belief.mass:
            marginal[atom.state] = marginal.get(atom.state, 0.0) + mass
        return marginal

    def get_detailed_stats(self):
        return {
            "key": str(self.key),
            "simulations": self.config.simulations,
            "modeled_simulations": self.config.modeled_simulations,
            "belief_size": len(self.belief.mass),
            "root_visit_count": self.root.visit_count,
            "action_values": {str(a): q for a, q in self.root.action_values.items()},
            "action_counts": {str(a): c for a, c in self.root.action_counts.items()},
        }
