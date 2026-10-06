"""I-POMCP: root-sampling Monte Carlo tree search over interactive beliefs.

Algorithm (one decision)
------------------------
The agent's belief b is a finite joint distribution over physical states and
complete opponent models (``belief.finite_belief``). For each of N simulations:

1. Descend the tree of the agent's private histories. At history h with
   remaining horizon H, choose an action by horizon-bounded UCB
   (``exploration.HorizonBoundUCB``), draw an interactive state (s, m_j) from
   the agent's posterior b_h at h, then simulate one joint step: the opponent
   acts by its own modeled policy at m_j, the state transitions, both agents
   observe, and the opponent's private model advances by its own subjective
   update (``generative_model``).
3. At a new child, estimate its value by one rollout with the domain's
   information-respecting rollout policy.
4. Back up with the empirical chance-weighted Bellman operator
   (``node.POMCPNode.bellman_backup``).

After the simulations, the finished tree is rescored bottom-up with the exact
chance law P(o, alive | b_h, a) in place of the sampled observation frequencies
(``chance_scoring``). The rescored values decide; the search values are kept
beside them for diagnostics. Real decisions and modeled opponents use the same
rule, so an opponent is predicted to act as the planner itself would.

Two quantities are integrated exactly instead of sampled, because doing so was
what made the estimator converge against exact references:

- Immediate reward at every tree history: E[R | b_h, a], where b_h is the
  agent's exact posterior at private history h. The planner carries b_h down
  the tree with the same belief update used for real observations. b_h never
  receives the sampled hidden state or the opponent's actual action, so the
  integration adds no information the agent does not have.
- The last decision (H = 1): Q_1(b_h, a) = E[R | b_h, a], maximized after
  integrating, which is the exact finite-horizon boundary condition.

Sampling each step's state from b_h rather than carrying one root particle
down the tree has the same distribution when b_h is exact (given the history,
the carried particle is distributed as b_h). With bounded beliefs it is what
keeps the tree consistent: a carried particle can sit on an atom that
compression dropped from b_h, and under deterministic sensors its observation
then has zero probability under b_h. Drawing from b_h makes the sampled
observation distribution exactly P(o | b_h, a), the one the Bellman backup
integrates, so every tree observation has positive evidence.

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
from ipomdp.planning.chance_scoring import score_tree
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
        # Last real decision: search Q, and the rescored values it was taken on.
        self.search_values = {}
        self.recommendation_values = {}
        self.scores = None
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
        self.search_values = {}
        self.recommendation_values = {}
        self.scores = None

    # ------------------------------------------------------------------ #
    # Decisions
    # ------------------------------------------------------------------ #

    def get_action(self, belief=None):
        return sample_policy(self.policy_for(self.model(belief)), self.rng)

    def policy_for(self, model, modeled=False):
        """Greedy policy at ``model``'s belief; ``modeled`` selects the opponent budget."""
        if any(self.pomdp_model.is_terminal(atom.state) for atom, _ in model.belief.mass):
            raise ValueError("Condition on public continuation before requesting a policy")
        if modeled and self.config.modeled_stopping is not None:
            return greedy_policy(self._stopped_values(model))
        simulations = self.config.modeled_simulations if modeled else self.config.simulations
        node, scores = self._search(model, simulations)
        policy = greedy_policy(scores.values)
        if not modeled:
            self.root = node
            self.search_values = dict(node.action_values)
            self.recommendation_values = dict(scores.values)
            self.scores = scores
        return policy

    def _stopped_values(self, model):
        """Scored values of a modeled solve by sequential stopping (``stopping``).

        The main tree is seeded as a flat search of the cap, so a solve that
        never separates returns exactly that search's values. Pilot trees are
        grown only once the main tree reaches its first checkpoint.
        """
        rule, cap = self.config.modeled_stopping, self.config.modeled_simulations
        checkpoints = rule.checkpoints(cap)
        main = self._tree(model, cap)
        calibration = None
        for simulations in checkpoints:
            values = main.grow(simulations).values
            if calibration is None:
                calibration = [values] + [
                    self._tree(model, cap, pilot).grow(rule.pilot).values
                    for pilot in range(1, rule.trees)
                ]
            if simulations == cap or rule.separated(
                values, calibration, simulations, len(checkpoints)
            ):
                return values

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #

    def _search(self, model, simulations):
        """One seeded search of ``simulations`` from ``model``; returns (tree, scores)."""
        tree = self._tree(model, simulations)
        return tree.root, tree.grow(simulations)

    def _tree(self, model, simulations, replicate=None):
        """An empty search tree at ``model``, seeded as a search of ``simulations``.

        The seed keys on the configuration without ``modeled_stopping``, so a
        flat search keeps its stream whether or not a rule is configured, and a
        stopping rule's main tree is a prefix of the flat search of its cap.
        Pilot trees add their index as ``replicate``.
        """
        config = {k: v for k, v in asdict(self.config).items() if k != "modeled_stopping"}
        settings = {"planner": "I-POMCP", "config": config, "simulations": simulations}
        if replicate is not None:
            settings["replicate"] = replicate
        model = self.solver_bank.search_model(model)
        return _GrowingSearch(
            self, model, random.Random(self.solver_bank.search_seed(model, settings))
        )

    def _simulate(self, node, depth, memory, history_model, rng):
        """One descent; returns the sampled-continuation Q estimate at ``node``."""
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
        particle = self.solver_bank.draw(history_model.belief, rng)
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
                continuation = self._rollout(depth + 1, next_memory, next_model, rng)
                child.rollout_value = continuation
            else:
                continuation = self._simulate(child, depth + 1, next_memory, next_model, rng)
            counts = node.continuation_counts.setdefault(action, {})
            counts[observation] = counts.get(observation, 0) + 1
        immediate = dict(self.solver_bank.expected_rewards(history_model))[action]
        node.action_counts[action] = node.action_counts.get(action, 0) + 1
        q = node.bellman_backup(action, immediate, self.config.gamma)
        node.visit_count += 1
        return q

    def _rollout(self, depth, memory, history_model, rng):
        """Sampled return of the domain rollout policy, with the exact last step.

        Rollout rewards stay sampled: integrating them would require the exact
        posterior at every rollout history, which is the cost the tree already
        pays. Opponent private models still advance at every rollout step,
        because freezing them would change the opponent's dynamics.
        """
        if self.config.depth - depth == 1:
            return max(self._leaf_values(history_model).values())
        action = self.pomdp_model.rollout_action(self.key.agent_id, memory, rng)
        particle = self.solver_bank.draw(history_model.belief, rng)
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
        return reward + self.config.gamma * self._rollout(depth + 1, next_memory, next_model, rng)

    # ------------------------------------------------------------------ #
    # Exact pieces
    # ------------------------------------------------------------------ #

    def _history_successor(self, model, action, observation):
        """The agent's exact posterior after its own action and observation."""
        posterior = self.solver_bank.search_filter.update(
            model, action, observation, terminal=False
        )
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
            "search_values": {str(a): q for a, q in self.search_values.items()},
            "action_values": {str(a): q for a, q in self.recommendation_values.items()},
            "action_counts": {str(a): c for a, c in self.root.action_counts.items()},
            "scores_exact": self.scores.exact if self.scores else None,
            "fallback_actions": self.scores.fallback_actions if self.scores else None,
            "frontiers": self.scores.frontiers if self.scores else None,
        }


class _GrowingSearch:
    """One search tree that grows in stages and is chance-scored at each stage.

    Growing a tree from n to m simulations continues the same random stream,
    so the tree at m equals a search of m simulations with this seed.
    """

    def __init__(self, planner, model, rng):
        self.planner, self.model, self.rng = planner, model, rng
        self.root = POMCPNode()
        self.memory = planner._physical_marginal(model)
        self.simulations = 0

    def grow(self, simulations):
        """Run simulations until the tree has ``simulations``; return its chance scores."""
        if simulations < len(self.planner.actions):
            raise ValueError("Search budget must evaluate every action")
        if simulations < self.simulations:
            raise ValueError("A search tree cannot shrink")
        planner = self.planner
        for _ in range(simulations - self.simulations):
            planner._simulate(self.root, 0, self.memory, self.model, self.rng)
        self.simulations = simulations
        return score_tree(
            self.root,
            self.model,
            planner.config.depth,
            planner.solver_bank,
            planner._history_successor,
            planner.config.gamma,
        )
