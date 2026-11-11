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

import math
import random
import time
from dataclasses import asdict

from ipomdp.belief.finite_belief import MentalModel
from ipomdp.belief.frame import AgentFrame
from ipomdp.planning.chance_scoring import deviation_bound, exact_two_step, score_tree
from ipomdp.planning.config import NEUTRAL_DEFAULTS, SCORING_FIELDS, PlannerConfig
from ipomdp.planning.exploration import SELECTORS
from ipomdp.planning.generative_model import InteractiveGenerativeModel
from ipomdp.planning.node import CertifiedState, POMCPNode
from ipomdp.planning.planner import Planner
from ipomdp.planning.policy import (
    TIE_TOLERANCE,
    greedy_policy,
    plausible_policy,
    plausible_sets,
    sample_policy,
)


class IPOMCPPlanner(Planner):
    def __init__(self, solver_key, pomdp_model, solver_bank, config: PlannerConfig):
        self.key, self.pomdp_model, self.solver_bank = solver_key, pomdp_model, solver_bank
        self.config = config
        self.actions = list(pomdp_model.get_all_actions(solver_key.agent_id))
        self.exploration = SELECTORS[config.selection].for_agent(
            pomdp_model, solver_key.agent_id, config.exploration
        )
        self._frontier_bounds = None
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
            scores = self._stopped_scores(model)
            self._modeled_exact = scores.exact
            return _policy(scores)
        if modeled and self.config.modeled_resolution is not None:
            return self.modeled_solution(model)[0]
        if not modeled and self.config.root_stopping is not None:
            node, scores = self._stopped_search(
                model, self.config.root_stopping, self.config.simulations, root=True
            )
        else:
            simulations = self.config.modeled_simulations if modeled else self.config.simulations
            node, scores = self._search(model, simulations, bounds=not modeled)
        policy = _policy(scores)
        if modeled:
            self._modeled_exact = scores.exact
        if not modeled:
            self.root = node
            self.search_values = dict(node.action_values)
            self.recommendation_values = dict(scores.values)
            self.scores = scores
        return policy

    def modeled_solution(self, model):
        """(policy, resolved, delta_hat) of this planner as a modeled agent at ``model`` (Theorem 1b).

        Without ``modeled_resolution``: today's modeled policy, resolved only
        when its scores are exact (then it is the exact law). With it: one tree
        grown until its widened intervals resolve (``_resolved_search``); the
        prediction is uniform over the plausible actions (``"plausible"``) or
        the greedy policy (``"argmax"``, resolved only if it equals the
        plausible law).
        """
        rule = self.config.modeled_resolution
        if rule is None:
            policy = self.policy_for(model, modeled=True)
            exact = self._modeled_exact
            return policy, exact, 0.0 if exact else 1.0
        if any(self.pomdp_model.is_terminal(atom.state) for atom, _ in model.belief.mass):
            raise ValueError("Condition on public continuation before requesting a policy")
        scores, plausible, resolved, delta = self._resolved_search(model, rule)
        if self.config.modeled_prediction == "plausible":
            return plausible, resolved, delta
        policy = _policy(scores)
        exact_law = resolved and set(policy) == set(plausible)
        return policy, exact_law, 0.0 if exact_law else 1.0

    def _resolved_search(self, model, rule):
        """(scores, plausible law, resolved, delta_hat) of one tree grown by ``rule`` (``stopping.ModeledResolution``).

        The tree is seeded as the flat search of the cap, so a solve that
        never resolves returns that search's scores. Intervals are widened by
        the deviation bound before the sets are formed.
        """
        cap, config = self.config.modeled_simulations, self.config
        tree = self._tree(model, cap)
        width = self.exploration.return_width(config.depth, config.gamma)
        for simulations in rule.checkpoints(cap):
            scores = tree.grow(simulations, bounds=True)
            deviation = deviation_bound(
                tree.root, tree.model, config.depth, self.solver_bank, self._history_successor,
                config.gamma,
            )
            widened = {
                a: (scores.bounds[a][0] - deviation[a], scores.bounds[a][1] + deviation[a])
                for a in self.actions
            }
            plausible, resolved, delta = plausible_policy(widened, rule.tie)
            m_lo = max(lo for lo, _ in widened.values())
            m_hi = max(hi for _, hi in widened.values())
            narrow = rule.rel_width > 0 and (m_hi - m_lo) / width <= rule.rel_width
            if resolved or narrow or simulations == cap:
                self.modeled_simulations_run = tree.simulations_run
                return scores, plausible, resolved, delta

    def anytime_scores(self, model, checkpoints):
        """Certified chance scores of the root search at each checkpoint, one tree grown through them.

        The tree is seeded as the flat search of ``config.simulations``, so the
        stage at that budget equals ``policy_for``'s scores; earlier stages are
        its prefixes. Returns [(simulations, scores, cpu_seconds, work)], CPU
        cumulative; ``work`` counts the simulations actually run (certified
        search stops once the root is resolved), the root actions pruned, and
        the belief updates and nested policy solves computed since the search
        started (cache misses).
        """
        if max(checkpoints) != self.config.simulations:
            raise ValueError("The last checkpoint must be the configured budget")
        bank = self.solver_bank
        updates0 = bank.search_filter.advance.cache_info().misses
        solves0 = bank._cached_policy.cache_info().misses
        started, tree, stages = time.process_time(), self._tree(model, self.config.simulations), []
        for simulations in sorted(checkpoints):
            scores = tree.grow(simulations, bounds=True)
            cpu = time.process_time() - started
            cert = tree.root.cert
            work = dict(
                sims_run=tree.simulations_run,
                pruned_root=sorted(map(str, cert.pruned)) if cert is not None else [],
                updates=bank.search_filter.advance.cache_info().misses - updates0,
                nested_solves=bank._cached_policy.cache_info().misses - solves0,
            )
            stages.append((simulations, scores, cpu, work))
        return stages

    def _stopped_values(self, model):
        return self._stopped_scores(model).values

    def _stopped_scores(self, model):
        """Chance scores of a modeled solve by sequential stopping (``stopping``)."""
        rule, cap = self.config.modeled_stopping, self.config.modeled_simulations
        return self._stopped_search(model, rule, cap)[1]

    def _stopped_search(self, model, rule, cap, root=False):
        """(root, chance scores) of one tree grown by ``rule`` up to ``cap`` simulations.

        The main tree is seeded as a flat search of the cap, so a solve that
        never separates returns exactly that search's values. Pilot trees are
        grown only once the main tree reaches its first checkpoint. For the
        agent's own decision (``root``) the scores carry certified bounds, and
        exact scores stop the growth at once, since more simulations cannot
        change them (Proposition 1); modeled solves keep their validated rule.
        """
        checkpoints = rule.checkpoints(cap)
        main = self._tree(model, cap)
        calibration = None
        for simulations in checkpoints:
            scores = main.grow(simulations, root)
            values = scores.values
            if root and scores.exact:
                return main.root, scores
            if calibration is None:
                calibration = [values] + [
                    self._tree(model, cap, pilot).grow(rule.pilot).values
                    for pilot in range(1, rule.trees)
                ]
            if simulations == cap or rule.separated(
                values, calibration, simulations, len(checkpoints)
            ):
                return main.root, scores

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #

    def _search(self, model, simulations, bounds=False):
        """One seeded search of ``simulations`` from ``model``; returns (tree, scores).

        ``bounds`` adds certified intervals to the scores (``chance_scoring``);
        real decisions request them, modeled solves do not need them.
        """
        tree = self._tree(model, simulations)
        return tree.root, tree.grow(simulations, bounds)

    def _tree(self, model, simulations, replicate=None):
        """An empty search tree at ``model``, seeded as a search of ``simulations``.

        The seed keys on the configuration without the stopping rules
        (``modeled_stopping``, ``root_stopping``), so a flat search keeps its
        stream whether or not a rule is configured (and every result from before
        a rule existed is reproduced), and a stopping rule's main tree is a
        prefix of the flat search of its cap. Pilot trees add their index as
        ``replicate``.
        """
        config = seed_config(self.config)
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
    # Certified search (audit spec 3.3; off unless a certified field is set)
    # ------------------------------------------------------------------ #

    def _bounds_for(self, model):
        """This agent's shared ``FrontierBounds`` (one table per agent, physics and horizon)."""
        if self._frontier_bounds is None:
            self._frontier_bounds = self.solver_bank.frontier_bounds(
                model, self.config.gamma, self.config.depth, self.config.frontier_bounds
            )
        return self._frontier_bounds

    def _simulate_certified(self, node, depth, memory, history_model, rng):
        """One descent of certified search; returns the backed-up Q estimate at ``node``.

        Every history keeps bounds on its value from its own belief (expanded)
        or from its parent's (frontier). Frontiers are valued by a point of
        their bound interval (``leaf_eval="bound"``) or by a rollout clipped
        into it; histories with two steps to go are solved exactly
        (``exact_tail``); bounds are recomputed bottom-up along the descent and
        only tighten; actions whose upper bound falls below the history's
        lower bound are never selected again (``prune``).
        """
        config = self.config
        remaining = config.depth - depth
        if node.cert is None or not node.cert.expanded:
            self._expand(node, history_model, remaining)
        cert = node.cert
        # Below the root only the value matters, so a solved history stops. At the root the decision
        # does: an action tied at the upper bound with a loose lower bound keeps it unresolved although
        # the value is exact, so search continues until the plausible and certain sets agree.
        if cert.solved and (depth > 0 or self._resolved(node)):
            node.visit_count += 1
            return cert.lo
        live = [action for action in self.actions if action not in cert.pruned]
        action = self.exploration.select_action(node, live, remaining, config.gamma, rng)
        particle = self.solver_bank.draw(history_model.belief, rng)
        following, joint, _, terminal = self.gen_model.tree_step(
            particle, action, self.key.agent_id, self.pomdp_model, rng, advance_opponent=False
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
            child = node.get_child(action, observation)
            if child is None:
                child = node.create_child(action, observation)
                probability = cert.law[action][observation]
                child.cert = CertifiedState(
                    cert.branch_lo[action][observation] / probability,
                    cert.branch_hi[action][observation] / probability,
                )
                if config.leaf_eval == "bound":
                    if not config.lazy_expansion:
                        next_model = self._history_successor(history_model, action, observation)
                        self._expand(child, next_model, remaining - 1)
                    continuation = _pick(child.cert.lo, child.cert.hi, config.unsampled_value)
                else:
                    next_model = self._history_successor(history_model, action, observation)
                    continuation = self._rollout(depth + 1, next_memory, next_model, rng)
                    continuation = min(max(continuation, child.cert.lo), child.cert.hi)
                child.cert.estimate = continuation
                child.rollout_value = continuation
            else:
                next_model = self._history_successor(history_model, action, observation)
                continuation = self._simulate_certified(
                    child, depth + 1, next_memory, next_model, rng
                )
            counts = node.continuation_counts.setdefault(action, {})
            counts[observation] = counts.get(observation, 0) + 1
        immediate = dict(self.solver_bank.expected_rewards(history_model))[action]
        node.action_counts[action] = node.action_counts.get(action, 0) + 1
        node.bellman_backup(action, immediate, config.gamma)
        node.visit_count += 1
        self._refresh(node, history_model, action)
        return node.action_values[action]

    def _expand(self, node, model, remaining):
        """Give ``node`` its own bounds (and, with two steps to go and ``exact_tail``, its exact value)."""
        bank, config = self.solver_bank, self.config
        nb = self._bounds_for(model).node_bounds(model, remaining, bank)
        cert = node.cert
        if cert is None:
            cert = node.cert = CertifiedState(None, None)
        cert.expanded = True
        cert.q_lo, cert.q_hi = dict(nb.q_lo), dict(nb.q_hi)
        cert.branch_lo, cert.branch_hi = nb.branch_lo, nb.branch_hi
        if remaining == 1:
            # Boundary history: exact, as on the legacy path.
            if not node.action_values:
                node.action_values.update(self._leaf_values(model))
            value = max(node.action_values.values())
            cert.lo = cert.hi = value
            cert.solved = True
            return
        cert.law = {action: dict(bank.continuation_law(model, action)) for action in self.actions}
        if config.prune:
            self._prune(cert)
        if remaining == 2 and config.exact_tail:
            self._exact_tail(node, model)
            return
        self._close(node)
        if config.selection != "ucb":
            width = self.exploration.return_width(remaining, config.gamma)
            live = [a for a in self.actions if a not in cert.pruned]
            top = max(cert.q_hi[a] for a in live)
            weights = {
                a: math.exp((cert.q_hi[a] - top) / (config.puct_temperature * width)) for a in live
            }
            total = math.fsum(weights.values())
            cert.prior = {a: w / total for a, w in weights.items()}

    def _exact_tail(self, node, model):
        """Solve a two-steps-to-go history: exact Q^2 for every action not ruled out by its bounds."""
        cert, gamma = node.cert, self.config.gamma
        for action in self.actions:
            if action in cert.pruned:
                continue
            value = exact_two_step(
                model, action, self.solver_bank, self._history_successor, gamma
            )
            cert.q_lo[action] = cert.q_hi[action] = value
            node.action_values[action] = value
        cert.lo = cert.hi = max(cert.q_lo[a] for a in self.actions if a not in cert.pruned)
        cert.solved = True

    def _refresh(self, node, model, action):
        """Recompute Q bounds of ``action`` from its children; keep the tighter; prune; clip."""
        cert, gamma = node.cert, self.config.gamma
        reward = dict(self.solver_bank.expected_rewards(model))[action]
        lows, highs = [], []
        for observation, probability in cert.law[action].items():
            child = node.get_child(action, observation)
            if child is not None and child.cert is not None:
                lows.append(probability * child.cert.lo)
                highs.append(probability * child.cert.hi)
            else:
                lows.append(cert.branch_lo[action][observation])
                highs.append(cert.branch_hi[action][observation])
        cert.q_lo[action] = max(cert.q_lo[action], reward + gamma * math.fsum(lows))
        cert.q_hi[action] = min(cert.q_hi[action], reward + gamma * math.fsum(highs))
        if self.config.prune:
            self._prune(cert)
        self._close(node)
        for evaluated, value in node.action_values.items():
            node.action_values[evaluated] = min(
                max(value, cert.q_lo[evaluated]), cert.q_hi[evaluated]
            )

    def _prune(self, cert):
        best = max(cert.q_lo.values())
        for action, upper in cert.q_hi.items():
            if upper < best - _slack(best):
                cert.pruned.add(action)

    @staticmethod
    def _close(node):
        """Node bounds from action bounds (intersected with the inherited ones); mark solved.

        A solved history may never have been searched (its bounds can close at
        expansion, e.g. an action that ends the episode with the largest
        possible reward), so every action whose own bounds have closed records
        that exact value; the best action's always have, so scoring and the
        parent's backup always see the history's value.
        """
        cert = node.cert
        lo, hi = max(cert.q_lo.values()), max(cert.q_hi.values())
        cert.lo = lo if cert.lo is None else max(cert.lo, lo)
        cert.hi = hi if cert.hi is None else min(cert.hi, hi)
        if cert.hi - cert.lo <= _slack(cert.hi):
            cert.solved = True
            for action, upper in cert.q_hi.items():
                if action not in node.action_values and upper - cert.q_lo[action] <= _slack(upper):
                    node.action_values[action] = cert.q_lo[action]

    def _resolved(self, node, tie=TIE_TOLERANCE):
        """True when bounds settle which root actions are optimal (within ``tie``)."""
        cert = node.cert
        if cert is None or not cert.expanded:
            return False
        if cert.solved and all(cert.q_lo[a] == cert.q_hi[a] for a in self.actions if a not in cert.pruned):
            return True
        # The same sets as a modeled solve's resolution (policy.plausible_sets), so search never
        # stops while the solve's own rule still calls the decision unresolved.
        certain, plausible = plausible_sets({a: (cert.q_lo[a], cert.q_hi[a]) for a in self.actions}, tie)
        return certain == plausible

    # ------------------------------------------------------------------ #
    # Exact pieces
    # ------------------------------------------------------------------ #

    def _history_successor(self, model, action, observation):
        """The agent's exact posterior after its own action and observation."""
        return self.solver_bank.search_filter.advance(model, action, observation, terminal=False)

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
            "bounds": ({str(a): b for a, b in self.scores.bounds.items()} if self.scores else None),
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
        self.simulations_run = 0

    def grow(self, simulations, bounds=False):
        """Run simulations until the tree has ``simulations``; return its chance scores."""
        if simulations < len(self.planner.actions):
            raise ValueError("Search budget must evaluate every action")
        if simulations < self.simulations:
            raise ValueError("A search tree cannot shrink")
        planner = self.planner
        config = planner.config
        if config.certified_search:
            for _ in range(simulations - self.simulations):
                if config.prune and planner._resolved(self.root):
                    break
                planner._simulate_certified(self.root, 0, self.memory, self.model, self.rng)
                self.simulations_run += 1
        else:
            for _ in range(simulations - self.simulations):
                planner._simulate(self.root, 0, self.memory, self.model, self.rng)
            self.simulations_run = simulations
        self.simulations = simulations
        # Probability-weighted scoring clips into certified intervals, so it always needs them.
        bounds = bounds or config.scoring == "pw"
        frontier = None
        if bounds and config.frontier_bounds != "scalar":
            frontier = planner.solver_bank.frontier_bounds(
                self.model, config.gamma, config.depth, config.frontier_bounds
            )
        return score_tree(
            self.root,
            self.model,
            config.depth,
            planner.solver_bank,
            planner._history_successor,
            config.gamma,
            bounds,
            frontier=frontier,
            scoring=config.scoring,
            unsampled=config.unsampled_value,
        )


_STOPPING_FIELDS = ("modeled_stopping", "root_stopping")


def _slack(x):
    """Outward rounding slack of every comparison that prunes or certifies (audit spec 3.1)."""
    return 1e-9 * max(1.0, abs(x))


def _pick(lo, hi, which):
    return {"lower": lo, "upper": hi, "mid": 0.5 * (lo + hi)}[which]


def seed_config(config):
    """The configuration as it enters the search seed.

    Drops the stopping rules, the scoring-only fields (``config.SCORING_FIELDS``)
    and every later field that holds its neutral default
    (``config.NEUTRAL_DEFAULTS``), so a default configuration hashes as it did
    before those fields existed and scoring variants search the same tree.
    """
    return {
        k: v
        for k, v in asdict(config).items()
        if k not in _STOPPING_FIELDS
        and k not in SCORING_FIELDS
        and not (k in NEUTRAL_DEFAULTS and v == NEUTRAL_DEFAULTS[k])
    }


def _policy(scores):
    """Greedy over scored values; exactly scored values tie within ``TIE_TOLERANCE``, as the exact reference's do."""
    return greedy_policy(scores.values, TIE_TOLERANCE if scores.exact else 0.0)
