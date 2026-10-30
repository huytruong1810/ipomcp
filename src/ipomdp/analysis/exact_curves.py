"""Exact expected reward curves of a pair of exact agents (zero variance).

The joint process (physical state, i's belief, j's belief) is enumerated
forward, merging equal nodes. Each agent plays its exact receding-horizon
policy (``reference.exact_agent``: uniform over exact maximizers; level 0
uniform over actions) and updates its belief with the exact filter, so the
curve is the expectation of exactly the pair that exact episodes play
(``study.ExactConfig``). Terminal successors absorb: they earn nothing after.

Nodes grow about 10x per step on Tiger (1.5M at step 6 for L1 against L1,
HANDOFF 0.00000), so the curve covers a short window; ``max_nodes`` bounds it.

``exact_decision_gains`` compares two levels at the same decision points instead:
it enumerates one agent's histories against the true opponent and scores both
levels' receding plans there, so the receding-horizon timing of whole episodes
does not enter (V3, PLAN.md).
"""

import math
import time
from collections import defaultdict

from ipomdp.analysis.hierarchy import level_posterior
from ipomdp.belief.finite_belief import FiniteBelief, MentalModel
from ipomdp.belief.finite_filter import InferenceBudgetExceeded, UnsupportedObservation
from ipomdp.planning.policy import TIE_TOLERANCE, greedy_policy
from ipomdp.reference.exact_agent import build_exact_agent

AGENTS = ("i", "j")


def uniform_nesting(level):
    """Uniform priors over lower levels at every nesting, as the level grid holds them."""
    return {n: {k: 1 / n for k in range(n)} for n in range(1, level + 1)}


def exact_pair_curve(physics, levels, steps, horizon, gamma=0.95, nested=None, max_nodes=2_000_000):
    """Yield, per step, both agents' exact expected cumulative rewards.

    Args:
        levels: {"i": level, "j": level}.
        nested: {agent: {level: {opponent level: weight}}}; uniform by default.

    Yields:
        {"step", "cum_reward_i", "cum_reward_j", "nodes", "cpu"} after each step;
        raises ``InferenceBudgetExceeded`` when a step's nodes exceed ``max_nodes``.
    """
    nested = nested or {}
    agents = {
        a: build_exact_agent(
            a,
            "j" if a == "i" else "i",
            levels[a],
            physics,
            nested.get(a, uniform_nesting(levels[a])),
            horizon,
            gamma,
        )
        for a in AGENTS
        if levels[a] > 0
    }
    uniform = {
        a: {x: 1 / len(physics.get_all_actions(a)) for x in physics.get_all_actions(a)}
        for a in AGENTS
    }

    def policy(agent, belief):
        if agent not in agents:
            return uniform[agent]
        planner = agents[agent]
        return planner.policy_for(planner.model(belief))

    def update(agent, belief, action, observation):
        if agent not in agents:
            return None
        planner = agents[agent]
        return planner.solver_bank.filter.update(
            planner.model(belief), action, observation, terminal=False
        ).belief

    start = {a: agents[a].belief if a in agents else None for a in AGENTS}
    nodes = {(s, start["i"], start["j"]): p for s, p in physics.initial_distribution()}
    cumulative = dict.fromkeys(AGENTS, 0.0)
    for step in range(1, steps + 1):
        clock = time.process_time()
        following = defaultdict(float)
        earned = {a: [] for a in AGENTS}  # one subtotal per node, summed exactly per step
        for (state, belief_i, belief_j), mass in nodes.items():
            here = {a: [] for a in AGENTS}
            for action_i, p_i in policy("i", belief_i).items():
                for action_j, p_j in policy("j", belief_j).items():
                    joint = {"i": action_i, "j": action_j}
                    for successor, p_t in physics.transition_distribution(state, joint):
                        weight = mass * p_i * p_j * p_t
                        for a in AGENTS:
                            here[a].append(weight * physics.get_reward(state, joint, successor, a))
                        if physics.is_terminal(successor):
                            continue
                        for o_i, q_i in physics.observation_distribution(successor, joint, "i"):
                            next_i = update("i", belief_i, action_i, o_i)
                            for o_j, q_j in physics.observation_distribution(successor, joint, "j"):
                                key = (successor, next_i, update("j", belief_j, action_j, o_j))
                                following[key] += weight * q_i * q_j
                                if len(following) > max_nodes:
                                    raise InferenceBudgetExceeded(
                                        f"Exact pair curve: over {max_nodes} nodes at step {step}"
                                    )
            for a in AGENTS:
                earned[a].append(math.fsum(here[a]))
        for a in AGENTS:
            cumulative[a] += math.fsum(earned[a])
        nodes = dict(following)
        yield {
            "step": step,
            "cum_reward_i": cumulative["i"],
            "cum_reward_j": cumulative["j"],
            "nodes": len(nodes),
            "cpu": time.process_time() - clock,
        }


def _restricted(model, level):
    """The model's belief conditioned on the opponent being at ``level``."""
    rows = tuple((atom, w) for atom, w in model.belief.mass if atom.opponent.frame.level == level)
    return MentalModel(model.frame, FiniteBelief(rows))


class _Stepper:
    """One exact agent of the stepping role (or uniform L0), its policy and filter by horizon."""

    def __init__(self, planner, actions):
        self.planner, self.actions = planner, actions

    def policy(self, belief, horizon):
        if self.planner is None:
            return {a: 1 / len(self.actions) for a in self.actions}
        q = self.planner.reference.q_values(self.planner.model(belief), horizon)
        return greedy_policy(q, TIE_TOLERANCE)

    def update(self, belief, action, observation):
        if self.planner is None:
            return None
        return self.planner.solver_bank.filter.update(
            self.planner.model(belief), action, observation, terminal=False
        ).belief


def _plan_value(truth, reference, stepper, belief, horizon, gamma):
    """True expected ``horizon``-step reward of the stepper's receding plan from this history.

    ``truth`` is the exact belief over (state, true opponent model) given the history, evaluated
    by ``reference``; the stepper follows its own greedy h-step policy at each remaining step h.
    """
    if horizon == 0:
        return 0.0
    terms = []
    for action, p_a in stepper.policy(belief, horizon).items():
        reward, branches = reference.branches(truth, action)
        future = [
            p_o * _plan_value(
                successor, reference, stepper, stepper.update(belief, action, o), horizon - 1, gamma
            )
            for o, p_o, successor in branches
        ] if horizon > 1 else []
        terms.append(p_a * (reward + gamma * math.fsum(future)))
    return math.fsum(terms)


def exact_decision_gains(physics, agent, k, m, steps, horizon, reference_level, gamma=0.95, max_nodes=2_000_000):
    """Yield, per step, the exact value of one more level at the decision points of that step.

    The stepping ``agent`` is at level k or k + 1 (uniform nesting) and faces an opponent at
    true level m (uniform nesting). Histories are generated by the stepping agent at
    ``reference_level`` (k or k + 1) against the true opponent, so both levels are compared at
    the same history; no receding-horizon episode value enters. At every history h:

    - plan gain: V_m(h, plan_{k+1}) - V_m(h, plan_k), the true expected ``horizon``-step reward
      of each level's receding plan (first action, then its greedy shorter-horizon responses);
    - immediate gain: the same for the first step's reward only;
    - Bayes residual: sum over the support levels m' of P_{k+1}(m' | h) times the plan gain
      against m', which is >= 0 when level k + 1 models each level in its support correctly
      and filters exactly; ``bayes_gap`` checks that level k + 1's mixture value equals its
      own maximal Q.

    Yields:
        {"step", "mass", "reward_ref", "immediate_gain", "plan_gain", "residual",
         "residual_min", "bayes_gap", "differ_mass", "unsupported", "nodes", "cpu"}; mass-weighted
        sums over the step's histories (``mass`` is the probability the episode is still running).
    """
    if reference_level not in (k, k + 1):
        raise ValueError("The reference level must be k or k + 1")
    other = "j" if agent == "i" else "i"
    actions = physics.get_all_actions(agent)

    def exact(level, nested):
        return build_exact_agent(agent, other, level, physics, nested, horizon, gamma)

    lower = _Stepper(exact(k, uniform_nesting(k)) if k else None, actions)
    upper = _Stepper(exact(k + 1, uniform_nesting(k + 1)), actions)
    up = upper.planner
    if m <= k:
        oracle = None
        reference = up.reference
    else:
        oracle = exact(m + 1, {**uniform_nesting(m + 1), m + 1: {m: 1.0}})
        reference = oracle.reference
    driver = lower if reference_level == k else upper

    def truth_of(node):
        b_lower, b_upper, b_oracle = node
        return _restricted(up.model(b_upper), m) if oracle is None else oracle.model(b_oracle)

    nodes = {(lower.planner.belief if k else None, up.belief, oracle.belief if oracle else None): 1.0}
    for step in range(1, steps + 1):
        clock = time.process_time()
        sums = defaultdict(list)
        residual_min, gap_max, unsupported = math.inf, 0.0, 0
        following = defaultdict(float)
        for node, mass in nodes.items():
            b_lower, b_upper, b_oracle = node
            truth = truth_of(node)
            pi_lo, pi_up = lower.policy(b_lower, horizon), upper.policy(b_upper, horizon)
            immediate = reference.q_values(truth, 1)
            sums["immediate_gain"].append(mass * (
                math.fsum(p * immediate[a] for a, p in pi_up.items())
                - math.fsum(p * immediate[a] for a, p in pi_lo.items())))
            sums["plan_gain"].append(mass * (
                _plan_value(truth, reference, upper, b_upper, horizon, gamma)
                - _plan_value(truth, reference, lower, b_lower, horizon, gamma)))
            if set(pi_up) != set(pi_lo) or any(abs(pi_up[a] - pi_lo[a]) > 1e-12 for a in pi_up):
                sums["differ_mass"].append(mass)
            # Bayes residual over level k + 1's own support, evaluated by its own reference.
            model = up.model(b_upper)
            terms, mixture = [], []
            try:
                for level, weight in level_posterior(b_upper).items():
                    part = _restricted(model, level)
                    v_up = _plan_value(part, up.reference, upper, b_upper, horizon, gamma)
                    v_lo = _plan_value(part, up.reference, lower, b_lower, horizon, gamma)
                    terms.append(weight * (v_up - v_lo))
                    mixture.append(weight * v_up)
                residual = math.fsum(terms)
                sums["residual"].append(mass * residual)
                residual_min = min(residual_min, residual)
                gap_max = max(gap_max, abs(math.fsum(mixture) - max(up.reference.q_values(model, horizon).values())))
            except UnsupportedObservation:
                unsupported += 1
            pi_ref = driver.policy(b_lower if driver is lower else b_upper, horizon)
            sums["reward_ref"].append(mass * math.fsum(p * immediate[a] for a, p in pi_ref.items()))
            for action, p_a in pi_ref.items():
                _, branches = reference.branches(truth, action)
                for o, p_o, successor in branches:
                    key = (
                        lower.update(b_lower, action, o),
                        upper.update(b_upper, action, o),
                        successor.belief if oracle else None,
                    )
                    following[key] += mass * p_a * p_o
                    if len(following) > max_nodes:
                        raise InferenceBudgetExceeded(
                            f"Exact decision gains: over {max_nodes} nodes at step {step}"
                        )
        row = {"step": step, "mass": math.fsum(nodes.values())}
        for name in ("reward_ref", "immediate_gain", "plan_gain", "residual", "differ_mass"):
            row[name] = math.fsum(sums[name])
        row.update(residual_min=residual_min, bayes_gap=gap_max, unsupported=unsupported,
                   nodes=len(nodes), cpu=time.process_time() - clock)
        yield row
        nodes = dict(following)
        if not nodes:
            return
