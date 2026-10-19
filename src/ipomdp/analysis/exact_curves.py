"""Exact expected reward curves of a pair of exact agents (zero variance).

The joint process (physical state, i's belief, j's belief) is enumerated
forward, merging equal nodes. Each agent plays its exact receding-horizon
policy (``reference.exact_agent``: uniform over exact maximizers; level 0
uniform over actions) and updates its belief with the exact filter, so the
curve is the expectation of exactly the pair that exact episodes play
(``study.ExactConfig``). Terminal successors absorb: they earn nothing after.

Nodes grow about 10x per step on Tiger (1.5M at step 6 for L1 against L1,
HANDOFF 0.00000), so the curve covers a short window; ``max_nodes`` bounds it.
"""

import math
import time
from collections import defaultdict

from ipomdp.belief.finite_filter import InferenceBudgetExceeded
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
