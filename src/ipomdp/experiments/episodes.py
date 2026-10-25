"""One episode of a study, recorded identically for both agents.

Each trial has its own environment streams (initial state and transitions, and
each agent's sensor) and its own planner seed, all derived from the trial id,
so equal trial ids give equal streams across conditions. Streams stay aligned
only while two conditions' actions agree, so comparisons pair by trial.

Every row is one step; step 0 is the prior. Per agent (suffix ``_i``/``_j``):

- ``action``, ``observation``, ``reward``, ``cum_reward``;
- ``planning_cpu``/``planning_wall`` and ``update_cpu``/``update_wall``: process
  CPU and wall seconds for the decision and for the belief update (each trial
  runs in its own single-threaded process, so CPU time is this agent's work);
- ``belief_support``: distinct interactive hypotheses held;
- ``prob_l{k}_{opponent}``: the agent's posterior over the opponent's level;
- ``physical``: the agent's physical-state marginal (JSON);
- ``action_values``: the root action values of its last decision (JSON);
- ``forced_share`` (I-NTMCP only): the share of force-filled particles in the
  belief the decision searched from;
- ``escalated`` and ``base_gap`` (escalating agents only): whether the shadow
  at the larger bound decided, and the base planner's top-two gap;
- ``hierarchy`` (JSON, when the study records hierarchies).

After the episode ends, rows are padded to the declared length with zero
rewards and carried cumulative rewards, so every trial contributes a full
panel and early termination is not silently averaged away.

Exact agents play only inside their exact window (``study.ExactConfig``):
from the first step past the declared ``max_steps``, or the first step whose
decision or belief update exceeds the enumeration budget, every row has
status ``"Window"`` and NaN rewards, so an unplayed tail can never be read as
reward.
"""

import json
import math
import random
import time

from ipomdp.analysis.hierarchy import level_posterior, nested_hierarchy
from ipomdp.belief.finite_filter import InferenceBudgetExceeded
from ipomdp.experiments.study import AGENTS, build_agents, exact_window
from ipomdp.infra.resources import process_rss_mb


def planner_seed(trial_id: int) -> int:
    """The planners' seed for a trial: fixed, distinct per trial, independent of the environment."""
    return trial_id * 1_000_003 + 17


def run_episode(spec, trial_id):
    env_transition = random.Random(f"{trial_id}/environment/transition")
    env_observation = {a: random.Random(f"{trial_id}/environment/observation/{a}") for a in AGENTS}
    physics, planners = build_agents(spec, planner_seed(trial_id))
    state = physics.sample_initial_state(env_transition)
    cumulative = dict.fromkeys(AGENTS, 0.0)
    rows = [_row(trial_id, 0, state, "Active", planners, spec, cumulative)]
    terminal = physics.is_terminal(state)
    window = exact_window(spec)
    for step in range(1, spec.steps + 1):
        if terminal:
            rows.append(_padding(rows[-1], step))
            continue
        if window is not None and step > window:
            rows += [_window(trial_id, later) for later in range(step, spec.steps + 1)]
            break
        try:
            row, state, terminal = _step(
                trial_id,
                step,
                state,
                physics,
                planners,
                spec,
                cumulative,
                env_transition,
                env_observation,
            )
        except InferenceBudgetExceeded:
            rows += [_window(trial_id, later) for later in range(step, spec.steps + 1)]
            break
        rows.append(row)
    return rows


def _step(
    trial_id, step, state, physics, planners, spec, cumulative, env_transition, env_observation
):
    """Play one joint step: (row, next state, whether it is terminal)."""
    timings, joint, decisions = {}, {}, {}
    for agent in AGENTS:
        joint[agent] = _timed(planners[agent].get_action, timings, "planning", agent)
        # Read before update_root, which clears the decision's statistics.
        stats = planners[agent].get_detailed_stats()
        decisions[f"action_values_{agent}"] = json.dumps(
            stats.get("action_values", {}), sort_keys=True
        )
        decisions[f"bounds_{agent}"] = json.dumps(stats.get("bounds"), sort_keys=True)
        if "forced_share" in stats:
            decisions[f"forced_share_{agent}"] = stats["forced_share"]
        if "escalated" in stats:
            decisions[f"escalated_{agent}"] = stats["escalated"]
            decisions[f"base_gap_{agent}"] = stats["base_gap"]
        decisions[f"prediction_{agent}"] = json.dumps(
            predicted_opponent_action(planners[agent]), sort_keys=True
        )
    following = physics.sample_transition(state, joint, env_transition)
    rewards = {a: physics.get_reward(state, joint, following, a) for a in AGENTS}
    for agent in AGENTS:
        cumulative[agent] += rewards[agent]
    terminal = physics.is_terminal(following)
    observations = dict.fromkeys(AGENTS)
    for agent in AGENTS:
        if terminal:
            timings[f"update_cpu_{agent}"] = timings[f"update_wall_{agent}"] = 0.0
            continue
        observations[agent] = physics.sample_observation(
            following, joint, agent, env_observation[agent]
        )
        update = planners[agent].update_root
        _timed(lambda: update(joint[agent], observations[agent]), timings, "update", agent)
    row = _row(
        trial_id,
        step,
        following,
        "Terminal" if terminal else "Active",
        planners,
        spec,
        cumulative,
    )
    for agent in AGENTS:
        row[f"action_{agent}"] = str(joint[agent])
        row[f"observation_{agent}"] = (
            None if observations[agent] is None else str(observations[agent])
        )
        row[f"reward_{agent}"] = rewards[agent]
    row.update(timings)
    row.update(decisions)
    return row, following, terminal


def predicted_opponent_action(planner):
    """The agent's predicted distribution of the opponent's next action, or None at level 0.

    sum over (s, m_j) of b(s, m_j) pi_j(a_j | m_j), with the same modeled policies
    the agent plans against. Compared with the opponent's actual action along the
    realized path, it measures whether predictions merge with play, the
    subjective-equilibrium criterion (Kalai & Lehrer 1993; Doshi & Gmytrasiewicz
    2006); see docs/EQUILIBRIA.md.
    """
    belief = getattr(planner, "belief", None)
    if belief is None:
        return None
    predicted = {}
    for atom, mass in belief.mass:
        for action, probability in planner.solver_bank.filter.action_distribution(atom.opponent):
            predicted[str(action)] = predicted.get(str(action), 0.0) + mass * probability
    return predicted


def _row(trial_id, step, state, status, planners, spec, cumulative):
    row = {
        "trial": trial_id,
        "step": step,
        "status": status,
        "state": str(state),
        "process_rss_mb": round(process_rss_mb(), 2),
    }
    for agent in AGENTS:
        other = "j" if agent == "i" else "i"
        planner = planners[agent]
        belief = getattr(planner, "belief", None)
        row.update(
            {
                f"action_{agent}": None,
                f"observation_{agent}": None,
                f"reward_{agent}": 0.0,
                f"cum_reward_{agent}": cumulative[agent],
                f"planning_cpu_{agent}": 0.0,
                f"planning_wall_{agent}": 0.0,
                f"update_cpu_{agent}": 0.0,
                f"update_wall_{agent}": 0.0,
                f"belief_support_{agent}": len(belief.mass) if belief is not None else 0,
            }
        )
        if belief is not None:
            posterior = level_posterior(belief)
            row.update(
                {f"prob_l{k}_{other}": posterior.get(k, 0.0) for k in range(planner.key.level)}
            )
            physical = {}
            for atom, weight in belief.mass:
                physical[str(atom.state)] = physical.get(str(atom.state), 0.0) + weight
            row[f"physical_{agent}"] = json.dumps(physical, sort_keys=True)
            # The decision taken at this step, recorded by run_episode (step 0: none).
            row[f"action_values_{agent}"] = json.dumps({})
            row[f"bounds_{agent}"] = json.dumps(None)
            row[f"prediction_{agent}"] = json.dumps(None)
            if spec.record_hierarchies:
                nodes = nested_hierarchy(belief, agent, planner.key.level)
                row[f"hierarchy_{agent}"] = json.dumps([node.to_dict() for node in nodes])
    return row


def _padding(previous, step):
    row = {
        key: value
        for key, value in previous.items()
        if key.startswith(("trial", "cum_reward", "state", "belief_support"))
    }
    row.update(step=step, status="Terminal", process_rss_mb=previous["process_rss_mb"])
    for agent in AGENTS:
        row.update(
            {
                f"reward_{agent}": 0.0,
                f"planning_cpu_{agent}": 0.0,
                f"planning_wall_{agent}": 0.0,
                f"update_cpu_{agent}": 0.0,
                f"update_wall_{agent}": 0.0,
            }
        )
    return row


def _window(trial_id, step):
    """A step past the exact window: nothing was played, so nothing is recorded as reward."""
    row = {"trial": trial_id, "step": step, "status": "Window"}
    for agent in AGENTS:
        row.update({f"reward_{agent}": math.nan, f"cum_reward_{agent}": math.nan})
    return row


def _timed(call, timings, kind, agent):
    """Run ``call``; record its process-CPU and wall seconds (CPU is the primary cost)."""
    wall, cpu = time.perf_counter(), time.process_time()
    result = call()
    timings[f"{kind}_wall_{agent}"] = time.perf_counter() - wall
    timings[f"{kind}_cpu_{agent}"] = time.process_time() - cpu
    return result
