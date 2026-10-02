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
- ``hierarchy`` (JSON, when the study records hierarchies).

After the episode ends, rows are padded to the declared length with zero
rewards and carried cumulative rewards, so every trial contributes a full
panel and early termination is not silently averaged away.
"""

import json
import random
import time

from ipomdp.analysis.hierarchy import level_posterior, nested_hierarchy
from ipomdp.experiments.study import AGENTS, build_agents
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
    for step in range(1, spec.steps + 1):
        if terminal:
            rows.append(_padding(rows[-1], step))
            continue
        timings, joint = {}, {}
        for agent in AGENTS:
            joint[agent] = _timed(planners[agent].get_action, timings, "planning", agent)
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
        rows.append(row)
        state = following
    return rows


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
            row[f"action_values_{agent}"] = json.dumps(
                {str(a): q for a, q in planner.recommendation_values.items()}, sort_keys=True
            )
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


def _timed(call, timings, kind, agent):
    """Run ``call``; record its process-CPU and wall seconds (CPU is the primary cost)."""
    wall, cpu = time.perf_counter(), time.process_time()
    result = call()
    timings[f"{kind}_wall_{agent}"] = time.perf_counter() - wall
    timings[f"{kind}_cpu_{agent}"] = time.process_time() - cpu
    return result
