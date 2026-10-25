"""Rao-Blackwellized rewards (analysis.rao_blackwell) against brute-force enumeration."""

import itertools
import math

import numpy as np
import pandas as pd
import pytest

from ipomdp.analysis.endpoints import final_rewards
from ipomdp.analysis.rao_blackwell import rao_blackwell
from ipomdp.experiments.episodes import run_episode
from ipomdp.experiments.study import AgentSpec, StudySpec

AGENTS = ("i", "j")


def random_pair(domain, steps, trials):
    spec = StudySpec(
        name="rb",
        domain=domain,
        trials=trials,
        steps=steps,
        agents={"i": AgentSpec(level=0), "j": AgentSpec(level=0)},
    )
    frame = pd.DataFrame([row for t in range(trials) for row in run_episode(spec, t)])
    return spec.physics(), frame


def brute_force(trial, physics):
    """E[r_t | actions, observations] by summing over every hidden state path."""
    rows = trial.sort_values("step")
    played = rows[(rows.step > 0) & rows.action_i.notna()]
    actions = {a: {str(x): x for x in physics.get_all_actions(a)} for a in AGENTS}
    joints = [{a: actions[a][row[f"action_{a}"]] for a in AGENTS} for _, row in played.iterrows()]
    seen = [
        {a: (None if pd.isna(row[f"observation_{a}"]) else row[f"observation_{a}"]) for a in AGENTS}
        for _, row in played.iterrows()
    ]
    states = {s for s, _ in physics.initial_distribution()}
    frontier = set(states)
    for joint in joints:
        frontier = {s2 for s in frontier for s2, _ in physics.transition_distribution(s, joint)}
        states |= frontier
    prior = dict(physics.initial_distribution())
    totals, weight = np.zeros((len(joints), 2)), 0.0
    for path in itertools.product(sorted(states, key=str), repeat=len(joints) + 1):
        p = prior.get(path[0], 0.0)
        for t, joint in enumerate(joints):
            if p == 0:
                break
            p *= dict(physics.transition_distribution(path[t], joint)).get(path[t + 1], 0.0)
            ended = physics.is_terminal(path[t + 1])
            if seen[t]["i"] is None:
                p *= 1.0 if ended else 0.0
                continue
            if ended:
                p = 0.0
                continue
            for a in AGENTS:
                p *= sum(
                    q
                    for o, q in physics.observation_distribution(path[t + 1], joint, a)
                    if str(o) == seen[t][a]
                )
        if p == 0:
            continue
        weight += p
        for t, joint in enumerate(joints):
            for k, a in enumerate(AGENTS):
                totals[t, k] += p * physics.get_reward(path[t], joint, path[t + 1], a)
    return totals / weight


@pytest.mark.parametrize("domain, steps", [("tiger", 4), ("tiger-joint", 4)])
def test_rewards_match_brute_force_enumeration(domain, steps):
    physics, frame = random_pair(domain, steps, trials=6)
    result = rao_blackwell(frame, physics)
    for trial, rows in result.groupby("trial"):
        rows = rows.sort_values("step")
        expected = brute_force(rows, physics)
        got = rows[rows.step > 0][["rb_reward_i", "rb_reward_j"]].to_numpy()
        np.testing.assert_allclose(got[: len(expected)], expected, atol=1e-9)


def test_next_state_rewards_and_termination_match_brute_force():
    # UAV pays on the joint move into the target's sector and ends the episode,
    # so the reward reads the next state; small grid paths keep enumeration exact.
    physics, frame = random_pair("uav", 3, trials=8)
    result = rao_blackwell(frame, physics)
    for trial, rows in result.groupby("trial"):
        rows = rows.sort_values("step")
        expected = brute_force(rows, physics)
        played = rows[(rows.step > 0) & rows.action_i.notna()]
        got = played[["rb_reward_i", "rb_reward_j"]].to_numpy()
        np.testing.assert_allclose(got, expected, atol=1e-9)
        # Padding after termination stays at zero reward.
        assert (rows[(rows.step > 0) & rows.action_i.isna()]["rb_reward_i"] == 0).all()


def test_the_mean_is_unchanged_and_the_variance_falls():
    # Uniform L0 pairs on published Tiger earn -91/3 per step in expectation
    # (listen -1; an open is -100 or +10 with equal probability).
    physics, frame = random_pair("tiger", 3, trials=400)
    result = rao_blackwell(frame, physics)
    sampled = final_rewards(result, "i")
    rb = final_rewards(result, "i", column="rb_cum_reward")
    half = 1.96 * rb.std() / math.sqrt(len(rb))
    assert abs(rb.mean() - (-91.0)) < 3 * half
    assert rb.var() < sampled.var()


def test_rows_past_the_window_stay_nan():
    physics, frame = random_pair("tiger", 3, trials=1)
    frame.loc[frame.step == 3, ["status", "reward_i", "reward_j"]] = ["Window", math.nan, math.nan]
    result = rao_blackwell(frame, physics).sort_values("step")
    assert math.isnan(result.iloc[-1]["rb_cum_reward_i"])
    assert not math.isnan(result.iloc[-2]["rb_cum_reward_i"])
