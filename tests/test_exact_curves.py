"""Exact expected reward curves of exact pairs (analysis.exact_curves)."""

import math

import pandas as pd
import pytest

from ipomdp.analysis.endpoints import final_rewards
from ipomdp.analysis.exact_curves import exact_pair_curve
from ipomdp.analysis.rao_blackwell import rao_blackwell
from ipomdp.belief.finite_filter import InferenceBudgetExceeded
from ipomdp.domains.tiger import TigerModel
from ipomdp.experiments.episodes import run_episode
from ipomdp.experiments.study import AgentSpec, StudySpec


def curve(level_i, level_j, steps, **options):
    return list(exact_pair_curve(TigerModel(), {"i": level_i, "j": level_j}, steps, 2, **options))


def test_uniform_pairs_lose_91_thirds_per_step():
    # Listen -1; an open is -100 or +10 with equal probability: (-1 - 45 - 45) / 3.
    for row in curve(0, 0, 3):
        assert row["cum_reward_i"] == pytest.approx(-91 / 3 * row["step"])
        assert row["cum_reward_j"] == pytest.approx(-91 / 3 * row["step"])


def test_a_level_one_agent_listens_first():
    first = curve(1, 0, 1)[0]
    assert first["cum_reward_i"] == pytest.approx(-1.0)
    assert first["cum_reward_j"] == pytest.approx(-91 / 3)


def test_level_one_pairs_reproduce_n0():
    # HANDOFF 0.00000: steps 3 and 4 are 2.148 and 1.308 for L1 against L1.
    rows = curve(1, 1, 4)
    assert [round(r["cum_reward_i"], 3) for r in rows] == [-1.0, -2.0, 2.148, 1.308]
    assert all(r["cum_reward_i"] == pytest.approx(r["cum_reward_j"]) for r in rows)


def test_the_node_cap_raises():
    with pytest.raises(InferenceBudgetExceeded):
        curve(1, 1, 3, max_nodes=100)


def test_exact_episodes_average_to_the_exact_curve():
    exact = {"kind": "exact", "horizon": 2}
    spec = StudySpec(
        name="pair",
        domain="tiger",
        trials=300,
        steps=3,
        agents={"i": AgentSpec(level=1, planner=exact), "j": AgentSpec(level=1, planner=exact)},
    )
    frame = pd.DataFrame([row for t in range(spec.trials) for row in run_episode(spec, t)])
    rb = final_rewards(rao_blackwell(frame, spec.physics()), "i", column="rb_cum_reward")
    target = curve(1, 1, 3)[-1]["cum_reward_i"]
    assert abs(rb.mean() - target) < 4 * rb.std() / math.sqrt(len(rb))
