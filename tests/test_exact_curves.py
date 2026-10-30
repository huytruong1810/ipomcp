"""Exact expected reward curves of exact pairs (analysis.exact_curves)."""

import math

import pandas as pd
import pytest

from ipomdp.analysis.endpoints import final_rewards
from ipomdp.analysis.exact_curves import exact_decision_gains, exact_pair_curve
from ipomdp.analysis.rao_blackwell import rao_blackwell
from ipomdp.belief.finite_filter import InferenceBudgetExceeded
from ipomdp.domains.tiger import JointTigerModel, TigerModel
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


def gains(physics, k, m, steps, reference_level):
    return list(exact_decision_gains(physics, "i", k, m, steps, 2, reference_level))


def test_decision_gains_follow_the_reference_pair_curve():
    # Histories are those of the reference level against the true opponent, so its rewards are the pair curve's.
    for reference_level in (1, 2):
        rows = gains(TigerModel(), 1, 0, 3, reference_level)
        pair = curve(reference_level, 0, 3)
        cumulative = 0.0
        for row, exact in zip(rows, pair):
            cumulative += row["reward_ref"]
            assert cumulative == pytest.approx(exact["cum_reward_i"], abs=1e-9)


def test_step_three_gain_against_random_matches_the_enumeration():
    # results/step3_exact_20261028/step3.json: E[r3] is 2.9335 at level 1 and 3.394 at level 2.
    rows = gains(JointTigerModel(together_safe=40.0), 1, 0, 3, 2)
    assert rows[2]["immediate_gain"] == pytest.approx(3.394 - 2.9335, abs=1e-9)


def test_the_bayes_residual_is_never_negative():
    # Level k + 1 is Bayes-optimal over its own support at every history (V3 proposition).
    for k, m in ((1, 0), (1, 1), (0, 1), (1, 2)):
        for row in gains(JointTigerModel(together_safe=40.0), k, m, 3, k + 1):
            assert row["residual_min"] >= -1e-9
            assert row["bayes_gap"] <= 1e-9
            assert row["unsupported"] == 0
            assert row["mass"] == pytest.approx(1.0)


def test_the_restricted_truth_equals_a_separately_built_oracle():
    # Against a level-1 opponent, histories of a level-1 agent: the truth is level 2's belief restricted
    # to level 1 when k = 1, and a level-2 oracle with a point prior when k = 0.
    physics = TigerModel()
    restricted = gains(physics, 1, 1, 3, 1)
    oracle = gains(physics, 0, 1, 3, 1)
    for a, b in zip(restricted, oracle):
        assert a["reward_ref"] == pytest.approx(b["reward_ref"], abs=1e-9)
        assert a["nodes"] == b["nodes"]


def test_node_records_sum_to_the_step_rows_and_respect_the_confidence_bound():
    # Against a level in its support, the deeper agent's loss at h is at most eps / (1 - eps) times the
    # return range, eps = 1 - P_{k+1}(m | h); here H = 2 and gamma = 0.95.
    physics = JointTigerModel(together_safe=40.0)
    records = []
    rows = list(exact_decision_gains(physics, "i", 1, 0, 3, 2, 2, on_node=records.append))
    low, high = physics.reward_bounds("i")
    width = (high - low) * 1.95
    for row in rows:
        mine = [r for r in records if r["step"] == row["step"]]
        assert math.fsum(r["mass"] * r["plan_gain"] for r in mine) == pytest.approx(row["plan_gain"], abs=1e-9)
    for r in records:
        eps = 1 - r["posterior_true"]
        assert r["plan_gain"] >= -eps / (1 - eps) * width - 1e-9
