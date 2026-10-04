"""Modeled-agent qualification: j-L1 scored at the beliefs i's search queries it at."""

import json

import pytest

from ipomdp.experiments.modeled_qualification import run_qualification
from ipomdp.experiments.oracle import evaluate_case


def test_observer_sees_every_modeled_policy_and_level_one_rejects_it():
    seen = []
    evaluate_case(
        "mcts",
        2,
        20,
        0,
        0.3,
        level=2,
        opponent_depth=2,
        modeled="5",
        reference="exact",
        opponent_observer=lambda model, policy: seen.append((model, policy)),
    )
    assert seen
    assert all(model.frame.agent_id == "j" and model.frame.level == 1 for model, _ in seen)
    assert all(abs(sum(policy.values()) - 1) < 1e-12 for _, policy in seen)
    with pytest.raises(ValueError, match="no modeled policy"):
        evaluate_case("mcts", 2, 20, 0, 0.3, opponent_observer=lambda *_: None)


def test_ladder_reproduces_the_search_and_reports_a_qualifying_rung(tmp_path):
    # The driver raises if the rung equal to ``modeled`` does not reproduce the
    # policy i's search used, so a complete run is itself that check.
    summary, qualifying, failures = run_qualification(
        tmp_path / "q",
        horizons=[2],
        budget=30,
        modeled="5",
        ladder=("5", "200"),
        seeds=[0],
        beliefs=(0.1, 0.9),
        opponent_beliefs=(0.5,),
        workers=1,
    )
    assert failures == []
    assert sorted(summary["rung"]) == ["200", "5"]
    assert len(set(summary["queries"])) == 1  # every rung scores the same queries
    assert (summary["wrong_beliefs"] <= summary["beliefs"]).all()
    assert (summary["beliefs"] <= summary["queries"]).all()
    by_rung = summary.set_index("rung")
    assert (
        by_rung["wrong"]
        == by_rung["wrong_where_decision_wrong"] + by_rung["wrong_where_decision_right"]
    ).all()
    assert list(qualifying) == [(2, 2, 30, "5", 0)]
    assert qualifying[(2, 2, 30, "5", 0)] in ("5", "200", None)
    rows = json.loads((tmp_path / "q" / "rows.json").read_text())
    assert {r["rung"] for r in rows} == {"5", "200"}
    assert all(abs(sum(r["physical_belief"].values()) - 1) < 1e-9 for r in rows)


def test_level_three_rungs_are_whole_stacks_and_reproduce_the_search(tmp_path):
    summary, qualifying, failures = run_qualification(
        tmp_path / "q3",
        level=3,
        horizons=[2],
        budget=30,
        modeled="20/5",
        ladder=("20/5", "40s[20,2,3.0]/exact"),
        seeds=[0],
        beliefs=(0.3,),
        opponent_beliefs=(0.5,),
        nested_beliefs=(0.2, 0.8),
        workers=1,
    )
    assert failures == []
    assert set(summary["rung"]) == {"20/5", "40s[20,2,3.0]/exact"}
    assert set(summary["level"]) == {3}
    assert (summary["mean_cpu_seconds"] > 0).all()
    rows = json.loads((tmp_path / "q3" / "rows.json").read_text())
    assert {r["nested_belief"] for r in rows} == {0.2, 0.8}
    assert qualifying[(3, 2, 30, "20/5", 0)] in ("20/5", "40s[20,2,3.0]/exact", None)


def test_unscored_decisions_skip_the_reference_but_still_qualify(tmp_path):
    row = evaluate_case(
        "mcts", 2, 20, 0, 0.3, level=2, opponent_depth=2, modeled="5", reference="none"
    )[0]
    assert "oracle_q" not in row and "first_action_loss" not in row
    assert set(row["estimated_q"]) == set(row["policy"]) | set(row["estimated_q"])
    summary, _, failures = run_qualification(
        tmp_path / "q",
        horizons=[2],
        budget=30,
        modeled="5",
        ladder=("5",),
        seeds=[0],
        beliefs=(0.1,),
        opponent_beliefs=(0.5,),
        score_decision=False,
        workers=1,
    )
    assert failures == []
    assert summary.iloc[0]["queries"] > 0
    assert summary["wrong_where_decision_wrong"].isna().all()
