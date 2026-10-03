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
        opponent_budget=5,
        reference="exact",
        opponent_observer=lambda model, policy: seen.append((model, policy)),
    )
    assert seen
    assert all(model.frame.agent_id == "j" and model.frame.level == 1 for model, _ in seen)
    assert all(abs(sum(policy.values()) - 1) < 1e-12 for _, policy in seen)
    with pytest.raises(ValueError, match="no modeled policy"):
        evaluate_case("mcts", 2, 20, 0, 0.3, opponent_observer=lambda *_: None)


def test_ladder_reproduces_the_search_and_reports_a_qualifying_budget(tmp_path):
    # The driver raises if the rung equal to ``modeled`` does not reproduce the
    # policy i's search used, so a complete run is itself that check.
    summary, qualifying, failures = run_qualification(
        tmp_path / "q",
        horizons=[2],
        budget=30,
        modeled=5,
        ladder=(5, 200),
        seeds=[0],
        beliefs=(0.1, 0.9),
        opponent_beliefs=(0.5,),
        workers=1,
    )
    assert failures == []
    assert sorted(summary["ladder_budget"]) == [5, 200]
    assert len(set(summary["queries"])) == 1  # every rung scores the same queries
    by_rung = summary.set_index("ladder_budget")
    assert (
        by_rung["wrong"]
        == by_rung["wrong_where_decision_wrong"] + by_rung["wrong_where_decision_right"]
    ).all()
    assert list(qualifying) == [(2, 30, 5, 0)]
    assert qualifying[(2, 30, 5, 0)] in (5, 200, None)
    rows = json.loads((tmp_path / "q" / "rows.json").read_text())
    assert {r["ladder_budget"] for r in rows} == {5, 200}
    assert all(abs(sum(r["physical_belief"].values()) - 1) < 1e-9 for r in rows)
