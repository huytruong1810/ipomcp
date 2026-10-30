"""Reachable-history decisions (experiments.reachable_decisions): problems, references and scored decisions."""

import json

import pytest

from ipomdp.experiments import reachable_decisions as rd


def test_reference_history_is_reproducible_and_reachable():
    first = rd.reference("tiger-joint", 2, 2, seed=3, steps=2)
    again = rd.reference("tiger-joint", 2, 2, seed=3, steps=2)
    assert first["history"] == again["history"] and first["q"] == again["q"]
    assert len(first["history"]) == 2 and set(first["q"]) == {"L", "OL", "OR"}


def test_decide_scores_the_policy_against_the_stored_reference():
    problem = json.loads(json.dumps(rd.reference("tiger-joint", 2, 2, seed=1, steps=1)))
    row = rd.decide(problem, "mcts", 100, seed=0, modeled=100)
    assert row["loss"] == pytest.approx(0.0, abs=1e-9) and row["exact_root"]
    assert abs(sum(row["policy"].values()) - 1) < 1e-12


def test_uav_histories_never_end_the_episode():
    problem = rd.reference("uav", 2, 2, seed=5, steps=3)
    assert len(problem["history"]) == 3


def test_exact_modeled_levels_with_exact_beliefs_reproduce_the_exact_q():
    problem = json.loads(json.dumps(rd.reference("tiger-joint", 2, 2, seed=1, steps=2)))
    row = rd.decide(problem, "mcts", 100, seed=0, modeled_exact=True)
    assert row["modeled"] == "exact" and row["exact_root"]
    assert row["values"] == pytest.approx(problem["q"], abs=1e-9)


def test_exact_modeled_levels_are_an_i_pomcp_arm():
    problem = json.loads(json.dumps(rd.reference("tiger-joint", 2, 2, seed=1, steps=1)))
    with pytest.raises(ValueError):
        rd.decide(problem, "rts", 2, seed=0, modeled_exact=True)


def test_intmcp_with_the_exact_opponent_records_the_diagnostic():
    pytest.importorskip("intmcp", reason="third_party/i-ntmcp is not on the path")
    problem = json.loads(json.dumps(rd.reference("tiger-joint", 2, 2, seed=1, steps=1)))
    row = rd.decide(problem, "intmcp", 200, seed=0, intmcp_opponent="exact")
    assert row["intmcp_opponent"] == "exact" and row["loss"] >= -1e-9


def test_digest_only_matches_the_solved_problem():
    solved = rd.reference("tiger-joint", 2, 2, seed=3, steps=2)
    bare = rd.reference("tiger-joint", 2, 2, seed=3, steps=2, solve=False)
    assert "q" not in bare and bare["digest"] == solved["digest"]
    assert bare["history"] == solved["history"]


def test_certified_bounds_contain_the_exact_q_and_narrow_with_budget():
    problem = json.loads(json.dumps(rd.reference("tiger-joint", 2, 3, seed=1, steps=1)))
    rows = [rd.decide(problem, "mcts", budget, seed=0, modeled_exact=True) for budget in (10, 400)]
    assert all(row["bounds_contain_reference"] for row in rows)
    assert rows[1]["bound_width"] <= rows[0]["bound_width"]
    assert set(rows[0]["bounds"]) == set(problem["q"])


def test_a_budget_ladder_scores_prefixes_of_the_same_tree_and_ends_at_the_decision():
    problem = json.loads(json.dumps(rd.reference("tiger-joint", 2, 3, seed=1, steps=1)))
    plain = rd.decide(problem, "mcts", 400, seed=0, modeled_exact=True)
    row = rd.decide(problem, "mcts", 400, seed=0, modeled_exact=True, ladder=[10, 100])
    assert [stage["budget"] for stage in row["ladder"]] == [10, 100, 400]
    assert row["values"] == plain["values"] and row["bounds"] == plain["bounds"]
    assert row["ladder"][-1]["values"] == plain["values"] and row["ladder"][-1]["bounds"] == plain["bounds"]
    assert all(stage["bounds_contain_reference"] for stage in row["ladder"])
