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
