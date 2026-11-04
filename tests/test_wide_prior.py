"""Wide prior (experiments.wide_prior): M level-1 models of j with different initial beliefs (T4)."""

import json

import pytest

from ipomdp.belief.finite_belief import FiniteBelief
from ipomdp.experiments import reachable_decisions as rd
from ipomdp.experiments import wide_prior


def _wide(m_count, w0=0.0):
    return dict(kind="wide", M=m_count, w0=w0)


def test_grid_is_the_symmetric_midpoint_grid():
    assert wide_prior.grid(1) == [0.5]
    assert wide_prior.grid(4) == [0.125, 0.375, 0.625, 0.875]


@pytest.mark.parametrize("steps", [0, 1, 2])
def test_one_model_is_the_point_mass_prior_bit_for_bit(steps):
    point = rd.reference("tiger-joint", 2, 2, seed=4, steps=steps)
    wide = rd.reference("tiger-joint", 2, 2, seed=4, steps=steps, prior=_wide(1))
    assert wide["history"] == point["history"]
    assert wide["digest"] == point["digest"] and wide["q"] == point["q"]
    assert wide["true_opponent_p"] == 0.5


def test_root_support_is_two_atoms_per_model_of_j():
    physics = rd.make_physics("tiger-joint")
    assert len(FiniteBelief(wide_prior.prior_rows(physics, 2, 0.0)).mass) == 4
    assert len(FiniteBelief(wide_prior.prior_rows(physics, 8, 0.5)).mass) == 18


def test_true_opponent_is_reproducible_and_drawn_from_the_prior():
    physics = rd.make_physics("tiger-joint")
    draws = {wide_prior.true_opponent(physics, 8, 0.0, seed) for seed in range(40)}
    again = {wide_prior.true_opponent(physics, 8, 0.0, seed) for seed in range(40)}
    assert draws == again and len(draws) > 1
    assert all(model.frame.level == 1 for model in draws)


def test_exact_beliefs_decide_optimally_under_a_wide_prior():
    problem = json.loads(json.dumps(rd.reference("tiger-joint", 2, 2, seed=2, steps=1, prior=_wide(8))))
    assert problem["prior_digest"] and problem["support"] > 2
    row = rd.decide(problem, "mcts", 100, seed=0, modeled=100)
    assert row["loss"] == pytest.approx(0.0, abs=1e-9) and row["prior"] == _wide(8)


def test_exact_modeled_levels_reproduce_the_wide_prior_exact_q():
    problem = json.loads(json.dumps(rd.reference("tiger-joint", 2, 2, seed=2, steps=1, prior=_wide(4))))
    row = rd.decide(problem, "mcts", 100, seed=0, modeled_exact=True)
    assert row["values"] == pytest.approx(problem["q"], abs=1e-9)


def test_wide_prior_differs_from_the_point_mass():
    point = rd.reference("tiger-joint", 2, 2, seed=2, steps=0)
    wide = rd.reference("tiger-joint", 2, 2, seed=2, steps=0, prior=_wide(8))
    assert wide["digest"] != point["digest"]


def test_compressed_root_is_bounded_by_n():
    problem = json.loads(json.dumps(
        rd.reference("tiger-joint", 2, 2, seed=0, steps=0, solve=False, prior=_wide(32))))
    assert problem["support"] == 64
    row = rd.decide(problem, "mcts", 20, seed=0, modeled=20, particles=16)
    assert row["root_support"] <= 16


def test_tampered_prior_is_refused():
    problem = json.loads(json.dumps(
        rd.reference("tiger-joint", 2, 2, seed=0, steps=0, solve=False, prior=_wide(2))))
    problem["prior"]["M"] = 3
    with pytest.raises(ValueError):
        rd.decide(problem, "mcts", 10, seed=0, modeled=10)


def test_intmcp_refuses_a_wide_prior():
    problem = json.loads(json.dumps(
        rd.reference("tiger-joint", 2, 2, seed=0, steps=0, solve=False, prior=_wide(2))))
    with pytest.raises(ValueError, match="I-NTMCP"):
        rd.decide(problem, "intmcp", 100, seed=0)


def test_a_given_history_replays_to_the_generated_reference():
    generated = json.loads(json.dumps(rd.reference("tiger-joint", 2, 2, seed=5, steps=2, prior=_wide(4))))
    replayed = rd.reference("tiger-joint", 2, 2, seed=99, steps=0, prior=_wide(4),
                            given_history=generated["history"])
    assert replayed["digest"] == generated["digest"] and replayed["q"] == generated["q"]
    assert replayed["steps"] == 2 and replayed["history_given"]
    narrow = rd.reference("tiger-joint", 2, 2, seed=99, steps=0, given_history=generated["history"])
    assert narrow["digest"] != generated["digest"]
