"""Analytic and cross-component checks of the fixed-depth L2 reference contract."""

import pytest

from examples.experiments.planner_oracle_experiment import evaluate_case, run_oracle_comparison
from examples.tiger.model.tiger_model import TIGER_LEFT, TIGER_RIGHT, TigerModel
from ipomdp.finite_belief import FiniteBelief, InteractiveState, MentalModel
from ipomdp.frame import AgentFrame
from solvers.exact.fixed_tiger_opponent import FixedTigerL1Policy
from solvers.exact.ipomdp_exact_vi import ExactIPOMDPSolver


def private_model(physics, p):
    l0 = MentalModel(AgentFrame("i", 0, physics))
    belief = FiniteBelief(
        (
            (InteractiveState(TIGER_LEFT, l0), p),
            (InteractiveState(TIGER_RIGHT, l0), 1 - p),
        )
    )
    return MentalModel(AgentFrame("j", 1, physics), belief)


def test_fixed_opponent_depth_does_not_count_down_with_protagonist():
    physics = TigerModel()
    countdown = ExactIPOMDPSolver(physics, horizon=2)
    fixed = ExactIPOMDPSolver(physics, horizon=2, opponent_horizon=2)
    # At p=.085 a one-step L1 opens, but a two-step L1 listens. The distinction
    # persists when i has only one step left; it is not a numerical tie.
    assert dict(countdown._opponent_policy(0.085, 1)) == {"OL": 1}
    assert dict(fixed._opponent_policy(0.085, 1)) == {"L": 1}
    fixed.solve(1)
    assert dict(fixed._opponent_policy(0.085, 1)) == {"L": 1}


def test_registered_opponent_uses_own_belief_and_declared_fixed_depth():
    physics = TigerModel()
    policy = FixedTigerL1Policy(physics, 2, 0.95)
    reference = ExactIPOMDPSolver(physics, horizon=3, opponent_horizon=2)
    for p in (0.01, 0.085, 0.5, 0.915, 0.99):
        model = private_model(physics, p)
        assert policy.policy_for(model) == dict(reference._opponent_policy(p, 1))
        assert policy.policy_for(model, modeled=True) == policy.policy_for(model)
    assert policy.policy_for(private_model(physics, 0.01)) == {"OL": 1}
    assert policy.policy_for(private_model(physics, 0.99)) == {"OR": 1}


@pytest.mark.parametrize("depth", [0, -1, True, 1.5])
def test_invalid_fixed_horizon_rejected(depth):
    with pytest.raises(ValueError):
        ExactIPOMDPSolver(opponent_horizon=depth)
    with pytest.raises(ValueError):
        FixedTigerL1Policy(TigerModel(), depth, 0.95)


def test_fixed_l2_uniform_two_step_hand_calculation_and_runner_binding():
    # Both agents listen at their initial uniform beliefs; one 85% growl
    # leaves opening worth -6.5, below listening -1. Thus Q(L)=-1-.95.
    reference = ExactIPOMDPSolver(horizon=2, opponent_horizon=2)
    assert reference.q_values(0.5)["L"] == pytest.approx(-1.95)
    row = evaluate_case(
        "mcts",
        2,
        300,
        71,
        0.5,
        0.95,
        "bounded",
        1,
        True,
        "empirical_bellman",
        level=2,
        opponent_depth=2,
    )[0]
    assert row["oracle_q"]["L"] == pytest.approx(-1.95)
    assert row["estimated_q"]["L"] == pytest.approx(-1.95)
    assert row["first_action_loss"] == 0
    assert row["level"] == 2
    assert row["opponent_model"] == {
        "policy": "exact_l1_fixed_horizon",
        "depth": 2,
        "initial_belief_p": 0.5,
        "tie_atol": 1e-10,
    }


def test_l2_one_step_reward_does_not_expose_opponent_private_belief():
    for p_j in (0.01, 0.99):
        row = evaluate_case(
            "mcts",
            1,
            3,
            0,
            0.98,
            0.3,
            level=2,
            opponent_depth=2,
            opponent_belief=p_j,
        )[0]
        assert row["oracle_q"] == pytest.approx({"L": -1, "OL": -97.8, "OR": 7.8})
        assert row["estimated_q"] == pytest.approx(row["oracle_q"])


def test_l2_requires_explicit_contract_before_creating_output(tmp_path):
    with pytest.raises(ValueError, match="explicit positive"):
        run_oracle_comparison(tmp_path / "missing", level=2, planners=("mcts",))
    with pytest.raises(ValueError, match="MCTS only"):
        run_oracle_comparison(tmp_path / "rts", level=2, opponent_depth=2)
    assert not list(tmp_path.iterdir())


def test_fixed_l1_policy_rejects_mismatched_physics():
    policy = FixedTigerL1Policy(TigerModel(), 2, 0.95)
    with pytest.raises(ValueError, match="frame and physics"):
        policy.policy_for(private_model(TigerModel(), 0.5))
