"""Analytic and cross-component checks of the fixed-depth L2 reference contract."""

import pytest

from ipomdp.experiments.oracle import evaluate_case, run_oracle_comparison


def test_fixed_l2_uniform_two_step_hand_calculation_and_runner_binding():
    # Both agents listen at their initial uniform beliefs; one 85% growl
    # leaves opening worth -6.5, below listening -1. Thus Q(L)=-1-.95.
    row = evaluate_case("mcts", 2, 300, 71, 0.5, level=2, opponent_depth=2)[0]
    assert row["oracle_q"]["L"] == pytest.approx(-1.95)
    assert row["estimated_q"]["L"] == pytest.approx(-1.95)
    assert row["first_action_loss"] == 0
    assert row["level"] == 2
    assert row["opponent"] == {
        "policy": "exact_l1_fixed_depth",
        "depth": 2,
        "initial_belief_p": 0.5,
    }


def test_l2_one_step_reward_does_not_expose_opponent_private_belief():
    for p_j in (0.01, 0.99):
        row = evaluate_case(
            "mcts", 1, 3, 0, 0.98, gamma=0.3, level=2, opponent_depth=2, opponent_belief=p_j
        )[0]
        assert row["oracle_q"] == pytest.approx({"L": -1, "OL": -97.8, "OR": 7.8})
        assert row["estimated_q"] == pytest.approx(row["oracle_q"])


def test_l2_requires_explicit_contract_before_creating_output(tmp_path):
    panel = dict(budgets=[10], horizons=[1], seeds=1, beliefs=[0.5], level=2)
    with pytest.raises(ValueError, match="explicit positive"):
        run_oracle_comparison(tmp_path / "missing", **panel)
    with pytest.raises(ValueError, match="Unknown planner"):
        run_oracle_comparison(tmp_path / "rts", planners=("rts",), opponent_depth=2, **panel)
    assert not list(tmp_path.iterdir())
