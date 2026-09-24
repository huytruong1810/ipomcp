"""Independent boundary checks for reference solver and matched experiment setup."""

import pytest

from examples.experiments.planner_oracle_experiment import evaluate_case
from examples.tiger.model.tiger_model import TigerModel
from solvers.exact.alpha_vector import AlphaVector2D, prune_2d
from solvers.exact.pomdp_exact_vi import ExactPOMDPSolver


def test_large_slopes_are_not_merged_by_relative_tolerance():
    first = AlphaVector2D(1e9, 0)
    second = AlphaVector2D(1e9 + 0.5, -0.5)
    envelope = prune_2d([first, second])
    assert len(envelope) == 2
    for p in [0, 0.25, 0.75, 1]:
        assert max(v.value(p) for v in envelope) == max(first.value(p), second.value(p))


def test_exact_l1_zero_horizon_and_invalid_inputs():
    solver = ExactPOMDPSolver(TigerModel(), horizon=3)
    assert solver.value(0.5, 0) == 0
    assert solver.q_values(0.5, 0) == {"L": 0, "OL": 0, "OR": 0}
    with pytest.raises(ValueError):
        solver.value(float("nan"))
    with pytest.raises(ValueError):
        solver.value(0.5, -1)
    with pytest.raises(ValueError):
        ExactPOMDPSolver(TigerModel(), opponent_policy={"L": 0.2})


@pytest.mark.parametrize("kind", ["mcts", "rts"])
def test_oracle_comparison_binds_discount_belief_and_horizon(kind):
    row = evaluate_case(kind, 1, 100, 17, 0.98, 0.3)[0]
    assert row["oracle_q"] == pytest.approx({"L": -1, "OL": -97.8, "OR": 7.8})
    assert row["first_action_loss"] == pytest.approx(0)
    assert row["horizon"] == 1 and row["gamma"] == 0.3
