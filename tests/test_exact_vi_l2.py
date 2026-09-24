"""Validation tests for Exact Level 2 Interactive Value Iteration."""

import pytest

from examples.tiger.model.tiger_model import (
    LISTEN,
    OPEN_LEFT,
    OPEN_RIGHT,
    TigerModel,
)
from solvers.exact.ipomdp_exact_vi import ExactIPOMDPSolver


def test_exact_l2_solver_initialization_and_solution():
    model = TigerModel()
    solver = ExactIPOMDPSolver(model, horizon=3, gamma=0.95)
    for rem in [1, 2, 3]:
        assert set(solver.q_values(0.5, horizon=rem)) == {LISTEN, OPEN_LEFT, OPEN_RIGHT}


def test_exact_l2_symmetry():
    """Level 2 value and Q-values must remain symmetric across left and right."""
    model = TigerModel()
    solver = ExactIPOMDPSolver(model, horizon=2, gamma=0.95)

    for p in [0.05, 0.2, 0.5]:
        q_left = solver.q_values(p, b_j_tl=0.5, horizon=2)
        q_right = solver.q_values(1.0 - p, b_j_tl=0.5, horizon=2)

        assert q_left[LISTEN] == pytest.approx(q_right[LISTEN], abs=1e-9)
        assert q_left[OPEN_LEFT] == pytest.approx(q_right[OPEN_RIGHT], abs=1e-9)
        assert q_left[OPEN_RIGHT] == pytest.approx(q_right[OPEN_LEFT], abs=1e-9)


def test_exact_l2_decisions():
    """Verify Level 2 decision boundaries at horizon 2."""
    model = TigerModel()
    solver = ExactIPOMDPSolver(model, horizon=2, gamma=0.95)

    # High confidence => open correct door
    assert solver.policy(0.02, 0.5, 2) == OPEN_LEFT
    assert solver.policy(0.98, 0.5, 2) == OPEN_RIGHT

    # Ambiguity => Listen
    assert solver.policy(0.5, 0.5, 2) == LISTEN


def test_exact_l2_two_step_value_has_no_clairvoyant_continuation():
    """One 85%-accurate growl cannot make a final opening profitable.

    At the uniform initial belief both agents listen. On the final decision,
    listening pays -1 and even the preferred opening pays .85*10-.15*100=-6.5.
    Therefore V2=-1-.95=-1.95. A statewise maximization incorrectly gives +8.5.
    This is an independent hand calculation, not another call to the same DP.
    """
    solver = ExactIPOMDPSolver(TigerModel(), horizon=2, gamma=0.95)
    assert solver.value(0.5) == pytest.approx(-1.95, abs=1e-12)
    assert solver.q_values(0.5)[LISTEN] == pytest.approx(-1.95, abs=1e-12)


def test_exact_l2_uninformative_sensors_cannot_reveal_hidden_state():
    solver = ExactIPOMDPSolver(TigerModel(growl_accuracy={"i": 0.5, "j": 0.5}), horizon=3)
    assert solver.value(0.5) == pytest.approx(-1 - 0.95 - 0.95**2)


def test_exact_l2_arbitrary_opponent_beliefs_and_zero_horizon():
    solver = ExactIPOMDPSolver(TigerModel(), horizon=2)
    # At H1 the physical reward does not depend on j's action. Arbitrary j
    # beliefs must be accepted exactly, with no nearest-grid substitution.
    for b in [0.0123456789, 0.4, 0.987654321]:
        assert solver.q_values(0.5, b, 1) == pytest.approx(
            {LISTEN: -1, OPEN_LEFT: -45, OPEN_RIGHT: -45}
        )
        assert solver.value(0.5, b, 0) == 0
    with pytest.raises(ValueError):
        solver.value(1.01)
    with pytest.raises(ValueError):
        solver.value(0.5, horizon=-1)
