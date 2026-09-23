"""Validation tests for Exact Level 2 Interactive Value Iteration."""

import pytest

from examples.tiger.model.tiger_model import (
    LISTEN,
    OPEN_LEFT,
    OPEN_RIGHT,
    TigerModel,
)
from solvers.exact.ipomdp_exact_vi import ExactIPOMDPSolver
from solvers.exact.pomdp_exact_vi import ExactPOMDPSolver


def test_exact_l2_solver_initialization_and_solution():
    model = TigerModel()
    solver = ExactIPOMDPSolver(model, horizon=3, gamma=0.95)
    for rem in [1, 2, 3]:
        assert rem in solver._v_table
        assert rem in solver._q_table
        assert len(solver._v_table[rem]) > 0


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


def test_exact_l2_value_strictly_greater_than_l1():
    """An optimizing L1 partner yields strictly higher expected value than random L0."""
    model = TigerModel()
    l1_solver = ExactPOMDPSolver(model, horizon=2, gamma=0.95)
    l1_solver.solve(2)
    l2_solver = ExactIPOMDPSolver(model, horizon=2, gamma=0.95)

    # At p=0.5, facing an intentional L1 partner allows joint listening and coordination
    v_l1 = l1_solver.value(0.5, 2)
    v_l2 = l2_solver.value(0.5, 0.5, 2)

    # v_l1 is negative (-1.95) due to random L0 door openings
    # v_l2 is positive (+8.50) because L1 cooperates and opens the correct door
    assert v_l2 > v_l1
    assert v_l2 > 0.0
