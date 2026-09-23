"""Validation tests for Exact POMDP / Level 1 I-POMDP Value Iteration."""

import pytest

from examples.tiger.model.tiger_model import (
    LISTEN,
    OPEN_LEFT,
    OPEN_RIGHT,
    TigerModel,
)
from solvers.exact.alpha_vector import (
    AlphaVector2D,
    AlphaVectorND,
    prune_2d,
    prune_2d_with_intervals,
    prune_nd,
)
from solvers.exact.pomdp_exact_vi import ExactPOMDPSolver
from tests.reference_tiger import finite_horizon_values


def test_alpha_vector_2d_pruning_basic():
    # v1 dominates v2 everywhere
    v1 = AlphaVector2D(10.0, 10.0, action="A")
    v2 = AlphaVector2D(5.0, 5.0, action="B")
    pruned = prune_2d([v1, v2])
    assert len(pruned) == 1
    assert pruned[0].action == "A"


def test_alpha_vector_2d_intervals():
    # Intersection at p = 0.5: v1 better for p > 0.5, v2 better for p < 0.5
    v1 = AlphaVector2D(10.0, 0.0, action="R")
    v2 = AlphaVector2D(0.0, 10.0, action="L")
    intervals = prune_2d_with_intervals([v1, v2])
    assert len(intervals) == 2
    assert intervals[0][0].action == "L"
    assert intervals[0][1] == pytest.approx(0.0)
    assert intervals[0][2] == pytest.approx(0.5)
    assert intervals[1][0].action == "R"
    assert intervals[1][1] == pytest.approx(0.5)
    assert intervals[1][2] == pytest.approx(1.0)


def test_exact_vi_matches_reference_oracle():
    """Verify that ExactPOMDPSolver matches finite_horizon_values to machine precision."""
    model = TigerModel()
    solver = ExactPOMDPSolver(model, horizon=5, gamma=0.95)
    solver.solve(5)

    test_beliefs = [0.01, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 0.99]

    for h in range(1, 6):
        for p in test_beliefs:
            ref_q = finite_horizon_values(p, h, gamma=0.95)
            ref_v = max(ref_q.values())

            solver_v = solver.value(p, h)
            solver_q = solver.q_values(p, h)

            # Test Value equivalence
            assert solver_v == pytest.approx(ref_v, abs=1e-9), (
                f"Value mismatch at h={h}, p={p}: solver={solver_v}, ref={ref_v}"
            )

            # Test Q-value equivalence for all actions
            for a in [LISTEN, OPEN_LEFT, OPEN_RIGHT]:
                assert solver_q[a] == pytest.approx(ref_q[a], abs=1e-9), (
                    f"Q-value mismatch for {a} at h={h}, p={p}: solver={solver_q[a]}, ref={ref_q[a]}"
                )


def test_exact_vi_symmetry():
    """Tiger domain is strictly symmetric across left and right."""
    model = TigerModel()
    solver = ExactPOMDPSolver(model, horizon=4, gamma=0.95)
    solver.solve(4)

    for h in range(1, 5):
        for p in [0.05, 0.15, 0.3, 0.5]:
            v_left = solver.value(p, h)
            v_right = solver.value(1.0 - p, h)
            assert v_left == pytest.approx(v_right, abs=1e-12)

            q_left = solver.q_values(p, h)
            q_right = solver.q_values(1.0 - p, h)
            assert q_left[LISTEN] == pytest.approx(q_right[LISTEN], abs=1e-12)
            assert q_left[OPEN_LEFT] == pytest.approx(q_right[OPEN_RIGHT], abs=1e-12)
            assert q_left[OPEN_RIGHT] == pytest.approx(q_right[OPEN_LEFT], abs=1e-12)


def test_exact_vi_decision_boundaries():
    """Verify analytical decision thresholds."""
    model = TigerModel()
    solver = ExactPOMDPSolver(model, horizon=4, gamma=0.95)
    solver.solve(4)

    # At Horizon 1: threshold is exactly 0.10 and 0.90
    segs_1 = solver.decision_boundaries(1)
    assert len(segs_1) == 3
    assert segs_1[0].action == OPEN_LEFT
    assert segs_1[0].right_p == pytest.approx(0.10, abs=1e-5)
    assert segs_1[1].action == LISTEN
    assert segs_1[1].left_p == pytest.approx(0.10, abs=1e-5)
    assert segs_1[1].right_p == pytest.approx(0.90, abs=1e-5)
    assert segs_1[2].action == OPEN_RIGHT
    assert segs_1[2].left_p == pytest.approx(0.90, abs=1e-5)

    # At Horizon 4: Listen threshold widens
    segs_4 = solver.decision_boundaries(4)
    # Action at extreme left must be OPEN_LEFT, extreme right OPEN_RIGHT
    assert segs_4[0].action == OPEN_LEFT
    assert segs_4[-1].action == OPEN_RIGHT
    # Listen region is active in between
    assert solver.policy(0.5, 4) == LISTEN
    assert solver.policy(0.02, 4) == OPEN_LEFT
    assert solver.policy(0.98, 4) == OPEN_RIGHT


def test_prune_nd_linear_programming():
    """Test LP-based pruning on multi-dimensional alpha vectors."""
    # 3D state space: (v0, v1, v2)
    # v1 dominates at corner 0, v2 dominates at corner 1, v3 dominates at corner 2
    # v4 is dominated everywhere
    v1 = AlphaVectorND((10.0, 0.0, 0.0), action="A1")
    v2 = AlphaVectorND((0.0, 10.0, 0.0), action="A2")
    v3 = AlphaVectorND((0.0, 0.0, 10.0), action="A3")
    v4 = AlphaVectorND((1.0, 1.0, 1.0), action="A4")  # strictly below convex hull

    pruned = prune_nd([v1, v2, v3, v4])
    assert len(pruned) == 3
    actions = {v.action for v in pruned}
    assert actions == {"A1", "A2", "A3"}
