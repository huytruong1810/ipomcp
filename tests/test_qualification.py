"""Small-scale convergence gate: I-POMCP against exact L1 value iteration.

This is the fast version of ``experiments.l1_qualification``. It guards the one
property every planner change must preserve: root decisions are exactly
optimal at a moderate budget, and Q error shrinks as the budget grows. The
repository once shipped a default estimator that passed every other test while
being 61-88% optimal here.
"""

import pytest

from ipomdp.experiments.l1_qualification import evaluate_cell, summarize

BELIEFS = (0.1, 0.5, 0.85, 0.98)
SEEDS = (0, 1)


@pytest.mark.parametrize("horizon", [2, 3, 4])
def test_root_decisions_are_exactly_optimal_at_one_thousand_simulations(horizon):
    rows = evaluate_cell(1000, horizon, SEEDS, BELIEFS)
    assert all(row["optimal"] for row in rows), [r for r in rows if not r["optimal"]]


def test_q_error_shrinks_with_budget():
    rows = [row for budget in (100, 3000) for row in evaluate_cell(budget, 4, SEEDS, BELIEFS)]
    errors = summarize(rows).set_index("budget")["mean_max_q_error"]
    assert errors[3000] < errors[100]
