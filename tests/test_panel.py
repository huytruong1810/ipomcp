"""The panel: one supervised job per (cell, seed), scored against the exact I-POMDP."""

import json
from functools import partial

import pytest

from ipomdp.domains.uav import UAVModel
from ipomdp.experiments.panel import CELL, evaluate_cell, run_panel, summarize
from ipomdp.experiments.problems import (
    TIGER_BELIEFS,
    TIGER_OPPONENT_BELIEFS,
    tiger_problems,
    uav_laws,
    uav_problems,
)


def test_panel_rows_cover_every_problem_and_name_their_cell(tmp_path):
    cells = [("rts", 2, 2, 2, "2", 16, "sampled", 0), ("mcts", 2, 2, 30, "exact", 0, "exact", 2)]
    summary, failures = run_panel(tmp_path / "panel", cells, seeds=[0], workers=2)
    assert failures == []
    assert list(summary["n"]) == [len(TIGER_BELIEFS) * len(TIGER_OPPONENT_BELIEFS)] * 2
    rows = json.loads((tmp_path / "panel" / "rows.json").read_text())
    assert {(r["planner"], r["particles"], r["reference"]) for r in rows} == {
        ("rts", 16, "exact"),
        ("mcts", 0, "exact"),
    }
    manifest = json.loads((tmp_path / "panel" / "manifest.json").read_text())
    assert manifest["panel"]["cells"][0] == dict(
        planner="rts",
        level=2,
        horizon=2,
        budget=2,
        modeled="2",
        particles=16,
        particle_filter="sampled",
        history=0,
    )
    assert {r["history_steps"] for r in rows if r["planner"] == "mcts"} == {2}
    assert {r["opponent"]["policy"] for r in rows if r["planner"] == "mcts"} == {
        "exact_fixed_depth"
    }


def test_rts_only_panels_summarize_without_search_columns(tmp_path):
    cells = [("rts", 1, 2, 2, "exact", 0, "exact", 0)]
    summary, failures = run_panel(
        tmp_path / "panel",
        cells,
        seeds=[0],
        workers=1,
        problems=partial(tiger_problems, beliefs=(0.5,)),
    )
    assert failures == []
    assert summary.iloc[0]["n"] == 1
    assert summary["search_optimal_rate"].isna().all()


def test_panel_takes_a_belief_grid_and_bounds_each_optimal_rate(tmp_path):
    cells = [("mcts", 2, 1, 20, "exact", 0, "exact", 0)]
    summary, failures = run_panel(
        tmp_path / "panel",
        cells,
        seeds=[0, 1],
        workers=1,
        problems=partial(tiger_problems, beliefs=(0.1, 0.9), opponent_beliefs=(0.5,)),
    )
    assert failures == []
    row = summary.iloc[0]
    assert row["n"] == 4
    # Horizon 1 against the exact reference: every decision is optimal, and 4/4
    # still leaves the exact 95% interval wide.
    assert row["optimal_rate"] == 1.0
    assert row["optimal_rate_high"] == 1.0
    assert 0.39 < row["optimal_rate_low"] < 0.40
    manifest = json.loads((tmp_path / "panel" / "manifest.json").read_text())
    assert manifest["panel"]["problems"] == {"2": ["0.1/0.5", "0.9/0.5"]}


def test_level_three_cells_cross_the_nested_beliefs(tmp_path):
    cells = [("mcts", 3, 1, 20, "exact/exact", 0, "exact", 0)]
    summary, failures = run_panel(
        tmp_path / "panel",
        cells,
        seeds=[0],
        workers=1,
        problems=partial(
            tiger_problems, beliefs=(0.2,), opponent_beliefs=(0.3, 0.7), nested_beliefs=(0.4, 0.6)
        ),
    )
    assert failures == []
    assert summary.iloc[0]["n"] == 4
    assert summary.iloc[0]["optimal_rate"] == 1.0  # horizon 1 is exact
    rows = json.loads((tmp_path / "panel" / "rows.json").read_text())
    assert {(r["opponent_belief"], *r["nested_beliefs"]) for r in rows} == {
        (0.3, 0.4),
        (0.3, 0.6),
        (0.7, 0.4),
        (0.7, 0.6),
    }
    assert {r["opponent"]["level"] for r in rows} == {2}


def test_uav_problems_cross_named_laws_per_level():
    physics = UAVModel()
    laws = uav_laws(physics)
    assert sorted(laws) == ["far-corner", "near-start", "top-row", "uniform"]
    for law in laws.values():
        assert sum(law.values()) == pytest.approx(1.0)
        assert min(law.values()) > 0
    assert laws["far-corner"][(0, 2)] == pytest.approx(0.75 + 0.25 / 9)
    assert len(uav_problems(physics, 1)) == 4
    level_two = uav_problems(physics, 2)
    assert len(level_two) == 16
    assert {p.name for p in level_two} >= {"uniform/uniform", "top-row/far-corner"}
    assert all(p.case["problem"] == p.name for p in level_two)


def test_tiger_level_one_problems_are_beliefs_alone():
    assert [p.case for p in tiger_problems(1, beliefs=(0.2, 0.8))] == [
        dict(belief_p=0.2),
        dict(belief_p=0.8),
    ]


def test_uav_cells_are_scored_at_levels_one_and_two(tmp_path):
    physics = UAVModel()
    uniform = {"uniform": uav_laws(physics)["uniform"]}
    cells = [
        ("mcts", 1, 2, 200, "exact", 0, "exact", 0),
        ("mcts", 2, 2, 200, "exact", 0, "exact", 0),
    ]
    summary, failures = run_panel(
        tmp_path / "panel",
        cells,
        seeds=[0],
        workers=2,
        problems=partial(uav_problems, physics, laws=uniform),
        physics=physics,
    )
    assert failures == []
    assert list(summary["n"]) == [1, 1]
    # Horizon 2 with chance scoring is exact for the searched model (exact opponents).
    assert list(summary["optimal_rate"]) == [1.0, 1.0]
    rows = json.loads((tmp_path / "panel" / "rows.json").read_text())
    assert {r["problem"] for r in rows} == {"uniform", "uniform/uniform"}
    assert all(r["bounds_contain_reference"] for r in rows)
    manifest = json.loads((tmp_path / "panel" / "manifest.json").read_text())
    assert manifest["panel"]["problems"] == {"1": ["uniform"], "2": ["uniform/uniform"]}
    assert manifest["panel"]["physics"][1] == "UAVModel"


def test_uav_cells_take_a_history_of_listening(tmp_path):
    physics = UAVModel()
    uniform = {"uniform": uav_laws(physics)["uniform"]}
    cells = [("mcts", 2, 2, 200, "exact", 0, "exact", 2)]
    summary, failures = run_panel(
        tmp_path / "panel",
        cells,
        seeds=[0],
        workers=1,
        problems=partial(uav_problems, physics, laws=uniform),
        physics=physics,
    )
    assert failures == []
    assert list(summary["optimal_rate"]) == [1.0]
    # Tiger's scripted history (listen-left, ...) is not a UAV action.
    rows = json.loads((tmp_path / "panel" / "rows.json").read_text())
    assert rows[0]["history_steps"] == 2


def test_level_one_takes_no_modeled_budget():
    with pytest.raises(ValueError, match="modeled label"):
        evaluate_cell("mcts", 1, 2, 10, "100", 0, "exact", 0, seed=0)


def test_summary_counts_epsilon_correct_decisions_and_bounds_their_failure_rate():
    cell = dict(zip(CELL, ("mcts", 3, 4, 1000, "exact", 0, "exact", 0)))
    losses = [0.0] * 76 + [0.03] * 4
    rows = [
        dict(
            cell,
            first_action_loss=loss,
            max_abs_q_error=0.0,
            optimal_q_error=0.0,
            planner_cpu_seconds=1.0,
            history_update_cpu_seconds=0.0,
            belief_support=2,
        )
        for loss in losses
    ]
    row = summarize(rows).iloc[0]
    assert row["optimal_rate"] == 76 / 80
    # Every loss is within epsilon = 0.05, and 0/80 failures bounds delta at
    # 1 - 0.025 ** (1 / 80), the two-sided exact 95% interval's upper end.
    assert row["eps_correct_rate"] == 1.0
    assert row["eps_failure_high"] == pytest.approx(1 - 0.025 ** (1 / 80))
