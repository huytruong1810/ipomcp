"""The panel: one supervised job per (cell, seed), scored against the exact I-POMDP."""

import json

from ipomdp.experiments.panel import BELIEFS, OPPONENT_BELIEFS, run_panel


def test_panel_rows_cover_every_problem_and_name_their_cell(tmp_path):
    cells = [("rts", 2, 2, 2, 2, 16, "sampled", 0), ("mcts", 2, 2, 30, 0, 0, "exact", 2)]
    summary, failures = run_panel(tmp_path / "panel", cells, seeds=[0], workers=2)
    assert failures == []
    assert list(summary["n"]) == [len(BELIEFS) * len(OPPONENT_BELIEFS)] * 2
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
        modeled=2,
        particles=16,
        particle_filter="sampled",
        history=0,
    )
    assert {r["history_steps"] for r in rows if r["planner"] == "mcts"} == {2}
    assert {r["opponent"]["policy"] for r in rows if r["planner"] == "mcts"} == {
        "exact_fixed_depth"
    }


def test_panel_takes_a_belief_grid_and_bounds_each_optimal_rate(tmp_path):
    cells = [("mcts", 2, 1, 20, 0, 0, "exact", 0)]
    summary, failures = run_panel(
        tmp_path / "panel",
        cells,
        seeds=[0, 1],
        workers=1,
        beliefs=(0.1, 0.9),
        opponent_beliefs=(0.5,),
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
    assert manifest["panel"]["beliefs"] == [0.1, 0.9]
    assert manifest["panel"]["opponent_beliefs"] == [0.5]


def test_level_three_cells_cross_the_nested_beliefs(tmp_path):
    cells = [("mcts", 3, 1, 20, 0, 0, "exact", 0)]
    summary, failures = run_panel(
        tmp_path / "panel",
        cells,
        seeds=[0],
        workers=1,
        beliefs=(0.2,),
        opponent_beliefs=(0.3, 0.7),
        nested_beliefs=(0.4, 0.6),
    )
    assert failures == []
    assert summary.iloc[0]["n"] == 4
    assert summary.iloc[0]["optimal_rate"] == 1.0  # horizon 1 is exact
    rows = json.loads((tmp_path / "panel" / "rows.json").read_text())
    assert {(r["opponent_belief"], r["nested_belief"]) for r in rows} == {
        (0.3, 0.4),
        (0.3, 0.6),
        (0.7, 0.4),
        (0.7, 0.6),
    }
    assert {r["opponent"]["level"] for r in rows} == {2}
