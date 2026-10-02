"""The L2 panel: one supervised job per (cell, seed), scored against the exact I-POMDP."""

import json

from ipomdp.experiments.l2_panel import BELIEFS, OPPONENT_BELIEFS, run_panel


def test_panel_rows_cover_every_problem_and_name_their_cell(tmp_path):
    cells = [("rts", 2, 2, 2, 16), ("mcts", 2, 30, 10, 0)]
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
        planner="rts", horizon=2, budget=2, modeled=2, particles=16
    )
