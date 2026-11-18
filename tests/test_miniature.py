"""The miniature campaign runs the whole pipeline cleanly, and its checker notices a wrong row."""

import json

import pytest

from ipomdp.experiments import miniature
from ipomdp.infra.job_queue import Job


@pytest.mark.slow
def test_the_miniature_campaign_runs_end_to_end_without_a_failure(tmp_path):
    summary = miniature.run(tmp_path / "mini", workers=4)
    assert summary["failures"] == []
    assert summary["ran"] == summary["jobs"] == 90
    assert set(summary["decisions"]) == {"ipomcp-sample", "ipomcp-gap", "ipf-rts", "intmcp"}
    assert all(arm["n"] == 18 for arm in summary["decisions"].values())
    assert summary["episode_cells"] == 9 and summary["one_level_comparisons"] > 0
    assert (tmp_path / "mini" / "rewards_i2_j1.png").stat().st_size > 0
    # A second run finds every output and starts nothing.
    assert miniature.run(tmp_path / "mini", workers=4)["ran"] == 0


def test_the_checker_reports_intervals_that_miss_and_settled_stages_that_lose(tmp_path):
    reference = tmp_path / "ref.json"
    reference.write_text(json.dumps({"q": {"L": 1.0, "OL": -5.0, "OR": 2.0}}))
    stages = [
        # Settled on OR, the exact maximizer, and contains every value: fine.
        dict(budget=10, loss=0.0, bounds={"L": [0.5, 1.5], "OL": [-6, -4], "OR": [2.0, 2.0]}),
        # Misses OL's exact value (-5 is outside [-4, -3]).
        dict(budget=30, loss=0.0, bounds={"L": [0.5, 1.5], "OL": [-4, -3], "OR": [2.0, 2.0]}),
        # Settled (OR certain) but the stage's decision lost 1.
        dict(budget=100, loss=1.0, bounds={"L": [0.5, 1.5], "OL": [-6, -4], "OR": [2.0, 2.0]}),
    ]
    row = tmp_path / "row.json"
    row.write_text(json.dumps(dict(level=2, horizon=2, problem_seed=0, steps=0, loss=0.0, ladder=stages)))
    baseline = tmp_path / "base.json"
    baseline.write_text(json.dumps(dict(level=2, horizon=2, problem_seed=0, steps=0, loss=None)))
    jobs = [Job("decide:ipomcp-gap", str(row), "true", needs=(str(reference),)),
            Job("decide:ipf-rts", str(baseline), "true", needs=(str(reference),)),
            Job("decide:intmcp", str(tmp_path / "missing.json"), "true", needs=(str(reference),))]
    _, failures = miniature.check_decisions(tmp_path, jobs)
    assert any("stage 30 intervals miss" in f for f in failures)
    assert any("stage 100 is settled but loses 1.0" in f for f in failures)
    assert any("base.json: no first-action loss" in f for f in failures)
    assert any("missing row" in f for f in failures)
    assert not any("stage 10 " in f for f in failures)  # the trailing space keeps "stage 100" out
