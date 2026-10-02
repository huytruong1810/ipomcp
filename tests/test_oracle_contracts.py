"""Independent boundary checks for reference solver and matched experiment setup."""

import pytest

from ipomdp.domains.tiger import TigerModel
from ipomdp.experiments.oracle import evaluate_case
from ipomdp.reference.alpha_vector import AlphaVector2D, prune_2d
from ipomdp.reference.pomdp_exact_vi import ExactPOMDPSolver


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
    row = evaluate_case(kind, 1, 100, 17, 0.98, gamma=0.3)[0]
    assert row["oracle_q"] == pytest.approx({"L": -1, "OL": -97.8, "OR": 7.8})
    assert row["first_action_loss"] == pytest.approx(0)
    assert row["horizon"] == 1 and row["gamma"] == 0.3


def test_exploration_constant_is_recorded_and_preserves_one_step_problem():
    row = evaluate_case("mcts", 1, 100, 17, 0.98, gamma=0.3, exploration=0.1)[0]
    assert row["estimated_q"] == pytest.approx(row["oracle_q"])
    assert row["planner_config"]["exploration"] == 0.1


PANEL = dict(planners=("mcts",), horizons=(1,), budgets=(3,), beliefs=(0.5,))


def test_oracle_panel_records_and_executes_requested_seed_range(tmp_path, monkeypatch):
    import json

    from ipomdp.experiments import oracle as experiment

    observed = []

    def inline_supervisor(jobs, **kwargs):
        for identifier, job in jobs.items():
            rows = job()
            observed.append(rows[0]["seed"])
            yield identifier, {"status": "complete", "rows": rows, "wall_seconds": 0.0}

    monkeypatch.setattr(experiment, "supervise_jobs", inline_supervisor)
    experiment.run_oracle_comparison(tmp_path, seeds=2, seed_start=100, **PANEL)
    assert observed == [100, 101]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["settings"]["seed_start"] == 100
    assert manifest["settings"]["seeds"] == 2
    for identifier, seed in enumerate(observed):
        case = json.loads((tmp_path / f"case-{identifier}.json").read_text())
        assert case["case"][3] == case["rows"][0]["seed"] == seed


@pytest.mark.parametrize("seed_start", [-1, 0.5, True])
def test_oracle_panel_rejects_invalid_seed_start_before_output(tmp_path, seed_start):
    from ipomdp.experiments.oracle import run_oracle_comparison

    out = tmp_path / "invalid"
    with pytest.raises(ValueError, match="first seed"):
        run_oracle_comparison(out, seeds=1, seed_start=seed_start, **PANEL)
    assert not out.exists()


def test_panel_preserves_summary_when_supervision_raises(tmp_path, monkeypatch):
    import json

    from ipomdp.experiments import oracle as experiment

    def interrupted(jobs, **kwargs):
        raise RuntimeError("injected parent failure")
        yield

    monkeypatch.setattr(experiment, "supervise_jobs", interrupted)
    with pytest.raises(RuntimeError, match="injected parent failure"):
        experiment.run_oracle_comparison(tmp_path, seeds=1, **PANEL)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["cases"] == []
