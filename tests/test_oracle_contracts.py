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


@pytest.mark.parametrize("strategy", ["normalized", "standard", "bounded"])
def test_exploration_ablation_preserves_one_step_problem_and_records_settings(strategy):
    row = evaluate_case("mcts", 1, 100, 17, 0.98, 0.3, strategy, 0.1)[0]
    assert row["estimated_q"] == pytest.approx(row["oracle_q"])
    assert row["first_action_loss"] == pytest.approx(0)
    assert row["exploration"]["strategy"] == strategy
    assert row["exploration"]["c"] == 0.1
    if strategy == "bounded":
        assert row["exploration"]["reward_min"] == -100
        assert row["exploration"]["reward_max"] == 10


def test_oracle_panel_records_and_executes_requested_seed_range(tmp_path, monkeypatch):
    import json

    from examples.experiments import planner_oracle_experiment as experiment

    observed = []

    def inline_supervisor(jobs, **kwargs):
        for identifier, job in jobs.items():
            rows = job()
            observed.append(rows[0]["seed"])
            yield identifier, {"status": "complete", "rows": rows, "wall_seconds": 0.0}

    monkeypatch.setattr(experiment, "supervise_jobs", inline_supervisor)
    experiment.run_oracle_comparison(
        tmp_path,
        planners=("mcts",),
        horizons=(1,),
        budgets=(3,),
        beliefs=(0.5,),
        seeds=2,
        seed_start=100,
    )
    assert observed == [100, 101]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["settings"]["seed_start"] == 100
    assert manifest["settings"]["seeds"] == 2
    for identifier, seed in enumerate(observed):
        case = json.loads((tmp_path / f"case-{identifier}.json").read_text())
        assert case["case"][3] == case["rows"][0]["seed"] == seed


@pytest.mark.parametrize("seed_start", [-1, 0.5, True])
def test_oracle_panel_rejects_invalid_seed_start_before_output(tmp_path, seed_start):
    from examples.experiments.planner_oracle_experiment import run_oracle_comparison

    out = tmp_path / "invalid"
    with pytest.raises(ValueError, match="seed_start"):
        run_oracle_comparison(out, seed_start=seed_start)
    assert not out.exists()


def test_panel_rejects_impossible_worker_timing_and_keeps_evidence(tmp_path, monkeypatch):
    import json

    from examples.experiments import planner_oracle_experiment as experiment

    def impossible(jobs, **kwargs):
        for identifier in jobs:
            yield identifier, {"status": "complete", "rows": [], "wall_seconds": 1e9}

    monkeypatch.setattr(experiment, "supervise_jobs", impossible)
    with pytest.raises(RuntimeError, match="concurrency bound"):
        experiment.run_oracle_comparison(
            tmp_path,
            planners=("mcts",),
            horizons=(1,),
            budgets=(3,),
            beliefs=(0.5,),
            seeds=1,
        )
    timing = json.loads((tmp_path / "timing.json").read_text())
    assert not timing["worker_concurrency_bound_satisfied"]
    assert timing["recorded_cases"] == 1
    assert (tmp_path / "case-0.json").exists()


def test_panel_preserves_timing_when_supervision_raises(tmp_path, monkeypatch):
    import json

    from examples.experiments import planner_oracle_experiment as experiment

    def interrupted(jobs, **kwargs):
        raise RuntimeError("injected parent failure")
        yield

    monkeypatch.setattr(experiment, "supervise_jobs", interrupted)
    with pytest.raises(RuntimeError, match="injected parent failure"):
        experiment.run_oracle_comparison(
            tmp_path,
            planners=("mcts",),
            horizons=(1,),
            budgets=(3,),
            beliefs=(0.5,),
            seeds=1,
        )
    timing = json.loads((tmp_path / "timing.json").read_text())
    assert timing["recorded_cases"] == 0 and timing["requested_cases"] == 1
