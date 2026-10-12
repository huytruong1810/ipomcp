"""Experiment suites: builders state the pre-registered designs; the runner pairs and resumes."""

from functools import partial

import pandas as pd
import pytest

from ipomdp.analysis.endpoints import (
    arm_differences,
    cell_table,
    one_level_endpoint,
    paired_difference,
)
from ipomdp.analysis.statistics import holm
from ipomdp.experiments.study import build_agents
from ipomdp.experiments.suite import (
    SuiteSpec,
    level_matrix,
    load_suite,
    planner_comparison,
    prior_arms,
    prior_sensitivity,
    run_suite,
)
from tests.study_fixtures import random_tiger, recording_episode


def test_level_matrix_agents_hold_uniform_priors_at_every_nesting():
    spec = level_matrix("tiger-joint", (0, 1, 3), trials=5)
    assert len(spec.cells) == 9
    cell = spec.cells["i3_j0"]
    i, j = cell.agents["i"], cell.agents["j"]
    assert j.level == 0 and j.planner is None
    assert i.level_weights == {0: 1 / 3, 1: 1 / 3, 2: 1 / 3}
    assert i.nested_level_weights == {1: {0: 1.0}, 2: {0: 0.5, 1: 0.5}}
    assert i.modeled == "100" and i.particles == 64 and i.planner["depth"] == 2
    assert spec.cells["i1_j1"].agents["i"].level_weights == {0: 1.0}
    # A modeled opponent at level l is the matrix's real agent at l.
    assert level_matrix(levels=(2, 3)).cells["i3_j2"].agents["j"].nested_level_weights == {
        1: {0: 1.0}
    }
    assert list(spec.cells)[0] == "i3_j3"  # expensive cells first in each round


def test_planner_comparison_models_the_real_opponent_with_its_own_solve():
    spec = planner_comparison("tiger", depth=3, opponent_simulations=1000, rts_samples=(2,))
    assert set(spec.cells) == {"ipomcp", "rts-k2"}
    ipomcp_i = spec.cells["ipomcp"].agents["i"]
    j = spec.cells["ipomcp"].agents["j"]
    assert ipomcp_i.modeled == str(j.planner["simulations"])
    assert ipomcp_i.planner["depth"] == j.planner["depth"] == 3
    rts_i = spec.cells["rts-k2"].agents["i"]
    assert rts_i.planner["kind"] == "rts" and rts_i.particle_filter == "sampled"
    assert spec.cells["rts-k2"].agents["j"] == j
    build_agents(spec.cells["rts-k2"], 0)


@pytest.mark.parametrize(
    "level, opponent, arms",
    [
        (3, 1, {"correct": {1: 1.0}, "over": {2: 1.0}, "under": {0: 1.0}}),
        (4, 2, {"correct": {2: 1.0}, "over": {3: 1.0}, "under": {1: 1.0}}),
        (2, 0, {"correct": {0: 1.0}, "over": {1: 1.0}}),
    ],
)
def test_prior_arms_misspecify_in_both_directions_where_possible(level, opponent, arms):
    expected = dict(arms, uniform={k: 1 / level for k in range(level)})
    assert prior_arms(level, opponent) == expected


def test_prior_sensitivity_uniform_arm_is_the_matrix_cell():
    arm = prior_sensitivity("tiger-joint", 3, 1, domain_options={"together_safe": 40.0})
    cell = level_matrix("tiger-joint", (1, 3), domain_options={"together_safe": 40.0})
    assert arm.cells["uniform"].agents == cell.cells["i3_j1"].agents
    assert arm.cells["correct"].agents["j"] == cell.cells["i3_j1"].agents["j"]


def _two_cells(trials):
    return SuiteSpec(
        name="test",
        cells={"a": random_tiger(steps=3), "b": random_tiger(steps=3)},
        trials=trials,
    )


def test_suite_runs_round_robin_and_resumes(tmp_path):
    log = tmp_path / "order.log"
    output = tmp_path / "run"
    cells = run_suite(_two_cells(2), output, workers=1, episode=partial(recording_episode, log))
    assert log.read_text().split("\n")[:-1] == [
        "random-tiger 0",
        "random-tiger 0",
        "random-tiger 1",
        "random-tiger 1",
    ]
    assert {label: frame["trial"].nunique() for label, frame in cells.items()} == {"a": 2, "b": 2}
    # Cells share trial ids, so equal configurations give equal episodes.
    pd.testing.assert_frame_equal(
        cells["a"].drop(columns=_timing(cells["a"])), cells["b"].drop(columns=_timing(cells["b"]))
    )
    # Growing ``trials`` resumes in place and runs only the new trial ids.
    log.unlink()
    run_suite(_two_cells(3), output, workers=1, episode=partial(recording_episode, log))
    assert log.read_text().split("\n")[:-1] == ["random-tiger 2", "random-tiger 2"]
    assert len(load_suite(output)["a"]) == 3 * 4
    changed = SuiteSpec(name="test", cells={"a": random_tiger(steps=4)}, trials=1)
    with pytest.raises(ValueError, match="different suite"):
        run_suite(changed, output, workers=1)


def _timing(frame):
    return [c for c in frame if c.startswith(("planning_", "update_", "process_rss"))]


def test_holm_matches_a_hand_computation():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def _frame(rewards_i, rewards_j):
    rows = []
    for trial, (ri, rj) in enumerate(zip(rewards_i, rewards_j)):
        for step in (0, 1):
            rows.append(
                {
                    "trial": trial,
                    "step": step,
                    "cum_reward_i": ri * step,
                    "cum_reward_j": rj * step,
                    "planning_cpu_i": 0.5,
                    "planning_cpu_j": 0.25,
                }
            )
    return pd.DataFrame(rows)


def test_differences_pair_by_trial():
    a = pd.Series([3.0, 5.0, 7.0], index=[0, 1, 2])
    b = pd.Series([1.0, 4.0, 4.0, 100.0], index=[0, 1, 2, 3])
    result = paired_difference(a, b)
    assert result["n"] == 3 and result["mean"] == pytest.approx(2.0)


def test_one_level_endpoint_compares_each_agent_against_the_same_opponent():
    cells = {
        (0, 0): _frame([0, 0, 0], [0, 0, 0]),
        (1, 0): _frame([5, 6, 7], [1, 1, 1]),
        (0, 1): _frame([0, 0, 0], [2, 3, 4]),
    }
    table = one_level_endpoint(cells).set_index(["agent", "opponent"])
    assert table.loc[("i", 0), "mean"] == pytest.approx(6.0)
    assert table.loc[("j", 0), "mean"] == pytest.approx(3.0)
    assert len(table) == 2 and (table["p_holm"] >= table["p"]).all()
    arms = arm_differences({"x": cells[(1, 0)], "ref": cells[(0, 0)]}, "ref")
    assert arms.loc[0, "mean"] == pytest.approx(6.0)
    summary = cell_table({"c": cells[(1, 0)]})
    assert summary.loc[0, "welfare"] == pytest.approx(7.0)
    assert summary.loc[0, "cpu_per_step"] == pytest.approx(0.75)
