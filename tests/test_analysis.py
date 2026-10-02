"""Analysis layer: hierarchies with content and error, long-format metrics, figures."""

import json

import pytest

from ipomdp.analysis import figures, metrics
from ipomdp.analysis.hierarchy import hierarchy_error, level_posterior, nested_hierarchy
from ipomdp.domains.tiger import TigerModel
from ipomdp.experiments.bootstrapper import HierarchyBuilder
from ipomdp.experiments.runner import run_study
from ipomdp.experiments.study import AgentSpec, StudySpec
from ipomdp.planning.config import PlannerConfig
from ipomdp.planning.solver_bank import SolverBank

NESTED = {3: {0: 1 / 3, 1: 1 / 3, 2: 1 / 3}, 2: {0: 0.5, 1: 0.5}, 1: {0: 1.0}}


def l3_agent(**bank_options):
    config = PlannerConfig(simulations=20, depth=2, modeled_simulations=10)
    return HierarchyBuilder(SolverBank(0, **bank_options)).build(
        "i", 3, TigerModel(), "j", config=config, nested_level_weights=NESTED
    )


def test_hierarchy_masses_partition_every_depth_and_carry_content():
    planner = l3_agent()
    nodes = nested_hierarchy(planner.belief, "i", 3)
    assert nodes[0].path == "i_L3" and nodes[0].mass == 1.0
    for depth in range(1, 4):
        ring = [n for n in nodes if n.depth == depth]
        parents = {n.parent for n in ring}
        for parent in parents:
            children = [n for n in ring if n.parent == parent]
            parent_mass = next(n.mass for n in nodes if n.path == parent)
            assert sum(c.mass for c in children) == pytest.approx(parent_mass)
    assert {n.path for n in nodes if n.depth == 1} == {"i_L3/j_L0", "i_L3/j_L1", "i_L3/j_L2"}
    l1 = next(n for n in nodes if n.path == "i_L3/j_L1")
    assert l1.physical == pytest.approx({"TL": 0.5, "TR": 0.5})
    assert next(n for n in nodes if n.path == "i_L3/j_L0").physical is None
    assert level_posterior(planner.belief) == pytest.approx({0: 1 / 3, 1: 1 / 3, 2: 1 / 3})


def test_hierarchy_error_is_zero_for_equal_and_positive_for_different_beliefs():
    exact, approximate = l3_agent(), l3_agent(particles=8)
    for planner in (exact, approximate):
        planner.update_root("L", ("GL", "S"))
    reference = nested_hierarchy(exact.belief, "i", 3)
    assert all(row["mass_tv"] == 0 for row in hierarchy_error(reference, reference).values())
    error = hierarchy_error(nested_hierarchy(approximate.belief, "i", 3), reference)
    assert error[1]["mass_tv"] + error[1]["physical_tv"] > 0


@pytest.fixture(scope="module")
def episodes(tmp_path_factory):
    planner = {"kind": "ipomcp", "simulations": 20, "depth": 2, "modeled_simulations": 20}
    spec = StudySpec(
        name="l1-vs-random",
        domain="tiger",
        trials=3,
        steps=3,
        agents={"i": AgentSpec(level=1, planner=planner), "j": AgentSpec(level=0)},
        record_hierarchies=True,
    )
    return run_study(spec, tmp_path_factory.mktemp("runs"), workers=3)


def test_metrics_tables_are_long_and_complete(episodes):
    rewards = metrics.rewards(episodes)
    assert set(rewards["agent"]) == {"i", "j"} and len(rewards) == 2 * len(episodes)
    costs = metrics.costs(episodes)
    assert set(costs["component"]) == {"planning", "update"}
    assert (costs[costs["agent"] == "i"]["cpu_seconds"] > 0).any()
    levels = metrics.level_posteriors(episodes)
    assert set(levels["opponent"]) == {"j"} and set(levels["level"]) == {0}
    summary = metrics.bootstrap_mean(rewards, "cumulative_reward", ("agent", "step"))
    assert (summary["low"] <= summary["mean"]).all() and (summary["mean"] <= summary["high"]).all()
    assert set(summary["n_trials"]) == {3}


def test_figures_render(episodes, tmp_path):
    figures.cumulative_rewards(episodes, tmp_path / "rewards.pdf")
    figures.decision_costs(episodes, tmp_path / "costs.pdf")
    figures.level_posteriors(episodes, tmp_path / "levels.pdf", "j")
    for name in ("rewards.pdf", "costs.pdf", "levels.pdf"):
        assert (tmp_path / name).stat().st_size > 0


def test_sunbursts_keep_slice_positions_and_show_error(tmp_path):
    planner = l3_agent()
    before = nested_hierarchy(planner.belief, "i", 3)
    planner.update_root("L", ("GL", "CL"))
    after = nested_hierarchy(planner.belief, "i", 3)
    error = {0: hierarchy_error(before, before), 1: hierarchy_error(after, before)}
    fig = figures.sunburst_over_time(
        {0: before, 1: after}, tmp_path / "over_time.html", "test", error
    )
    assert fig.frames[0].data[0].ids == fig.frames[1].data[0].ids
    assert fig.data[0].sort is False
    assert any("error" in text for text in fig.frames[1].data[0].hovertext)
    figures.sunburst(after, tmp_path / "one.html", "test")
    assert (tmp_path / "one.html").stat().st_size > 0


def test_episodes_record_hierarchies_and_both_agents_costs(episodes):
    first = json.loads(episodes["hierarchy_i"].iloc[0])
    assert first[0]["path"] == "i_L1" and first[0]["mass"] == 1.0
    assert {"planning_cpu_j", "update_cpu_j", "physical_i"} <= set(episodes.columns)
