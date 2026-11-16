"""Study specifications: per-level modeled solves (experiments.study, experiments.modeled_levels)."""

import pytest

from ipomdp.experiments.study import AgentSpec, StudySpec, build_agents
from ipomdp.planning.solver_types import SolverKey
from ipomdp.planning.stopping import ModeledStopping

PLANNER = {"kind": "ipomcp", "simulations": 50, "depth": 2, "modeled_simulations": 50}


def spec(modeled, level=3):
    return StudySpec(
        name="stack",
        domain="tiger",
        trials=1,
        steps=1,
        agents={
            "i": AgentSpec(level=level, planner=PLANNER, modeled=modeled, particles=16),
            "j": AgentSpec(level=0),
        },
    )


def test_a_stack_label_sets_each_modeled_level_for_both_agents():
    _, planners = build_agents(spec("400s/30"), seed=0)
    bank = planners["i"].solver_bank
    for agent in ("i", "j"):
        two = bank.get_solver(SolverKey(agent, 2)).config
        one = bank.get_solver(SolverKey(agent, 1)).config
        assert (two.modeled_simulations, two.modeled_stopping) == (400, ModeledStopping())
        assert (one.modeled_simulations, one.modeled_stopping) == (30, None)
        assert two.depth == one.depth == PLANNER["depth"]
    # The real agent keeps its own planner.
    assert planners["i"].config.simulations == PLANNER["simulations"]


def test_a_stack_label_loads_from_toml(tmp_path):
    path = tmp_path / "study.toml"
    path.write_text(
        'name = "s"\ndomain = "tiger"\ntrials = 1\nsteps = 1\n'
        "[agents.i]\nlevel = 2\n"
        'planner = { kind = "ipomcp", simulations = 50, depth = 2, modeled_simulations = 50 }\n'
        'modeled = "6400s"\n'
        "[agents.j]\nlevel = 0\n"
    )
    assert StudySpec.from_toml(path).agents["i"].modeled == "6400s"


@pytest.mark.parametrize(
    "modeled, level, message",
    [("400", 1, "models no intentional level"), ("exact/30", 3, "Exact modeled levels")],
)
def test_invalid_modeled_stacks_are_rejected(modeled, level, message):
    with pytest.raises(ValueError, match=message):
        spec(modeled, level)


def test_episode_rows_record_each_decision_and_its_bounds():
    # Rows are written after update_root, which clears a planner's statistics;
    # the decision must be captured when it is made.
    import json

    from ipomdp.experiments.episodes import run_episode

    study = StudySpec(
        name="rows",
        domain="tiger",
        trials=1,
        steps=2,
        agents={"i": AgentSpec(level=1, planner=PLANNER), "j": AgentSpec(level=0)},
    )
    rows = run_episode(study, 0)
    for row in rows[1:]:
        values, bounds = json.loads(row["action_values_i"]), json.loads(row["bounds_i"])
        assert set(values) == set(bounds) == {"L", "OL", "OR"}
        assert all(lo <= hi for lo, hi in bounds.values())
        # i-L1 faces uniform L0 only, so it predicts uniform play.
        prediction = json.loads(row["prediction_i"])
        assert prediction == pytest.approx(dict.fromkeys(["L", "OL", "OR"], 1 / 3))
        assert json.loads(row["prediction_j"]) is None


def test_a_study_planner_can_carry_the_certified_profile():
    from ipomdp.experiments.study import _planner_config
    from ipomdp.planning.stopping import ModeledResolution

    config = _planner_config({
        "kind": "ipomcp", "simulations": 1000, "depth": 3, "modeled_simulations": 6400,
        "frontier_bounds": "ifib", "scoring": "pw", "exact_tail": True, "prune": True,
        "leaf_eval": "bound", "lazy_expansion": True, "selection": "iemax",
        "modeled_resolution": {"pilot": 50}, "modeled_prediction": "plausible",
    })
    assert config.certified_search and config.modeled_resolution == ModeledResolution(pilot=50)


# Planner profiles (experiments.profile): one definition for `decide` and for study specs.

PROFILED = {"kind": "ipomcp", "simulations": 50, "depth": 2, "modeled_simulations": 30, "profile": "certified"}


def _decide_arguments(**changes):
    """The `decide` flags at their defaults under --profile certified, with a modeled budget of 30."""
    from types import SimpleNamespace

    flags = dict(frontier_bounds="scalar", scoring="legacy", unsampled="lower", exact_tail=False, prune=False,
                 leaf_eval="rollout", lazy=False, selection=None, profile="certified", modeled_exact=False,
                 modeled=30, budget=50, modeled_resolve=False, modeled_pilot=50, modeled_prediction="argmax")
    return SimpleNamespace(**flags | changes)


@pytest.mark.parametrize(
    "level, particles, modeled, selection",
    [(2, 16, "30", None), (3, 16, "30/30", None), (3, None, None, None), (2, 16, "30", "ucb")],
)
def test_a_profile_study_builds_the_configurations_decide_builds(level, particles, modeled, selection):
    from ipomdp.experiments.reachable_decisions import _planner, certified_options, modeled_options
    from ipomdp.planning.i_pomcp import seed_config

    arguments = _decide_arguments(selection=selection)
    study = StudySpec(
        name="profile",
        domain="tiger",
        trials=1,
        steps=1,
        agents={
            "i": AgentSpec(level=level, planner=PROFILED | ({"selection": selection} if selection else {}),
                           modeled=modeled, particles=particles),
            "j": AgentSpec(level=0),
        },
    )
    decided = _planner(study.physics(), "mcts", 0, 50, 2, level, 30, particles or 0, "compressed",
                       certified=certified_options(arguments), modeled_options=modeled_options(arguments))
    _, planners = build_agents(study, seed=0)
    assert planners["i"].config == decided.config
    assert seed_config(planners["i"].config) == seed_config(decided.config)
    assert decided.config.selection == (selection or "iemax") and decided.config.modeled_resolution is None
    for agent in ("i", "j"):
        for lower in range(1, level):
            key = SolverKey(agent, lower)
            ours = planners["i"].solver_bank.get_solver(key).config
            theirs = decided.solver_bank.get_solver(key).config
            assert ours == theirs and seed_config(ours) == seed_config(theirs)
            # Resolved to the budget as the cap (pilot min(50, 30)), predicted by the plausible set.
            assert (ours.simulations, ours.modeled_resolution.pilot, ours.modeled_prediction) == (30, 30, "plausible")
            assert ours.certified_search and ours.selection == (selection or "iemax")


def test_without_a_profile_every_builder_gives_the_default_configuration():
    from ipomdp.experiments.modeled_levels import planner_config
    from ipomdp.experiments.study import _modeled_configs
    from ipomdp.experiments.suite import anytime_comparison, ipomcp, level_matrix, prior_sensitivity
    from ipomdp.planning.config import PlannerConfig

    assert ipomcp(100, 2) == {"kind": "ipomcp", "simulations": 100, "depth": 2, "modeled_simulations": 100}
    suites = (level_matrix("tiger", pairs=[(3, 2)]), prior_sensitivity("tiger", 3, 1),
              anytime_comparison("tiger", intmcp_simulations=(), rts_samples=()))
    for suite in suites:
        for cell in suite.cells.values():
            for agent in cell.agents.values():
                assert agent.level == 0 or "profile" not in agent.planner
    agent = level_matrix("tiger", pairs=[(3, 2)]).cells["i3_j2"].agents["i"]
    flat = planner_config("mcts", 100, 2, 0.95, 1.0)
    assert flat == PlannerConfig(simulations=100, depth=2, modeled_simulations=100)
    assert _modeled_configs(agent) == {2: flat, 1: flat}
    assert _modeled_configs(AgentSpec(level=3, planner=PLANNER)) is None


def test_the_suite_builders_pass_a_profile_to_every_agent_and_modeled_level():
    from ipomdp.experiments.suite import anytime_comparison, level_matrix, prior_sensitivity

    suites = (level_matrix("tiger", pairs=[(3, 2)], profile="certified"),
              prior_sensitivity("tiger", 3, 1, profile="certified"),
              anytime_comparison("tiger", intmcp_simulations=(), rts_samples=(), profile="certified"))
    for suite in suites:
        for cell in suite.cells.values():
            assert all(a.planner["profile"] == "certified" for a in cell.agents.values() if a.level)
    _, planners = build_agents(suites[0].cells["i3_j2"], seed=0)
    for agent in ("i", "j"):
        assert planners[agent].config.selection == "iemax" and planners[agent].config.modeled_resolution is None
    bank = planners["i"].solver_bank
    for lower in (1, 2):
        config = bank.get_solver(SolverKey("j", lower)).config
        assert config.modeled_resolution.pilot == 50 and config.simulations == 100


@pytest.mark.parametrize(
    "planner, modeled, message",
    [
        (PROFILED | {"prune": True}, "30", "The profile sets prune"),
        (PROFILED | {"modeled_prediction": "argmax"}, "30", "The profile sets modeled_prediction"),
        (PROFILED, "400s", "drop the stopping rule"),
        (PROFILED | {"profile": "fast"}, "30", "Unknown profile"),
    ],
)
def test_a_profile_refuses_settings_it_owns(planner, modeled, message):
    with pytest.raises(ValueError, match=message):
        AgentSpec(level=2, planner=planner, modeled=modeled, particles=16)


def test_an_episode_runs_under_the_certified_profile():
    import json

    from ipomdp.experiments.episodes import run_episode
    from ipomdp.experiments.suite import level_matrix

    study = level_matrix("tiger", pairs=[(2, 1)], steps=3, simulations=30, particles=16,
                         profile="certified").cells["i2_j1"]
    rows = run_episode(study, 0)
    assert len(rows) == 4
    for row in rows[1:]:
        for agent in ("i", "j"):
            assert row[f"action_{agent}"] in {"L", "OL", "OR"}
            bounds = json.loads(row[f"bounds_{agent}"])
            assert set(bounds) == {"L", "OL", "OR"} and all(lo <= hi + 1e-9 for lo, hi in bounds.values())
    # Deterministic given the trial: a second run decides alike (clock and memory fields aside).
    decided = [[row[f"{field}_{agent}"] for agent in "ij" for field in ("action", "action_values", "bounds")]
               for row in rows]
    again = [[row[f"{field}_{agent}"] for agent in "ij" for field in ("action", "action_values", "bounds")]
             for row in run_episode(study, 0)]
    assert decided == again


def test_a_profile_study_passes_the_gap_options_to_every_modeled_level_as_decide_does():
    from ipomdp.experiments.reachable_decisions import _planner, certified_options, modeled_options

    arguments = _decide_arguments(outcome="gap", root_rule="open")
    study = StudySpec(
        name="profile-gap",
        domain="tiger",
        trials=1,
        steps=1,
        agents={
            "i": AgentSpec(level=3, planner=PROFILED | {"outcome": "gap", "root_rule": "open"}, modeled="30/30",
                           particles=16),
            "j": AgentSpec(level=0),
        },
    )
    decided = _planner(study.physics(), "mcts", 0, 50, 2, 3, 30, 16, "compressed",
                       certified=certified_options(arguments), modeled_options=modeled_options(arguments))
    _, planners = build_agents(study, seed=0)
    assert planners["i"].config == decided.config and planners["i"].config.outcome == "gap"
    for agent in ("i", "j"):
        for lower in (1, 2):
            key = SolverKey(agent, lower)
            ours = planners["i"].solver_bank.get_solver(key).config
            assert ours == decided.solver_bank.get_solver(key).config
            assert (ours.outcome, ours.root_rule) == ("gap", "open")
