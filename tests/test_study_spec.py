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
