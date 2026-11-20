"""The planner-options path of the oracle, the panel and the qualification drivers (experiments.profile)."""

import json
import pickle
import sys
from dataclasses import asdict, replace
from functools import partial
from types import SimpleNamespace

import pandas as pd
import pytest

from ipomdp.domains.tiger import TigerModel
from ipomdp.domains.wumpus.model import WumpusModel
from ipomdp.experiments import l1_qualification, modeled_levels, oracle, panel, profile
from ipomdp.experiments import reachable_decisions as rd
from ipomdp.experiments.modeled_levels import parse_stack, planner_config, register_levels
from ipomdp.experiments.problems import tiger_problems
from ipomdp.experiments.study import AgentSpec, _modeled_configs, _planner_config, _profiled
from ipomdp.experiments.wumpus_qualification import human
from ipomdp.planning.config import PlannerConfig
from ipomdp.planning.solver_bank import SolverBank
from ipomdp.planning.solver_types import SolverKey
from ipomdp.reference.best_response import BestResponseReference
from ipomdp.reference.exact_agent import ExactIPOMDPPlanner

CERTIFIED = profile.planner_options("certified")
LEVELS = partial(profile.level_options, "certified")
# Level 3, stack 30/20: j-L2 searches 30 simulations, the i-L1 it models 20.
LEVEL_THREE = dict(level=3, opponent_depth=2, modeled="30/20", reference="none")
J2, I1 = SolverKey("j", 2), SolverKey("i", 1)


@pytest.fixture
def built(monkeypatch):
    """{solver key: configuration} of every planner ``evaluate_case`` builds."""
    configs = {}
    original = modeled_levels.make_planner

    def recording(planner_kind, key, physics, bank, config):
        configs[key] = config
        return original(planner_kind, key, physics, bank, config)

    monkeypatch.setattr(oracle, "make_planner", recording)
    monkeypatch.setattr(modeled_levels, "make_planner", recording)
    return configs


def _timeless(rows):
    return [{k: v for k, v in row.items() if "cpu_seconds" not in k} for row in rows]


def _flat(budget):
    return planner_config("mcts", budget, 2, 0.95, 1.0)


def _decide_flags(**changes):
    """The ``decide`` flags under --profile certified (as in test_reachable_decisions)."""
    base = dict(frontier_bounds="scalar", scoring="legacy", exact_tail=False, prune=False, selection=None,
                leaf_eval="rollout", lazy=False, outcome=None, root_rule=None, profile="certified",
                profile_off=None, modeled_exact=False, modeled=400, budget=1000, modeled_resolve=False,
                modeled_pilot=50, modeled_prediction="argmax")
    return SimpleNamespace(**(base | changes))


# (a) Without options every driver builds and scores as before.


def test_without_options_evaluate_case_gives_the_same_rows_and_configurations(built):
    case = ("mcts", 2, 30, 1, 0.3)
    problem = dict(level=2, opponent_depth=2, modeled="20", reference="exact")
    rows = oracle.evaluate_case(*case, **problem)
    before = dict(built)
    again = oracle.evaluate_case(*case, **problem, planner_options=None, level_options=None)
    assert _timeless(again) == _timeless(rows)
    assert built == before == {SolverKey("i", 2): _flat(30), SolverKey("j", 1): _flat(20)}
    assert "certified" not in rows[0] and "modeled_options" not in rows[0]
    assert rows[0]["planner_config"] == asdict(_flat(30))


def test_without_options_register_levels_registers_the_same_solvers():
    def registered(**options):
        physics = TigerModel()
        bank = SolverBank(0)
        register_levels(bank, physics, "i", 3, parse_stack("exact/40", 2), 2,
                        BestResponseReference(physics, bank, 0.95), **options)
        return type(bank.get_solver(J2)), bank.get_solver(I1).config

    assert registered() == registered(level_options=None) == (ExactIPOMDPPlanner, _flat(40))


def test_without_options_the_level_one_driver_gives_the_same_rows():
    rows = l1_qualification.evaluate_cell(30, 2, (0, 1), (0.3, 0.5))
    again = l1_qualification.evaluate_cell(30, 2, (0, 1), (0.3, 0.5), planner_options=None)
    assert _timeless(again) == _timeless(rows)
    assert all("certified" not in row for row in rows)


def _wumpus_configs(solver):
    bank = solver.solver_bank
    return solver.config, bank.get_solver(SolverKey("i", 1)).config, bank.get_solver(SolverKey("j", 1)).config


def test_without_options_the_wumpus_human_has_the_same_configurations():
    physics = WumpusModel(3, 3, 1)
    plain = _wumpus_configs(human(physics, 0, 20, 2, None, level=2, modeled=10))
    again = _wumpus_configs(human(physics, 0, 20, 2, None, level=2, modeled=10, planner_options=None,
                                  level_options=None))
    lower = PlannerConfig(simulations=10, depth=2, modeled_simulations=10, gamma=0.95)
    assert plain == again == (replace(lower, simulations=20), lower, lower)


# (b) The certified profile: one definition for decide, the panel and study specs.


def test_the_certified_profile_reaches_i_and_every_modeled_level_as_the_study_builds_them(built):
    row = oracle.evaluate_case("mcts", 2, 20, 0, 0.3, **LEVEL_THREE, planner_options=CERTIFIED,
                               level_options=LEVELS)[0]
    own = PlannerConfig(simulations=20, depth=2, modeled_simulations=20, **profile.own_options("certified"))
    assert built[SolverKey("i", 3)] == own and row["planner_config"] == asdict(own)
    agent = AgentSpec(level=3, planner={"kind": "ipomcp", "simulations": 20, "depth": 2, "modeled_simulations": 20,
                                        "profile": "certified"}, modeled="30/20")
    study = _modeled_configs(agent)
    base = _planner_config(agent.planner)
    assert built[J2] == study[2] == _profiled(base, 30, "certified")
    assert built[I1] == study[1] == _profiled(base, 20, "certified")
    assert (built[J2].modeled_resolution.pilot, built[I1].modeled_resolution.pilot) == (30, 20)
    # The row records the non-default options, as decide does.
    assert row["certified"] == profile.recorded(CERTIFIED)
    assert row["modeled_options"] == [profile.recorded(LEVELS(30)), profile.recorded(LEVELS(20))]
    assert row["modeled_options"][1]["modeled_resolution"] == str(built[I1].modeled_resolution)
    json.dumps(row)


def test_decide_builds_its_options_with_the_profile_helpers():
    flags = _decide_flags(modeled=30, budget=50)
    assert rd.certified_options(flags) == CERTIFIED
    assert rd.modeled_options(flags) == LEVELS(30)
    assert list(rd.modeled_options(flags))[:2] == ["modeled_resolution", "modeled_prediction"]


def test_level_options_apply_to_finite_levels_only():
    physics = TigerModel()
    bank = SolverBank(0)
    register_levels(bank, physics, "i", 3, parse_stack("exact/40", 2), 2,
                    BestResponseReference(physics, bank, 0.95), level_options=LEVELS)
    assert type(bank.get_solver(J2)) is ExactIPOMDPPlanner
    assert bank.get_solver(I1).config == replace(_flat(40), **LEVELS(40))


def test_a_level_two_case_with_exact_modeled_levels_records_no_modeled_options(built):
    row = oracle.evaluate_case("mcts", 2, 20, 0, 0.3, level=2, opponent_depth=2, reference="none",
                               planner_options=CERTIFIED, level_options=LEVELS)[0]
    assert "modeled_options" not in row and row["certified"] == profile.recorded(CERTIFIED)
    assert set(built) == {SolverKey("i", 2)}


def test_the_level_one_driver_takes_planner_options(monkeypatch):
    planners = []

    class Recording(l1_qualification.HierarchyBuilder):
        def build(self, *args, **kwargs):
            planners.append(super().build(*args, **kwargs))
            return planners[-1]

    monkeypatch.setattr(l1_qualification, "HierarchyBuilder", Recording)
    rows = l1_qualification.evaluate_cell(30, 2, (0,), (0.5,), planner_options={"exploration": 2.0})
    configs = [planner.config for planner in planners if hasattr(planner, "config")]
    assert configs == [PlannerConfig(simulations=30, depth=2, modeled_simulations=30, exploration=2.0)]
    assert rows[0]["certified"] == {"exploration": 2.0}


def test_the_wumpus_human_takes_the_profile_as_the_study_builds_it():
    solver = human(WumpusModel(3, 3, 1), 0, 20, 2, None, level=2, modeled=10, planner_options=CERTIFIED,
                   level_options=LEVELS)
    own, lower_i, lower_j = _wumpus_configs(solver)
    base = _planner_config({"kind": "ipomcp", "simulations": 20, "depth": 2, "modeled_simulations": 10,
                            "profile": "certified"})
    assert own == base
    assert lower_i == lower_j == _profiled(base, 10, "certified")


# (c) Options are I-POMCP options.


@pytest.mark.parametrize("options", [dict(planner_options=CERTIFIED), dict(level_options=LEVELS)])
def test_options_with_another_planner_are_refused(options):
    with pytest.raises(ValueError, match="I-POMCP"):
        oracle.evaluate_case("rts", 2, 2, 0, 0.3, level=2, opponent_depth=2, modeled="2", **options)
    with pytest.raises(ValueError, match="I-POMCP"):
        human(WumpusModel(3, 3, 1), 0, 2, 2, None, planner="rts", **options)


# (d) Components switched off reach i and every modeled level, as in decide.


@pytest.mark.parametrize("off, changed", [
    (["exact_tail"], dict(exact_tail=False)),
    (["pw_scoring"], dict(scoring="legacy")),
    (["gap"], dict(outcome="sample", root_rule="upper")),
    (["gap", "prune"], dict(outcome="sample", root_rule="upper", prune=False)),
    (["gap", "bound_leaves"], dict(outcome="sample", root_rule="upper", leaf_eval="rollout", lazy_expansion=False)),
    (["resolution"], dict()),
])
def test_each_component_switched_off_reaches_i_and_every_modeled_level(built, off, changed):
    oracle.evaluate_case("mcts", 2, 20, 0, 0.3, **LEVEL_THREE, planner_options=profile.planner_options(
        "certified", off=off), level_options=partial(profile.level_options, "certified", off=off))
    own = PlannerConfig(simulations=20, depth=2, modeled_simulations=20)
    assert built[SolverKey("i", 3)] == replace(own, **(CERTIFIED | changed))
    assert built[SolverKey("i", 3)] == replace(own, **rd.certified_options(_decide_flags(profile_off=off)))
    for key, budget in ((J2, 30), (I1, 20)):
        decided = rd.modeled_options(_decide_flags(profile_off=off, modeled=budget))
        assert built[key] == replace(_flat(budget), **decided)
        if off == ["resolution"]:
            assert (built[key].modeled_resolution, built[key].modeled_prediction) == (None, "argmax")
        else:
            assert built[key] == replace(_flat(budget), **(LEVELS(budget) | changed))


# The panel passes the options through and builds them from --profile and --profile-off.


def test_the_panel_cell_passes_the_options_to_every_case(monkeypatch):
    calls = []
    monkeypatch.setattr(panel, "evaluate_case", lambda *args, **kwargs: calls.append(kwargs) or [{}])
    problems = partial(tiger_problems, beliefs=(0.3,), opponent_beliefs=(0.4, 0.6))
    panel.evaluate_cell("mcts", 2, 2, 20, "30", 0, "exact", 0, seed=0, problems=problems)
    panel.evaluate_cell("mcts", 2, 2, 20, "30", 0, "exact", 0, seed=0, problems=problems,
                        planner_options=CERTIFIED, level_options=LEVELS)
    assert [(c["planner_options"], c["level_options"]) for c in calls] == [(None, None)] * 2 + [(CERTIFIED, LEVELS)] * 2


def test_a_profile_panel_gives_every_i_pomcp_job_the_options(monkeypatch, tmp_path):
    jobs = {}
    monkeypatch.setattr(panel, "supervise_jobs", lambda found, **_: jobs.update(found) or iter(()))
    cells = [("mcts", 3, 2, 20, "30/20", 0, "exact", 0), ("rts", 2, 2, 2, "2", 0, "exact", 0)]
    panel.run_panel(tmp_path / "panel", cells, [0], 1, profile="certified", profile_off=["gap"])
    by_planner = {job.args[0]: job.keywords for job in jobs.values()}
    assert by_planner["mcts"]["planner_options"] == profile.planner_options("certified", off=["gap"])
    level_options = by_planner["mcts"]["level_options"]
    assert level_options(30) == profile.level_options("certified", 30, off=["gap"])
    assert "planner_options" not in by_planner["rts"] and "level_options" not in by_planner["rts"]
    # Jobs run in spawned workers, so they must pickle.
    assert pickle.loads(pickle.dumps(level_options))(20) == level_options(20)
    for job in jobs.values():
        pickle.dumps(job)
    manifest = json.loads((tmp_path / "panel" / "manifest.json").read_text())["panel"]
    assert (manifest["profile"], manifest["profile_off"]) == ("certified", ["gap"])


def test_a_panel_without_a_profile_gives_its_jobs_no_options(monkeypatch, tmp_path):
    jobs = {}
    monkeypatch.setattr(panel, "supervise_jobs", lambda found, **_: jobs.update(found) or iter(()))
    panel.run_panel(tmp_path / "panel", [("mcts", 2, 2, 20, "30", 0, "exact", 0)], [0], 1)
    (job,) = jobs.values()
    assert set(job.keywords) == {"problems", "physics"}
    assert "profile" not in json.loads((tmp_path / "panel" / "manifest.json").read_text())["panel"]


@pytest.mark.parametrize("arguments, message", [
    (dict(profile_off=["gap"]), "needs a profile"),
    (dict(profile="certified", profile_off=["prune"]), "needs 'gap' off too"),
    (dict(profile="fast"), "Unknown profile"),
])
def test_a_panel_refuses_a_bad_profile_before_writing(tmp_path, arguments, message):
    with pytest.raises(ValueError, match=message):
        panel.run_panel(tmp_path / "panel", [("mcts", 2, 2, 20, "30", 0, "exact", 0)], [0], 1, **arguments)
    assert not (tmp_path / "panel").exists()


def test_the_panel_command_line_takes_the_profile(monkeypatch, tmp_path):
    called = {}

    def run_panel(*args, **kwargs):
        called.update(kwargs)
        return pd.DataFrame(), []

    monkeypatch.setattr(panel, "run_panel", run_panel)
    argv = ["panel", "--output", str(tmp_path / "panel"), "--profile", "certified", "--profile-off", "gap", "prune"]
    monkeypatch.setattr(sys, "argv", argv)
    panel.main()
    assert (called["profile"], called["profile_off"]) == ("certified", ["gap", "prune"])
    monkeypatch.setattr(sys, "argv", argv[:3])
    panel.main()
    assert (called["profile"], called["profile_off"]) == (None, None)


# Scoring certified rows: pruned root actions carry no value.


def test_a_certified_row_with_an_exact_reference_lists_the_pruned_actions_and_scores_the_optimal_one():
    row = oracle.evaluate_case("mcts", 2, 30, 0, 0.3, level=2, opponent_depth=2, modeled="exact",
                               reference="exact", planner_options=CERTIFIED)[0]
    valued, legal = set(row["estimated_q"]), set(row["oracle_q"])
    assert valued < legal and row["unvalued_actions"] == sorted(legal - valued, key=str)
    assert row["max_abs_q_error"] is None
    best = max(row["oracle_q"].values())
    optimal = {a for a, q in row["oracle_q"].items() if q >= best - oracle.OPTIMALITY_TOLERANCE}
    assert optimal <= valued  # exact beliefs and opponents: the bounds hold, so no optimal action is pruned
    assert row["optimal_q_error"] == max(abs(row["estimated_q"][a] - row["oracle_q"][a]) for a in optimal)
    assert row["first_action_loss"] >= -1e-9


def test_a_default_row_values_every_action_and_has_no_unvalued_field():
    row = oracle.evaluate_case("mcts", 2, 30, 0, 0.3, level=2, opponent_depth=2, modeled="exact", reference="exact")[0]
    assert "unvalued_actions" not in row and row["max_abs_q_error"] is not None


def test_the_level_one_driver_scores_a_certified_row():
    rows = l1_qualification.evaluate_cell(30, 2, (0,), (0.5,), planner_options=CERTIFIED)
    row = rows[0]
    if set(row["estimated_q"]) != set(row["exact_q"]):
        assert row["max_abs_q_error"] is None and row["unvalued_actions"]
    assert row["optimal_q_error"] is not None


def test_q_errors_over_partial_estimates():
    exact = {"L": 1.0, "OL": -5.0, "OR": 1.0}
    assert oracle.max_abs_q_error({"L": 0.5, "OL": -5.0, "OR": 1.25}, exact) == 0.5
    assert oracle.max_abs_q_error({"L": 0.5}, exact) is None
    assert oracle.optimal_q_error({"L": 0.5}, exact) == 0.5
    assert oracle.optimal_q_error({"OL": -5.0}, exact) is None
    assert oracle.unvalued({"L": 0.5}, exact) == {"unvalued_actions": ["OL", "OR"]}
    assert oracle.unvalued(exact, exact) == {}
