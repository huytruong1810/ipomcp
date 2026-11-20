"""Reachable-history decisions (experiments.reachable_decisions): problems, references and scored decisions."""

import json

import pytest

from ipomdp.experiments import reachable_decisions as rd
from ipomdp.planning.config import PlannerConfig


def test_reference_history_is_reproducible_and_reachable():
    first = rd.reference("tiger-joint", 2, 2, seed=3, steps=2)
    again = rd.reference("tiger-joint", 2, 2, seed=3, steps=2)
    assert first["history"] == again["history"] and first["q"] == again["q"]
    assert len(first["history"]) == 2 and set(first["q"]) == {"L", "OL", "OR"}


def test_decide_scores_the_policy_against_the_stored_reference():
    problem = json.loads(json.dumps(rd.reference("tiger-joint", 2, 2, seed=1, steps=1)))
    row = rd.decide(problem, "mcts", 100, seed=0, modeled=100)
    assert row["loss"] == pytest.approx(0.0, abs=1e-9) and row["exact_root"]
    assert abs(sum(row["policy"].values()) - 1) < 1e-12


def test_uav_histories_never_end_the_episode():
    problem = rd.reference("uav", 2, 2, seed=5, steps=3)
    assert len(problem["history"]) == 3


def test_exact_modeled_levels_with_exact_beliefs_reproduce_the_exact_q():
    problem = json.loads(json.dumps(rd.reference("tiger-joint", 2, 2, seed=1, steps=2)))
    row = rd.decide(problem, "mcts", 100, seed=0, modeled_exact=True)
    assert row["modeled"] == "exact" and row["exact_root"]
    assert row["values"] == pytest.approx(problem["q"], abs=1e-9)


def test_exact_modeled_levels_are_an_i_pomcp_arm():
    problem = json.loads(json.dumps(rd.reference("tiger-joint", 2, 2, seed=1, steps=1)))
    with pytest.raises(ValueError):
        rd.decide(problem, "rts", 2, seed=0, modeled_exact=True)


def test_intmcp_with_the_exact_opponent_records_the_diagnostic():
    pytest.importorskip("intmcp", reason="third_party/i-ntmcp is not on the path")
    problem = json.loads(json.dumps(rd.reference("tiger-joint", 2, 2, seed=1, steps=1)))
    row = rd.decide(problem, "intmcp", 200, seed=0, intmcp_opponent="exact")
    assert row["intmcp_opponent"] == "exact" and row["loss"] >= -1e-9


def test_digest_only_matches_the_solved_problem():
    solved = rd.reference("tiger-joint", 2, 2, seed=3, steps=2)
    bare = rd.reference("tiger-joint", 2, 2, seed=3, steps=2, solve=False)
    assert "q" not in bare and bare["digest"] == solved["digest"]
    assert bare["history"] == solved["history"]


def test_certified_bounds_contain_the_exact_q_and_narrow_with_budget():
    problem = json.loads(json.dumps(rd.reference("tiger-joint", 2, 3, seed=1, steps=1)))
    rows = [rd.decide(problem, "mcts", budget, seed=0, modeled_exact=True) for budget in (10, 400)]
    assert all(row["bounds_contain_reference"] for row in rows)
    assert rows[1]["bound_width"] <= rows[0]["bound_width"]
    assert set(rows[0]["bounds"]) == set(problem["q"])


def test_a_budget_ladder_scores_prefixes_of_the_same_tree_and_ends_at_the_decision():
    problem = json.loads(json.dumps(rd.reference("tiger-joint", 2, 3, seed=1, steps=1)))
    plain = rd.decide(problem, "mcts", 400, seed=0, modeled_exact=True)
    row = rd.decide(problem, "mcts", 400, seed=0, modeled_exact=True, ladder=[10, 100])
    assert [stage["budget"] for stage in row["ladder"]] == [10, 100, 400]
    assert row["values"] == plain["values"] and row["bounds"] == plain["bounds"]
    assert row["ladder"][-1]["values"] == plain["values"] and row["ladder"][-1]["bounds"] == plain["bounds"]
    assert all(stage["bounds_contain_reference"] for stage in row["ladder"])


CERTIFIED_OPTIONS = dict(frontier_bounds="ifib", scoring="pw", exact_tail=True, prune=True, leaf_eval="bound",
                         lazy_expansion=True, selection="iemax")


@pytest.mark.slow
@pytest.mark.parametrize("extra", [{}, dict(outcome="gap", root_rule="open")])
def test_counting_work_changes_no_score_and_counts_from_a_cold_start(extra):
    problem = json.loads(json.dumps(rd.reference("tiger-joint", 2, 3, seed=1, steps=1)))
    options = CERTIFIED_OPTIONS | extra
    plain = rd.decide(problem, "mcts", 100, seed=0, modeled_exact=True, ladder=[10, 30], certified=options)
    counted = rd.decide(problem, "mcts", 100, seed=0, modeled_exact=True, ladder=[10, 30], certified=options,
                        count_work=True)
    assert counted["work_counted"] and "work_counted" not in plain
    for key in ("loss", "policy", "values", "bounds", "bounds_contain_reference"):
        assert counted[key] == plain[key]
    for cold, warm in zip(counted["ladder"], plain["ladder"]):
        for key in ("budget", "loss", "bounds", "sims_run", "pruned_root", "bounds_contain_reference"):
            assert cold[key] == warm[key]
        assert warm["updates"] == 0  # scored after a warm first search: nothing is counted
    assert counted["ladder"][-1]["updates"] > 0
    assert all(stage["bounds_contain_reference"] for stage in counted["ladder"])
    assert counted["loss"] == pytest.approx(0.0, abs=1e-9)


def test_counting_work_needs_a_ladder():
    problem = json.loads(json.dumps(rd.reference("tiger-joint", 2, 2, seed=1, steps=1)))
    with pytest.raises(ValueError, match="needs a ladder"):
        rd.decide(problem, "mcts", 100, seed=0, modeled_exact=True, count_work=True)


def _profile_args(**changes):
    from types import SimpleNamespace

    base = dict(frontier_bounds="scalar", scoring="legacy", exact_tail=False, prune=False, selection=None,
                leaf_eval="rollout", lazy=False, outcome="sample", root_rule="upper", profile="certified",
                profile_off=None, modeled_exact=False, modeled=400, budget=1000, modeled_resolve=False,
                modeled_pilot=50, modeled_prediction="argmax")
    return SimpleNamespace(**(base | changes))


def test_the_certified_profile_is_the_gap_rule_and_switching_nothing_off_changes_nothing():
    from ipomdp.experiments.reachable_decisions import certified_options, modeled_options

    own, modeled = certified_options(_profile_args()), modeled_options(_profile_args())
    assert (own["outcome"], own["root_rule"], own["selection"], own["prune"]) == ("gap", "open", "iemax", True)
    assert (modeled["outcome"], modeled["root_rule"], modeled["modeled_prediction"]) == ("gap", "open", "plausible")
    assert certified_options(_profile_args(profile_off=[])) == own
    PlannerConfig(simulations=10, depth=3, modeled_simulations=10, **own)  # a valid planner


@pytest.mark.parametrize("off, changed", [
    (["exact_tail"], dict(exact_tail=False)),
    (["pw_scoring"], dict(scoring="legacy")),
    (["gap"], dict(outcome="sample", root_rule="upper")),
    (["gap", "prune"], dict(outcome="sample", root_rule="upper", prune=False)),
    (["gap", "bound_leaves"], dict(outcome="sample", root_rule="upper", leaf_eval="rollout", lazy_expansion=False)),
])
def test_each_profile_component_switches_off_alone_in_the_agent_and_every_modeled_level(off, changed):
    from ipomdp.experiments.reachable_decisions import certified_options, modeled_options

    base_own, base_modeled = certified_options(_profile_args()), modeled_options(_profile_args())
    own, modeled = certified_options(_profile_args(profile_off=off)), modeled_options(_profile_args(profile_off=off))
    assert own == base_own | changed
    assert modeled == base_modeled | changed
    PlannerConfig(simulations=10, depth=3, modeled_simulations=10, **own)


def test_switching_off_resolution_acts_on_modeled_levels_only():
    from ipomdp.experiments.reachable_decisions import certified_options, modeled_options

    assert certified_options(_profile_args(profile_off=["resolution"])) == certified_options(_profile_args())
    modeled = modeled_options(_profile_args(profile_off=["resolution"]))
    assert "modeled_resolution" not in modeled and modeled["modeled_prediction"] == "argmax"


@pytest.mark.parametrize("off", [["prune"], ["bound_leaves"]])
def test_components_the_gap_rule_needs_cannot_be_switched_off_while_it_runs(off):
    from ipomdp.experiments.reachable_decisions import certified_options

    with pytest.raises(ValueError, match="needs 'gap' off too"):
        certified_options(_profile_args(profile_off=off))
