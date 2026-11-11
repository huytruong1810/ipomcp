"""Plausible sets, deterministic resolution and the deviation bound (audit spec 2.2, 3.5; Theorem 1b)."""

import random

import pytest

from ipomdp.belief.finite_belief import FiniteBelief, InteractiveState, MentalModel
from ipomdp.belief.frame import AgentFrame
from ipomdp.domains.tiger import TIGER_LEFT, TIGER_RIGHT, JointTigerModel
from ipomdp.experiments import reachable_decisions
from ipomdp.experiments.bootstrapper import HierarchyBuilder
from ipomdp.planning.config import PlannerConfig
from ipomdp.planning.i_pomcp import IPOMCPPlanner
from ipomdp.planning.policy import TIE_TOLERANCE, greedy_policy, plausible_policy, plausible_sets
from ipomdp.planning.solver_bank import SolverBank
from ipomdp.planning.stopping import ModeledResolution
from ipomdp.reference.best_response import BestResponseReference

GAMMA = 0.95


def exact_optimal(q):
    return set(greedy_policy(q, TIE_TOLERANCE))


@pytest.mark.parametrize("seed", range(200))
def test_certain_is_inside_the_optimal_set_which_is_inside_plausible(seed):
    rng = random.Random(seed)
    actions = "abcde"
    q = {a: rng.choice([rng.uniform(-50, 50), 10.0]) for a in actions}  # frequent exact ties at 10
    bounds = {}
    for a, value in q.items():
        lo = value - (0 if rng.random() < 0.3 else rng.uniform(0, 20))
        hi = value + (0 if rng.random() < 0.3 else rng.uniform(0, 20))
        bounds[a] = (lo, hi)
    certain, plausible = plausible_sets(bounds)
    optimal = exact_optimal(q)
    assert certain <= optimal <= plausible
    policy, resolved, delta = plausible_policy(bounds)
    assert set(policy) == plausible and 0 <= delta < 1
    if resolved:
        assert policy == greedy_policy(q, TIE_TOLERANCE) or set(policy) == optimal


def test_point_intervals_resolve_to_the_exact_law_ties_included():
    q = {"L": -1.0, "OL": 3.0, "OR": 3.0 + 1e-13}
    policy, resolved, delta = plausible_policy({a: (v, v) for a, v in q.items()})
    assert resolved and delta == 0 and set(policy) == {"OL", "OR"}


def j_planner(cap, pilot=10, **options):
    physics = JointTigerModel(together_safe=40.0)
    bank = SolverBank(seed=3)
    config = PlannerConfig(cap, 3, cap, gamma=GAMMA,
                           modeled_resolution=ModeledResolution(pilot=pilot), **options)
    planner = HierarchyBuilder(bank).build("j", 1, physics, "i", config=config, level_weights={0: 1.0})
    return planner, physics, bank


def j_model(physics, p):
    i0 = MentalModel(AgentFrame("i", 0, physics))
    return MentalModel(AgentFrame("j", 1, physics), FiniteBelief(
        ((InteractiveState(TIGER_LEFT, i0), p), (InteractiveState(TIGER_RIGHT, i0), 1 - p))))


PROFILE = dict(frontier_bounds="ifib", scoring="pw", exact_tail=True, prune=True, leaf_eval="bound",
               lazy_expansion=True, modeled_prediction="plausible")


@pytest.mark.parametrize("p", [0.02, 0.3, 0.5, 0.7, 0.98])
@pytest.mark.parametrize("options", [dict(modeled_prediction="plausible"), PROFILE])
def test_the_plausible_set_contains_the_exact_optimal_set_at_every_checkpoint(p, options, monkeypatch):
    planner, physics, bank = j_planner(640, **options)
    model = j_model(physics, p)
    exact = exact_optimal(BestResponseReference(physics, bank, GAMMA).q_values(model, 3))
    import ipomdp.planning.i_pomcp as module

    seen = []
    original = module.plausible_policy

    def spying(bounds, tie):
        result = original(bounds, tie)
        seen.append(set(result[0]))
        return result

    monkeypatch.setattr(module, "plausible_policy", spying)
    policy, resolved, delta = planner.modeled_solution(model)
    assert seen and all(exact <= plausible for plausible in seen)
    assert exact <= set(policy)
    if resolved:
        assert set(policy) == exact and delta == 0


@pytest.mark.parametrize("p", [0.02, 0.5, 0.98])
def test_a_resolved_policy_is_the_exact_policy(p):
    planner, physics, bank = j_planner(6400, pilot=50, **PROFILE)
    model = j_model(physics, p)
    policy, resolved, delta = planner.modeled_solution(model)
    assert resolved and delta == 0
    exact = BestResponseReference(physics, bank, GAMMA).q_values(model, 3)
    assert policy == pytest.approx(greedy_policy(exact, TIE_TOLERANCE))


def test_resolution_needs_no_pilot_trees(monkeypatch):
    planner, physics, bank = j_planner(320, **PROFILE)
    trees = []
    original = IPOMCPPlanner._tree

    def counting(self, model, simulations, replicate=None):
        trees.append(replicate)
        return original(self, model, simulations, replicate)

    monkeypatch.setattr(IPOMCPPlanner, "_tree", counting)
    planner.modeled_solution(j_model(physics, 0.3))
    assert trees == [None]


def test_the_bank_caches_policy_and_status_together():
    planner, physics, bank = j_planner(20, pilot=10)
    model = j_model(physics, 0.5)
    policy = bank.policy(model)
    resolved, delta = bank.policy_status(model)
    assert set(policy) and isinstance(resolved, bool) and 0 <= delta <= 1
    if not resolved:
        assert 1 in bank.unresolved_levels
    bank.clear_caches()
    if not resolved:
        assert 1 in bank.unresolved_levels  # never forgotten: forgetting would be unsound


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("modeled", [10, 40])
def test_unresolved_solves_widen_the_callers_intervals_and_stay_sound(seed, modeled):
    # Level-2 coordination tiger, searched level-1 opponents with a tiny cap: the widened
    # root intervals must contain the exact I-POMDP values.
    problem = reachable_decisions.reference("tiger-joint", 2, 3, seed, 1)
    row = reachable_decisions.decide(
        problem, "mcts", 60, 0, modeled=modeled,
        certified=dict(frontier_bounds="ifib", scoring="pw", exact_tail=True, prune=True,
                       leaf_eval="bound", lazy_expansion=True),
        modeled_options=dict(modeled_resolution=ModeledResolution(pilot=10), modeled_prediction="plausible"),
    )
    assert row["bounds_widened_contain_reference"]
    if row["model_certified"]:
        assert row["bounds_contain_reference"]
