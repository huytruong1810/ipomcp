"""Certified search (audit spec 3.3-3.4): bounds at every history contain the exact values."""

import math
import random
from dataclasses import replace

import pytest

from ipomdp.belief.finite_belief import FiniteBelief, InteractiveState, MentalModel
from ipomdp.belief.frame import AgentFrame
from ipomdp.domains.tiger import TIGER_LEFT, TIGER_RIGHT, JointTigerModel, TigerModel
from ipomdp.domains.uav import UAVModel
from ipomdp.experiments.bootstrapper import HierarchyBuilder
from ipomdp.planning.chance_scoring import exact_two_step, score_tree
from ipomdp.planning.config import PlannerConfig
from ipomdp.planning.solver_bank import SolverBank
from ipomdp.reference.best_response import BestResponseReference

GAMMA = 0.95
SLACK = 1e-9
CERTIFIED = dict(frontier_bounds="ifib", scoring="pw", exact_tail=True, prune=True,
                 leaf_eval="bound", lazy_expansion=True)


def planner_at(physics, states, weights, depth, simulations, level=1, **options):
    bank = SolverBank(seed=11)
    config = PlannerConfig(simulations, depth, 20, gamma=GAMMA, **options)
    planner = HierarchyBuilder(bank).build(
        "i", level, physics, "j", config=config, level_weights={k: 1 / level for k in range(level)}
    )
    opponent = MentalModel(AgentFrame("j", 0, physics))
    total = sum(weights)
    planner.set_initial_belief(
        FiniteBelief(tuple((InteractiveState(s, opponent), w / total) for s, w in zip(states, weights)))
    )
    return planner


def tiger(p, depth, simulations, joint=False, **options):
    physics = JointTigerModel(together_safe=40.0) if joint else TigerModel()
    return planner_at(physics, [TIGER_LEFT, TIGER_RIGHT], [p, 1 - p], depth, simulations, **options)


def small_uav(seed, depth, simulations, **options):
    physics = UAVModel(rows=2, cols=2)
    rng = random.Random(seed)
    states = [((1, 0), t) for t in physics.cells() if t != (1, 0)]
    return planner_at(physics, states, [rng.random() for _ in states], depth, simulations, **options)


def walk(planner):
    """(node, model, steps to go) for every expanded history of the last search."""
    stack = [(planner.root, planner.model(), planner.config.depth)]
    while stack:
        node, model, remaining = stack.pop()
        if node.cert is not None and node.cert.expanded:
            yield node, model, remaining
        for action, children in node.children.items():
            for observation, child in children.items():
                if child.cert is not None and child.cert.expanded:
                    stack.append(
                        (child, planner._history_successor(model, action, observation), remaining - 1)
                    )


def check_every_history(planner):
    reference = BestResponseReference(planner.pomdp_model, planner.solver_bank, GAMMA)
    histories = 0
    for node, model, remaining in walk(planner):
        histories += 1
        q = reference.q_values(model, remaining)
        best = max(q.values())
        cert = node.cert
        assert cert.lo - SLACK <= best <= cert.hi + SLACK
        for action, value in q.items():
            assert cert.q_lo[action] - SLACK <= value <= cert.q_hi[action] + SLACK
        for action in cert.pruned:
            assert q[action] < best  # never prune an exact maximizer
        if cert.solved:
            assert math.isclose(cert.lo, best, rel_tol=0, abs_tol=1e-8)
    return histories


CONFIGS = [
    dict(CERTIFIED),
    dict(CERTIFIED, lazy_expansion=False),
    dict(CERTIFIED, selection="puct"),
    dict(CERTIFIED, selection="iemax"),
    dict(frontier_bounds="ifib", exact_tail=True),
    dict(frontier_bounds="ifib", prune=True),
    dict(frontier_bounds="ifib", prune=True, leaf_eval="bound"),
]


@pytest.mark.parametrize("options", CONFIGS)
@pytest.mark.parametrize("p", [0.08, 0.5, 0.85])
@pytest.mark.parametrize("depth, simulations", [(3, 40), (4, 120)])
def test_bounds_at_every_history_contain_the_exact_values_in_tiger(options, p, depth, simulations):
    planner = tiger(p, depth, simulations, **options)
    planner.policy_for(planner.model())
    assert check_every_history(planner) >= 1
    exact = BestResponseReference(planner.pomdp_model, planner.solver_bank, GAMMA).q_values(
        planner.model(), depth
    )
    for action, (lo, hi) in planner.scores.bounds.items():
        assert lo - SLACK <= exact[action] <= hi + SLACK
    assert set(planner.scores.bounds) == set(planner.actions)


@pytest.mark.parametrize("options", [CONFIGS[0], CONFIGS[3]])
@pytest.mark.parametrize("seed", range(4))
def test_bounds_at_every_history_contain_the_exact_values_in_a_small_uav(options, seed):
    planner = small_uav(seed, 3, 60, **options)
    planner.policy_for(planner.model())
    assert check_every_history(planner) >= 1


@pytest.mark.parametrize("p", [0.08, 0.3, 0.5, 0.7, 0.85])
@pytest.mark.parametrize("joint", [False, True])
def test_h3_root_is_resolved_early_and_decides_optimally(p, joint):
    planner = tiger(p, 3, 4000, joint=joint, **CERTIFIED)
    tree = planner._tree(planner.model(), 4000)
    scores = tree.grow(4000, bounds=True)
    assert tree.simulations_run < 400
    assert planner._resolved(tree.root)
    exact = BestResponseReference(planner.pomdp_model, planner.solver_bank, GAMMA).q_values(
        planner.model(), 3
    )
    best = max(exact.values())
    chosen = max(scores.values, key=scores.values.get)
    assert exact[chosen] >= best - 1e-9


def test_bounds_only_tighten():
    planner = tiger(0.3, 4, 600, **dict(CERTIFIED, selection="puct"))
    tree = planner._tree(planner.model(), 600)
    previous = None
    for budget in range(10, 601, 10):
        tree.grow(budget)
        cert = tree.root.cert
        current = (dict(cert.q_lo), dict(cert.q_hi), cert.lo, cert.hi)
        if previous is not None:
            for action in planner.actions:
                assert current[0][action] >= previous[0][action]
                assert current[1][action] <= previous[1][action]
            assert current[2] >= previous[2] and current[3] <= previous[3]
        previous = current


def test_exact_two_step_matches_the_reference_and_scoring():
    planner = tiger(0.7, 2, 300)
    planner.policy_for(planner.model())
    bank, model = planner.solver_bank, planner.model()
    exact = BestResponseReference(planner.pomdp_model, bank, GAMMA).q_values(model, 2)
    scored = score_tree(planner.root, model, 2, bank, planner._history_successor, GAMMA)
    for action in planner.actions:
        value = exact_two_step(model, action, bank, planner._history_successor, GAMMA)
        assert value == scored.values[action]
        assert math.isclose(value, exact[action], rel_tol=0, abs_tol=1e-12)


def test_the_default_configuration_keeps_the_legacy_descent():
    planner = tiger(0.3, 3, 200)
    planner.policy_for(planner.model())
    assert not planner.config.certified_search
    assert all(node.cert is None for node in _nodes(planner.root))


def test_certified_options_need_belief_dependent_bounds():
    with pytest.raises(ValueError):
        PlannerConfig(10, 3, 10, prune=True)
    with pytest.raises(ValueError):
        PlannerConfig(10, 3, 10, frontier_bounds="ifib", lazy_expansion=True)
    with pytest.raises(ValueError):
        PlannerConfig(10, 3, 10, frontier_bounds="ifib", selection="other")
    assert PlannerConfig(10, 3, 10, frontier_bounds="ifib").certified_search is False


def test_a_lazy_frontier_is_scored_without_its_belief():
    # Scoring computes one belief update per expanded child of an unsolved
    # expanded history, and none for frontiers or below solved histories.
    planner = tiger(0.3, 4, 60, **CERTIFIED)
    planner.policy_for(planner.model())
    expected, frontiers, stack = 0, 0, [planner.root]
    while stack:
        node = stack.pop()
        if node.cert.solved:
            continue
        for children in node.children.values():
            for child in children.values():
                if child.cert.expanded:
                    expected += 1
                    stack.append(child)
                else:
                    frontiers += 1
    assert frontiers
    calls = []
    original = planner._history_successor

    def counting(model, action, observation):
        calls.append(1)
        return original(model, action, observation)

    rescored = score_tree(planner.root, planner.model(), 4, planner.solver_bank, counting, GAMMA,
                          True, frontier=planner._frontier_bounds, scoring="pw")
    assert len(calls) == expected
    assert rescored.frontiers == frontiers


def _nodes(root):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        for children in node.children.values():
            stack.extend(children.values())
