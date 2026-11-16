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
from ipomdp.planning.node import POMCPNode
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
# Gap-directed outcome rules: every soundness test below also runs under each of them.
DIRECTED = [
    dict(CERTIFIED, selection="iemax", outcome="gap", root_rule="open"),
    dict(CERTIFIED, selection="iemax", outcome="gap_sample", root_rule="open"),
    dict(CERTIFIED, selection="iemax", outcome="gap", root_rule="challenger"),
    dict(frontier_bounds="ifib", prune=True, selection="iemax", outcome="gap", root_rule="open"),
]


@pytest.mark.parametrize("options", CONFIGS + DIRECTED)
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


@pytest.mark.parametrize("options", [CONFIGS[0], CONFIGS[3]] + DIRECTED)
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


@pytest.mark.parametrize("options", [CONFIGS[0], CONFIGS[3]])
def test_a_root_solved_at_expansion_still_decides(options):
    # When an action's bounds close at the largest upper bound before any search (UAV problem
    # s4_k3: spotting for certain), the root is solved at expansion; it must still decide.
    from ipomdp.planning.frontier_bounds import NodeBounds

    planner = tiger(0.5, 3, 200, **options)
    model = planner.model()
    frontier = planner._bounds_for(planner.solver_bank.search_model(model))
    original = frontier.node_bounds

    def closing(belief_model, r, bank):
        nb = original(belief_model, r, bank)
        if r != 3:
            return nb
        top = max(nb.q_hi.values()) + 1.0
        q_lo = dict(nb.q_lo, L=top)
        q_hi = dict(nb.q_hi, L=top)
        return NodeBounds(q_lo, q_hi, nb.branch_lo, nb.branch_hi)

    frontier.node_bounds = closing
    policy = planner.policy_for(model)
    assert planner.root.cert.solved and planner.root.visit_count >= 1
    assert policy == {"L": 1.0}
    assert set(planner.scores.values) == {"L"}


def test_a_solved_but_unresolved_root_keeps_searching():
    # The root's value is exact (bounds closed) while a second action, tied at the upper bound, still
    # has a loose lower bound: the decision is unresolved, so a descent must search, not return.
    planner = tiger(0.5, 3, 200, **CERTIFIED)
    model = planner.solver_bank.search_model(planner.model())
    root = POMCPNode()
    planner._expand(root, model, 3)
    cert = root.cert
    best = max(cert.q_lo, key=cert.q_lo.get)
    top = cert.q_lo[best]
    other = next(a for a in planner.actions if a != best)
    cert.q_hi[best] = top
    cert.q_hi[other], cert.q_lo[other] = top, top - 5.0
    for a in planner.actions:
        if a not in (best, other):
            cert.q_hi[a] = top - 10.0
            cert.q_lo[a] = min(cert.q_lo[a], cert.q_hi[a])
    cert.pruned = set(planner.actions) - {best, other}
    cert.lo = cert.hi = top
    cert.solved = True
    assert not planner._resolved(root)
    import random as stdlib_random

    planner._simulate_certified(root, 0, planner._physical_marginal(model), model, stdlib_random.Random(0))
    assert sum(root.action_counts.values()) == 1
    # A resolved solved root returns at once.
    cert.q_lo[other] = top - 20.0
    cert.q_hi[other] = top - 15.0
    assert planner._resolved(root)
    before = dict(root.action_counts)
    planner._simulate_certified(root, 0, planner._physical_marginal(model), model, stdlib_random.Random(1))
    assert root.action_counts == before


def test_scored_intervals_are_no_wider_than_the_searchs_own_bounds():
    planner = tiger(0.3, 4, 300, **dict(CERTIFIED, selection="iemax"))
    planner.policy_for(planner.model())
    cert = planner.root.cert
    for action, (lo, hi) in planner.scores.bounds.items():
        assert lo >= cert.q_lo[action] - 1e-12 and hi <= cert.q_hi[action] + 1e-12


def test_the_certified_profile_selects_by_upper_bound_unless_told_otherwise():
    from types import SimpleNamespace
    from ipomdp.experiments.reachable_decisions import certified_options

    base = dict(frontier_bounds="scalar", scoring="legacy", unsampled="lower", exact_tail=False,
                prune=False, leaf_eval="rollout", lazy=False)
    assert certified_options(SimpleNamespace(**base, selection=None, profile=None))["selection"] == "ucb"
    assert certified_options(SimpleNamespace(**base, selection=None, profile="certified"))["selection"] == "iemax"
    assert certified_options(SimpleNamespace(**base, selection="ucb", profile="certified"))["selection"] == "ucb"


# Gap-directed outcome rules (config.outcome, config.root_rule).

def test_the_default_configuration_and_its_seed_are_unchanged_by_the_gap_options():
    from ipomdp.planning.i_pomcp import seed_config

    default = PlannerConfig(100, 3, 20)
    assert (default.outcome, default.root_rule) == ("sample", "upper")
    assert not {"outcome", "root_rule"} & set(seed_config(default))
    certified = PlannerConfig(100, 3, 20, **CERTIFIED, selection="iemax")
    assert not {"outcome", "root_rule"} & set(seed_config(certified))
    assert {"outcome", "root_rule"} <= set(seed_config(replace(certified, outcome="gap", root_rule="open")))


@pytest.mark.parametrize(
    "options, message",
    [
        (dict(CERTIFIED, selection="iemax", root_rule="open"), "belong to a gap-directed"),
        (dict(CERTIFIED, selection="iemax", outcome="gap"), "needs the root rule"),
        (dict(CERTIFIED, selection="puct", outcome="gap", root_rule="open"), "needs pruning and selection"),
        (dict(frontier_bounds="ifib", exact_tail=True, selection="iemax", outcome="gap", root_rule="open"),
         "needs pruning and selection"),
        (dict(CERTIFIED, selection="iemax", outcome="widest", root_rule="open"), "outcome must be one of"),
    ],
)
def test_gap_options_are_validated(options, message):
    with pytest.raises(ValueError, match=message):
        PlannerConfig(100, 3, 20, **options)


def _tree_size(node):
    return 1 + sum(_tree_size(child) for children in node.children.values() for child in children.values())


@pytest.mark.parametrize("options", DIRECTED)
@pytest.mark.parametrize("p", [0.08, 0.3, 0.5, 0.7, 0.85])
@pytest.mark.parametrize("joint", [False, True])
def test_a_gap_directed_search_resolves_the_root_and_decides_optimally(options, p, joint):
    planner = tiger(p, 3, 400, joint=joint, **options)
    policy = planner.policy_for(planner.model())
    exact = BestResponseReference(planner.pomdp_model, planner.solver_bank, GAMMA).q_values(planner.model(), 3)
    best = max(exact.values())
    assert all(math.isclose(exact[a], best, rel_tol=0, abs_tol=1e-8) for a, mass in policy.items() if mass > 0)
    assert planner._resolved(planner.root)
    # Every simulation but the first expands one history or closes one, so the count of simulations is
    # bounded by the tree, far below the budget (Theorem 2 of the method review).
    assert planner.root.visit_count <= 2 * _tree_size(planner.root) + len(planner.actions)
    assert planner.root.visit_count < 400


def test_the_largest_gap_rule_does_not_use_the_random_stream():
    trees = []
    for seed in (11, 12):
        bank = SolverBank(seed=seed)
        physics = TigerModel()
        config = PlannerConfig(400, 3, 20, gamma=GAMMA, **DIRECTED[0])
        planner = HierarchyBuilder(bank).build("i", 1, physics, "j", config=config, level_weights={0: 1.0})
        opponent = MentalModel(AgentFrame("j", 0, physics))
        planner.set_initial_belief(FiniteBelief(tuple(
            (InteractiveState(s, opponent), w) for s, w in zip([TIGER_LEFT, TIGER_RIGHT], [0.3, 0.7]))))
        planner.policy_for(planner.model())
        trees.append((planner.root.visit_count, planner.root.to_dict(max_depth=4), dict(planner.scores.bounds)))
    assert trees[0] == trees[1]


def test_the_open_rule_refines_an_undecided_action_that_is_not_the_largest_upper_bound():
    # The liveness repair: with a closed leader, the adopted rule keeps selecting it; the open rule
    # selects the action that still keeps the decision unresolved.
    planner = tiger(0.5, 3, 40, **DIRECTED[0])
    planner.policy_for(planner.model())
    cert = planner.root.cert
    actions = list(planner.actions)
    leader, other = actions[0], actions[1]
    saved = dict(q_lo=dict(cert.q_lo), q_hi=dict(cert.q_hi), pruned=set(cert.pruned))
    try:
        cert.pruned.clear()
        for a in actions:
            cert.q_lo[a], cert.q_hi[a] = -100.0, -50.0
        cert.q_lo[leader] = cert.q_hi[leader] = 10.0           # closed leader
        cert.q_lo[other], cert.q_hi[other] = 0.0, 10.0 - 5e-9  # within the tolerance below it, open
        assert planner._directed_action(cert, actions, 0) == other
        assert planner._directed_action(cert, actions, 1) == leader  # below the root: the largest upper bound
    finally:
        cert.q_lo.update(saved["q_lo"])
        cert.q_hi.update(saved["q_hi"])
        cert.pruned.update(saved["pruned"])


def _bare_node(q_lo, q_hi):
    from ipomdp.planning.node import CertifiedState

    node = POMCPNode()
    node.cert = CertifiedState(None, None)
    node.cert.expanded = True
    node.cert.q_lo, node.cert.q_hi = dict(q_lo), dict(q_hi)
    return node


def test_a_gap_rule_never_prunes_an_action_the_stop_test_still_counts_as_plausible():
    # Largest lower bound 2, largest upper bound 1000: the pruning slack of the sampled rule is relative to 2,
    # the plausible set's tolerance is relative to 1000, so an upper bound of 2 - 5e-7 falls between the two.
    from ipomdp.planning.policy import plausible_sets

    sampled = tiger(0.5, 3, 40, **CERTIFIED, selection="iemax")
    directed = tiger(0.5, 3, 40, **DIRECTED[0])
    first, second, third = list(directed.actions)
    q_lo = {first: 2.0, second: -5.0, third: -500.0}
    q_hi = {first: 1000.0, second: 2.0 - 5e-7, third: -400.0}
    _, plausible = plausible_sets({a: (q_lo[a], q_hi[a]) for a in q_lo})
    assert second in plausible and third not in plausible
    # The adopted rule prunes the plausible action: it would never be refined and the root never resolves.
    node = _bare_node(q_lo, q_hi)
    sampled._prune(node.cert)
    assert second in node.cert.pruned
    # The gap rules prune exactly what is not plausible, and the root can select the action.
    node = _bare_node(q_lo, q_hi)
    directed._prune(node.cert)
    assert node.cert.pruned == {third}
    # The pruned set and the stop test's sets now agree at every history: unpruned means plausible.
    assert {a for a in directed.actions if a not in node.cert.pruned} == set(plausible)
    # The root refines the largest upper bound among the plausible, open actions; the second action stays
    # eligible and is taken once the first one's interval has closed below the tolerance.
    assert directed._directed_action(node.cert, [first, second], 0) == first
    node.cert.q_lo[first] = node.cert.q_hi[first] = 1000.0 * (1 - 1e-12) + 2.0 * 1e-12  # still the top, now closed
    node.cert.q_lo[second], node.cert.q_hi[second] = 990.0, node.cert.q_hi[first] - 5e-7
    assert directed._directed_action(node.cert, [first, second], 0) == second


def test_closed_children_of_large_magnitude_close_a_parent_near_zero_under_a_gap_rule():
    # A parent whose value is near zero averages two closed children near +50 and -50. With a slack relative to
    # each history's own value the children count as closed (gap 4e-8 <= 5e-8) and the parent does not
    # (gap 3.8e-8 > 1e-9): nothing is left to expand and the parent stays open. One absolute tolerance per
    # search composes: closed children give a closed parent.
    sampled = tiger(0.5, 3, 40, **CERTIFIED, selection="iemax")
    directed = tiger(0.5, 3, 40, **DIRECTED[0])
    gap = 4e-8
    assert gap <= 1e-9 * 50.0 and gap <= directed._epsilon  # each child is closed under both rules
    action = directed.actions[0]
    low = GAMMA * (0.5 * (50.0 - gap / 2) + 0.5 * (-50.0 - gap / 2))
    high = GAMMA * (0.5 * (50.0 + gap / 2) + 0.5 * (-50.0 + gap / 2))
    others = {a: -1000.0 for a in directed.actions if a != action}
    for planner, closes in ((sampled, False), (directed, True)):
        node = _bare_node({action: low} | others, {action: high} | others)
        planner._close(node)
        assert node.cert.solved is closes
    assert directed._epsilon == 1e-9 * directed.exploration.return_width(3, GAMMA)
    assert directed._tolerance(1e6) == directed._tolerance(0.0) == directed._epsilon
    assert sampled._epsilon is None and sampled._tolerance(1e6) == 1e-3
