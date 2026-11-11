"""Head-tail compression and the deterministic compression slack (audit spec 2.3)."""

import math
import random

import pytest

from ipomdp.belief.resampling import head_tail
from ipomdp.experiments import reachable_decisions


def random_belief(rng, n):
    raw = [rng.random() ** 4 for _ in range(n)]
    total = sum(raw)
    return [(f"x{i}", w / total) for i, w in enumerate(raw)]


def l1(rows, belief):
    merged = {}
    for atom, mass in rows:
        merged[atom] = merged.get(atom, 0.0) + mass
    target = dict(belief)
    return math.fsum(abs(merged.get(a, 0.0) - target.get(a, 0.0)) for a in set(merged) | set(target))


@pytest.mark.parametrize("seed", range(100))
def test_head_tail_l1_error_is_at_most_twice_the_loss(seed):
    rng = random.Random(seed)
    belief = random_belief(rng, rng.randint(20, 200))
    count = rng.randint(4, 16)
    tail = rng.randint(0, count - 1)
    rows, loss = head_tail(belief, count, tail, rng)
    assert len({a for a, _ in rows}) <= count
    assert math.isclose(math.fsum(m for _, m in rows), 1.0, abs_tol=1e-12)
    assert l1(rows, belief) <= 2 * loss + 1e-12


@pytest.mark.parametrize("seed", range(30))
def test_truncation_attains_the_l1_minimum(seed):
    rng = random.Random(seed)
    belief = random_belief(rng, 50)
    rows, loss = head_tail(belief, 10, 0, rng)
    assert math.isclose(l1(rows, belief), 2 * loss, abs_tol=1e-12)
    heaviest = sorted((w for _, w in belief), reverse=True)
    assert math.isclose(loss, math.fsum(heaviest[10:]), abs_tol=1e-12)


def test_head_tail_is_unbiased():
    # Per-atom mean mass over 2,000 seeds within 4 standard errors of the true mass. Atoms
    # expected fewer than 20 times in all runs are pooled, so every tested bucket has spread.
    rng = random.Random(0)
    belief = random_belief(rng, 40)
    runs, count, tail = 2000, 8, 4
    _, loss = head_tail(belief, count, tail, random.Random(0))
    rare = {a for a, w in belief if w / loss * tail * runs < 20}
    buckets = {a: [] for a, _ in belief if a not in rare} | {"rare": []}
    for seed in range(runs):
        rows, _ = head_tail(belief, count, tail, random.Random(seed))
        merged = dict.fromkeys(buckets, 0.0)
        for atom, mass in rows:
            merged["rare" if atom in rare else atom] += mass
        for key in buckets:
            buckets[key].append(merged[key])
    truth = {a: w for a, w in belief if a not in rare} | {"rare": math.fsum(w for a, w in belief if a in rare)}
    for key, values in buckets.items():
        mean = math.fsum(values) / runs
        sd = math.sqrt(math.fsum((v - mean) ** 2 for v in values) / (runs - 1))
        assert abs(mean - truth[key]) <= 4 * sd / math.sqrt(runs) + 1e-12


def test_small_beliefs_are_untouched():
    rng = random.Random(1)
    belief = random_belief(rng, 6)
    rows, loss = head_tail(belief, 6, 2, rng)
    assert loss == 0.0 and rows == tuple(belief)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_compression_slack_contains_the_uncompressed_searched_value(seed):
    problem = reachable_decisions.reference(
        "tiger-joint", 2, 3, seed, 1, prior=dict(kind="wide", M=8, w0=0.0)
    )
    row = reachable_decisions.decide(problem, "mcts", 200, 0, modeled=50, particles=8,
                                     compression="head-tail")
    assert row["max_loss"] > 0  # the beliefs were compressed
    assert row["bounds_compression_widened_contain_searched"]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_compression_slack_covers_the_exact_tail_of_certified_search(seed):
    # With the exact two-step tail, histories with two steps to go are solved from compressed
    # successor beliefs that never become tree nodes; their losses must enter the slack too.
    problem = reachable_decisions.reference(
        "tiger-joint", 2, 3, seed, 1, prior=dict(kind="wide", M=8, w0=0.0)
    )
    certified = dict(frontier_bounds="ifib", scoring="pw", exact_tail=True, prune=True,
                     leaf_eval="bound", lazy_expansion=True)
    row = reachable_decisions.decide(problem, "mcts", 200, 0, modeled=50, particles=8,
                                     compression="head-tail", certified=certified)
    assert row["bounds_compression_widened_contain_searched"]


def test_two_step_branches_are_charged_even_without_tree_nodes():
    # A depth-2 certified search is solved by the exact tail at expansion: no child node exists,
    # yet each branch is valued from its own compressed belief, so its loss must enter the slack.
    from ipomdp.experiments.reachable_decisions import GAMMA, _planner, make_physics
    from ipomdp.planning.chance_scoring import compression_bound

    physics = make_physics("tiger-joint")
    certified = dict(frontier_bounds="ifib", scoring="pw", exact_tail=True, prune=True,
                     leaf_eval="bound", lazy_expansion=True)
    planner = _planner(physics, "mcts", 0, 50, 2, 2, 20, 4, "sampled", certified=certified,
                       compression="head-tail", prior=(8, 0.0))
    model = planner.model()
    planner.policy_for(model)
    bank = planner.solver_bank
    slack = compression_bound(planner.root, model, 2, bank, planner._history_successor, GAMMA)
    charged = 0
    for action in planner.actions:
        losses = [bank.search_filter.loss(planner._history_successor(model, action, o).belief)
                  for o, _ in bank.continuation_law(model, action)]
        if any(loss > 0 for loss in losses):
            assert slack[action] > 0
            charged += 1
    assert charged >= 1
