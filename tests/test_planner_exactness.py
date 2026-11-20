"""Small-scale exactness check of the planner at levels 1 to 3, for every search rule under test.

``test_certified_search.py`` checks the bounds of certified search against the
exact reference at level 1 (a uniformly random opponent). The claims of the
method are about nested reasoning, so this file repeats the check where the
opponent is itself an optimizing agent: levels 2 and 3, with every modeled
level solved exactly, in two domains (one of which ends episodes), at the
prior and after a first step, for each of the two observation and root rules:

- A: the next observation is sampled with its probability;
- B: the open branch with the largest probability-weighted gap.

With exact modeled levels and exact beliefs the searched model is the exact
I-POMDP, so the reference (``BestResponseReference`` on the same bank, itself
checked against brute force in ``test_reference_independent.py``) gives the
true value of every action at every history the search expanded. Checked:

1. the interval of every history and of every action at it contains the exact value;
2. no pruned action is an exact maximizer;
3. the root's scored intervals contain the exact values, and the values the
   decision is taken on lie inside them (so the reported width bounds the error);
4. a decision the search calls resolved plays exactly the exact maximizers;
5. a history marked solved, and scores marked exact, equal the exact values.

The budgets are small on purpose: the properties must hold at any budget, and
an unresolved search is still required to pass 1 to 3.
"""

import pytest

from ipomdp.experiments import profile
from ipomdp.experiments.reachable_decisions import GAMMA, _planner, make_physics
from ipomdp.planning.policy import TIE_TOLERANCE, greedy_policy
from ipomdp.reference.best_response import BestResponseReference

SLACK = 1e-9
ARMS = {
    "A": {},
    "B": dict(outcome="gap", root_rule="open"),
}
# (domain, level, horizon, simulations): every cell runs in about a second or less.
CELLS = [
    ("tiger-joint", 1, 3, 60),
    ("tiger-joint", 2, 2, 30),
    ("tiger-joint", 2, 3, 120),
    ("tiger-joint", 3, 3, 120),
    ("uav2", 1, 3, 80),
    ("uav2", 2, 3, 120),
    ("uav2", 3, 2, 40),
]


def physics_of(domain):
    if domain == "uav2":
        from ipomdp.domains.uav import UAVModel

        return UAVModel(rows=2, cols=2)
    return make_physics(domain)


def expanded_histories(planner):
    """(node, model, steps to go) for every history the last search expanded."""
    stack = [(planner.root, planner.solver_bank.search_model(planner.model()), planner.config.depth)]
    while stack:
        node, model, remaining = stack.pop()
        if node.cert is None or not node.cert.expanded:
            continue
        yield node, model, remaining
        for action, children in node.children.items():
            for observation, child in children.items():
                if child.cert is not None and child.cert.expanded:
                    stack.append((child, planner._history_successor(model, action, observation), remaining - 1))


def first_step(planner):
    """One real step: the first action and its most likely observation (positive evidence by construction)."""
    model = planner.model()
    action = planner.actions[0]
    law = planner.solver_bank.continuation_law(model, action)
    observation = max(law, key=lambda row: row[1])[0]
    planner.update_root(action, observation)


# After a first step a level-3 Tiger belief makes the exact reference cost about 100 s per case, so the
# one-step variant runs at levels 1 and 2 only; level 3 is checked at the prior.
CASES = [(*cell, steps) for cell in CELLS for steps in (0, 1) if steps == 0 or cell[1] < 3]


@pytest.mark.slow
@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("domain, level, horizon, simulations, steps", CASES)
def test_intervals_contain_the_exact_values_and_resolved_decisions_are_optimal(
    arm, domain, level, horizon, simulations, steps
):
    physics = physics_of(domain)
    certified = profile.own_options("certified") | ARMS[arm]
    planner = _planner(physics, "mcts", 5, simulations, horizon, level, simulations, 0, "sampled",
                       modeled_exact=True, certified=certified)
    for _ in range(steps):
        first_step(planner)
    policy = planner.policy_for(planner.model())
    reference = BestResponseReference(physics, planner.solver_bank, GAMMA, max_branches=5_000_000)

    histories = 0
    for node, model, remaining in expanded_histories(planner):
        histories += 1
        q = reference.q_values(model, remaining)
        best, cert = max(q.values()), node.cert
        assert cert.lo - SLACK <= best <= cert.hi + SLACK
        if cert.solved:
            # "Solved" is a claim that the bounds have met at the exact value, not merely come close.
            assert cert.hi - cert.lo <= 1e-6 and abs(cert.lo - best) <= 1e-6
        optimal = set(greedy_policy(q, TIE_TOLERANCE))
        for action, value in q.items():
            assert cert.q_lo[action] - SLACK <= value <= cert.q_hi[action] + SLACK
            if action in cert.pruned:
                assert action not in optimal
    assert histories >= 1

    exact = reference.q_values(planner.model(), horizon)
    bounds = planner.scores.bounds
    assert set(bounds) == set(exact)
    for action, (lo, hi) in bounds.items():
        assert lo - SLACK <= exact[action] <= hi + SLACK
    for action, value in planner.recommendation_values.items():
        lo, hi = bounds[action]
        assert lo - SLACK <= value <= hi + SLACK
        assert abs(value - exact[action]) <= (hi - lo) + SLACK
    if planner.scores.exact:
        assert planner.recommendation_values == pytest.approx(exact, abs=1e-6)
    if planner._resolved(planner.root):
        assert set(policy) == set(greedy_policy(exact, TIE_TOLERANCE))


@pytest.mark.slow
@pytest.mark.parametrize("domain, level, horizon", [("tiger-joint", 2, 3), ("uav2", 2, 3), ("tiger-joint", 3, 3)])
def test_a_gap_directed_search_resolves_every_small_decision_within_its_budget(domain, level, horizon):
    """The gap rule terminates with a certificate on a finite tree; 2,000 simulations is far above these trees."""
    physics = physics_of(domain)
    certified = profile.own_options("certified") | ARMS["B"]
    planner = _planner(physics, "mcts", 5, 2000, horizon, level, 2000, 0, "sampled",
                       modeled_exact=True, certified=certified)
    policy = planner.policy_for(planner.model())
    assert planner._resolved(planner.root)
    exact = BestResponseReference(physics, planner.solver_bank, GAMMA, max_branches=5_000_000).q_values(
        planner.model(), horizon)
    assert set(policy) == set(greedy_policy(exact, TIE_TOLERANCE))
