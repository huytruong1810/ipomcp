"""The Wumpus qualification's problems (experiments.wumpus_qualification)."""

from ipomdp.domains.wumpus.model import WumpusModel
from ipomdp.experiments.wumpus_qualification import history, human


def test_histories_are_prefixes_and_reproducible():
    physics = WumpusModel()
    full = history(physics, 4, seed=1)
    assert history(physics, 2, seed=1) == full[:2]
    assert history(physics, 4, seed=1) == full
    assert all(action in physics.get_all_actions("i") for action, _ in full)


def test_the_human_filters_exactly_and_searches_compressed():
    planner = human(WumpusModel(), seed=0, simulations=10, horizon=2, search_particles=64)
    bank = planner.solver_bank
    assert len(planner.belief.mass) == 94_500
    assert len(bank.search_model(planner.model()).belief.mass) <= 64


def test_random_scripts_are_reproducible_prefixes_that_differ_across_seeds():
    physics = WumpusModel()
    full = history(physics, 4, seed=3, script="random")
    assert history(physics, 2, seed=3, script="random") == full[:2]
    assert history(physics, 4, seed=3, script="random") == full
    assert len({tuple(map(str, history(physics, 4, s, script="random"))) for s in range(10)}) > 1


def test_level_two_problems_replay_the_exact_wumpus_history_to_the_planner():
    from ipomdp.experiments.wumpus_qualification import evaluate, reference

    board = (3, 3, 1)
    q, _, support, _, taken = reference(1, 2, 1, "random", board, 2)
    again = reference(1, 2, 1, "random", board, 2)
    assert again[4] == taken and again[0] == q
    assert len(taken) == 2 and support > 0
    row = evaluate(1, 2, 64, 50, 1, q, "random", board, 2, 20, taken)
    assert row["level"] == 2 and row["board"] == [3, 3, 1]
    assert "loss" in row or "failure" in row


def test_zero_search_particles_searches_exact_beliefs():
    planner = human(WumpusModel(3, 3, 1), 0, 10, 1, 0, level=2, modeled=10)
    bank = planner.solver_bank
    assert bank.search_filter is bank.filter
    assert bank.search_model(planner.model()) == planner.model()


def test_level_three_histories_from_the_modeled_human_have_positive_evidence():
    from ipomdp.experiments.wumpus_qualification import evaluate, reference

    board = (3, 3, 1)
    q, _, _, _, taken = reference(0, 2, 1, "modeled", board, 3)
    assert len(taken) == 2
    row = evaluate(0, 2, 0, 10, 1, q, "modeled", board, 3, 5, taken)
    assert "loss" in row or "failure" in row
