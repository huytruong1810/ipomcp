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
