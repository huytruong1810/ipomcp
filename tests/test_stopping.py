"""Search seeds and trees grown in stages (planning.i_pomcp)."""

import random

import pytest

from ipomdp.domains.tiger import TigerModel
from ipomdp.experiments.problems import chain_model, tiger_chain
from ipomdp.planning.config import PlannerConfig
from ipomdp.planning.i_pomcp import IPOMCPPlanner
from ipomdp.planning.random_planner import RandomPlanner
from ipomdp.planning.solver_bank import SolverBank
from ipomdp.planning.solver_types import SolverKey


def j_planner(seed, simulations=600, depth=3):
    physics = TigerModel()
    bank = SolverBank(seed)
    bank.register_solver(
        SolverKey("i", 0), RandomPlanner(physics.get_all_actions("i"), random.Random(0))
    )
    config = PlannerConfig(
        simulations=simulations,
        depth=depth,
        modeled_simulations=simulations,
    )
    planner = IPOMCPPlanner(SolverKey("j", 1), physics, bank, config)
    bank.register_solver(SolverKey("j", 1), planner)
    return planner, physics


def test_the_search_seed_holds_every_search_field_and_no_scoring_field():
    from dataclasses import replace

    from ipomdp.belief import canonical
    from ipomdp.planning.i_pomcp import seed_config

    def digest(config):
        settings = {"planner": "I-POMCP", "config": seed_config(config), "simulations": 400}
        return canonical.digest([canonical.stable_value(settings)])

    config = PlannerConfig(400, 3, 400)
    # Digest of the default configuration since the consolidation release (every search field in the seed).
    assert digest(config) == "c5c498837181983383f4045757be11c4e59aa425b9ee557ea37b4a108a83eb9d"
    assert seed_config(config) == {
        "simulations": 400, "depth": 3, "modeled_simulations": 400, "gamma": 0.95, "exploration": 1.0,
        "exact_tail": False, "prune": False, "selection": "ucb", "leaf_eval": "rollout", "lazy_expansion": False,
        "outcome": "sample", "root_rule": "upper",
    }
    # Every search field changes the seed.
    with_bounds = replace(config, frontier_bounds="ifib")
    assert digest(replace(with_bounds, exact_tail=True)) != digest(with_bounds)
    # Scoring-only fields never enter the seed: their variants score the same tree.
    varied = replace(config, frontier_bounds="ifib", scoring="pw")
    assert digest(varied) == digest(config)
    with pytest.raises(ValueError):
        PlannerConfig(400, 3, 400, scoring="other")


def test_a_grown_tree_equals_a_flat_search_of_its_size():
    planner, physics = j_planner(4, simulations=400)
    model = chain_model(physics, "j", 1, tiger_chain(0.3))
    node, scores = planner._search(model, 400)
    grown = planner._tree(model, 400)
    grown.grow(150)
    assert grown.grow(400).values == scores.values
    assert grown.root.action_counts == node.action_counts
