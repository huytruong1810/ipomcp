"""Sequential replicate searches for modeled policies (planning.replicates)."""

import random

import pytest

from ipomdp.domains.tiger import OPEN_LEFT, TigerModel
from ipomdp.experiments.oracle import _point_model
from ipomdp.planning.config import PlannerConfig
from ipomdp.planning.i_pomcp import IPOMCPPlanner
from ipomdp.planning.random_planner import RandomPlanner
from ipomdp.planning.replicates import ModeledStopping, mean_values
from ipomdp.planning.solver_bank import SolverBank
from ipomdp.planning.solver_types import SolverKey


def test_separation_needs_min_replicates_and_a_gap_of_z_standard_errors():
    rule = ModeledStopping(batch=10, min_replicates=3, z=3.0)
    clear = [{"a": 10.0 + d, "b": 0.0 - d} for d in (0.1, -0.1, 0.0)]
    assert not rule.separated(clear[:2])
    assert rule.separated(clear)
    noisy = [{"a": x, "b": y} for x, y in ((1.0, 0.0), (-1.0, 1.0), (1.0, 0.0))]
    assert not rule.separated(noisy)
    assert mean_values(clear) == pytest.approx({"a": 10.0, "b": 0.0})


def test_batch_cannot_exceed_the_modeled_cap():
    with pytest.raises(ValueError, match="cap"):
        PlannerConfig(10, 2, 100, modeled_stopping=ModeledStopping(batch=200))


def j_planner(seed, stopping, simulations=600, depth=3):
    physics = TigerModel()
    bank = SolverBank(seed)
    bank.register_solver(
        SolverKey("i", 0), RandomPlanner(physics.get_all_actions("i"), random.Random(0))
    )
    config = PlannerConfig(
        simulations=simulations,
        depth=depth,
        modeled_simulations=simulations,
        modeled_stopping=stopping,
    )
    planner = IPOMCPPlanner(SolverKey("j", 1), physics, bank, config)
    bank.register_solver(SolverKey("j", 1), planner)
    return planner, physics


def test_the_rule_does_not_change_single_search_streams():
    # Seeds key on the configuration without the rule, so real decisions (and
    # every result produced before the rule existed) are reproduced exactly.
    plain, physics = j_planner(3, None)
    ruled, _ = j_planner(3, ModeledStopping(batch=100))
    model = _point_model(physics, "j", 1, (0.3,))
    plain.policy_for(model)
    ruled.policy_for(_point_model(ruled.pomdp_model, "j", 1, (0.3,)))
    assert plain.search_values == ruled.search_values


def test_modeled_solves_stop_early_on_clear_beliefs_and_are_pure_functions(monkeypatch):
    calls = []
    original = IPOMCPPlanner._search

    def counted(self, model, simulations, replicate=None):
        calls.append(replicate)
        return original(self, model, simulations, replicate)

    monkeypatch.setattr(IPOMCPPlanner, "_search", counted)
    rule = ModeledStopping(batch=100, min_replicates=3, z=3.0)
    first, physics = j_planner(5, rule, simulations=1000)
    # P(tiger left) = 0.02: opening left is far ahead of every alternative.
    policy = first.policy_for(_point_model(physics, "j", 1, (0.02,)), modeled=True)
    assert policy == {OPEN_LEFT: 1.0}
    assert calls == [0, 1, 2]
    again, physics = j_planner(5, rule, simulations=1000)
    assert again.policy_for(_point_model(physics, "j", 1, (0.02,)), modeled=True) == policy
