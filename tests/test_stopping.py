"""Sequential stopping for modeled policies (planning.stopping)."""

import random

import pytest

from ipomdp.domains.tiger import OPEN_LEFT, TigerModel
from ipomdp.experiments.problems import chain_model, tiger_chain
from ipomdp.planning.config import PlannerConfig
from ipomdp.planning.i_pomcp import IPOMCPPlanner, _GrowingSearch
from ipomdp.planning.random_planner import RandomPlanner
from ipomdp.planning.solver_bank import SolverBank
from ipomdp.planning.solver_types import SolverKey
from ipomdp.planning.stopping import ModeledStopping


def test_checkpoints_double_from_the_pilot_to_the_cap():
    rule = ModeledStopping(pilot=200)
    assert rule.checkpoints(1600) == (200, 400, 800, 1600)
    assert rule.checkpoints(1000) == (200, 400, 800, 1000)
    assert rule.checkpoints(200) == (200,)
    with pytest.raises(ValueError, match="cap"):
        rule.checkpoints(100)


def test_separation_scales_the_pilot_spread_by_the_main_tree_size():
    rule = ModeledStopping(pilot=100, trees=4, alpha=0.01)
    # Leading-pair differences over the calibration trees: sd sqrt(2 / 3).
    calibration = [{"a": d, "b": 0.0} for d in (0.0, 1.0, 1.0, 2.0)]
    # t(1 - 0.01 / 8, 3) = 9.46, so the threshold is 7.73 at the pilot and
    # 7.73 * 100 / 1600 = 0.483 at 1,600 simulations (spread falling as 1 / n).
    assert not rule.separated({"a": 7.5, "b": 0.0}, calibration, 100, 4)
    assert rule.separated({"a": 8.0, "b": 0.0}, calibration, 100, 4)
    assert not rule.separated({"a": 0.48, "b": 0.0}, calibration, 1600, 4)
    assert rule.separated({"a": 0.49, "b": 0.0}, calibration, 1600, 4)
    assert rule.separated({"a": 1.0}, calibration, 100, 4)


def test_the_pilot_cannot_exceed_the_modeled_cap():
    with pytest.raises(ValueError, match="cap"):
        PlannerConfig(10, 2, 100, modeled_stopping=ModeledStopping(pilot=200))


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


def test_the_rule_does_not_change_flat_search_streams():
    # Seeds key on the configuration without the rule, so real decisions (and
    # every result produced before the rule existed) are reproduced exactly.
    plain, physics = j_planner(3, None)
    ruled, _ = j_planner(3, ModeledStopping(pilot=100))
    plain.policy_for(chain_model(physics, "j", 1, tiger_chain(0.3)))
    ruled.policy_for(chain_model(ruled.pomdp_model, "j", 1, tiger_chain(0.3)))
    assert plain.search_values == ruled.search_values


def test_a_grown_tree_equals_a_flat_search_of_its_size():
    planner, physics = j_planner(4, None, simulations=400)
    model = chain_model(physics, "j", 1, tiger_chain(0.3))
    node, scores = planner._search(model, 400)
    grown = planner._tree(model, 400)
    grown.grow(150)
    assert grown.grow(400).values == scores.values
    assert grown.root.action_counts == node.action_counts


def counting(monkeypatch):
    sizes = []
    original = _GrowingSearch.grow

    def grow(self, simulations, bounds=False):
        sizes.append(simulations)
        return original(self, simulations, bounds)

    monkeypatch.setattr(_GrowingSearch, "grow", grow)
    return sizes


def test_clear_beliefs_stop_at_the_pilot_and_solves_are_pure_functions(monkeypatch):
    sizes = counting(monkeypatch)
    rule = ModeledStopping(pilot=100, trees=4, alpha=0.01)
    first, physics = j_planner(5, rule, simulations=1600)
    # P(tiger left) = 0.02: opening left is far ahead of every alternative.
    policy = first.policy_for(chain_model(physics, "j", 1, tiger_chain(0.02)), modeled=True)
    assert policy == {OPEN_LEFT: 1.0}
    assert sizes[:4] == [100, 100, 100, 100]  # the main tree's pilot checkpoint, then 3 pilots
    assert max(sizes) < 1600
    again, physics = j_planner(5, rule, simulations=1600)
    assert again.policy_for(chain_model(physics, "j", 1, tiger_chain(0.02)), modeled=True) == policy


def test_an_unseparated_solve_returns_the_flat_search_of_the_cap(monkeypatch):
    # An unreachable alpha keeps the test from passing while the pilots disagree.
    # Depth 4: at depth 3 every root branch has two steps left and is scored
    # exactly, so the pilots agree and the rule rightly stops at once.
    rule = ModeledStopping(pilot=100, trees=2, alpha=1e-12)
    ruled, physics = j_planner(6, rule, simulations=800, depth=4)
    plain, _ = j_planner(6, None, simulations=800, depth=4)
    model = chain_model(physics, "j", 1, tiger_chain(0.5))
    sizes = counting(monkeypatch)
    values = ruled._stopped_values(model)
    assert sizes == [100, 100, 200, 400, 800]
    assert (
        values
        == plain._search(chain_model(plain.pomdp_model, "j", 1, tiger_chain(0.5)), 800)[1].values
    )
