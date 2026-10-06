"""Per-level modeled stacks: labels, and the hierarchy they register."""

import pytest

from ipomdp.domains.tiger import TigerModel
from ipomdp.experiments.modeled_levels import (
    ModeledSolve,
    modeled_keys,
    parse_stack,
    register_levels,
    stack_label,
)
from ipomdp.planning.i_pomcp import IPOMCPPlanner
from ipomdp.planning.solver_bank import SolverBank
from ipomdp.planning.solver_types import SolverKey
from ipomdp.planning.stopping import ModeledStopping
from ipomdp.reference.best_response import BestResponseReference
from ipomdp.reference.exact_agent import ExactIPOMDPPlanner


def test_labels_parse_per_level_and_round_trip():
    assert parse_stack("400", 2) == (ModeledSolve(400), ModeledSolve(400))
    assert parse_stack("6400s/exact", 2) == (ModeledSolve(6400, ModeledStopping()), None)
    custom = parse_stack("800s[100,3,0.05]", 1)[0]
    assert custom.stopping == ModeledStopping(pilot=100, trees=3, alpha=0.05)
    for label in ("400", "exact", "6400s/400", "800s[100,3,0.05]", "exact/400"):
        assert stack_label(parse_stack(label, len(label.split("/")))) == label
    assert stack_label(parse_stack("400/400", 2)) == "400"


@pytest.mark.parametrize("label, levels", [("400/400", 1), ("400/400/400", 2), ("4x", 1), ("0", 1)])
def test_malformed_or_mismatched_labels_are_rejected(label, levels):
    with pytest.raises(ValueError):
        parse_stack(label, levels)


def test_modeled_keys_alternate_agents_down_to_level_one():
    assert modeled_keys("i", 3) == [SolverKey("j", 2), SolverKey("i", 1)]
    assert modeled_keys("j", 2) == [SolverKey("i", 1)]
    assert modeled_keys("j", 1) == []


def test_each_level_gets_its_own_solve():
    physics = TigerModel()
    bank = SolverBank(0)
    reference = BestResponseReference(physics, bank, 0.95)
    register_levels(bank, physics, "i", 3, parse_stack("exact/1600s", 2), 3, reference)
    assert type(bank.get_solver(SolverKey("j", 2))) is ExactIPOMDPPlanner
    level_one = bank.get_solver(SolverKey("i", 1))
    assert type(level_one) is IPOMCPPlanner
    assert level_one.config.modeled_simulations == 1600
    assert level_one.config.modeled_stopping == ModeledStopping()
    assert {bank.has_solver(SolverKey(a, 0)) for a in "ij"} == {True}
    with pytest.raises(ValueError, match="needs 2 modeled levels"):
        register_levels(SolverBank(0), physics, "i", 3, parse_stack("400", 1), 3, reference)
