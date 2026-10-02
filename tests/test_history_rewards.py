"""Immediate rewards at tree histories are integrated over the agent's posterior.

The sampled hidden state of a simulation must never set the reward used in a
backup: that would let the tree condition on information the agent lacks.
"""

import random
from types import SimpleNamespace

import pytest

from ipomdp.belief.finite_belief import FiniteBelief, InteractiveState, MentalModel
from ipomdp.belief.frame import AgentFrame
from ipomdp.domains.tiger import TIGER_LEFT, TIGER_RIGHT, TigerModel
from ipomdp.planning.config import PlannerConfig
from ipomdp.planning.i_pomcp import IPOMCPPlanner
from ipomdp.planning.node import POMCPNode
from ipomdp.planning.solver_bank import SolverBank
from ipomdp.planning.solver_types import SolverKey


def setup(depth=2):
    # Perfect creaks make the conditioned posterior analytic: silence means j listened.
    physics = TigerModel(creak_accuracy=1.0)
    bank = SolverBank(seed=1)
    opponent = MentalModel(AgentFrame("j", 0, physics))
    belief = FiniteBelief(
        (
            (InteractiveState(TIGER_LEFT, opponent), 0.07),
            (InteractiveState(TIGER_RIGHT, opponent), 0.93),
        )
    )
    model = MentalModel(AgentFrame("i", 1, physics), belief)
    planner = IPOMCPPlanner(
        SolverKey("i", 1),
        physics,
        bank,
        PlannerConfig(simulations=10, depth=depth, modeled_simulations=10),
    )
    planner.exploration = SimpleNamespace(select_action=lambda *args: "OL")
    return planner, model


@pytest.mark.parametrize("state", [TIGER_LEFT, TIGER_RIGHT])
@pytest.mark.parametrize("conditioned", [False, True])
def test_internal_reward_uses_posterior_not_sampled_hidden_state(state, conditioned):
    planner, model = setup()
    p = 0.07
    if conditioned:
        # Silence in creak means j listened, so the growl update is analytic.
        model = planner._history_successor(model, "L", ("GL", "S"))
        p = 0.07 * 0.85 / (0.07 * 0.85 + 0.93 * 0.15)
    opponent = model.belief.mass[0][0].opponent
    node = POMCPNode()
    memory = planner._physical_marginal(model)
    value = planner._simulate(
        InteractiveState(state, opponent), node, 0, memory, model, random.Random(0)
    )
    # Opening resets the tiger, so the next (last) decision is at a uniform
    # belief, where listening (-1) is optimal.
    expected = (10 - 110 * p) + 0.95 * -1
    assert value == pytest.approx(expected)
    assert node.action_values["OL"] == pytest.approx(expected)
    assert node.action_counts == {"OL": 1}


def test_terminal_outcome_has_reward_and_no_continuation(monkeypatch):
    planner, model = setup(depth=3)
    atom = model.belief.mass[0][0]
    monkeypatch.setattr(
        planner.gen_model, "tree_step", lambda *args: (atom, {"i": "OL", "j": "L"}, -100, True)
    )
    monkeypatch.setattr(planner.solver_bank, "expected_rewards", lambda model: (("OL", 3.0),))

    def unexpected(*args):
        raise AssertionError("A terminal outcome has no continuing posterior")

    monkeypatch.setattr(planner, "_history_successor", unexpected)
    node = POMCPNode()
    assert planner._simulate(atom, node, 0, {}, model, random.Random(0)) == 3.0
    assert node.continuation_counts == {}
