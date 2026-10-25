"""Level-3 oracle problems: i-L3 against a j-L2 who models i as L1, scored exactly."""

import pytest

from ipomdp.belief.finite_belief import MentalModel
from ipomdp.belief.frame import AgentFrame
from ipomdp.domains.tiger import TigerModel
from ipomdp.experiments.oracle import evaluate_case
from ipomdp.experiments.problems import chain_model, tiger_chain
from ipomdp.reference.exact_agent import build_exact_agent


def test_level_three_reference_matches_the_independently_built_exact_hierarchy():
    # build_exact_agent assembles the exact L3 -> L2 -> L1 -> L0 hierarchy from
    # the uniform prior; with every point belief at 0.5 the oracle's problem is
    # the same one, built along a different path.
    row = evaluate_case(
        "mcts", 2, 50, 0, 0.5, level=3, opponent_depth=2, opponent_belief=0.5, reference="exact"
    )[0]
    agent = build_exact_agent(
        "i", "j", 3, TigerModel(), {3: {2: 1.0}, 2: {1: 1.0}, 1: {0: 1.0}}, 2, 0.95
    )
    exact = agent.q_values(agent.model())
    assert {str(a): q for a, q in exact.items()} == pytest.approx(
        {str(a): q for a, q in row["oracle_q"].items()}, abs=1e-12
    )


def test_level_three_point_belief_nests_three_private_beliefs():
    physics = TigerModel()
    belief = chain_model(physics, "i", 3, tiger_chain(0.2, 0.3, (0.4,))).belief
    (atom, p), _ = sorted(belief.mass, key=lambda item: -item[1])
    j = atom.opponent
    assert (p, j.frame.agent_id, j.frame.level) == (0.8, "j", 2)
    i = next(iter(j.belief.mass))[0].opponent
    assert (i.frame.agent_id, i.frame.level) == ("i", 1)
    assert sorted(w for _, w in i.belief.mass) == pytest.approx([0.4, 0.6])
    assert next(iter(i.belief.mass))[0].opponent == MentalModel(AgentFrame("j", 0, physics))


def test_level_three_decisions_with_finite_opponents_are_scored_and_described():
    row = evaluate_case(
        "mcts",
        2,
        200,
        1,
        0.15,
        level=3,
        opponent_depth=2,
        opponent_belief=0.7,
        nested_beliefs=(0.4,),
        modeled="50",
        reference="exact",
    )[0]
    assert row["level"] == 3
    assert row["opponent"] == {
        "policy": "same_planner",
        "modeled": "50",
        "level": 2,
        "depth": 2,
        "initial_belief_p": 0.7,
        "nested_initial_beliefs_p": [0.4],
    }
    assert row["first_action_loss"] >= 0
    assert set(row["oracle_q"]) == set(row["estimated_q"])
