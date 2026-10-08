"""Level-4 oracle problems and payoff variants: i-L4 -> j-L3 -> i-L2 -> j-L1 -> uniform i-L0."""

import json

import pytest

from ipomdp.domains.tiger import JointTigerModel
from ipomdp.experiments.oracle import _two_state_belief, evaluate_case
from ipomdp.experiments.panel import run_panel
from ipomdp.reference.exact_agent import build_exact_agent


def test_level_four_reference_on_a_payoff_variant_matches_the_exact_hierarchy():
    # With every point belief at 0.5 the oracle's problem is build_exact_agent's
    # point-level hierarchy, built along a different path, on the same physics.
    physics = JointTigerModel(together_safe=40.0)
    row = evaluate_case(
        "mcts",
        2,
        50,
        0,
        0.5,
        level=4,
        opponent_depth=2,
        reference="exact",
        physics=physics,
    )[0]
    nested = {4: {3: 1.0}, 3: {2: 1.0}, 2: {1: 1.0}, 1: {0: 1.0}}
    agent = build_exact_agent("i", "j", 4, physics, nested, 2, 0.95)
    exact = agent.q_values(agent.model())
    assert {str(a): q for a, q in exact.items()} == pytest.approx(
        {str(a): q for a, q in row["oracle_q"].items()}, abs=1e-12
    )
    assert row["opponent"]["nested_initial_beliefs_p"] == [0.5, 0.5]
    # Two steps left: chance scoring is exact for the searched (here exact) model.
    assert row["first_action_loss"] == pytest.approx(0, abs=1e-9)


def test_level_four_point_belief_nests_the_whole_chain():
    physics = JointTigerModel()
    model = None
    belief = _two_state_belief(physics, 4, 0.2, 0.3, (0.4, 0.6))
    found = []
    while belief is not None:
        atom, p = max(belief.mass, key=lambda item: item[1])
        found.append(round(p, 12))
        model = atom.opponent
        belief = model.belief
    assert found == [0.8, 0.7, 0.6, 0.6]
    assert (model.frame.agent_id, model.frame.level) == ("i", 0)


def test_nested_chain_length_must_match_the_level():
    with pytest.raises(ValueError, match="nested beliefs"):
        evaluate_case("mcts", 1, 10, 0, 0.5, level=3, opponent_depth=1, nested_beliefs=(0.5, 0.5))


def test_level_four_panel_crosses_every_nested_level_on_the_variant(tmp_path):
    cells = [("mcts", 4, 1, 20, "exact", 0, "exact", 0)]
    summary, failures = run_panel(
        tmp_path / "panel",
        cells,
        seeds=[0],
        workers=1,
        beliefs=(0.2,),
        opponent_beliefs=(0.5,),
        nested_beliefs=(0.4, 0.6),
        physics=JointTigerModel(together_safe=40.0),
    )
    assert failures == []
    assert summary.iloc[0]["n"] == 4 and summary.iloc[0]["optimal_rate"] == 1.0
    rows = json.loads((tmp_path / "panel" / "rows.json").read_text())
    assert {tuple(r["nested_beliefs"]) for r in rows} == {
        (0.4, 0.4),
        (0.4, 0.6),
        (0.6, 0.4),
        (0.6, 0.6),
    }
    manifest = json.loads((tmp_path / "panel" / "manifest.json").read_text())
    assert manifest["panel"]["physics"][1] == "JointTigerModel"
