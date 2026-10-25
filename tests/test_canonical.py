"""Canonical keys and digests built from cached fragments equal the plain JSON definitions."""

import hashlib
import json

import pytest

from ipomdp.belief import canonical
from ipomdp.belief.finite_belief import FiniteBelief
from ipomdp.domains.tiger import (
    ACTIONS,
    OBSERVATIONS,
    TIGER_LEFT,
    TIGER_RIGHT,
    JointTigerModel,
    TigerModel,
)
from ipomdp.domains.uav import UAVModel
from ipomdp.domains.wumpus.model import WumpusModel
from ipomdp.experiments.bootstrapper import HierarchyBuilder
from ipomdp.planning.config import PlannerConfig
from ipomdp.planning.solver_bank import SolverBank

NESTED = {2: {1: 0.5, 0: 0.5}, 1: {0: 1.0}}


def reference_digest(belief):
    """The definition: the digest of the atom rows sorted by their JSON."""
    rows = sorted([canonical.atom_row(atom, mass) for atom, mass in belief.mass], key=json.dumps)
    return hashlib.sha256(
        json.dumps(rows, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def nested_beliefs(model):
    """``model``'s belief and every belief nested in it."""
    if model.belief is None:
        return []
    found = [model.belief]
    for atom, _ in model.belief.mass:
        found += nested_beliefs(atom.opponent)
    return found


def tiger_beliefs():
    """An L3 Tiger hierarchy's beliefs, before and after exact updates (uneven masses)."""
    config = PlannerConfig(simulations=20, depth=2, modeled_simulations=20)
    bank = SolverBank(3)
    model = (
        HierarchyBuilder(bank)
        .build(
            "i",
            2,
            JointTigerModel(together_safe=40.0),
            "j",
            config=config,
            nested_level_weights=NESTED,
        )
        .model()
    )
    beliefs = nested_beliefs(model)
    for observation in OBSERVATIONS[:2]:
        beliefs += nested_beliefs(
            bank.filter.advance(model, ACTIONS[0], observation, terminal=False)
        )
    return beliefs


def prior_belief(physics, atoms):
    """A level-1 belief over the first ``atoms`` prior states (uniform L0 opponent)."""
    config = PlannerConfig(simulations=20, depth=2, modeled_simulations=20)
    planner = HierarchyBuilder(SolverBank(5)).build("i", 1, physics, "j", config=config)
    return FiniteBelief(planner.belief.mass[:atoms])


@pytest.mark.parametrize(
    "beliefs",
    [
        tiger_beliefs,
        lambda: [prior_belief(UAVModel(), 200)],
        lambda: [prior_belief(WumpusModel(), 300)],
    ],
    ids=["tiger-l2", "uav-l1", "wumpus-l1"],
)
def test_keys_and_digests_equal_their_json_definitions(beliefs):
    for belief in beliefs():
        for row in belief.mass:
            assert canonical.atom_sort_key(row) == json.dumps(canonical.atom_row(*row))
        assert canonical.belief_digest(belief) == reference_digest(belief)
        assert canonical.canonical_mass_and_digest(belief) == (
            canonical.canonical_mass(belief),
            reference_digest(belief),
        )
        assert canonical.canonical_mass(belief) == tuple(
            sorted(belief.mass, key=lambda row: json.dumps(canonical.atom_row(*row)))
        )


def test_rollout_memory_memo_equals_the_update_and_returns_fresh_dicts():
    physics = TigerModel()
    for memory in ({TIGER_LEFT: 0.3, TIGER_RIGHT: 0.7}, {TIGER_RIGHT: 1.0}):
        for action in ACTIONS:
            for observation in OBSERVATIONS:
                expected = physics._rollout_update(memory, action, observation, "i")
                first = physics.update_rollout_memory(memory, action, observation, "i")
                second = physics.update_rollout_memory(memory, action, observation, "i")
                assert first == expected and list(first) == list(expected)
                assert first is not second
