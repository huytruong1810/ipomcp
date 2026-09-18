"""Executable counterexamples for unresolved model semantics.

Strict expected failures are deliberately visible: green engineering regressions
do not certify an I-POMDP filter. Removing an xfail requires a principled fix, not
a looser assertion or a target reward chosen after observing benchmark results.
"""

import random

import pytest

from core.config import IPOMCPConfig, JITConfig, MCTSConfig
from examples.tiger.model.tiger_model import CREAK_RIGHT, GROWL_LEFT, LISTEN, TigerModel
from solvers.solver_bank import SolverBank
from utils.bootstrapper import I_POMDP_Bootstrapper


@pytest.mark.xfail(
    strict=True, reason="P0: epoch reset discards diagnostic opponent-action evidence"
)
def test_creak_updates_type_under_fixed_known_policies(monkeypatch):
    random.seed(512)
    model = TigerModel(creak_accuracy=1.0)
    bank = SolverBank()
    planner = I_POMDP_Bootstrapper(bank).create_solver(
        "i",
        2,
        model,
        ["j"],
        n_particles=200,
        level_weights={1: 0.8, 0: 0.2},
        config=IPOMCPConfig(
            mcts=MCTSConfig(n_sims=10, node_capacity=200, max_depth=2), jit=JITConfig(sims=1)
        ),
    )
    # Controlled policy fixture: L1 certainly listens; L0 opens right with 1/3
    # probability. A perfect right creak therefore has zero likelihood under L1.
    # The newly reset tiger/growl is independent and cancels from the type odds.
    for particle in planner.root.belief_particles:
        frame, node = particle.models["j"]
        if frame.level == 1:
            node.visit_count = 10
            node.action_values = {LISTEN: 0.0}
            node.action_counts = {LISTEN: 10}
            monkeypatch.setattr(
                bank.get_solver_for_frame(frame), "extend_search", lambda *a, **k: None
            )
            assert planner.gen_model._sample_action_from_node(node) == LISTEN
    planner.update_root(LISTEN, (GROWL_LEFT, CREAK_RIGHT), min_particles=200)
    posterior = sum(p.models["j"][0].level == 1 for p in planner.root.belief_particles) / len(
        planner.root.belief_particles
    )
    assert posterior == 0.0


@pytest.mark.xfail(
    strict=True, reason="P0: deeper private models are copied without recursive observation updates"
)
def test_nested_private_model_advances_with_a_simulated_step(monkeypatch):
    random.seed(182)
    model = TigerModel()
    bank = SolverBank()
    planner = I_POMDP_Bootstrapper(bank).create_solver(
        "i",
        3,
        model,
        ["j"],
        n_particles=20,
        level_weights={2: 1.0},
        nested_level_weights={2: {1: 1.0}, 1: {0: 1.0}},
        config=IPOMCPConfig(mcts=MCTSConfig(n_sims=10, node_capacity=20, max_depth=2)),
    )
    particle = planner.root.belief_particles[0]
    frame, opponent = particle.models["j"]
    opponent.visit_count = 10
    opponent.action_values = {LISTEN: 0.0}
    opponent.action_counts = {LISTEN: 10}
    monkeypatch.setattr(bank.get_solver_for_frame(frame), "extend_search", lambda *a, **k: None)
    original_private_nodes = {id(p.models["i"][1]) for p in opponent.belief_particles}
    next_particle, _, _, _ = planner.gen_model.tree_step(particle, LISTEN, "i", model)
    updated = next_particle.models["j"][1]
    # A new private observation changes the nested belief history, even if the
    # finite sample physical-state histogram happens to coincide by chance.
    assert all(id(p.models["i"][1]) not in original_private_nodes for p in updated.belief_particles)
