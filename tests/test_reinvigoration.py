# Absolute Path: <project_root>/tests/test_reinvigoration.py

import pytest
from solvers.solver_types import SolverKey
from core.config import IPOMCPConfig, MCTSConfig, JITConfig, ReinvigorationConfig
from solvers.solver_bank import SolverBank
from solvers.i_pomcp import IPOMCPPlanner
from utils.bootstrapper import I_POMDP_Bootstrapper
from examples.tiger.model.tiger_model import (
    TigerModel, LISTEN, OPEN_LEFT, OPEN_RIGHT,
    GROWL_LEFT, GROWL_RIGHT, CREAK_LEFT, CREAK_RIGHT, SILENCE,
    TIGER_LEFT, TIGER_RIGHT
)


def _setup_l3_planner(prior_weights=None, n_particles=500):
    if prior_weights is None:
        prior_weights = {2: 0.80, 1: 0.10, 0: 0.10}
    env = TigerModel(creak_accuracy=1.0)
    bank = SolverBank()
    boot = I_POMDP_Bootstrapper(bank)
    cfg = IPOMCPConfig(
        mcts=MCTSConfig(n_sims=100, max_depth=3, node_capacity=n_particles),
        jit=JITConfig(sims=20),
        reinvigoration=ReinvigorationConfig(alpha=0.20, min_particles=100, preserve_levels=True)
    )
    planner = boot.create_solver(
        agent_id="i",
        level=3,
        model=env,
        other_agent_ids=["j"],
        level_weights=prior_weights,
        n_particles=n_particles,
        config=cfg
    )
    return env, planner, bank


def test_consistent_state_sampling():
    """Verify Bayesian physical state sampling given observations in Tiger."""
    env = TigerModel(growl_accuracy={'i': 0.85, 'j': 0.85})

    # When hearing GL, state should be predominantly TL (85%)
    tl_count = sum(1 for _ in range(1000) if env.sample_state_consistent_with_obs(LISTEN, (GROWL_LEFT, SILENCE), agent_id='i') == TIGER_LEFT)
    assert 800 <= tl_count <= 900

    # When hearing GR, state should be predominantly TR (85%)
    tr_count = sum(1 for _ in range(1000) if env.sample_state_consistent_with_obs(LISTEN, (GROWL_RIGHT, SILENCE), agent_id='i') == TIGER_RIGHT)
    assert 800 <= tr_count <= 900

    # When deafened, state should be uniform 50/50
    tl_deaf = sum(1 for _ in range(1000) if env.sample_state_consistent_with_obs(OPEN_LEFT, (SILENCE, SILENCE), agent_id='i') == TIGER_LEFT)
    assert 430 <= tl_deaf <= 570


def test_opponent_door_opening_preserves_opponent_level_prior():
    """Verify that when opponent J opens a door (CREAK heard), Level-2 does NOT collapse to Level-0."""
    env, planner, bank = _setup_l3_planner(prior_weights={2: 0.80, 1: 0.10, 0: 0.10}, n_particles=500)

    # Verify initial belief
    initial_l2_ratio = sum(1 for p in planner.root.belief_particles if p.models["j"][0].level == 2) / len(planner.root.belief_particles)
    assert initial_l2_ratio >= 0.70

    # Simulate step: agent i listens, opponent j opens right door -> agent i observes (GL, CR)
    a_i = LISTEN
    o_i = (GROWL_LEFT, CREAK_RIGHT)

    assert env.is_epoch_reset(a_i, o_i) is True

    planner.update_root(a_i, o_i, min_particles=200)

    particles_after = planner.root.belief_particles
    assert len(particles_after) >= 200

    count_l0 = sum(1 for p in particles_after if p.models["j"][0].level == 0)
    count_l1 = sum(1 for p in particles_after if p.models["j"][0].level == 1)
    count_l2 = sum(1 for p in particles_after if p.models["j"][0].level == 2)
    total = len(particles_after)

    prob_l2 = count_l2 / total
    prob_l1 = count_l1 / total
    prob_l0 = count_l0 / total

    # P(L2) must be preserved around prior (80%), absolutely MUST NOT collapse to 0
    assert prob_l2 >= 0.65, f"Level-2 collapsed! prob_l2 = {prob_l2}"
    assert prob_l0 <= 0.25, f"Level-0 dominated! prob_l0 = {prob_l0}"
    assert prob_l1 > 0.0, f"Level-1 went extinct! prob_l1 = {prob_l1}"

    # Opponent models must have valid fresh root nodes
    for p in particles_after:
        frame, node = p.models["j"]
        if frame.level > 0:
            assert node is not None
            assert len(node.belief_particles) > 0


def test_agent_door_opening_preserves_prior():
    """Verify that when agent I opens a door, opponent level distribution is preserved."""
    env, planner, bank = _setup_l3_planner(prior_weights={2: 0.80, 1: 0.10, 0: 0.10}, n_particles=500)

    a_i = OPEN_LEFT
    o_i = (SILENCE, SILENCE)

    assert env.is_epoch_reset(a_i, o_i) is True

    planner.update_root(a_i, o_i, min_particles=200)

    particles_after = planner.root.belief_particles
    total = len(particles_after)
    assert total >= 200

    prob_l2 = sum(1 for p in particles_after if p.models["j"][0].level == 2) / total
    assert prob_l2 >= 0.65, f"Level-2 collapsed! prob_l2 = {prob_l2}"


def test_normal_step_mixture_reinvigoration_prevents_extinction():
    """Verify that during normal listening steps, mixture reinvigoration maintains representation of all prior levels."""
    env, planner, bank = _setup_l3_planner(prior_weights={2: 0.80, 1: 0.10, 0: 0.10}, n_particles=500)

    # Perform a normal update (listen, silence creak)
    a_i = LISTEN
    o_i = (GROWL_LEFT, SILENCE)

    planner.update_root(a_i, o_i, min_particles=200)

    particles_after = planner.root.belief_particles
    total = len(particles_after)
    assert total >= 200

    count_l0 = sum(1 for p in particles_after if p.models["j"][0].level == 0)
    count_l1 = sum(1 for p in particles_after if p.models["j"][0].level == 1)
    count_l2 = sum(1 for p in particles_after if p.models["j"][0].level == 2)

    # All levels from prior must be represented
    assert count_l0 > 0, "Level 0 went extinct"
    assert count_l1 > 0, "Level 1 went extinct"
    assert count_l2 > 0, "Level 2 went extinct"
    assert count_l2 / total >= 0.60


def test_deep_hierarchy_l3_vs_l2_multi_step(monkeypatch):
    """Verify end-to-end multi-step batch execution retains Level-2 modeling without collapse."""
    from examples.experiments import deep_hierarchy_prior_experiment as dhpe
    from core.config import ExperimentConfig

    monkeypatch.setattr(dhpe, "SIM_SCHEDULE", {0: 0, 1: 50, 2: 100, 3: 150})
    monkeypatch.setattr(dhpe, "PARTICLE_SCHEDULE", {0: 0, 1: 100, 2: 200, 3: 300})

    exp_config = ExperimentConfig(
        n_trials=2,
        max_steps=6,
        export_trees=False,
        verbose=False
    )
    runner = dhpe.DeepHierarchyTigerRunner(
        config=exp_config,
        log_dir=None,
        level_i=3,
        level_j=2,
        prior_weights_i={2: 0.80, 1: 0.10, 0: 0.10},
        prior_weights_j={1: 0.80, 0: 0.20},
        planning_depth=3
    )
    df = runner.run_batch(max_workers=1)
    assert not df.empty
    assert "prob_l2_j" in df.columns
    active_df = df[df["status"] == "Active"]
    assert len(active_df) > 0
    mean_prob_l2 = active_df["prob_l2_j"].mean()
    assert mean_prob_l2 >= 0.60, f"Mean prob_l2 across steps was {mean_prob_l2}, expected >= 0.60"
    min_prob_l2 = active_df["prob_l2_j"].min()
    assert min_prob_l2 > 0.10, f"Min prob_l2 was {min_prob_l2}, collapsed"

