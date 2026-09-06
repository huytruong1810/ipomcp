# Absolute Path: tests/test_oracle_rts.py

import pytest
import os
from solvers.solver_bank import SolverBank
from solvers.solver_types import SolverKey
from solvers.rts_planner import RTSPlanner
from core.config import RTSConfig, ExperimentConfig
from examples.tiger.model.tiger_model import TigerModel
from utils.bootstrapper import I_POMDP_Bootstrapper
from utils.generic_batch_runner import GenericBatchRunner


def test_level1_rts_planner_creation_and_action_values():
    bank = SolverBank()
    env = TigerModel()
    boot = I_POMDP_Bootstrapper(bank)
    
    rts_l1 = boot.create_level1_rts_solver("i", env, ["j"], n_particles=30, max_depth=2, obs_branching=2)
    assert isinstance(rts_l1, RTSPlanner)
    assert rts_l1.key == SolverKey("i", 1)
    
    action = rts_l1.get_action()
    assert action in env.get_all_actions("i")
    
    q_vals = rts_l1.get_action_values()
    assert isinstance(q_vals, dict)
    assert len(q_vals) == len(env.get_all_actions("i"))
    for a in env.get_all_actions("i"):
        assert a in q_vals
        assert isinstance(q_vals[a], float)
        
    stats = rts_l1.get_detailed_stats()
    assert "action_values" in stats
    assert stats["belief_size"] == 30


def test_level2_rts_planner_creation_and_execution():
    bank = SolverBank()
    env = TigerModel()
    boot = I_POMDP_Bootstrapper(bank)
    
    # Prerequisite models
    boot.create_level0_solver("j", env)
    boot.create_level1_solver("j", env, ["i"], n_particles=30)
    
    rts_l2 = boot.create_level2_rts_solver("i", env, ["j"], level_weights={0: 0.3, 1: 0.7}, n_particles=30, max_depth=2, obs_branching=2)
    assert isinstance(rts_l2, RTSPlanner)
    assert rts_l2.key == SolverKey("i", 2)
    
    action = rts_l2.get_action()
    assert action in env.get_all_actions("i")


def test_general_create_rts_solver_interface():
    bank = SolverBank()
    env = TigerModel()
    boot = I_POMDP_Bootstrapper(bank)
    
    rts_l0 = boot.create_rts_solver("i", 0, env, ["j"])
    assert bank.has_solver(SolverKey("i", 0))
    
    rts_l1 = boot.create_rts_solver("i", 1, env, ["j"], n_particles=20)
    assert isinstance(rts_l1, RTSPlanner)
    assert rts_l1.key == SolverKey("i", 1)
    
    rts_l2 = boot.create_rts_solver("i", 2, env, ["j"], level_weights={1: 1.0}, n_particles=20)
    assert isinstance(rts_l2, RTSPlanner)
    assert rts_l2.key == SolverKey("i", 2)


class SimpleTigerRTSBatchRunner(GenericBatchRunner):
    def _setup_domain(self):
        env = TigerModel()
        bank = SolverBank()
        boot = I_POMDP_Bootstrapper(bank)
        
        planner_i = boot.create_rts_solver("i", 1, env, ["j"], n_particles=20, config=RTSConfig(max_depth=2, obs_branching=2, num_particles=20))
        planner_j = boot.create_solver("j", 0, env, ["i"])
        return env, planner_i, planner_j, env.get_initial_state()


def test_rts_batch_runner_execution(tmp_path):
    exp_cfg = ExperimentConfig(n_trials=2, max_steps=3, export_trees=False, verbose=False)
    runner = SimpleTigerRTSBatchRunner(config=exp_cfg, log_dir=str(tmp_path))
    df = runner.run_batch(max_workers=2)
    
    assert not df.empty
    assert len(df) == 8  # 2 trials * (max_steps + 1)
    assert "cum_reward_i" in df.columns
    assert "n_particles_i" in df.columns
    assert df["n_particles_i"].iloc[0] == 20
    assert "prob_l0_j" in df.columns
