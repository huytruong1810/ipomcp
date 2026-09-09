import os
import sys
import json
import time
import numpy as np
import pandas as pd
from core.config import ExperimentConfig, IPOMCPConfig, MCTSConfig, JITConfig
from core.logger import get_logger
from examples.tiger.model.tiger_model import TigerModel
from solvers.solver_bank import SolverBank
from solvers.exploration import NormalizedUCB
from utils.bootstrapper import I_POMDP_Bootstrapper
from utils.generic_batch_runner import GenericBatchRunner

logger = get_logger("TigerBenchmark")

class TigerBenchmarkRunner(GenericBatchRunner):
    """
    Standardized benchmark runner for Tiger L2 vs L1.
    Uses independent solver banks for Agent I and Agent J to guarantee complete isolation.
    """
    def _setup_domain(self):
        growl_dict = {'i': 0.85, 'j': 0.85}
        env = TigerModel(growl_accuracy=growl_dict, creak_accuracy=1.0)

        mcts_cfg = MCTSConfig(n_sims=10000, max_depth=6, node_capacity=500)
        jit_cfg = JITConfig(entropy_threshold=0.6, visit_threshold=5, sims=10)
        agent_config = IPOMCPConfig(mcts=mcts_cfg, jit=jit_cfg)

        # Agent J (Level 1) with isolated bank
        bank_j = SolverBank()
        boot_j = I_POMDP_Bootstrapper(bank_j)
        boot_j.create_level0_solver('i', TigerModel(growl_accuracy=growl_dict))
        planner_j = boot_j.create_level1_solver(
            'j', TigerModel(growl_accuracy=growl_dict), ['i'],
            n_particles=2000,
            config=agent_config,
            exploration_strategy=NormalizedUCB(exploration_const=2**(0.5))
        )

        # Agent I (Level 2) with isolated bank
        bank_i = SolverBank()
        boot_i = I_POMDP_Bootstrapper(bank_i)
        boot_i.create_level0_solver('j', TigerModel(growl_accuracy=growl_dict))
        boot_i.create_level1_solver(
            'j', TigerModel(growl_accuracy=growl_dict), ['i'],
            n_particles=2000,
            config=agent_config,
            exploration_strategy=NormalizedUCB(exploration_const=2**(0.5))
        )
        planner_i = boot_i.create_level2_solver(
            'i', TigerModel(growl_accuracy=growl_dict), ['j'],
            l1_probability=1.0,
            n_particles=2000,
            config=agent_config,
            exploration_strategy=NormalizedUCB(exploration_const=2**(0.5))
        )

        return env, planner_i, planner_j, env.get_initial_state()

def run_benchmark(tag: str, n_trials: int = 30, max_steps: int = 6, max_workers: int = 6) -> dict:
    from core.paths import get_results_dir
    output_dir = get_results_dir("benchmarks", tag)

    config = ExperimentConfig(
        n_trials=n_trials,
        max_steps=max_steps,
        export_trees=False,
        verbose=False
    )

    runner = TigerBenchmarkRunner(config=config, log_dir=output_dir)
    t0 = time.time()
    df = runner.run_batch(max_workers=max_workers)
    wall_time = time.time() - t0

    # Extract metrics at the final horizon step
    final_step = df[df["step"] == max_steps]
    cum_r_i = final_step["cum_reward_i"]
    cum_r_j = final_step["cum_reward_j"]

    # Compute step-level stats
    active_steps_i = df[df["step"] > 0]
    time_i = active_steps_i["planning_time_i"]

    summary = {
        "tag": tag,
        "n_trials": n_trials,
        "max_steps": max_steps,
        "wall_time_sec": wall_time,
        "cum_reward_i_mean": float(cum_r_i.mean()),
        "cum_reward_i_std": float(cum_r_i.std()),
        "cum_reward_i_sem": float(cum_r_i.sem()),
        "cum_reward_i_median": float(cum_r_i.median()),
        "cum_reward_i_min": float(cum_r_i.min()),
        "cum_reward_i_max": float(cum_r_i.max()),
        "cum_reward_j_mean": float(cum_r_j.mean()),
        "cum_reward_j_std": float(cum_r_j.std()),
        "cum_reward_j_sem": float(cum_r_j.sem()),
        "cum_reward_j_median": float(cum_r_j.median()),
        "cum_reward_j_min": float(cum_r_j.min()),
        "cum_reward_j_max": float(cum_r_j.max()),
        "mean_planning_time_i_sec": float(time_i.mean()),
        "std_planning_time_i_sec": float(time_i.std()),
    }

    summary_file = os.path.join(output_dir, f"summary_{tag}.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"=== BENCHMARK COMPLETED: {tag} ===")
    logger.info(f"Wall Time: {wall_time:.2f}s")
    logger.info(f"Agent I (L2) Cumulative Reward: {summary['cum_reward_i_mean']:.2f} +/- {summary['cum_reward_i_sem']:.2f} (Median: {summary['cum_reward_i_median']:.2f})")
    logger.info(f"Agent J (L1) Cumulative Reward: {summary['cum_reward_j_mean']:.2f} +/- {summary['cum_reward_j_sem']:.2f} (Median: {summary['cum_reward_j_median']:.2f})")
    logger.info(f"Mean Planning Time (I): {summary['mean_planning_time_i_sec']:.3f}s / step")
    print(json.dumps(summary, indent=2))
    return summary

if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "baseline_pre_edits"
    run_benchmark(tag=tag, n_trials=30, max_steps=6, max_workers=6)
