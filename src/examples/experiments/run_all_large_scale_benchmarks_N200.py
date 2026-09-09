# Absolute Path: <project_root>/examples/experiments/run_all_large_scale_benchmarks_N200.py

"""
run_all_large_scale_benchmarks_N200.py — Master Large-Scale Orchestration Suite (N=200 Trials, T=12 Steps).

Sequentially executes the three core experimental suites with high statistical power (N=200, T=12):
1. Deep Hierarchy Prior Benchmark (7 conditions x 200 trials = 1,400 episodes, Lv1-Lv4)
2. Apples-to-Apples Doshi-Gmytrasiewicz RTS vs I-POMCP (7 conditions x 200 trials = 1,400 episodes)
3. Full 5x5 Strategy-Level Payoff Matrix (25 cells x 200 trials = 5,000 episodes, Lv0-Lv4)
"""

import os
import sys
import argparse
import time
from datetime import datetime
from typing import Optional

from core.logger import get_logger
from examples.experiments.deep_hierarchy_prior_experiment import run_deep_prior_experiment
from examples.tiger.runners.apples_to_apples_oracle_benchmark import run_apples_to_apples_benchmark
from examples.experiments.level_convergence_matrix_experiment import run_payoff_matrix_experiment

logger = get_logger("MasterLargeScaleRunnerN200")


def run_all(n_trials: int = 200, max_steps: int = 12, run_suite: str = "all", resume_dir: Optional[str] = None):
    start_time = time.time()
    logger.info(f"================================================================================")
    logger.info(f"=== MASTER LARGE-SCALE BENCHMARK SUITE: N={n_trials} TRIALS, T={max_steps} STEPS ===")
    logger.info(f"================================================================================")

    # Cleanly isolate resume directories by suite to avoid directory collisions
    prior_resume = resume_dir if (run_suite == "prior" or (resume_dir and "deep_prior" in resume_dir)) else None
    oracle_resume = resume_dir if (run_suite == "oracle" or (resume_dir and "oracle" in resume_dir)) else None
    matrix_resume = resume_dir if (run_suite == "matrix" or (resume_dir and "payoff_matrix" in resume_dir)) else None

    # Suite 1: Deep Hierarchy Prior Benchmark (7 conditions, Lv1-Lv4)
    if run_suite in ["all", "prior"]:
        logger.info(f"\n>>> [1/3] LAUNCHING DEEP HIERARCHY PRIOR BENCHMARK (7 Conditions x {n_trials} Trials, T={max_steps}) <<<")
        t0 = time.time()
        run_deep_prior_experiment(n_trials=n_trials, max_steps=max_steps, planning_depth=5, resume_dir=prior_resume)
        logger.info(f">>> [1/3] Deep Hierarchy Prior Benchmark Completed in {(time.time() - t0)/60:.1f} minutes.")

    # Suite 2: Apples-to-Apples Oracle RTS vs I-POMCP Benchmark
    if run_suite in ["all", "oracle"]:
        logger.info(f"\n>>> [2/3] LAUNCHING APPLES-TO-APPLES ORACLE RTS BENCHMARK (7 Conditions x {n_trials} Trials, T={max_steps}) <<<")
        t0 = time.time()
        run_apples_to_apples_benchmark(n_trials=n_trials, max_steps=max_steps, planning_depth=3, resume_dir=oracle_resume)
        logger.info(f">>> [2/3] Apples-to-Apples Oracle Benchmark Completed in {(time.time() - t0)/60:.1f} minutes.")

    # Suite 3: Full 5x5 Strategy Level Payoff Matrix (Lv0-Lv4)
    if run_suite in ["all", "matrix"]:
        logger.info(f"\n>>> [3/3] LAUNCHING FULL 5x5 STRATEGY LEVEL PAYOFF MATRIX (25 Cells x {n_trials} Trials, T={max_steps}) <<<")
        t0 = time.time()
        run_payoff_matrix_experiment(max_level=4, n_trials=n_trials, max_steps=max_steps, planning_depth=5, resume_dir=matrix_resume)
        logger.info(f">>> [3/3] Full 5x5 Payoff Matrix Benchmark Completed in {(time.time() - t0)/60:.1f} minutes.")

    total_mins = (time.time() - start_time) / 60.0
    logger.info(f"\n================================================================================")
    logger.info(f"=== ALL REQUESTED BENCHMARKS FINISHED SUCCESSFULLY IN {total_mins:.1f} MINUTES ===")
    logger.info(f"================================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master Large-Scale Runner (N=200 Trials, T=12 Steps)")
    parser.add_argument("--trials", type=int, default=200, help="Number of trials per condition")
    parser.add_argument("--steps", type=int, default=12, help="Environment steps per trial (default: 12)")
    parser.add_argument("--suite", type=str, default="all", choices=["all", "prior", "oracle", "matrix"], help="Which suite to execute")
    parser.add_argument("--resume-dir", type=str, default=None, help="Directory of an existing benchmark run to resume")
    args = parser.parse_args()

    run_all(n_trials=args.trials, max_steps=args.steps, run_suite=args.suite, resume_dir=args.resume_dir)
