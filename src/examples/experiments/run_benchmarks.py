"""Run prior sensitivity, sampled planner comparison, and empirical payoff suites.

Trial count and interaction horizon are command-line parameters; planning depth
is specified for each suite. A payoff matrix over finitely many heuristic policies
does not establish convergence of reasoning hierarchies or exact Nash equilibrium.
"""

import argparse
import time
from typing import Optional

from core.logger import get_logger
from examples.experiments.deep_hierarchy_prior_experiment import run_deep_prior_experiment
from examples.experiments.level_convergence_matrix_experiment import run_payoff_matrix_experiment
from examples.experiments.planner_comparison_experiment import run_planner_comparison

logger = get_logger("BenchmarkSuites")


def run_all(
    n_trials: int = 200,
    max_steps: int = 20,
    run_suite: str = "all",
    resume_dir: Optional[str] = None,
    workers: int = 8,
    timeout: float = 2400.0,
    max_rss_mb: float = 4096.0,
):
    start_time = time.time()
    logger.info("================================================================================")
    logger.info(
        f"=== MASTER LARGE-SCALE BENCHMARK SUITE: N={n_trials} TRIALS, T={max_steps} STEPS ==="
    )
    logger.info("================================================================================")

    if resume_dir and run_suite in {"all", "optimal"}:
        raise ValueError("Resume requires --suite prior, comparison, or matrix")
    prior_resume = resume_dir if run_suite == "prior" else None
    comparison_resume = resume_dir if run_suite == "comparison" else None
    matrix_resume = resume_dir if run_suite == "matrix" else None

    # Suite 1: Deep Hierarchy Prior Benchmark (7 conditions, Lv1-Lv4)
    if run_suite in ["all", "prior"]:
        logger.info(
            f"\n>>> [1/3] LAUNCHING DEEP HIERARCHY PRIOR BENCHMARK (7 Conditions x {n_trials} Trials, T={max_steps}) <<<"
        )
        t0 = time.time()
        run_deep_prior_experiment(
            n_trials=n_trials,
            max_steps=max_steps,
            planning_depth=5,
            resume_dir=prior_resume,
            workers=workers,
            timeout=timeout,
            max_rss_mb=max_rss_mb,
        )
        logger.info(
            f">>> [1/3] Deep Hierarchy Prior Benchmark Completed in {(time.time() - t0) / 60:.1f} minutes."
        )

    # Suite 2: Sampled Planner Comparison Sampled RTS RTS vs I-POMCP Benchmark
    if run_suite in ["all", "comparison"]:
        logger.info(
            f"\n>>> [2/3] LAUNCHING SAMPLED RTS COMPARISON BENCHMARK (7 Conditions x {n_trials} Trials, T={max_steps}) <<<"
        )
        t0 = time.time()
        run_planner_comparison(
            n_trials=n_trials,
            max_steps=max_steps,
            planning_depth=3,
            resume_dir=comparison_resume,
            workers=workers,
            timeout=timeout,
            max_rss_mb=max_rss_mb,
        )
        logger.info(
            f">>> [2/3] Sampled Planner Comparison Sampled RTS Benchmark Completed in {(time.time() - t0) / 60:.1f} minutes."
        )

    # Suite 3: Full 5x5 Strategy Level Payoff Matrix (Lv0-Lv4)
    if run_suite in ["all", "matrix"]:
        logger.info(
            f"\n>>> [3/3] LAUNCHING FULL 5x5 STRATEGY LEVEL PAYOFF MATRIX (25 Cells x {n_trials} Trials, T={max_steps}) <<<"
        )
        t0 = time.time()
        run_payoff_matrix_experiment(
            max_level=4,
            n_trials=n_trials,
            max_steps=max_steps,
            planning_depth=5,
            resume_dir=matrix_resume,
            workers=workers,
            timeout=timeout,
            max_rss_mb=max_rss_mb,
        )
        logger.info(
            f">>> [3/3] Full 5x5 Payoff Matrix Benchmark Completed in {(time.time() - t0) / 60:.1f} minutes."
        )

    # Suite 4: Master Bayes-Optimal Cross-Suite Benchmark
    if run_suite in ["all", "optimal"]:
        logger.info("\n>>> LAUNCHING MASTER BAYES-OPTIMAL BENCHMARK SUITE <<<")
        t0 = time.time()
        from datetime import datetime

        from core.paths import get_results_dir
        from examples.experiments.planner_oracle_experiment import run_oracle_comparison

        run_oracle_comparison(
            get_results_dir("oracle", datetime.now().strftime("%Y%m%d_%H%M%S")),
            seeds=n_trials,
            workers=workers,
            timeout=timeout,
            max_rss_mb=max_rss_mb,
        )
        logger.info(
            f">>> Master Bayes-Optimal Suite Completed in {(time.time() - t0) / 60:.1f} minutes."
        )

    total_mins = (time.time() - start_time) / 60.0
    logger.info(
        "\n================================================================================"
    )
    logger.info(
        f"=== ALL REQUESTED BENCHMARKS FINISHED SUCCESSFULLY IN {total_mins:.1f} MINUTES ==="
    )
    logger.info("================================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Master Large-Scale Runner (N=200 Trials, T=20 Steps)"
    )
    parser.add_argument("--trials", type=int, default=200, help="Number of trials per condition")
    parser.add_argument(
        "--steps", type=int, default=20, help="Environment steps per trial (default: 20)"
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Number of parallel worker processes (default: 8)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=2400.0,
        help="Timeout in seconds per trial (default: 2400.0)",
    )
    parser.add_argument(
        "--max-rss-mb",
        type=float,
        default=4096.0,
        help="Maximum resident memory in MB per trial (default: 4096.0)",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="all",
        choices=["all", "prior", "comparison", "matrix", "optimal"],
        help="Which suite to execute",
    )
    parser.add_argument(
        "--resume-dir",
        type=str,
        default=None,
        help="Directory of an existing benchmark run to resume",
    )
    args = parser.parse_args()

    run_all(
        n_trials=args.trials,
        max_steps=args.steps,
        run_suite=args.suite,
        resume_dir=args.resume_dir,
        workers=args.workers,
        timeout=args.timeout,
        max_rss_mb=args.max_rss_mb,
    )
