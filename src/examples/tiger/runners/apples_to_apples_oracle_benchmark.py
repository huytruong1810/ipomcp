# Absolute Path: <project_root>/examples/tiger/runners/apples_to_apples_oracle_benchmark.py

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from core.pomdp_model import POMDPModel, State
from core.config import ExperimentConfig, IPOMCPConfig, MCTSConfig, JITConfig, RTSConfig
from core.paths import get_results_dir
from core.logger import get_logger
from solvers.planner import Planner
from solvers.solver_bank import SolverBank
from solvers.exploration import NormalizedUCB
from utils.bootstrapper import I_POMDP_Bootstrapper
from utils.generic_batch_runner import GenericBatchRunner
from utils.plotting import plot_all_metrics
from utils.paper_plots import generate_paper_plots
from examples.tiger.model.tiger_model import TigerModel

logger = get_logger("ApplesToApplesOracleBenchmark")


class ControlledConditionRunner(GenericBatchRunner):
    """
    Executes a single paired condition comparing either exact RTS Oracle or
    sample-based I-POMCP under strictly controlled planning depth and parameters.
    """
    def __init__(self, config: ExperimentConfig, log_dir: str,
                 solver_type: str, level_i: int, level_j: int,
                 n_sims: int, planning_depth: int,
                 n_particles: int = 2000, obs_branching: int = 6):
        super().__init__(config=config, log_dir=log_dir)
        self.solver_type = solver_type
        self.level_i = level_i
        self.level_j = level_j
        self.n_sims = n_sims
        self.planning_depth = planning_depth
        self.n_particles = n_particles
        self.obs_branching = obs_branching

    def _setup_domain(self) -> Tuple[POMDPModel, Planner, Planner, State]:
        growl_dict = {"i": 0.85, "j": 0.85}
        env = TigerModel(growl_accuracy=growl_dict, creak_accuracy=0.90)

        # 1. Opponent Agent J Setup (Level-1 I-POMCP)
        bank_j = SolverBank()
        boot_j = I_POMDP_Bootstrapper(bank_j)
        cfg_j = IPOMCPConfig(
            mcts=MCTSConfig(n_sims=50000, max_depth=self.planning_depth, node_capacity=2000),
            jit=JITConfig(entropy_threshold=0.6, visit_threshold=5, sims=10)
        )
        planner_j = boot_j.create_solver(
            agent_id="j",
            level=self.level_j,
            model=TigerModel(growl_accuracy=growl_dict),
            other_agent_ids=["i"],
            n_particles=2000,
            config=cfg_j,
            exploration_strategy=NormalizedUCB(exploration_const=2**0.5)
        )

        # 2. Protagonist Agent I Setup (Exact RTS Oracle vs I-POMCP at depth D)
        bank_i = SolverBank()
        boot_i = I_POMDP_Bootstrapper(bank_i)
        if self.solver_type == "rts":
            rts_cfg = RTSConfig(
                max_depth=self.planning_depth,
                obs_branching=self.obs_branching,
                num_particles=self.n_particles
            )
            planner_i = boot_i.create_rts_solver(
                agent_id="i",
                level=self.level_i,
                model=TigerModel(growl_accuracy=growl_dict),
                other_agent_ids=["j"],
                level_weights={1: 1.0},
                n_particles=self.n_particles,
                config=rts_cfg
            )
        else:
            cfg_i = IPOMCPConfig(
                mcts=MCTSConfig(n_sims=self.n_sims, max_depth=self.planning_depth, node_capacity=2000),
                jit=JITConfig(entropy_threshold=0.6, visit_threshold=5, sims=10)
            )
            planner_i = boot_i.create_solver(
                agent_id="i",
                level=self.level_i,
                model=TigerModel(growl_accuracy=growl_dict),
                other_agent_ids=["j"],
                level_weights={1: 1.0},
                n_particles=self.n_particles,
                config=cfg_i,
                exploration_strategy=NormalizedUCB(exploration_const=2**0.5)
            )

        return env, planner_i, planner_j, env.get_initial_state()


def run_apples_to_apples_benchmark(n_trials: int = 50, max_steps: int = 6, planning_depth: int = 3, resume_dir: Optional[str] = None):
    if resume_dir and os.path.exists(resume_dir):
        master_dir = resume_dir
        logger.info(f"Resuming existing oracle benchmark from: {master_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        master_dir = get_results_dir("oracle", f"apples_to_apples_oracle_{timestamp}_N{n_trials}_T{max_steps}_D{planning_depth}")

    logger.info(f"=== STARTING APPLES-TO-APPLES ORACLE BENCHMARK (N={n_trials}, Horizon={max_steps}, Depth={planning_depth}) ===")
    logger.info(f"Results Directory: {master_dir}")

    conditions = [
        {"name": f"Exact RTS Oracle (Depth {planning_depth})", "type": "rts", "sims": 0, "particles": 500},
        {"name": f"I-POMCP 1k (Depth {planning_depth})", "type": "ipomcp", "sims": 1000, "particles": 2000},
        {"name": f"I-POMCP 5k (Depth {planning_depth})", "type": "ipomcp", "sims": 5000, "particles": 2000},
        {"name": f"I-POMCP 10k (Depth {planning_depth})", "type": "ipomcp", "sims": 10000, "particles": 2000},
        {"name": f"I-POMCP 25k (Depth {planning_depth})", "type": "ipomcp", "sims": 25000, "particles": 2000},
        {"name": f"I-POMCP 50k (Depth {planning_depth})", "type": "ipomcp", "sims": 50000, "particles": 2000},
        {"name": f"I-POMCP 100k (Depth {planning_depth})", "type": "ipomcp", "sims": 100000, "particles": 2000},
    ]

    exp_config = ExperimentConfig(n_trials=n_trials, max_steps=max_steps, export_trees=False, verbose=False)
    all_dfs = []
    condition_dfs = {}

    for cond_idx, cond in enumerate(conditions, 1):
        cond_name = cond["name"]
        sanitized_name = cond_name.replace(" ", "_").replace("(", "").replace(")", "").replace(":", "")
        cond_dir = os.path.join(master_dir, f"cond_{cond_idx}_{sanitized_name}")
        csv_path = os.path.join(cond_dir, "batch_results.csv")
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 1000:
            logger.info(f"[{cond_idx}/{len(conditions)}] Condition '{cond_name}' already completed. Skipping batch execution.")
            df = pd.read_csv(csv_path)
            all_dfs.append(df)
            condition_dfs[cond_name] = df
            continue

        logger.info(f"[{cond_idx}/{len(conditions)}] Evaluating Condition: {cond_name}...")

        runner = ControlledConditionRunner(
            config=exp_config,
            log_dir=cond_dir,
            solver_type=cond["type"],
            level_i=2,
            level_j=1,
            n_sims=cond["sims"],
            planning_depth=planning_depth,
            n_particles=cond["particles"],
            obs_branching=6
        )

        df = runner.run_batch(max_workers=6)
        if not df.empty:
            df["condition"] = cond_name
            df["solver_type"] = cond["type"]
            df["sims"] = cond["sims"]
            all_dfs.append(df)
            condition_dfs[cond_name] = df
            plot_all_metrics(
                df,
                agent_labels={'i': f"Agent I ({cond_name})", 'j': "Agent J (L1)"},
                title_prefix=cond_name,
                save_dir=cond_dir
            )
            generate_paper_plots(os.path.join(cond_dir, "batch_results.csv"), cond_dir)

    if not all_dfs:
        logger.error("No experimental data collected.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    summary_csv = os.path.join(master_dir, "apples_to_apples_summary.csv")
    combined.to_csv(summary_csv, index=False)

    # 1. Oracle Reference Baseline Extraction
    oracle_name = conditions[0]["name"]
    oracle_df = condition_dfs.get(oracle_name)
    oracle_final = oracle_df[oracle_df["step"] == max_steps] if oracle_df is not None else None
    oracle_mean_r = float(oracle_final["cum_reward_i"].mean()) if oracle_final is not None else 0.0
    oracle_median_r = float(oracle_final["cum_reward_i"].median()) if oracle_final is not None else 0.0
    oracle_time = float(oracle_df["planning_time_i"].mean()) if oracle_df is not None else 0.0

    # 2. Comparative Analysis & Policy Agreement
    results_table = []
    oracle_actions = oracle_df.set_index(["trial", "step"])["action_i"].to_dict() if oracle_df is not None else {}

    for cond in conditions:
        c_name = cond["name"]
        c_df = condition_dfs.get(c_name)
        if c_df is None:
            continue

        c_final = c_df[c_df["step"] == max_steps]
        mean_r = float(c_final["cum_reward_i"].mean())
        sem_r = float(c_final["cum_reward_i"].std() / (len(c_final) ** 0.5))
        median_r = float(c_final["cum_reward_i"].median())
        mean_time = float(c_df["planning_time_i"].mean())

        # Compute Policy Agreement Rate vs Oracle
        agreement_count = 0
        total_steps = 0
        for _, row in c_df.iterrows():
            t_key = (row["trial"], row["step"])
            if t_key in oracle_actions:
                total_steps += 1
                if row["action_i"] == oracle_actions[t_key]:
                    agreement_count += 1
        agreement_pct = (agreement_count / total_steps * 100.0) if total_steps > 0 else 0.0

        # Optimality percentage vs Oracle median/mean
        optimality_pct = (mean_r / oracle_mean_r * 100.0) if oracle_mean_r != 0 else 100.0
        speedup = (oracle_time / mean_time) if mean_time > 0 else 1.0

        results_table.append({
            "condition": c_name,
            "solver_type": cond["type"],
            "sims": cond["sims"],
            "mean_reward": mean_r,
            "sem_reward": sem_r,
            "median_reward": median_r,
            "mean_latency_s": mean_time,
            "policy_agreement_pct": agreement_pct,
            "optimality_pct": optimality_pct,
            "speedup_vs_oracle": speedup
        })

    report = {
        "timestamp": timestamp,
        "n_trials": n_trials,
        "max_steps": max_steps,
        "planning_depth": planning_depth,
        "oracle_mean_reward": oracle_mean_r,
        "oracle_median_reward": oracle_median_r,
        "oracle_mean_latency_s": oracle_time,
        "results": results_table
    }

    with open(os.path.join(master_dir, "apples_to_apples_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # 3. Publication Visualizations
    # Figure 1: Pareto Frontier (Latency vs Cumulative Reward)
    plt.figure(figsize=(9, 6))
    pareto_df = pd.DataFrame(results_table)
    
    # Plot I-POMCP points
    ipomcp_pts = pareto_df[pareto_df["solver_type"] == "ipomcp"]
    plt.plot(ipomcp_pts["mean_latency_s"], ipomcp_pts["mean_reward"], marker="o", color="#ff7f0e", linewidth=2.5, label="I-POMCP (1k - 100k Sims)")
    for _, row in ipomcp_pts.iterrows():
        plt.annotate(f"{row['sims']//1000}k sims\n({row['policy_agreement_pct']:.1f}% agreement)", 
                     (row["mean_latency_s"], row["mean_reward"]),
                     textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8)

    # Plot Oracle Point
    oracle_pt = pareto_df[pareto_df["solver_type"] == "rts"].iloc[0]
    plt.scatter([oracle_pt["mean_latency_s"]], [oracle_pt["mean_reward"]], color="#1f77b4", s=180, zorder=5, marker="*", label=f"Exact RTS Oracle (Depth {planning_depth})")
    plt.annotate(f"Exact Oracle\n({oracle_pt['mean_latency_s']*1000:.1f}ms)", 
                 (oracle_pt["mean_latency_s"], oracle_pt["mean_reward"]),
                 textcoords="offset points", xytext=(0, 12), ha="center", fontsize=9, fontweight="bold", color="#1f77b4")

    plt.title(f"Pareto Efficiency: Cumulative Reward vs Planning Latency (Depth D={planning_depth}, T={max_steps})", fontsize=12, fontweight="bold")
    plt.xlabel("Mean Planning Time per Step (seconds, log scale)", fontsize=11)
    plt.ylabel("Mean Cumulative Reward ($R_i$)", fontsize=11)
    plt.xscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(frameon=True, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(master_dir, "fig_pareto_runtime_vs_optimality.pdf"), format="pdf", dpi=300)
    plt.close()

    # Figure 2: Policy Agreement Rate vs MCTS Simulation Count
    plt.figure(figsize=(8, 5))
    plt.plot(ipomcp_pts["sims"], ipomcp_pts["policy_agreement_pct"], marker="s", color="#2ca02c", linewidth=2)
    plt.axhline(100.0, color="gray", linestyle="--", alpha=0.7, label="Exact Oracle Agreement (100%)")
    plt.title(f"Policy Action Agreement with Exact Oracle vs MCTS Simulations (Depth D={planning_depth})", fontsize=12, fontweight="bold")
    plt.xlabel("MCTS Simulations ($N_{sims}$)", fontsize=11)
    plt.ylabel("Policy Agreement with Oracle (%)", fontsize=11)
    plt.ylim(50, 105)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(master_dir, "fig_policy_agreement_rate.pdf"), format="pdf", dpi=300)
    plt.close()

    # Figure 3: Apples-to-Apples Bar Plot
    plt.figure(figsize=(10, 5.5))
    final_df = combined[combined["step"] == max_steps]
    palette = ["#1f77b4" if "Oracle" in c else "#ff7f0e" for c in final_df["condition"].unique()]
    sns.barplot(data=final_df, x="condition", y="cum_reward_i", hue="condition", errorbar=("ci", 95), palette=palette, legend=False)
    plt.title(f"Apples-to-Apples Controlled Evaluation: Oracle vs I-POMCP (Depth D={planning_depth}, T={max_steps})", fontsize=12, fontweight="bold")
    plt.xlabel("Planner Configuration", fontsize=11)
    plt.ylabel("Agent I Cumulative Reward ($R_i$)", fontsize=11)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(master_dir, "fig_apples_to_apples_reward_bars.pdf"), format="pdf", dpi=300)
    plt.close()

    logger.info(f"=== APPLES-TO-APPLES BENCHMARK COMPLETE. ARTIFACTS IN: {master_dir} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apples-to-Apples Controlled Oracle vs I-POMCP Benchmark")
    parser.add_argument("--trials", type=int, default=50, help="Number of Monte Carlo trials per condition")
    parser.add_argument("--steps", type=int, default=6, help="Number of environment steps per trial")
    parser.add_argument("--depth", type=int, default=3, help="Planning horizon depth D for both Oracle and I-POMCP")
    args = parser.parse_args()

    run_apples_to_apples_benchmark(n_trials=args.trials, max_steps=args.steps, planning_depth=args.depth)
