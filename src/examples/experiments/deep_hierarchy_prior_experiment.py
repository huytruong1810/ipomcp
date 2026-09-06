import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

from core.pomdp_model import POMDPModel, State
from core.config import ExperimentConfig, IPOMCPConfig, MCTSConfig, JITConfig
from core.logger import get_logger
from core.paths import get_results_dir
from solvers.solver_bank import SolverBank
from solvers.exploration import NormalizedUCB
from solvers.planner import Planner
from utils.bootstrapper import I_POMDP_Bootstrapper
from utils.generic_batch_runner import GenericBatchRunner, extract_nested_belief_hierarchy
from utils.plotting import plot_all_metrics, plot_nested_belief_sunburst, plot_episode_sunburst_slider
from utils.paper_plots import generate_paper_plots, generate_nested_sunburst_pdf
from examples.tiger.model.tiger_model import TigerModel

logger = get_logger("DeepHierarchyPriorExperiment")

# Tractable, memory-safe simulation and particle schedule across reasoning levels
SIM_SCHEDULE = {
    0: 0,
    1: 10000,
    2: 15000,
    3: 20000,
    4: 25000,
    5: 30000
}

PARTICLE_SCHEDULE = {
    0: 0,
    1: 1000,
    2: 1500,
    3: 2000,
    4: 2000,
    5: 2000
}


class DeepHierarchyTigerRunner(GenericBatchRunner):
    def __init__(self, config: ExperimentConfig, log_dir: str,
                 level_i: int, level_j: int,
                 prior_weights_i: Optional[Dict[int, float]] = None,
                 prior_weights_j: Optional[Dict[int, float]] = None,
                 planning_depth: int = 5):
        super().__init__(config=config, log_dir=log_dir)
        self.level_i = level_i
        self.level_j = level_j
        self.prior_weights_i = prior_weights_i
        self.prior_weights_j = prior_weights_j
        self.planning_depth = planning_depth

    def _setup_domain(self) -> Tuple[POMDPModel, Planner, Planner, State]:
        growl_dict = {"i": 0.85, "j": 0.85}
        env = TigerModel(growl_accuracy=growl_dict, creak_accuracy=0.90)

        # 1. Opponent Agent J Setup
        bank_j = SolverBank()
        boot_j = I_POMDP_Bootstrapper(bank_j)
        if self.level_j == 0:
            planner_j = boot_j.create_solver(agent_id="j", level=0, model=env, other_agent_ids=["i"])
        else:
            sims_j = SIM_SCHEDULE.get(self.level_j, 50000 * self.level_j)
            particles_j = PARTICLE_SCHEDULE.get(self.level_j, 5000 * self.level_j)
            cfg_j = IPOMCPConfig(
                mcts=MCTSConfig(n_sims=sims_j, max_depth=self.planning_depth, node_capacity=2000),
                jit=JITConfig(entropy_threshold=0.6, visit_threshold=5, sims=10)
            )

            planner_j = boot_j.create_solver(
                agent_id="j",
                level=self.level_j,
                model=TigerModel(growl_accuracy=growl_dict),
                other_agent_ids=["i"],
                level_weights=self.prior_weights_j,
                n_particles=particles_j,
                config=cfg_j,
                exploration_strategy=NormalizedUCB(exploration_const=2**0.5)
            )

        # 2. Protagonist Agent I Setup
        bank_i = SolverBank()
        boot_i = I_POMDP_Bootstrapper(bank_i)
        if self.level_i == 0:
            planner_i = boot_i.create_solver(agent_id="i", level=0, model=env, other_agent_ids=["j"])
        else:
            sims_i = SIM_SCHEDULE.get(self.level_i, 50000 * self.level_i)
            particles_i = PARTICLE_SCHEDULE.get(self.level_i, 5000 * self.level_i)
            cfg_i = IPOMCPConfig(
                mcts=MCTSConfig(n_sims=sims_i, max_depth=self.planning_depth, node_capacity=2000),
                jit=JITConfig(entropy_threshold=0.6, visit_threshold=5, sims=10)
            )

            planner_i = boot_i.create_solver(
                agent_id="i",
                level=self.level_i,
                model=TigerModel(growl_accuracy=growl_dict),
                other_agent_ids=["j"],
                level_weights=self.prior_weights_i,
                n_particles=particles_i,
                config=cfg_i,
                exploration_strategy=NormalizedUCB(exploration_const=2**0.5)
            )

        return env, planner_i, planner_j, env.get_initial_state()


def run_deep_prior_experiment(n_trials: int = 50, max_steps: int = 20, planning_depth: int = 5, resume_dir: Optional[str] = None):
    if resume_dir and os.path.exists(resume_dir):
        master_dir = resume_dir
        logger.info(f"Resuming existing benchmark from: {master_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        master_dir = get_results_dir("deep_prior", f"deep_prior_benchmark_{timestamp}_N{n_trials}_T{max_steps}")

    logger.info(f"=== STARTING DEEP HIERARCHY PRIOR BENCHMARK (N={n_trials}, T={max_steps}, Depth={planning_depth}) ===")
    logger.info(f"Results Directory: {master_dir}")

    conditions = [
        # Level 3 Variations
        {
            "name": "L3 vs L2 (80% L2 Prior)",
            "level_i": 3, "level_j": 2,
            "prior_i": {2: 0.80, 1: 0.10, 0: 0.10},
            "prior_j": {1: 0.80, 0: 0.20}
        },
        {
            "name": "L3 vs L1 (80% L1 Prior)",
            "level_i": 3, "level_j": 1,
            "prior_i": {1: 0.80, 2: 0.10, 0: 0.10},
            "prior_j": {0: 1.0}
        },
        {
            "name": "L3 vs L1 (80% Over-estimated L2 Prior)",
            "level_i": 3, "level_j": 1,
            "prior_i": {2: 0.80, 1: 0.10, 0: 0.10},
            "prior_j": {0: 1.0}
        },
        {
            "name": "L3 vs L1 (Uniform 1/3 Mixture Prior)",
            "level_i": 3, "level_j": 1,
            "prior_i": {0: 1/3, 1: 1/3, 2: 1/3},
            "prior_j": {0: 1.0}
        },

        # Level 4 Variations
        {
            "name": "L4 vs L3 (80% L3 Prior)",
            "level_i": 4, "level_j": 3,
            "prior_i": {3: 0.80, 2: 0.20/3, 1: 0.20/3, 0: 0.20/3},
            "prior_j": {2: 0.80, 1: 0.10, 0: 0.10}
        },
        {
            "name": "L4 vs L1 (80% Over-estimated L3 Prior)",
            "level_i": 4, "level_j": 1,
            "prior_i": {3: 0.80, 2: 0.20/3, 1: 0.20/3, 0: 0.20/3},
            "prior_j": {0: 1.0}
        },
        {
            "name": "L4 vs L1 (Uniform 1/4 Mixture Prior)",
            "level_i": 4, "level_j": 1,
            "prior_i": {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25},
            "prior_j": {0: 1.0}
        },

        # Level 5 Variations
        {
            "name": "L5 vs L4 (80% L4 Prior)",
            "level_i": 5, "level_j": 4,
            "prior_i": {4: 0.80, 3: 0.05, 2: 0.05, 1: 0.05, 0: 0.05},
            "prior_j": {3: 0.80, 2: 0.20/3, 1: 0.20/3, 0: 0.20/3}
        },
        {
            "name": "L5 vs L1 (80% Over-estimated L4 Prior)",
            "level_i": 5, "level_j": 1,
            "prior_i": {4: 0.80, 3: 0.05, 2: 0.05, 1: 0.05, 0: 0.05},
            "prior_j": {0: 1.0}
        },
        {
            "name": "L5 vs L1 (Uniform 1/5 Mixture Prior)",
            "level_i": 5, "level_j": 1,
            "prior_i": {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2},
            "prior_j": {0: 1.0}
        }
    ]

    all_dfs = []
    exp_config = ExperimentConfig(n_trials=n_trials, max_steps=max_steps, export_trees=False, verbose=False)

    for cond_idx, cond in enumerate(conditions, 1):
        cond_name = cond["name"]
        sanitized_name = cond_name.replace("/", "_div_").replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct")
        cond_dir = os.path.join(master_dir, f"cond_{cond_idx}_{sanitized_name}")
        csv_path = os.path.join(cond_dir, "batch_results.csv")
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 1000:
            logger.info(f"[{cond_idx}/{len(conditions)}] Condition '{cond_name}' already completed. Skipping batch execution.")
            df = pd.read_csv(csv_path)
            df["condition"] = cond_name
            all_dfs.append(df)
            continue

        logger.info(f"[{cond_idx}/{len(conditions)}] Running Prior Condition: {cond_name}...")

        runner = DeepHierarchyTigerRunner(
            config=exp_config,
            log_dir=cond_dir,
            level_i=cond["level_i"],
            level_j=cond["level_j"],
            prior_weights_i=cond["prior_i"],
            prior_weights_j=cond["prior_j"],
            planning_depth=planning_depth
        )

        # 1. Capture snapshots for Sunburst (Trial 0)
        _, snapshots = runner.run_single_trial_with_snapshots(trial_id=0)
        if snapshots:
            if 0 in snapshots:
                plot_nested_belief_sunburst(snapshots[0], title=f"{cond_name} - Prior Hierarchy (t=0)", save_dir=cond_dir, filename="sunburst_t0")
                generate_nested_sunburst_pdf(snapshots[0], os.path.join(cond_dir, "fig_sunburst_t0.pdf"), title=f"{cond_name} (t=0)")
            final_step = max(snapshots.keys())
            plot_nested_belief_sunburst(snapshots[final_step], title=f"{cond_name} - Posterior Hierarchy (t={final_step})", save_dir=cond_dir, filename="sunburst_final")
            generate_nested_sunburst_pdf(snapshots[final_step], os.path.join(cond_dir, "fig_sunburst_final.pdf"), title=f"{cond_name} (t={final_step})")
            plot_episode_sunburst_slider(snapshots, title_prefix=cond_name, save_dir=cond_dir, filename="sunburst_animated")

        # Memory-safe worker throttling to guarantee execution inside physical RAM
        max_lvl = max(cond["level_i"], cond["level_j"])
        if max_lvl <= 2:
            workers = 6
        elif max_lvl == 3:
            workers = 4
        else:
            workers = 2

        # 2. Run batch
        df = runner.run_batch(max_workers=workers)
        if not df.empty:
            df["condition"] = cond_name
            all_dfs.append(df)
            plot_all_metrics(df, agent_labels={"i": f"Agent I (L{cond['level_i']})", "j": f"Agent J (L{cond['level_j']})"}, title_prefix=cond_name, save_dir=cond_dir)
            generate_paper_plots(os.path.join(cond_dir, "batch_results.csv"), cond_dir)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(os.path.join(master_dir, "deep_prior_summary.csv"), index=False)

        # Cross-Condition Visualizations
        plt.figure(figsize=(12, 6))
        sns.barplot(data=combined[combined["step"] == max_steps], x="condition", y="cum_reward_i", errorbar=("ci", 95))
        plt.title(f"Impact of Strategic Prior Specification on Agent I Cumulative Reward (t={max_steps})", fontsize=12, fontweight="bold")
        plt.xlabel("Prior Modeling Condition")
        plt.ylabel("Agent I Final Cumulative Reward")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(master_dir, "fig_prior_comparison_bars.pdf"), format="pdf", dpi=300)
        plt.close()

        logger.info(f"=== DEEP PRIOR BENCHMARK COMPLETE. ALL ARTIFACTS IN: {master_dir} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=200, help="Number of trials per condition")
    parser.add_argument("--steps", type=int, default=12, help="Decision steps per trial")
    parser.add_argument("--planning-depth", type=int, default=5, help="MCTS tree search max depth")
    args = parser.parse_args()

    run_deep_prior_experiment(n_trials=args.trials, max_steps=args.steps, planning_depth=args.planning_depth)
