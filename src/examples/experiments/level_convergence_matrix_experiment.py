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
from core.config import (
    ExperimentConfig,
    IPOMCPConfig,
    MCTSConfig,
    JITConfig,
    DEFAULT_SIM_SCHEDULE as SIM_SCHEDULE,
    DEFAULT_PARTICLE_SCHEDULE as PARTICLE_SCHEDULE,
)
from core.paths import get_results_dir
from core.logger import get_logger
from solvers.solver_bank import SolverBank
from solvers.exploration import NormalizedUCB
from solvers.planner import Planner
from utils.bootstrapper import I_POMDP_Bootstrapper
from utils.generic_batch_runner import GenericBatchRunner, extract_nested_belief_hierarchy, is_batch_complete
from utils.plotting import plot_all_metrics, plot_nested_belief_sunburst, plot_episode_sunburst_slider
from utils.paper_plots import generate_paper_plots, generate_nested_sunburst_pdf
from examples.tiger.model.tiger_model import TigerModel

logger = get_logger("LevelConvergenceMatrix")


class MatrixCellTigerRunner(GenericBatchRunner):
    def __init__(self, config: ExperimentConfig, log_dir: str,
                 level_i: int, level_j: int,
                 prior_mode_i: str = "exact",
                 prior_mode_j: str = "exact",
                 planning_depth: int = 5):
        super().__init__(config=config, log_dir=log_dir)
        self.level_i = level_i
        self.level_j = level_j
        self.prior_mode_i = prior_mode_i
        self.prior_mode_j = prior_mode_j
        self.planning_depth = planning_depth

    def _setup_domain(self) -> Tuple[POMDPModel, Planner, Planner, State]:
        growl_dict = {"i": 0.85, "j": 0.85}
        env = TigerModel(growl_accuracy=growl_dict, creak_accuracy=1.0)

        # 1. Independent Opponent Agent J Setup
        bank_j = SolverBank()
        boot_j = I_POMDP_Bootstrapper(bank_j)
        if self.level_j == 0:
            planner_j = boot_j.create_solver(agent_id="j", level=0, model=env, other_agent_ids=["i"])
        else:
            sims_j = SIM_SCHEDULE.get(self.level_j, 10000 * self.level_j)
            particles_j = PARTICLE_SCHEDULE.get(self.level_j, 1000 * self.level_j)
            cfg_j = IPOMCPConfig(
                mcts=MCTSConfig(n_sims=sims_j, max_depth=self.planning_depth, node_capacity=2000),
                jit=JITConfig()
            )

            # Determine J prior over I
            target_model_for_j = min(self.level_i, self.level_j - 1)
            weights_j = {target_model_for_j: 1.0} if self.prior_mode_j == "exact" else None

            planner_j = boot_j.create_solver(
                agent_id="j",
                level=self.level_j,
                model=TigerModel(growl_accuracy=growl_dict, creak_accuracy=1.0),
                other_agent_ids=["i"],
                level_weights=weights_j,
                n_particles=particles_j,
                config=cfg_j,
                exploration_strategy=NormalizedUCB(exploration_const=2**0.5)
            )

        # 2. Independent Protagonist Agent I Setup (Completely isolated bank prevents cross-agent contamination)
        bank_i = SolverBank()
        boot_i = I_POMDP_Bootstrapper(bank_i)
        if self.level_i == 0:
            planner_i = boot_i.create_solver(agent_id="i", level=0, model=env, other_agent_ids=["j"])
        else:
            sims_i = SIM_SCHEDULE.get(self.level_i, 10000 * self.level_i)
            particles_i = PARTICLE_SCHEDULE.get(self.level_i, 1000 * self.level_i)
            cfg_i = IPOMCPConfig(
                mcts=MCTSConfig(n_sims=sims_i, max_depth=self.planning_depth, node_capacity=2000),
                jit=JITConfig()
            )

            # Determine I prior over J
            target_model_for_i = min(self.level_j, self.level_i - 1)
            weights_i = {target_model_for_i: 1.0} if self.prior_mode_i == "exact" else None

            planner_i = boot_i.create_solver(
                agent_id="i",
                level=self.level_i,
                model=TigerModel(growl_accuracy=growl_dict, creak_accuracy=1.0),
                other_agent_ids=["j"],
                level_weights=weights_i,
                n_particles=particles_i,
                config=cfg_i,
                exploration_strategy=NormalizedUCB(exploration_const=2**0.5)
            )

        return env, planner_i, planner_j, env.get_initial_state()


def plot_payoff_heatmaps(payoff_i: np.ndarray, payoff_j: np.ndarray, levels: List[int], out_dir: str):
    """Plots game-theoretic Payoff Heatmaps and Joint Welfare Matrix."""
    labels = [f"Level {l}" for l in levels]
    joint = payoff_i + payoff_j

    # 1. Agent I Payoff Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(payoff_i, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=labels, yticklabels=labels, cbar_kws={"label": "Mean Cum. Reward $R_i$"})
    plt.title("Agent I Payoff Matrix $R_i(m, n)$ (Row: Level $m$, Col: Opponent Level $n$)")
    plt.xlabel("Opponent Agent J Level ($n$)")
    plt.ylabel("Protagonist Agent I Level ($m$)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_payoff_matrix_agent_i.pdf"), format="pdf", dpi=300)
    plt.close()

    # 2. Agent J Payoff Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(payoff_j, annot=True, fmt=".2f", cmap="YlOrRd", xticklabels=labels, yticklabels=labels, cbar_kws={"label": "Mean Cum. Reward $R_j$"})
    plt.title("Agent J Payoff Matrix $R_j(m, n)$ (Row: Level $m$, Col: Opponent Level $n$)")
    plt.xlabel("Opponent Agent J Level ($n$)")
    plt.ylabel("Protagonist Agent I Level ($m$)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_payoff_matrix_agent_j.pdf"), format="pdf", dpi=300)
    plt.close()

    # 3. Joint Welfare Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(joint, annot=True, fmt=".2f", cmap="viridis", xticklabels=labels, yticklabels=labels, cbar_kws={"label": "Joint Reward $R_i + R_j$"})
    plt.title("Joint Social Welfare Matrix $(R_i + R_j)(m, n)$")
    plt.xlabel("Opponent Agent J Level ($n$)")
    plt.ylabel("Protagonist Agent I Level ($m$)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_joint_welfare_matrix.pdf"), format="pdf", dpi=300)
    plt.close()

    # 4. Asymptotic Convergence Curves: R_i(m | n) vs m
    plt.figure(figsize=(9, 5))
    for col_idx, n in enumerate(levels):
        plt.plot(levels, payoff_i[:, col_idx], marker="o", linewidth=2, label=f"vs Opponent L{n}")
    plt.title(r"Asymptotic Level Convergence: $R_i(m \mid n)$ vs Reasoning Level $m$")
    plt.xlabel("Protagonist Reasoning Level $m$")
    plt.ylabel("Agent I Mean Cumulative Reward")
    plt.xticks(levels)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_level_convergence_curves.pdf"), format="pdf", dpi=300)
    plt.close()


def find_nash_equilibria(payoff_i: np.ndarray, payoff_j: np.ndarray, levels: List[int]) -> Dict[str, Any]:
    """Identifies best responses and pure-strategy empirical Nash equilibria."""
    n_levels = len(levels)
    # Best response for i: for each column n (opponent level), argmax row m
    best_response_i = {}
    for j_idx, n in enumerate(levels):
        max_r_i = np.max(payoff_i[:, j_idx])
        best_m = [levels[m_idx] for m_idx in np.where(payoff_i[:, j_idx] == max_r_i)[0]]
        best_response_i[f"L{n}"] = {"best_m": best_m, "max_payoff": float(max_r_i)}

    # Best response for j: for each row m (agent i level), argmax col n
    best_response_j = {}
    for i_idx, m in enumerate(levels):
        max_r_j = np.max(payoff_j[i_idx, :])
        best_n = [levels[n_idx] for n_idx in np.where(payoff_j[i_idx, :] == max_r_j)[0]]
        best_response_j[f"L{m}"] = {"best_n": best_n, "max_payoff": float(max_r_j)}

    # Check for Nash Equilibria: (m, n) where m in BR_i(n) and n in BR_j(m)
    nash_points = []
    for i_idx, m in enumerate(levels):
        for j_idx, n in enumerate(levels):
            is_br_i = payoff_i[i_idx, j_idx] == np.max(payoff_i[:, j_idx])
            is_br_j = payoff_j[i_idx, j_idx] == np.max(payoff_j[i_idx, :])
            if is_br_i and is_br_j:
                nash_points.append({
                    "m_level": m,
                    "n_level": n,
                    "payoff_i": float(payoff_i[i_idx, j_idx]),
                    "payoff_j": float(payoff_j[i_idx, j_idx]),
                    "joint_payoff": float(payoff_i[i_idx, j_idx] + payoff_j[i_idx, j_idx])
                })

    return {
        "best_response_i": best_response_i,
        "best_response_j": best_response_j,
        "empirical_nash_equilibria": nash_points
    }


def run_payoff_matrix_experiment(max_level: int = 4, n_trials: int = 50, max_steps: int = 20, planning_depth: int = 5, resume_dir: Optional[str] = None):
    if resume_dir and os.path.exists(resume_dir):
        master_dir = resume_dir
        logger.info(f"Resuming existing payoff matrix benchmark from: {master_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        master_dir = get_results_dir("payoff_matrix", f"payoff_matrix_L0toL{max_level}_{timestamp}_N{n_trials}_T{max_steps}")

    levels = list(range(max_level + 1))
    n_levels = len(levels)
    logger.info(f"=== STARTING EMPIRICAL PAYOFF MATRIX BENCHMARK (Levels 0..{max_level}, N={n_trials}, T={max_steps}, Depth={planning_depth}) ===")
    logger.info(f"Results Directory: {master_dir}")

    payoff_matrix_i = np.zeros((n_levels, n_levels))
    payoff_matrix_j = np.zeros((n_levels, n_levels))
    sem_matrix_i = np.zeros((n_levels, n_levels))

    all_cell_dfs = []
    total_cells = n_levels * n_levels
    cell_count = 0

    exp_config = ExperimentConfig(n_trials=n_trials, max_steps=max_steps, export_trees=False, verbose=False)

    for i_idx, m in enumerate(levels):
        for j_idx, n in enumerate(levels):
            cell_count += 1
            cell_name = f"Cell_L{m}_vs_L{n}"
            cell_dir = os.path.join(master_dir, cell_name)
            csv_path = os.path.join(cell_dir, "batch_results.csv")
            if is_batch_complete(csv_path, n_trials):
                logger.info(f"[{cell_count}/{total_cells}] Cell '{cell_name}' already completed ({n_trials} trials). Skipping batch execution.")
                df = pd.read_csv(csv_path)
                df["cell"] = cell_name
                df["level_i"] = m
                df["level_j"] = n
                all_cell_dfs.append(df)
                payoff_matrix_i[i_idx, j_idx] = df[df["step"] == max_steps]["cum_reward_i"].mean()
                payoff_matrix_j[i_idx, j_idx] = df[df["step"] == max_steps]["cum_reward_j"].mean()
                sem_matrix_i[i_idx, j_idx] = df[df["step"] == max_steps]["cum_reward_i"].sem()
                continue

            logger.info(f"[{cell_count}/{total_cells}] Evaluating Cell: Agent I Level {m} vs Agent J Level {n}...")

            runner = MatrixCellTigerRunner(
                config=exp_config,
                log_dir=cell_dir,
                level_i=m,
                level_j=n,
                planning_depth=planning_depth
            )

            # Single trial snapshot if level >= 3
            if m >= 3 and cell_count == 1:
                _, snapshots = runner.run_single_trial_with_snapshots(trial_id=0)
                if snapshots:
                    plot_episode_sunburst_slider(snapshots, title_prefix=cell_name, save_dir=cell_dir, filename="sunburst_animated")

            # Memory-safe worker throttling to guarantee execution inside physical RAM
            max_lvl = max(m, n)
            if max_lvl <= 2:
                workers = 6
            elif max_lvl == 3:
                workers = 4
            else:
                workers = 2

            # Run batch
            df = runner.run_batch(max_workers=workers)
            if not df.empty:
                df["cell"] = cell_name
                df["level_i"] = m
                df["level_j"] = n
                all_cell_dfs.append(df)

                # Generate per-cell interactive and publication vector plots
                plot_all_metrics(
                    df,
                    agent_labels={"i": f"Agent I (Level {m})", "j": f"Agent J (Level {n})"},
                    title_prefix=cell_name,
                    save_dir=cell_dir
                )
                generate_paper_plots(os.path.join(cell_dir, "batch_results.csv"), cell_dir)

                # Final step rewards
                final_step = df["step"].max()
                final_df = df[df["step"] == final_step]
                r_i_mean = final_df["cum_reward_i"].mean()
                r_j_mean = final_df["cum_reward_j"].mean()
                r_i_sem = final_df["cum_reward_i"].std() / (len(final_df)**0.5)

                payoff_matrix_i[i_idx, j_idx] = r_i_mean
                payoff_matrix_j[i_idx, j_idx] = r_j_mean
                sem_matrix_i[i_idx, j_idx] = r_i_sem
                logger.info(f"  -> Finished Cell L{m} vs L{n}: Mean R_i = {r_i_mean:+.3f} (SEM={r_i_sem:.3f}), Mean R_j = {r_j_mean:+.3f}")

    # Save Payoff Matrix Grid CSV
    grid_rows = []
    for i_idx, m in enumerate(levels):
        for j_idx, n in enumerate(levels):
            grid_rows.append({
                "level_i": m,
                "level_j": n,
                "mean_reward_i": payoff_matrix_i[i_idx, j_idx],
                "sem_reward_i": sem_matrix_i[i_idx, j_idx],
                "mean_reward_j": payoff_matrix_j[i_idx, j_idx],
                "joint_reward": payoff_matrix_i[i_idx, j_idx] + payoff_matrix_j[i_idx, j_idx]
            })
    grid_df = pd.DataFrame(grid_rows)
    grid_df.to_csv(os.path.join(master_dir, "payoff_matrix_grid.csv"), index=False)

    if all_cell_dfs:
        combined = pd.concat(all_cell_dfs, ignore_index=True)
        combined.to_csv(os.path.join(master_dir, "combined_all_trials.csv"), index=False)

    # Generate Heatmaps & Convergence Curves
    logger.info("Generating Payoff Heatmaps and Asymptotic Convergence Curves...")
    plot_payoff_heatmaps(payoff_matrix_i, payoff_matrix_j, levels, master_dir)

    # Compute Empirical Nash Equilibria
    nash_analysis = find_nash_equilibria(payoff_matrix_i, payoff_matrix_j, levels)
    with open(os.path.join(master_dir, "empirical_nash_equilibria.json"), "w") as f:
        json.dump(nash_analysis, f, indent=2)

    logger.info(f"=== PAYOFF MATRIX BENCHMARK COMPLETE. ARTIFACTS IN: {master_dir} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-level", type=int, default=4, help="Maximum reasoning level to evaluate (0..max_level)")
    parser.add_argument("--trials", type=int, default=200, help="Number of trials per cell")
    parser.add_argument("--steps", type=int, default=20, help="Decision steps per trial")
    parser.add_argument("--planning-depth", type=int, default=5, help="MCTS tree search max depth")
    args = parser.parse_args()

    run_payoff_matrix_experiment(max_level=args.max_level, n_trials=args.trials, max_steps=args.steps, planning_depth=args.planning_depth)
