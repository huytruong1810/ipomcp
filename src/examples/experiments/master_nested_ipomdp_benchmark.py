import argparse
import os

import matplotlib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

matplotlib.use("Agg")
from datetime import datetime
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import seaborn as sns

from core.config import ExperimentConfig, IPOMCPConfig, MCTSConfig, OpponentPolicyConfig
from core.logger import get_logger
from core.paths import get_results_dir
from core.pomdp_model import POMDPModel, State
from examples.tiger.model.tiger_model import TigerModel
from solvers.exploration import NormalizedUCB
from solvers.planner import Planner
from solvers.solver_bank import SolverBank
from utils.bootstrapper import I_POMDP_Bootstrapper
from utils.generic_batch_runner import GenericBatchRunner
from utils.paper_plots import generate_nested_sunburst_pdf, generate_paper_plots
from utils.plotting import (
    plot_all_metrics,
    plot_episode_sunburst_slider,
    plot_nested_belief_sunburst,
)
from utils.statistics import compute_statistical_significance

logger = get_logger("MasterBenchmark")


class ConfigurableTigerRunner(GenericBatchRunner):
    def __init__(
        self,
        config: ExperimentConfig,
        log_dir: str,
        protagonist_level: int,
        opponent_level: int,
        protagonist_weights: Dict[int, Any],
        opponent_weights: Dict[int, Any],
        planning_depth: int = 20,
    ):
        super().__init__(config=config, log_dir=log_dir)
        self.protagonist_level = protagonist_level
        self.opponent_level = opponent_level
        self.protagonist_weights = protagonist_weights
        self.opponent_weights = opponent_weights
        self.planning_depth = planning_depth

    def _setup_domain(self) -> Tuple[POMDPModel, Planner, Planner, State]:
        growl_dict = {"i": 0.85, "j": 0.85}
        env = TigerModel(growl_accuracy=growl_dict, creak_accuracy=1.0)

        mcts_cfg = MCTSConfig(n_sims=10000, max_depth=self.planning_depth, node_capacity=2000)
        opponent_cfg = OpponentPolicyConfig(n_sims=10)
        agent_config = IPOMCPConfig(mcts=mcts_cfg, opponent=opponent_cfg)

        # 1. Opponent Agent J with dedicated SolverBank
        bank_j = SolverBank()
        boot_j = I_POMDP_Bootstrapper(bank_j)
        if self.opponent_level == 0:
            key_j = boot_j.create_level0_solver("j", TigerModel(growl_accuracy=growl_dict))
            planner_j = bank_j.get_solver(key_j)
        else:
            planner_j = boot_j.create_solver(
                agent_id="j",
                level=self.opponent_level,
                model=TigerModel(growl_accuracy=growl_dict),
                other_agent_ids=["i"],
                level_weights=self.opponent_weights.get("level_weights"),
                nested_level_weights=self.opponent_weights.get("nested_level_weights"),
                n_particles=2000,
                config=agent_config,
                exploration_strategy=NormalizedUCB(exploration_const=2**0.5),
            )

        # 2. Protagonist Agent I with isolated SolverBank
        bank_i = SolverBank()
        boot_i = I_POMDP_Bootstrapper(bank_i)
        if self.protagonist_level == 0:
            key_i = boot_i.create_level0_solver("i", TigerModel(growl_accuracy=growl_dict))
            planner_i = bank_i.get_solver(key_i)
        else:
            planner_i = boot_i.create_solver(
                agent_id="i",
                level=self.protagonist_level,
                model=TigerModel(growl_accuracy=growl_dict),
                other_agent_ids=["j"],
                level_weights=self.protagonist_weights.get("level_weights"),
                nested_level_weights=self.protagonist_weights.get("nested_level_weights"),
                n_particles=2000,
                config=agent_config,
                exploration_strategy=NormalizedUCB(exploration_const=2**0.5),
            )

        return env, planner_i, planner_j, env.get_initial_state()


def plot_cross_condition_comparisons(combined_df: pd.DataFrame, out_dir: str):
    """Generates comprehensive cross-condition comparison plots (HTML and PDF)."""
    step_final = combined_df["step"].max()
    final_df = combined_df[combined_df["step"] == step_final]

    # 1. Final Reward Distribution Box/Violin Plot
    fig = px.box(
        final_df,
        x="condition",
        y="cum_reward_i",
        color="condition",
        points="all",
        title=f"Cross-Condition Comparison: Agent I Final Cumulative Reward (t={step_final})",
        labels={"cum_reward_i": "Final Cumulative Reward", "condition": "Condition"},
    )
    fig.update_layout(showlegend=False, paper_bgcolor="#ffffff")
    fig.write_html(os.path.join(out_dir, "cross_condition_reward_box.html"))

    # 2. Mean Cumulative Reward Trajectory per Condition
    grouped = (
        combined_df.groupby(["condition", "step"])["cum_reward_i"]
        .agg(["mean", "std", "sem"])
        .reset_index()
    )
    fig_traj = go.Figure()
    for cond in combined_df["condition"].unique():
        sub = grouped[grouped["condition"] == cond]
        fig_traj.add_trace(
            go.Scatter(
                x=sub["step"],
                y=sub["mean"],
                mode="lines+markers",
                name=cond,
                error_y=dict(type="data", array=sub["sem"] * 1.96, visible=True),
            )
        )
    fig_traj.update_layout(
        title="Mean Cumulative Reward Trajectory with 95% CI (t=0 is Initial Prior)",
        xaxis_title="Simulation Step",
        yaxis_title="Cumulative Reward",
        paper_bgcolor="#ffffff",
        hovermode="x unified",
    )
    fig_traj.write_html(os.path.join(out_dir, "cross_condition_reward_trajectories.html"))

    # 3. High-DPI Publication PDF (Seaborn)
    plt.figure(figsize=(9, 5))
    sns.lineplot(
        data=combined_df,
        x="step",
        y="cum_reward_i",
        hue="condition",
        marker="o",
        errorbar=("ci", 95),
    )
    plt.title("Finitely Nested I-POMDP Multi-Level Benchmark: Cumulative Reward")
    plt.xlabel("Simulation Step (t=0 is Initial Prior)")
    plt.ylabel("Agent I Cumulative Reward")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_cross_condition_rewards.pdf"), format="pdf", dpi=300)
    plt.close()


def run_master_benchmark(n_trials: int = 100, max_steps: int = 20, planning_depth: int = 20):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    master_dir = get_results_dir(
        "deep_prior", f"master_benchmark_{timestamp}_N{n_trials}_T{max_steps}_D{planning_depth}"
    )

    logger.info(
        f"=== STARTING MASTER FINITELY NESTED I-POMDP BENCHMARK (N={n_trials}, Horizon T={max_steps}, Depth D={planning_depth}) ==="
    )
    logger.info(f"Master Results Directory: {master_dir}")

    conditions = [
        {
            "name": "C1: Level-1 (100% L0) vs L1 Opponent",
            "protagonist_level": 1,
            "opponent_level": 1,
            "protagonist_weights": {"level_weights": {0: 1.0}},
            "opponent_weights": {"level_weights": {0: 1.0}},
        },
        {
            "name": "C2: Level-2 (100% L1) vs L1 Opponent",
            "protagonist_level": 2,
            "opponent_level": 1,
            "protagonist_weights": {"level_weights": {1: 1.0, 0: 0.0}},
            "opponent_weights": {"level_weights": {0: 1.0}},
        },
        {
            "name": "C3: Level-2 (50% L0 / 50% L1) vs L1 Opponent",
            "protagonist_level": 2,
            "opponent_level": 1,
            "protagonist_weights": {"level_weights": {0: 0.5, 1: 0.5}},
            "opponent_weights": {"level_weights": {0: 1.0}},
        },
        {
            "name": "C4: Level-2 (10% L0 / 90% L1) vs L1 Opponent",
            "protagonist_level": 2,
            "opponent_level": 1,
            "protagonist_weights": {"level_weights": {0: 0.1, 1: 0.9}},
            "opponent_weights": {"level_weights": {0: 1.0}},
        },
        {
            "name": "C5: Level-3 (Uniform 1/3 Prior) vs L1 Opponent",
            "protagonist_level": 3,
            "opponent_level": 1,
            "protagonist_weights": {
                "nested_level_weights": {
                    3: {0: 1 / 3, 1: 1 / 3, 2: 1 / 3},
                    2: {0: 0.5, 1: 0.5},
                    1: {0: 1.0},
                }
            },
            "opponent_weights": {"level_weights": {0: 1.0}},
        },
        {
            "name": "C6: Level-3 (Uniform 1/3 Prior) vs L2 Opponent (50/50)",
            "protagonist_level": 3,
            "opponent_level": 2,
            "protagonist_weights": {
                "nested_level_weights": {
                    3: {0: 1 / 3, 1: 1 / 3, 2: 1 / 3},
                    2: {0: 0.5, 1: 0.5},
                    1: {0: 1.0},
                }
            },
            "opponent_weights": {"level_weights": {0: 0.5, 1: 0.5}},
        },
    ]

    all_dfs = []
    exp_config = ExperimentConfig(
        n_trials=n_trials, max_steps=max_steps, export_trees=False, verbose=False
    )

    for cond_idx, cond in enumerate(conditions, 1):
        cond_name = cond["name"]
        cond_dir = os.path.join(master_dir, f"cond_{cond_idx}_{cond_name.split(':')[0].strip()}")
        logger.info(
            f"[{cond_idx}/{len(conditions)}] Running Condition: {cond_name} (Horizon T={max_steps}, Planning Depth D={planning_depth})..."
        )

        runner = ConfigurableTigerRunner(
            config=exp_config,
            log_dir=cond_dir,
            protagonist_level=cond["protagonist_level"],
            opponent_level=cond["opponent_level"],
            protagonist_weights=cond["protagonist_weights"],
            opponent_weights=cond["opponent_weights"],
            planning_depth=planning_depth,
        )

        # 1. Capture snapshot for Sunburst visualization (Trial 0)
        _, snapshots = runner.run_single_trial_with_snapshots(trial_id=0)
        if snapshots:
            if 0 in snapshots:
                plot_nested_belief_sunburst(
                    snapshots[0],
                    title=f"{cond_name} - Prior Hierarchy (t=0)",
                    save_dir=cond_dir,
                    filename="sunburst_t0",
                )
                generate_nested_sunburst_pdf(
                    snapshots[0],
                    os.path.join(cond_dir, "fig_sunburst_t0.pdf"),
                    title=f"{cond_name} (t=0)",
                )
            final_step = max(snapshots.keys())
            plot_nested_belief_sunburst(
                snapshots[final_step],
                title=f"{cond_name} - Posterior Hierarchy (t={final_step})",
                save_dir=cond_dir,
                filename="sunburst_final",
            )
            generate_nested_sunburst_pdf(
                snapshots[final_step],
                os.path.join(cond_dir, "fig_sunburst_final.pdf"),
                title=f"{cond_name} (t={final_step})",
            )
            plot_episode_sunburst_slider(
                snapshots, title_prefix=cond_name, save_dir=cond_dir, filename="sunburst_animated"
            )

        # 2. Run full batch
        df = runner.run_batch()
        if not df.empty:
            df["condition"] = cond_name
            all_dfs.append(df)
            plot_all_metrics(
                df,
                agent_labels={"i": "Agent I", "j": "Agent J"},
                title_prefix=cond_name,
                save_dir=cond_dir,
            )
            generate_paper_plots(os.path.join(cond_dir, "batch_results.csv"), cond_dir)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(os.path.join(master_dir, "master_benchmark_summary.csv"), index=False)

        logger.info("Generating Cross-Condition Meta-Visualizations & Statistical Tests...")
        plot_cross_condition_comparisons(combined, master_dir)
        compute_statistical_significance(combined, master_dir)
        logger.info(f"=== MASTER BENCHMARK COMPLETE. ALL ARTIFACTS IN: {master_dir} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100, help="Number of trials per condition")
    parser.add_argument(
        "--steps", type=int, default=20, help="Actual decision horizon (timesteps per episode)"
    )
    parser.add_argument("--planning-depth", type=int, default=20, help="MCTS tree search max depth")
    args = parser.parse_args()

    run_master_benchmark(
        n_trials=args.trials, max_steps=args.steps, planning_depth=args.planning_depth
    )
