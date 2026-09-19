import argparse
import json
import os
import random

import matplotlib
import pandas as pd

matplotlib.use("Agg")
from datetime import datetime
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import seaborn as sns

from core.config import ExperimentConfig, IPOMCPConfig, MCTSConfig, OpponentPolicyConfig, RTSConfig
from core.logger import get_logger
from core.paths import get_results_dir
from core.pomdp_model import POMDPModel, State
from examples.tiger.model.tiger_model import TigerModel
from solvers.exploration import NormalizedUCB
from solvers.planner import Planner
from solvers.policy import search_randomness
from solvers.solver_bank import SolverBank
from utils.bootstrapper import I_POMDP_Bootstrapper
from utils.generic_batch_runner import GenericBatchRunner
from utils.paper_plots import generate_paper_plots
from utils.plotting import plot_all_metrics

logger = get_logger("PlannerComparison")


class ControlledConditionRunner(GenericBatchRunner):
    """
    Executes a single paired condition comparing either approximate RTS or
    sample-based I-POMCP under strictly controlled planning depth and parameters.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        log_dir: str,
        solver_type: str,
        level_i: int,
        level_j: int,
        n_sims: int,
        planning_depth: int,
        n_particles: int = 5000,
        obs_branching: int = 6,
        modeled_opponent_sims: int = 50000,
    ):
        super().__init__(config=config, log_dir=log_dir)
        self.solver_type = solver_type
        self.level_i = level_i
        self.level_j = level_j
        self.n_sims = n_sims
        self.planning_depth = planning_depth
        self.n_particles = n_particles
        self.obs_branching = obs_branching
        self.modeled_opponent_sims = modeled_opponent_sims

    def _setup_domain(self) -> Tuple[POMDPModel, Planner, Planner, State]:
        growl_dict = {"i": 0.85, "j": 0.85}
        env = TigerModel(growl_accuracy=growl_dict, creak_accuracy=1.0)

        # This controlled comparison makes computation part of the declared
        # opponent frame: both agents know the initial empirical prior and the
        # reproducible search seed. Only this initialization is common knowledge;
        # subsequent private observations/beliefs remain separate in separate banks.
        # Independent unknown solver seeds would require a distribution over those
        # seeds in the opponent model, not a point hypothesis pretending to match.
        policy_seed, prior_seed = random.getrandbits(64), random.getrandbits(64)
        initial_samples = 2500
        bank_j = SolverBank(seed=policy_seed)
        boot_j = I_POMDP_Bootstrapper(bank_j)
        cfg_j = IPOMCPConfig(
            mcts=MCTSConfig(n_sims=50000, max_depth=self.planning_depth, node_capacity=2000),
            opponent=OpponentPolicyConfig(n_sims=self.modeled_opponent_sims),
        )
        with search_randomness(prior_seed):
            planner_j = boot_j.create_solver(
                agent_id="j",
                level=self.level_j,
                model=TigerModel(growl_accuracy=growl_dict, creak_accuracy=1.0),
                other_agent_ids=["i"],
                n_particles=initial_samples,
                config=cfg_j,
                exploration_strategy=NormalizedUCB(exploration_const=2**0.5),
            )

        # Both protagonists model the same MCTS opponent family and settings.
        # The modeled budget matches the executing budget by default. An explicit
        # budget override is a misspecification experiment and may fail inference.
        # Unsupported evidence must remain a recorded experimental failure.
        # 2. Protagonist Agent I Setup (Approximate RTS vs I-POMCP at depth D)
        bank_i = SolverBank(seed=policy_seed)
        boot_i = I_POMDP_Bootstrapper(bank_i)
        with search_randomness(prior_seed):
            if self.solver_type == "rts":
                rts_cfg = RTSConfig(
                    max_depth=self.planning_depth,
                    obs_branching=self.obs_branching,
                    num_particles=self.n_particles,
                )
                planner_i = boot_i.create_rts_solver(
                    agent_id="i",
                    level=self.level_i,
                    model=TigerModel(growl_accuracy=growl_dict, creak_accuracy=1.0),
                    other_agent_ids=["j"],
                    level_weights={1: 1.0},
                    n_particles=initial_samples,
                    config=rts_cfg,
                    modeled_config=cfg_j,
                    modeled_exploration=NormalizedUCB(exploration_const=2**0.5),
                )
            else:
                cfg_i = IPOMCPConfig(
                    mcts=MCTSConfig(
                        n_sims=self.n_sims, max_depth=self.planning_depth, node_capacity=2000
                    ),
                    opponent=OpponentPolicyConfig(),
                )
                planner_i = boot_i.create_solver(
                    agent_id="i",
                    level=self.level_i,
                    model=TigerModel(growl_accuracy=growl_dict, creak_accuracy=1.0),
                    other_agent_ids=["j"],
                    level_weights={1: 1.0},
                    n_particles=initial_samples,
                    config=cfg_i,
                    modeled_config=cfg_j,
                    modeled_exploration=NormalizedUCB(exploration_const=2**0.5),
                    exploration_strategy=NormalizedUCB(exploration_const=2**0.5),
                )

        return env, planner_i, planner_j, env.get_initial_state()


def run_planner_comparison(
    n_trials: int = 50,
    max_steps: int = 20,
    planning_depth: int = 3,
    resume_dir: Optional[str] = None,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if resume_dir and os.path.exists(resume_dir):
        master_dir = resume_dir
        logger.info(f"Resuming existing reference benchmark from: {master_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        master_dir = get_results_dir(
            "reference",
            f"planner_comparison_reference_{timestamp}_N{n_trials}_T{max_steps}_D{planning_depth}",
        )

    logger.info(
        f"=== STARTING PLANNER COMPARISON BENCHMARK (N={n_trials}, Horizon={max_steps}, Depth={planning_depth}) ==="
    )
    logger.info(f"Results Directory: {master_dir}")

    conditions = [
        {
            "name": f"Approximate RTS (Depth {planning_depth})",
            "type": "rts",
            "sims": 0,
            "particles": 500,
        },
        {
            "name": f"I-POMCP 1k (Depth {planning_depth})",
            "type": "ipomcp",
            "sims": 1000,
            "particles": 2000,
        },
        {
            "name": f"I-POMCP 5k (Depth {planning_depth})",
            "type": "ipomcp",
            "sims": 5000,
            "particles": 2000,
        },
        {
            "name": f"I-POMCP 10k (Depth {planning_depth})",
            "type": "ipomcp",
            "sims": 10000,
            "particles": 2000,
        },
        {
            "name": f"I-POMCP 25k (Depth {planning_depth})",
            "type": "ipomcp",
            "sims": 25000,
            "particles": 2000,
        },
        {
            "name": f"I-POMCP 50k (Depth {planning_depth})",
            "type": "ipomcp",
            "sims": 50000,
            "particles": 2000,
        },
        {
            "name": f"I-POMCP 100k (Depth {planning_depth})",
            "type": "ipomcp",
            "sims": 100000,
            "particles": 2000,
        },
    ]

    exp_config = ExperimentConfig(
        n_trials=n_trials, max_steps=max_steps, export_trees=False, verbose=False
    )
    all_dfs = []
    condition_dfs = {}

    for cond_idx, cond in enumerate(conditions, 1):
        cond_name = cond["name"]
        sanitized_name = (
            cond_name.replace(" ", "_").replace("(", "").replace(")", "").replace(":", "")
        )
        cond_dir = os.path.join(master_dir, f"cond_{cond_idx}_{sanitized_name}")

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
            obs_branching=6,
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
                agent_labels={"i": f"Agent I ({cond_name})", "j": "Agent J (L1)"},
                title_prefix=cond_name,
                save_dir=cond_dir,
            )
            generate_paper_plots(os.path.join(cond_dir, "batch_results.csv"), cond_dir)

    if not all_dfs:
        logger.error("No experimental data collected.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    summary_csv = os.path.join(master_dir, "planner_comparison_summary.csv")
    combined.to_csv(summary_csv, index=False)

    # 1. Sampled RTS Reference Baseline Extraction
    reference_name = conditions[0]["name"]
    reference_df = condition_dfs[reference_name]
    reference_final = (
        reference_df[reference_df["step"] == max_steps] if reference_df is not None else None
    )
    reference_mean_r = (
        float(reference_final["cum_reward_i"].mean()) if reference_final is not None else 0.0
    )
    reference_median_r = (
        float(reference_final["cum_reward_i"].median()) if reference_final is not None else 0.0
    )
    reference_time = (
        float(reference_df.loc[reference_df["step"] > 0, "planning_time_i"].mean())
        if reference_df is not None
        else 0.0
    )

    # 2. Comparative Analysis
    results_table = []

    for cond in conditions:
        c_name = cond["name"]
        c_df = condition_dfs[c_name]
        if c_df is None:
            continue

        c_final = c_df[c_df["step"] == max_steps]
        mean_r = float(c_final["cum_reward_i"].mean())
        sem_r = float(c_final["cum_reward_i"].std() / (len(c_final) ** 0.5))
        median_r = float(c_final["cum_reward_i"].median())
        mean_time = float(c_df.loc[c_df["step"] > 0, "planning_time_i"].mean())

        # Independent closed-loop histories cannot measure policy agreement.
        # Reward ratios are also not reward percentages, especially with
        # negative returns. Report differences and latency instead.
        reward_difference = mean_r - reference_mean_r
        speedup = (reference_time / mean_time) if mean_time > 0 else 1.0

        results_table.append(
            {
                "condition": c_name,
                "solver_type": cond["type"],
                "sims": cond["sims"],
                "mean_reward": mean_r,
                "sem_reward": sem_r,
                "median_reward": median_r,
                "mean_latency_s": mean_time,
                "reward_difference_vs_rts": reward_difference,
                "speedup_vs_reference": speedup,
            }
        )

    report = {
        "timestamp": timestamp,
        "n_trials": n_trials,
        "max_steps": max_steps,
        "planning_depth": planning_depth,
        "reference_mean_reward": reference_mean_r,
        "reference_median_reward": reference_median_r,
        "reference_mean_latency_s": reference_time,
        "results": results_table,
    }

    with open(os.path.join(master_dir, "planner_comparison_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # 3. Publication Visualizations
    # Figure 1: Pareto Frontier (Latency vs Cumulative Reward)
    plt.figure(figsize=(9, 6))
    pareto_df = pd.DataFrame(results_table)

    # Plot I-POMCP points
    ipomcp_pts = pareto_df[pareto_df["solver_type"] == "ipomcp"]
    plt.plot(
        ipomcp_pts["mean_latency_s"],
        ipomcp_pts["mean_reward"],
        marker="o",
        color="#ff7f0e",
        linewidth=2.5,
        label="I-POMCP (1k - 100k Sims)",
    )
    for _, row in ipomcp_pts.iterrows():
        plt.annotate(
            f"{row['sims'] // 1000}k sims",
            (row["mean_latency_s"], row["mean_reward"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
        )

    # Plot Sampled RTS Point
    reference_pt = pareto_df[pareto_df["solver_type"] == "rts"].iloc[0]
    plt.scatter(
        [reference_pt["mean_latency_s"]],
        [reference_pt["mean_reward"]],
        color="#1f77b4",
        s=180,
        zorder=5,
        marker="*",
        label=f"Approximate RTS (Depth {planning_depth})",
    )
    plt.annotate(
        f"Approximate RTS\n({reference_pt['mean_latency_s'] * 1000:.1f}ms)",
        (reference_pt["mean_latency_s"], reference_pt["mean_reward"]),
        textcoords="offset points",
        xytext=(0, 12),
        ha="center",
        fontsize=9,
        fontweight="bold",
        color="#1f77b4",
    )

    plt.title(
        f"Pareto Efficiency: Cumulative Reward vs Planning Latency (Depth D={planning_depth}, T={max_steps})",
        fontsize=12,
        fontweight="bold",
    )
    plt.xlabel("Mean Planning Time per Step (seconds, log scale)", fontsize=11)
    plt.ylabel("Mean Cumulative Reward ($R_i$)", fontsize=11)
    plt.xscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(frameon=True, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(master_dir, "fig_pareto_runtime_vs_reward.pdf"), format="pdf", dpi=300)
    plt.close()

    # Figure 3: Sampled Planner Comparison Bar Plot
    plt.figure(figsize=(10, 5.5))
    final_df = combined[combined["step"] == max_steps]
    palette = [
        "#1f77b4" if "Sampled RTS" in c else "#ff7f0e" for c in final_df["condition"].unique()
    ]
    sns.barplot(
        data=final_df,
        x="condition",
        y="cum_reward_i",
        hue="condition",
        errorbar=("ci", 95),
        palette=palette,
        legend=False,
    )
    plt.title(
        f"Sampled Planner Comparison Controlled Evaluation: Sampled RTS vs I-POMCP (Depth D={planning_depth}, T={max_steps})",
        fontsize=12,
        fontweight="bold",
    )
    plt.xlabel("Planner Configuration", fontsize=11)
    plt.ylabel("Agent I Cumulative Reward ($R_i$)", fontsize=11)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.tight_layout()
    plt.savefig(
        os.path.join(master_dir, "fig_planner_comparison_reward_bars.pdf"), format="pdf", dpi=300
    )
    plt.close()

    logger.info(f"=== PLANNER COMPARISON BENCHMARK COMPLETE. ARTIFACTS IN: {master_dir} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sampled Planner Comparison Approximate RTS vs I-POMCP Benchmark"
    )
    parser.add_argument(
        "--trials", type=int, default=200, help="Number of Monte Carlo trials per condition"
    )
    parser.add_argument(
        "--steps", type=int, default=20, help="Number of environment steps per trial"
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="Planning horizon depth D for both Sampled RTS and I-POMCP",
    )
    args = parser.parse_args()

    run_planner_comparison(n_trials=args.trials, max_steps=args.steps, planning_depth=args.depth)
