import argparse
import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")
from datetime import datetime
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import seaborn as sns

from core.config import (
    DEFAULT_PARTICLE_SCHEDULE as PARTICLE_SCHEDULE,
)
from core.config import (
    DEFAULT_SIM_SCHEDULE as SIM_SCHEDULE,
)
from core.config import (
    ExperimentConfig,
    IPOMCPConfig,
    MCTSConfig,
    OpponentPolicyConfig,
)
from core.logger import get_logger
from core.paths import get_results_dir
from core.pomdp_model import POMDPModel, State
from examples.tiger.model.tiger_model import TigerModel
from solvers.exploration import NormalizedUCB
from solvers.planner import Planner
from solvers.solver_bank import SolverBank
from utils.bootstrapper import I_POMDP_Bootstrapper
from utils.generic_batch_runner import (
    GenericBatchRunner,
)
from utils.paper_plots import generate_nested_sunburst_pdf, generate_paper_plots
from utils.plotting import (
    plot_all_metrics,
    plot_episode_sunburst_slider,
    plot_nested_belief_sunburst,
)

logger = get_logger("DeepHierarchyPriorExperiment")


class DeepHierarchyTigerRunner(GenericBatchRunner):
    def __init__(
        self,
        config: ExperimentConfig,
        log_dir: str,
        level_i: int,
        level_j: int,
        prior_weights_i: Optional[Dict[int, float]] = None,
        prior_weights_j: Optional[Dict[int, float]] = None,
        planning_depth: int = 5,
    ):
        super().__init__(config=config, log_dir=log_dir)
        self.level_i = level_i
        self.level_j = level_j
        self.prior_weights_i = prior_weights_i
        self.prior_weights_j = prior_weights_j
        self.planning_depth = planning_depth

    def _setup_domain(self) -> Tuple[POMDPModel, Planner, Planner, State]:
        growl_dict = {"i": 0.85, "j": 0.85}
        env = TigerModel(growl_accuracy=growl_dict, creak_accuracy=1.0)

        # 1. Opponent Agent J Setup
        bank_j = SolverBank()
        boot_j = I_POMDP_Bootstrapper(bank_j)
        if self.level_j == 0:
            planner_j = boot_j.create_solver(
                agent_id="j", level=0, model=env, other_agent_ids=["i"]
            )
        else:
            sims_j = SIM_SCHEDULE.get(self.level_j, 10000 * self.level_j)
            particles_j = PARTICLE_SCHEDULE.get(self.level_j, 1000 * self.level_j)
            cfg_j = IPOMCPConfig(
                mcts=MCTSConfig(n_sims=sims_j, max_depth=self.planning_depth, node_capacity=1000),
                opponent=OpponentPolicyConfig(),
            )

            planner_j = boot_j.create_solver(
                agent_id="j",
                level=self.level_j,
                model=TigerModel(growl_accuracy=growl_dict, creak_accuracy=1.0),
                other_agent_ids=["i"],
                level_weights=self.prior_weights_j,
                n_particles=particles_j,
                config=cfg_j,
                exploration_strategy=NormalizedUCB(exploration_const=2**0.5),
            )

        # 2. Protagonist Agent I Setup
        bank_i = SolverBank()
        boot_i = I_POMDP_Bootstrapper(bank_i)
        if self.level_i == 0:
            planner_i = boot_i.create_solver(
                agent_id="i", level=0, model=env, other_agent_ids=["j"]
            )
        else:
            sims_i = SIM_SCHEDULE.get(self.level_i, 10000 * self.level_i)
            particles_i = PARTICLE_SCHEDULE.get(self.level_i, 1000 * self.level_i)
            cfg_i = IPOMCPConfig(
                mcts=MCTSConfig(n_sims=sims_i, max_depth=self.planning_depth, node_capacity=1000),
                opponent=OpponentPolicyConfig(),
            )

            planner_i = boot_i.create_solver(
                agent_id="i",
                level=self.level_i,
                model=TigerModel(growl_accuracy=growl_dict, creak_accuracy=1.0),
                other_agent_ids=["j"],
                level_weights=self.prior_weights_i,
                n_particles=particles_i,
                config=cfg_i,
                exploration_strategy=NormalizedUCB(exploration_const=2**0.5),
            )

        return env, planner_i, planner_j, env.get_initial_state()


def experiment_conditions(planning_depth=5):
    """One condition table shared by suite execution and qualification."""
    return [
        # Level 3 Variations
        {
            "name": "L3 vs L2 (80% L2 Prior)",
            "level_i": 3,
            "level_j": 2,
            "prior_i": {2: 0.80, 1: 0.10, 0: 0.10},
            "prior_j": {1: 0.80, 0: 0.20},
        },
        {
            "name": "L3 vs L1 (80% L1 Prior)",
            "level_i": 3,
            "level_j": 1,
            "prior_i": {1: 0.80, 2: 0.10, 0: 0.10},
            "prior_j": {0: 1.0},
        },
        {
            "name": "L3 vs L1 (80% Over-estimated L2 Prior)",
            "level_i": 3,
            "level_j": 1,
            "prior_i": {2: 0.80, 1: 0.10, 0: 0.10},
            "prior_j": {0: 1.0},
        },
        {
            "name": "L3 vs L1 (Uniform 1/3 Mixture Prior)",
            "level_i": 3,
            "level_j": 1,
            "prior_i": {0: 1 / 3, 1: 1 / 3, 2: 1 / 3},
            "prior_j": {0: 1.0},
        },
        # Level 4 Variations
        {
            "name": "L4 vs L3 (80% L3 Prior)",
            "level_i": 4,
            "level_j": 3,
            "prior_i": {3: 0.80, 2: 0.20 / 3, 1: 0.20 / 3, 0: 0.20 / 3},
            "prior_j": {2: 0.80, 1: 0.10, 0: 0.10},
        },
        {
            "name": "L4 vs L1 (80% Over-estimated L3 Prior)",
            "level_i": 4,
            "level_j": 1,
            "prior_i": {3: 0.80, 2: 0.20 / 3, 1: 0.20 / 3, 0: 0.20 / 3},
            "prior_j": {0: 1.0},
        },
        {
            "name": "L4 vs L1 (Uniform 1/4 Mixture Prior)",
            "level_i": 4,
            "level_j": 1,
            "prior_i": {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25},
            "prior_j": {0: 1.0},
        },
    ]


def run_deep_prior_experiment(
    n_trials: int = 50,
    max_steps: int = 20,
    planning_depth: int = 5,
    resume_dir: Optional[str] = None,
    condition_idx: Optional[int] = None,
    workers: int = 8,
):
    if resume_dir and os.path.exists(resume_dir):
        master_dir = resume_dir
        logger.info(f"Resuming existing benchmark from: {master_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cond_tag = f"_cond{condition_idx}" if condition_idx is not None else ""
        master_dir = get_results_dir(
            "deep_prior", f"deep_prior_benchmark_{timestamp}{cond_tag}_N{n_trials}_T{max_steps}"
        )

    logger.info(
        f"=== STARTING DEEP HIERARCHY PRIOR BENCHMARK (N={n_trials}, T={max_steps}, Depth={planning_depth}) ==="
    )
    logger.info(f"Results Directory: {master_dir}")

    conditions = experiment_conditions(planning_depth)

    if condition_idx is not None:
        if 1 <= condition_idx <= len(conditions):
            cond_offset = condition_idx
            conditions = [conditions[condition_idx - 1]]
        else:
            raise ValueError(
                f"Invalid condition_idx {condition_idx}; must be between 1 and {len(conditions)}"
            )
    else:
        cond_offset = 1

    all_dfs = []
    exp_config = ExperimentConfig(
        n_trials=n_trials,
        max_steps=max_steps,
        export_trees=False,
        verbose=False,
        max_workers=workers,
    )

    for idx_offset, cond in enumerate(conditions):
        cond_idx = cond_offset + idx_offset
        cond_name = cond["name"]
        sanitized_name = (
            cond_name.replace("/", "_div_")
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("%", "pct")
        )
        cond_dir = os.path.join(master_dir, f"cond_{cond_idx}_{sanitized_name}")

        logger.info(f"[{cond_idx}/{len(conditions)}] Running Prior Condition: {cond_name}...")

        runner = DeepHierarchyTigerRunner(
            config=exp_config,
            log_dir=cond_dir,
            level_i=cond["level_i"],
            level_j=cond["level_j"],
            prior_weights_i=cond["prior_i"],
            prior_weights_j=cond["prior_j"],
            planning_depth=planning_depth,
        )

        # 1. Capture snapshots for Sunburst (Trial 0)
        snapshots_path = os.path.join(cond_dir, "nested_belief_snapshots_trial_0.json")
        if os.path.exists(snapshots_path) and os.path.exists(
            os.path.join(cond_dir, "sunburst_final.html")
        ):
            logger.info(
                f"Snapshots and Sunburst plots already exist in {cond_dir}. Skipping snapshot trial."
            )
        else:
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
                    snapshots,
                    title_prefix=cond_name,
                    save_dir=cond_dir,
                    filename="sunburst_animated",
                )

        # Memory-safe worker scaling respecting user configuration
        max_lvl = max(cond["level_i"], cond["level_j"])
        cell_workers = min(workers, 12) if max_lvl <= 3 else min(workers, 8)

        # 2. Run batch
        df = runner.run_batch(max_workers=cell_workers)
        if not df.empty:
            df["condition"] = cond_name
            all_dfs.append(df)
            plot_all_metrics(
                df,
                agent_labels={
                    "i": f"Agent I (L{cond['level_i']})",
                    "j": f"Agent J (L{cond['level_j']})",
                },
                title_prefix=cond_name,
                save_dir=cond_dir,
            )
            generate_paper_plots(os.path.join(cond_dir, "batch_results.csv"), cond_dir)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(os.path.join(master_dir, "deep_prior_summary.csv"), index=False)

        # Cross-Condition Visualizations
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=combined[combined["step"] == max_steps],
            x="condition",
            y="cum_reward_i",
            errorbar=("ci", 95),
        )
        plt.title(
            f"Impact of Strategic Prior Specification on Agent I Cumulative Reward (t={max_steps})",
            fontsize=12,
            fontweight="bold",
        )
        plt.xlabel("Prior Modeling Condition")
        plt.ylabel("Agent I Final Cumulative Reward")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(
            os.path.join(master_dir, "fig_prior_comparison_bars.pdf"), format="pdf", dpi=300
        )
        plt.close()

        logger.info(f"=== DEEP PRIOR BENCHMARK COMPLETE. ALL ARTIFACTS IN: {master_dir} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50, help="Number of trials per condition")
    parser.add_argument("--steps", type=int, default=10, help="Decision steps per trial")
    parser.add_argument("--planning-depth", type=int, default=5, help="MCTS tree search max depth")
    parser.add_argument(
        "--condition", type=int, default=None, help="Specific condition index (1-7) to run"
    )
    parser.add_argument("--resume-dir", type=str, default=None, help="Existing directory to resume")
    args = parser.parse_args()

    run_deep_prior_experiment(
        n_trials=args.trials,
        max_steps=args.steps,
        planning_depth=args.planning_depth,
        resume_dir=args.resume_dir,
        condition_idx=args.condition,
    )
