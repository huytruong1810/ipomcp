import os
from datetime import datetime

from core.config import ExperimentConfig, IPOMCPConfig, MCTSConfig, OpponentPolicyConfig
from core.logger import get_logger
from examples.tiger.model.tiger_model import TigerModel
from solvers.exploration import NormalizedUCB
from solvers.solver_bank import SolverBank
from utils.bootstrapper import I_POMDP_Bootstrapper
from utils.generic_batch_runner import GenericBatchRunner
from utils.paper_plots import generate_paper_plots
from utils.plotting import plot_all_metrics

logger = get_logger("TigerLevel3Experiment")


class TigerLevel3Runner(GenericBatchRunner):
    """
    Executes Level-3 Agent I vs Level-1 Agent J in Tiger World.

    Theoretical Formulation:
    - Agent I (Level 3): Interactive state space IS_{i,3} = S x (Theta_j^0 union Theta_j^1 union Theta_j^2).
      Prior belief: 1/3 Level 0, 1/3 Level 1, 1/3 Level 2.
    - Inside Level 2 j: IS_{j,2} = S x (Theta_i^0 union Theta_i^1).
      Prior belief: 1/2 Level 0, 1/2 Level 1.
    - Inside Level 1 i (nested in Level 2 j): IS_{i,1} = S x Theta_j^0 (100% Level 0).
    - Agent J (Real-world Opponent): True Level-1 agent modeling Agent I as Level-0 with 100%.
    """

    def _setup_domain(self):
        growl_dict = {"i": 0.85, "j": 0.85}
        env = TigerModel(growl_accuracy=growl_dict, creak_accuracy=1.0)

        mcts_cfg = MCTSConfig(n_sims=10000, max_depth=6, node_capacity=2000)
        opponent_cfg = OpponentPolicyConfig(n_sims=10)
        agent_config = IPOMCPConfig(mcts=mcts_cfg, opponent=opponent_cfg)

        # 1. Real-World Opponent: Agent J at Level 1 (models i at Level 0 with 100%)
        bank_j = SolverBank()
        boot_j = I_POMDP_Bootstrapper(bank_j)
        planner_j = boot_j.create_solver(
            agent_id="j",
            level=1,
            model=TigerModel(growl_accuracy=growl_dict),
            other_agent_ids=["i"],
            level_weights={0: 1.0},
            n_particles=2000,
            config=agent_config,
            exploration_strategy=NormalizedUCB(exploration_const=2**0.5),
        )

        # 2. Protagonist: Agent I at Level 3 (isolated bank prevents mental model contamination)
        nested_weights = {3: {0: 1 / 3, 1: 1 / 3, 2: 1 / 3}, 2: {0: 0.5, 1: 0.5}, 1: {0: 1.0}}

        bank_i = SolverBank()
        boot_i = I_POMDP_Bootstrapper(bank_i)
        planner_i = boot_i.create_solver(
            agent_id="i",
            level=3,
            model=TigerModel(growl_accuracy=growl_dict),
            other_agent_ids=["j"],
            nested_level_weights=nested_weights,
            n_particles=2000,
            config=agent_config,
            exploration_strategy=NormalizedUCB(exploration_const=2**0.5),
        )

        return env, planner_i, planner_j, env.get_initial_state()


def run_level3_experiment(n_trials: int = 30, max_steps: int = 6):
    from core.paths import get_results_dir

    log_dir = get_results_dir(
        "tiger", f"tiger_level3_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    exp_config = ExperimentConfig(
        n_trials=n_trials, max_steps=max_steps, export_trees=False, verbose=False
    )

    logger.info(f"Starting Level-3 I-POMCP Experiment (N={n_trials}, T={max_steps})...")
    runner = TigerLevel3Runner(config=exp_config, log_dir=log_dir)

    # 1. Capture detailed episode trial with nested belief snapshots across t=0...T
    logger.info("Executing detailed episode trial to capture multi-level belief snapshots...")
    single_df, snapshots_by_step = runner.run_single_trial_with_snapshots(trial_id=0)

    if snapshots_by_step:
        logger.info("Generating Top-Down Nested Mental Model Sunburst Visualizations...")
        from utils.paper_plots import generate_nested_sunburst_pdf
        from utils.plotting import plot_episode_sunburst_slider, plot_nested_belief_sunburst

        # Sunburst at t=0 (Prior belief)
        if 0 in snapshots_by_step:
            plot_nested_belief_sunburst(
                snapshots_by_step[0],
                title="Level-3 Agent I: Initial Prior Mental Model Hierarchy (t=0)",
                save_dir=log_dir,
                filename="nested_belief_sunburst_t0",
            )
            generate_nested_sunburst_pdf(
                snapshots_by_step[0],
                out_path=os.path.join(log_dir, "fig_nested_sunburst_t0.pdf"),
                title="Level-3 Agent I: Prior Mental Model Hierarchy (t=0)",
            )

        # Sunburst at final step
        final_step = max(snapshots_by_step.keys())
        plot_nested_belief_sunburst(
            snapshots_by_step[final_step],
            title=f"Level-3 Agent I: Final Posterior Mental Model Hierarchy (t={final_step})",
            save_dir=log_dir,
            filename="nested_belief_sunburst_final",
        )
        generate_nested_sunburst_pdf(
            snapshots_by_step[final_step],
            out_path=os.path.join(log_dir, "fig_nested_sunburst_final.pdf"),
            title=f"Level-3 Agent I: Posterior Mental Model Hierarchy (t={final_step})",
        )

        # Interactive Step Scrubber Slider (Full Episode Animation)
        plot_episode_sunburst_slider(
            snapshots_by_step,
            title_prefix="Tiger Level-3 vs Level-1",
            save_dir=log_dir,
            filename="nested_belief_sunburst_animated",
        )

    # 2. Run batch execution for statistical aggregation
    logger.info("Executing full batch simulation for population statistics...")
    df = runner.run_batch()

    if not df.empty:
        logger.info("Generating interactive diagnostic charts...")
        plot_all_metrics(
            df,
            agent_labels={
                "i": "Agent I (Level-3: 1/3 L0, 1/3 L1, 1/3 L2)",
                "j": "Agent J (Level-1: 100% L0)",
            },
            title_prefix="Tiger Domain Level-3 Experiment:",
            save_dir=log_dir,
        )

        logger.info("Generating publication vector graphics...")
        generate_paper_plots(os.path.join(log_dir, "batch_results.csv"), log_dir)
        logger.info(f"Level-3 evaluation completed successfully. Artifacts saved to: {log_dir}")


if __name__ == "__main__":
    run_level3_experiment(n_trials=30, max_steps=6)
