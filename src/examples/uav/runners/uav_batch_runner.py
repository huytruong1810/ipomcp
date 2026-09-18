import os
from datetime import datetime

from core.config import ExperimentConfig, IPOMCPConfig, JITConfig, MCTSConfig
from core.logger import get_logger
from examples.uav.model.uav_model import UAVModel
from solvers.exploration import NormalizedUCB
from solvers.solver_bank import SolverBank
from utils.bootstrapper import I_POMDP_Bootstrapper
from utils.generic_batch_runner import GenericBatchRunner
from utils.paper_plots import generate_paper_plots
from utils.plotting import plot_all_metrics

logger = get_logger("UAVRunner")


class UAVBatchRunner(GenericBatchRunner):
    """
    Experimental Group: Executes Level-2 UAV (Hunter) vs Level-1 Target (Hunted) agents.
    """

    def _setup_domain(self):
        env = UAVModel()
        agent_uav = "i"
        agent_target = "j"

        mcts_cfg = MCTSConfig(n_sims=10000, max_depth=5, node_capacity=500)
        jit_cfg = JITConfig(entropy_threshold=0.6, visit_threshold=5, sims=10)
        agent_config = IPOMCPConfig(mcts=mcts_cfg, jit=jit_cfg)

        # 1. Target (L1) with dedicated SolverBank
        bank_j = SolverBank()
        boot_j = I_POMDP_Bootstrapper(bank_j)
        boot_j.create_level0_solver(agent_uav, UAVModel())
        boot_j.create_level0_solver(agent_target, UAVModel())
        planner_j = boot_j.create_level1_solver(
            agent_target,
            UAVModel(),
            [agent_uav],
            n_particles=2000,
            config=agent_config,
            exploration_strategy=NormalizedUCB(exploration_const=1.0),
        )

        # 2. UAV (L2) with isolated SolverBank
        bank_i = SolverBank()
        boot_i = I_POMDP_Bootstrapper(bank_i)
        boot_i.create_level0_solver(agent_uav, UAVModel())
        boot_i.create_level0_solver(agent_target, UAVModel())
        boot_i.create_level1_solver(
            agent_target,
            UAVModel(),
            [agent_uav],
            n_particles=2000,
            config=agent_config,
            exploration_strategy=NormalizedUCB(exploration_const=1.0),
        )
        planner_i = boot_i.create_level2_solver(
            agent_uav,
            UAVModel(),
            [agent_target],
            n_particles=2000,
            config=agent_config,
            exploration_strategy=NormalizedUCB(exploration_const=1.0),
        )

        return env, planner_i, planner_j, env.get_initial_state()

    def _get_custom_metrics(self, state, next_state, env, planner_i=None, planner_j=None):
        # Calculate Chebyshev distance (grid steps)
        dist = max(
            abs(next_state.uav_pos[0] - next_state.target_pos[0]),
            abs(next_state.uav_pos[1] - next_state.target_pos[1]),
        )

        # Manhattan distance
        manhattan = abs(next_state.uav_pos[0] - next_state.target_pos[0]) + abs(
            next_state.uav_pos[1] - next_state.target_pos[1]
        )

        metrics = {
            "distance_chebyshev": dist,
            "distance_manhattan": manhattan,
            "uav_pos": f"{next_state.uav_pos}",
            "target_pos": f"{next_state.target_pos}",
            "is_same_row": next_state.uav_pos[0] == next_state.target_pos[0],
            "is_same_col": next_state.uav_pos[1] == next_state.target_pos[1],
            "is_caught": next_state.uav_pos == next_state.target_pos,
        }

        # Mechanistic Explanation: What does the UAV believe about the target?
        if planner_i and hasattr(planner_i, "root") and planner_i.root.belief_particles:
            particles = planner_i.root.belief_particles
            # Count occurrences of target positions in particles
            pos_counts = {}
            for p in particles:
                t_pos = p.state.target_pos
                pos_counts[t_pos] = pos_counts.get(t_pos, 0) + 1

            if pos_counts:
                most_likely_t_pos = max(pos_counts, key=pos_counts.get)
                metrics["uav_belief_target_pos_mode"] = f"{most_likely_t_pos}"
                metrics["uav_belief_target_pos_mode_prob"] = pos_counts[most_likely_t_pos] / len(
                    particles
                )

        return metrics


if __name__ == "__main__":
    from core.paths import get_results_dir

    log_dir = get_results_dir("uav", f"uav_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    exp_config = ExperimentConfig(n_trials=100, max_steps=25, export_trees=False, verbose=False)

    logger.info("Initializing UAV Batch Runner...")
    logger.info(f"Configuration: N={exp_config.n_trials}, Horizon={exp_config.max_steps}")

    runner = UAVBatchRunner(config=exp_config, log_dir=log_dir)
    df = runner.run_batch()

    if not df.empty:
        logger.info("Generating interactive debug plots...")
        plot_all_metrics(
            df,
            agent_labels={"i": "UAV (Hunter - L2)", "j": "Target (Hunted - L1)"},
            title_prefix="UAV Domain:",
            save_dir=log_dir,
        )

        logger.info("Generating publication vector graphics...")
        generate_paper_plots(os.path.join(log_dir, "batch_results.csv"), log_dir)
        logger.info("UAV Batch execution pipeline finalized successfully.")
    else:
        logger.warning("Dataframe returned empty. Aborting plot generation.")
