import os
from datetime import datetime

from core.config import ExperimentConfig, IPOMCPConfig, MCTSConfig, RTSConfig
from core.logger import get_logger
from examples.uav.model.uav_model import UAVModel
from solvers.exploration import NormalizedUCB
from solvers.solver_bank import SolverBank
from utils.bootstrapper import I_POMDP_Bootstrapper
from utils.generic_batch_runner import GenericBatchRunner
from utils.paper_plots import generate_paper_plots
from utils.plotting import plot_all_metrics

logger = get_logger("UAV_RTS_Runner")


class UAV_RTS_Runner(GenericBatchRunner):
    """
    Evaluates sampled reachability-tree baseline against I-POMCP.
    """

    def _setup_domain(self):
        env = UAVModel()
        target_mcts = MCTSConfig(n_sims=3000, max_depth=5, node_capacity=500)
        target_cfg = IPOMCPConfig(mcts=target_mcts)

        # 1. Target (J) uses I-POMCP with dedicated SolverBank
        bank_j = SolverBank()
        boot_j = I_POMDP_Bootstrapper(bank_j)
        boot_j.create_level0_solver("i", UAVModel())
        boot_j.create_level0_solver("j", UAVModel())
        planner_j = boot_j.create_level1_solver(
            "j",
            UAVModel(),
            ["i"],
            n_particles=1000,
            config=target_cfg,
            exploration_strategy=NormalizedUCB(exploration_const=1.0),
        )

        # 2. UAV (I) uses RTS sampled lookahead baseline with isolated SolverBank
        rts_cfg = RTSConfig(max_depth=3, obs_branching=3, num_particles=50)
        bank_i = SolverBank()
        boot_i = I_POMDP_Bootstrapper(bank_i)
        boot_i.create_level0_solver("i", UAVModel())
        boot_i.create_level0_solver("j", UAVModel())
        boot_i.create_level1_solver(
            "j",
            UAVModel(),
            ["i"],
            n_particles=1000,
            config=target_cfg,
            exploration_strategy=NormalizedUCB(exploration_const=1.0),
        )
        planner_i = boot_i.create_level2_rts_solver("i", UAVModel(), ["j"], config=rts_cfg)

        return env, planner_i, planner_j, env.get_initial_state()

    def _get_custom_metrics(self, state, next_state, env, planner_i=None, planner_j=None):
        dist = max(
            abs(next_state.uav_pos[0] - next_state.target_pos[0]),
            abs(next_state.uav_pos[1] - next_state.target_pos[1]),
        )
        return {"distance": dist}


if __name__ == "__main__":
    from core.paths import get_results_dir

    log_dir = get_results_dir("uav", f"uav_rts_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    exp_config = ExperimentConfig(n_trials=100, max_steps=25, export_trees=False, verbose=False)

    logger.info("Initializing UAV RTS Runner...")
    logger.info(f"Configuration: N={exp_config.n_trials}, Horizon={exp_config.max_steps}")

    runner = UAV_RTS_Runner(config=exp_config, log_dir=log_dir)
    df = runner.run_batch()

    if not df.empty:
        logger.info("Generating interactive debug plots...")
        plot_all_metrics(
            df,
            agent_labels={"i": "UAV (RTS - L2)", "j": "Target (I-POMCP - L1)"},
            title_prefix="RTS Performance:",
            save_dir=log_dir,
        )

        logger.info("Generating publication vector graphics...")
        generate_paper_plots(os.path.join(log_dir, "batch_results.csv"), log_dir)
        logger.info("UAV RTS execution pipeline finalized successfully.")
    else:
        logger.warning("Dataframe returned empty. Aborting plot generation.")
