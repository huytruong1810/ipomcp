import os
from datetime import datetime

from core.config import ExperimentConfig
from core.logger import get_logger
from examples.uav.model.uav_model import UAVModel
from solvers.solver_bank import SolverBank
from utils.bootstrapper import I_POMDP_Bootstrapper
from utils.generic_batch_runner import GenericBatchRunner
from utils.paper_plots import generate_paper_plots
from utils.plotting import plot_all_metrics

logger = get_logger("UAVBaseline")


class UAVBaselineRunner(GenericBatchRunner):
    """
    Control Group: Executes Level-0 (Random) vs Level-0 (Random) agents.
    Used to establish the performance floor of the UAV domain.
    """

    def _setup_domain(self):
        env = UAVModel()
        # 1. Register L0 Solver for Agent J
        bank_j = SolverBank()
        boot_j = I_POMDP_Bootstrapper(bank_j)
        key_j = boot_j.create_level0_solver("j", env)
        planner_j = bank_j.get_solver(key_j)

        # 2. Register L0 Solver for Agent I
        bank_i = SolverBank()
        boot_i = I_POMDP_Bootstrapper(bank_i)
        key_i = boot_i.create_level0_solver("i", env)
        planner_i = bank_i.get_solver(key_i)

        return env, planner_i, planner_j, env.get_initial_state()

    def _get_custom_metrics(self, state, next_state, env, planner_i=None, planner_j=None):
        dist = max(
            abs(next_state.uav_pos[0] - next_state.target_pos[0]),
            abs(next_state.uav_pos[1] - next_state.target_pos[1]),
        )
        return {"distance": dist}


if __name__ == "__main__":
    from core.paths import get_results_dir

    log_dir = get_results_dir("uav", f"uav_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    exp_config = ExperimentConfig(n_trials=1000, max_steps=25, export_trees=False, verbose=False)

    logger.info("Initializing UAV Baseline Runner...")
    logger.info(f"Configuration: N={exp_config.n_trials}, Horizon={exp_config.max_steps}")

    runner = UAVBaselineRunner(config=exp_config, log_dir=log_dir)
    df = runner.run_batch()

    if not df.empty:
        logger.info("Generating interactive debug plots...")
        plot_all_metrics(
            df,
            agent_labels={"i": "UAV (Random - L0)", "j": "Target (Random - L0)"},
            title_prefix="UAV Baseline (Random vs Random):",
            save_dir=log_dir,
        )

        logger.info("Generating publication vector graphics...")
        generate_paper_plots(os.path.join(log_dir, "batch_results.csv"), log_dir)
        logger.info("UAV Baseline execution pipeline finalized successfully.")
    else:
        logger.warning("Dataframe returned empty. Aborting plot generation.")
