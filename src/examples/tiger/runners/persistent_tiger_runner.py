# Absolute Path: <project_root>/examples/tiger/runners/persistent_tiger_runner.py

from utils.generic_batch_runner import GenericBatchRunner
from examples.tiger.model.tiger_model import TigerModel
from solvers.solver_bank import SolverBank
from solvers.exploration import NormalizedUCB
from utils.bootstrapper import I_POMDP_Bootstrapper
from utils.plotting import plot_all_metrics
from utils.paper_plots import generate_paper_plots
from core.config import ExperimentConfig, IPOMCPConfig, MCTSConfig, JITConfig
from core.logger import get_logger

import os
from datetime import datetime

logger = get_logger("PersistentTigerRunner")


class PersistentTigerRunner(GenericBatchRunner):
    """
    Experimental Group: L2 (Blind) vs L1 (Omniscient) in a Static Environment.

    Theoretically, L1 will listen once, find the gold with 100% certainty,
    and then farm the gold door repeatedly.
    L2 will receive pure noise from the tiger, wait to hear L1's door creak,
    deduce the gold's location via theory of mind, and begin free-riding on L1.
    """

    def _setup_domain(self):
        # i is perfectly blind (0.5), j is perfectly informed (1.0)
        growl_dict = {'i': 0.5, 'j': 1.0}

        # persistent=True keeps the gold behind the same door forever
        env = TigerModel(growl_accuracy=growl_dict, creak_accuracy=0.90, persistent=True)

        mcts_cfg = MCTSConfig(n_sims=10000, max_depth=6, node_capacity=500)
        jit_cfg = JITConfig(entropy_threshold=0.6, visit_threshold=5, sims=10)
        agent_config = IPOMCPConfig(mcts=mcts_cfg, jit=jit_cfg)

        # 1. Opponent Agent J (Level-1 I-POMCP)
        bank_j = SolverBank()
        boot_j = I_POMDP_Bootstrapper(bank_j)
        boot_j.create_level0_solver('i', TigerModel(growl_accuracy=growl_dict, persistent=True))
        planner_j = boot_j.create_level1_solver(
            'j', TigerModel(growl_accuracy=growl_dict, persistent=True), ['i'],
            n_particles=2000,
            config=agent_config,
            exploration_strategy=NormalizedUCB(exploration_const=2**(0.5))
        )

        # 2. Protagonist Agent I (Level-2 I-POMCP) with isolated bank
        bank_i = SolverBank()
        boot_i = I_POMDP_Bootstrapper(bank_i)
        boot_i.create_level0_solver('j', TigerModel(growl_accuracy=growl_dict, persistent=True))
        boot_i.create_level1_solver(
            'j', TigerModel(growl_accuracy=growl_dict, persistent=True), ['i'],
            n_particles=2000,
            config=agent_config,
            exploration_strategy=NormalizedUCB(exploration_const=2**(0.5))
        )
        planner_i = boot_i.create_level2_solver(
            'i', TigerModel(growl_accuracy=growl_dict, persistent=True), ['j'],
            l1_probability=1.0,
            n_particles=2000,
            config=agent_config,
            exploration_strategy=NormalizedUCB(exploration_const=2**(0.5))
        )

        return env, planner_i, planner_j, env.get_initial_state()


if __name__ == "__main__":
    from core.paths import get_results_dir
    log_dir = get_results_dir("tiger", f"persistent_tiger_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    exp_config = ExperimentConfig(
        n_trials=1000,
        max_steps=10,
        export_trees=False,
        verbose=False
    )

    logger.info(f"Initializing Persistent Tiger Runner...")
    logger.info(f"Configuration: N={exp_config.n_trials}, Horizon={exp_config.max_steps}")

    runner = PersistentTigerRunner(config=exp_config, log_dir=log_dir)
    df = runner.run_batch()

    if not df.empty:
        logger.info("Generating interactive debug plots...")
        plot_all_metrics(df,
                         agent_labels={'i': 'Agent I (L2 - Blind)', 'j': 'Agent J (L1 - Omniscient)'},
                         title_prefix="Persistent Tiger (Blind vs Omniscient)",
                         save_dir=log_dir)

        logger.info("Generating IEEE/ACM compliant vector graphics...")
        generate_paper_plots(os.path.join(log_dir, "batch_results.csv"), log_dir)

        logger.info("Persistent Tiger execution pipeline finalized successfully.")
    else:
        logger.warning("Dataframe returned empty. Aborting plot generation.")