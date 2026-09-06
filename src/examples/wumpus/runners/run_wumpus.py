import time
from core.config import IPOMCPConfig, MCTSConfig
from solvers.exploration import NormalizedUCB
from examples.wumpus.model.wumpus_model import WumpusModel, AGENT_HUMAN, AGENT_WUMPUS
from examples.wumpus.model.wumpus_viz import WumpusVisualizer
from solvers.solver_bank import SolverBank
from utils.bootstrapper import I_POMDP_Bootstrapper
from utils.visualizer import ForestVisualizer


def run_interactive_wumpus():
    print("=== Interactive Wumpus World (Human L2 vs Wumpus L1) ===")

    # 1. Setup Environment (4x4, 2 Pits)
    real_env = WumpusModel(width=4, height=4, n_pits=2)
    viz = WumpusVisualizer()

    # 2. Build Hierarchy with Isolated Banks
    mcts_wumpus = MCTSConfig(n_sims=5000, max_depth=10, node_capacity=500)
    config_wumpus = IPOMCPConfig(mcts=mcts_wumpus)

    mcts_human = MCTSConfig(n_sims=5000, max_depth=10, node_capacity=500)
    config_human = IPOMCPConfig(mcts=mcts_human)

    # Isolated bank for Wumpus (Agent J / Level 1)
    bank_wumpus = SolverBank()
    boot_wumpus = I_POMDP_Bootstrapper(bank_wumpus)
    boot_wumpus.create_level0_solver(AGENT_HUMAN, real_env)
    boot_wumpus.create_level0_solver(AGENT_WUMPUS, real_env)
    planner_wumpus_l1 = boot_wumpus.create_level1_solver(
        AGENT_WUMPUS, real_env, [AGENT_HUMAN],
        n_particles=5000,
        config=config_wumpus,
        exploration_strategy=NormalizedUCB(exploration_const=1.0)
    )

    # Isolated bank for Human (Agent I / Level 2)
    bank_human = SolverBank()
    boot_human = I_POMDP_Bootstrapper(bank_human)
    boot_human.create_level0_solver(AGENT_HUMAN, real_env)
    boot_human.create_level0_solver(AGENT_WUMPUS, real_env)
    boot_human.create_level1_solver(
        AGENT_WUMPUS, real_env, [AGENT_HUMAN],
        n_particles=5000,
        config=config_wumpus,
        exploration_strategy=NormalizedUCB(exploration_const=1.0)
    )
    planner_human_l2 = boot_human.create_level2_solver(
        AGENT_HUMAN, real_env, [AGENT_WUMPUS],
        l1_probability=0.9,
        n_particles=5000,
        config=config_human,
        exploration_strategy=NormalizedUCB(exploration_const=1.0)
    )

    # 3. Initialization
    true_state = real_env.get_initial_state()
    print("Initial State:")
    viz.print_grid(true_state)

    total_reward = 0.0

    # 4. Loop
    for t in range(50):
        print(f"\n--- Step {t + 1} ---")

        # Plan
        start = time.time()
        print("Human (L2) Planning...")
        a_human = planner_human_l2.get_action()

        print("Wumpus (L1) Planning...")
        a_wumpus = planner_wumpus_l1.get_action()

        plan_time = time.time() - start

        # Act
        joint_action = {AGENT_HUMAN: a_human, AGENT_WUMPUS: a_wumpus}
        print(f"Actions: Human={a_human}, Wumpus={a_wumpus} ({plan_time:.2f}s)")

        next_state = real_env.sample_transition(true_state, joint_action)

        # Observe
        o_human = real_env.sample_observation(next_state, joint_action, AGENT_HUMAN)
        o_wumpus = real_env.sample_observation(next_state, joint_action, AGENT_WUMPUS)

        # Reward
        r_human = real_env.get_reward(true_state, joint_action, next_state, AGENT_HUMAN)
        total_reward += r_human

        print(f"Obs Human: {o_human}")
        print(f"Obs Wumpus: {o_wumpus}")
        print(f"Reward: {r_human} (Total: {total_reward})")

        viz.print_grid(next_state)

        # Visualization Export (Every 5 steps)
        if t % 25 == 0:
            forest_viz.export_forest(planner_human_l2, f"../results/wumpus_forest_step_{t}", t)

        # Check Terminal
        if real_env.is_terminal(next_state):
            if next_state.has_gold:
                print("\nVICTORY! Human obtained Gold.")
            else:
                print("\nDEFEAT! Human died.")
            break

        # Belief Update
        planner_human_l2.update_root(a_human, o_human, min_human_particles)
        planner_wumpus_l1.update_root(a_wumpus, o_wumpus, min_wumpus_particles)

        true_state = next_state


if __name__ == "__main__":
    run_interactive_wumpus()