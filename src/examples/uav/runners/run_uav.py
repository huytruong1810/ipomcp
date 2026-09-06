from core.config import MCTSConfig, IPOMCPConfig
from solvers.exploration import NormalizedUCB
from examples.uav.model.uav_model import UAVModel
from examples.uav.model.uav_viz import UAVVisualizer
from solvers.solver_bank import SolverBank
from utils.bootstrapper import I_POMDP_Bootstrapper
from utils.visualizer import ForestVisualizer


def run_uav_recon():
    print("=== UAV Reconnaissance (Hunter vs Hunted) ===")

    real_env = UAVModel()
    viz = UAVVisualizer()

    agent_uav = 'i'
    agent_target = 'j'
    min_uav_particles = 5000
    min_target_particles = 5000

    mcts_target = MCTSConfig(n_sims=5000, max_depth=5, node_capacity=500)
    config_target = IPOMCPConfig(mcts=mcts_target)

    mcts_uav = MCTSConfig(n_sims=9000, max_depth=5, node_capacity=500)
    config_uav = IPOMCPConfig(mcts=mcts_uav)

    # Isolated bank for Target (Agent J)
    bank_target = SolverBank()
    boot_target = I_POMDP_Bootstrapper(bank_target)
    boot_target.create_level0_solver(agent_uav, UAVModel())
    boot_target.create_level0_solver(agent_target, UAVModel())
    planner_target = boot_target.create_level1_solver(
        agent_target, UAVModel(), [agent_uav],
        n_particles=min_target_particles,
        config=config_target,
        exploration_strategy=NormalizedUCB(exploration_const=1.0)
    )

    # Isolated bank for UAV (Agent I)
    bank_uav = SolverBank()
    boot_uav = I_POMDP_Bootstrapper(bank_uav)
    boot_uav.create_level0_solver(agent_uav, UAVModel())
    boot_uav.create_level0_solver(agent_target, UAVModel())
    boot_uav.create_level1_solver(
        agent_target, UAVModel(), [agent_uav],
        n_particles=min_target_particles,
        config=config_target,
        exploration_strategy=NormalizedUCB(exploration_const=1.0)
    )
    planner_uav = boot_uav.create_level2_solver(
        agent_uav, UAVModel(), [agent_target],
        n_particles=min_uav_particles,
        config=config_uav,
        exploration_strategy=NormalizedUCB(exploration_const=1.0)
    )

    true_state = real_env.get_initial_state()
    print(f"Start State: {true_state}")
    viz.print_grid(true_state)

    for t in range(15):
        print(f"\n--- Step {t + 1} ---")

        # Plan
        print("UAV Planning...")
        a_uav = planner_uav.get_action()

        # Target Planning (Simulating the opponent)
        # Note: In real world, Target acts on its own info.
        # Here we give Target valid belief updates based on its own obs.
        print("Target Planning...")
        a_target = planner_target.get_action()

        joint_action = {agent_uav: a_uav, agent_target: a_target}
        print(f"(Action UAV, Action Target)=({a_uav}, {a_target})")

        # Act
        next_state = real_env.sample_transition(true_state, joint_action)

        # Observe
        o_uav = real_env.sample_observation(next_state, joint_action, agent_uav)
        o_target = real_env.sample_observation(next_state, joint_action, agent_target)

        r_uav = real_env.get_reward(true_state, joint_action, next_state, agent_uav)
        r_t = real_env.get_reward(true_state, joint_action, next_state, agent_target)

        print(f"Observations: UAV sees {o_uav}, Target sees {o_target}")
        print(f"Reward UAV: {r_uav}")
        print(f"Reward Target: {r_t}")
        viz.print_grid(next_state)

        # Visualize Belief Forest
        # if t % 25 == 0:
        #     forest_viz.export_forest(planner_uav, f"../results/uav_forest_step_{t}", t)

        # Update Beliefs
        planner_uav.update_root(a_uav, o_uav, min_uav_particles)
        planner_target.update_root(a_target, o_target, min_target_particles)

        true_state = next_state

        if real_env.is_terminal(true_state):
            print("\n!!! TARGET SPOTTED !!!")
            break


if __name__ == "__main__":
    run_uav_recon()