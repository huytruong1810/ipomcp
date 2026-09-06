import pytest
from examples.tiger.model.tiger_model import TigerModel, LISTEN, OPEN_LEFT, OPEN_RIGHT
from solvers.solver_bank import SolverBank
from solvers.solver_types import SolverKey, AgentFrame
from utils.bootstrapper import I_POMDP_Bootstrapper
from core.config import IPOMCPConfig, MCTSConfig, JITConfig, RTSConfig
from solvers.random_planner import RandomPlanner
from solvers.i_pomcp import IPOMCPPlanner
from solvers.rts_planner import RTSPlanner


def test_random_planner():
    planner = RandomPlanner([LISTEN, OPEN_LEFT, OPEN_RIGHT])
    a = planner.get_action()
    assert a in [LISTEN, OPEN_LEFT, OPEN_RIGHT]


def test_solver_bank_registration():
    bank = SolverBank()
    env = TigerModel()
    key = SolverKey("i", 0)
    planner = RandomPlanner(env.get_all_actions("i"))
    bank.register_solver(key, planner)
    
    assert bank.get_solver(key) is planner
    frame = AgentFrame("i", 0, env)
    assert bank.get_solver_for_frame(frame) is planner


def test_bootstrapper_and_ipomcp_hierarchy():
    bank = SolverBank()
    env = TigerModel()
    boot = I_POMDP_Bootstrapper(bank)
    
    boot.create_level0_solver("i", env)
    boot.create_level0_solver("j", env)
    
    cfg = IPOMCPConfig(mcts=MCTSConfig(n_sims=50, max_depth=3))
    l1 = boot.create_level1_solver("j", env, ["i"], n_particles=50, config=cfg)
    l2 = boot.create_level2_solver("i", env, ["j"], l1_probability=1.0, n_particles=50, config=cfg)
    
    assert isinstance(l1, IPOMCPPlanner)
    assert isinstance(l2, IPOMCPPlanner)
    
    action = l2.get_action()
    assert action in env.get_all_actions("i")


def test_bootstrapper_level_weights_mixture():
    bank = SolverBank()
    env = TigerModel()
    boot = I_POMDP_Bootstrapper(bank)
    
    boot.create_level0_solver("i", env)
    boot.create_level0_solver("j", env)
    boot.create_level1_solver("j", env, ["i"], n_particles=50)
    
    # 50% Level-0 / 50% Level-1 mixture
    l2 = boot.create_level2_solver("i", env, ["j"], level_weights={0: 0.5, 1: 0.5}, n_particles=500)
    particles = l2.root.belief_particles
    assert len(particles) == 500
    
    l0_count = sum(1 for p in particles if p.models["j"][0].level == 0)
    l1_count = sum(1 for p in particles if p.models["j"][0].level == 1)
    
    # Check that both levels are populated around 50%
    assert 200 <= l0_count <= 300
    assert 200 <= l1_count <= 300


def test_rts_planner_creation():
    bank = SolverBank()
    env = TigerModel()
    boot = I_POMDP_Bootstrapper(bank)
    boot.create_level0_solver("i", env)
    boot.create_level0_solver("j", env)
    boot.create_level1_solver("j", env, ["i"], n_particles=20)
    
    rts = boot.create_level2_rts_solver("i", env, ["j"], level_weights={0: 0.5, 1: 0.5}, n_particles=20, max_depth=2, obs_branching=2)
    assert isinstance(rts, RTSPlanner)
    a = rts.get_action()
    assert a in env.get_all_actions("i")


def test_recursive_arbitrary_level_bootstrapper_level3():
    bank = SolverBank()
    env = TigerModel()
    boot = I_POMDP_Bootstrapper(bank)
    
    cfg = IPOMCPConfig(mcts=MCTSConfig(n_sims=50, max_depth=3, node_capacity=1000))
    
    # Define exact nested level mixture:
    # Level 3: 1/3 L0, 1/3 L1, 1/3 L2
    # Level 2: 1/2 L0, 1/2 L1
    # Level 1: 1.0 L0
    nested_weights = {
        3: {0: 1/3, 1: 1/3, 2: 1/3},
        2: {0: 0.5, 1: 0.5},
        1: {0: 1.0}
    }
    
    l3_planner = boot.create_solver(
        agent_id="i",
        level=3,
        model=env,
        other_agent_ids=["j"],
        nested_level_weights=nested_weights,
        n_particles=600,
        config=cfg
    )
    
    assert isinstance(l3_planner, IPOMCPPlanner)
    assert l3_planner.key == SolverKey("i", 3)
    
    # 1. Verify complete hierarchy is registered in SolverBank
    assert bank.has_solver(SolverKey("i", 0))
    assert bank.has_solver(SolverKey("j", 0))
    assert bank.has_solver(SolverKey("i", 1))
    assert bank.has_solver(SolverKey("j", 1))
    assert bank.has_solver(SolverKey("i", 2))
    assert bank.has_solver(SolverKey("j", 2))
    assert bank.has_solver(SolverKey("i", 3))
    
    # 2. Verify Level-3 particle mixture distribution (1/3 L0, 1/3 L1, 1/3 L2)
    particles_l3 = l3_planner.root.belief_particles
    assert len(particles_l3) == 600
    
    count_l0 = sum(1 for p in particles_l3 if p.models["j"][0].level == 0)
    count_l1 = sum(1 for p in particles_l3 if p.models["j"][0].level == 1)
    count_l2 = sum(1 for p in particles_l3 if p.models["j"][0].level == 2)
    
    assert 140 <= count_l0 <= 260
    assert 140 <= count_l1 <= 260
    assert 140 <= count_l2 <= 260
    
    # 3. Verify nested Level-2 belief structure inside Level-3 particles
    l2_particles = [p for p in particles_l3 if p.models["j"][0].level == 2]
    sample_l2_root = l2_particles[0].models["j"][1]
    assert sample_l2_root is not None
    assert len(sample_l2_root.belief_particles) == 600
    
    l2_count_sub_l0 = sum(1 for p in sample_l2_root.belief_particles if p.models["i"][0].level == 0)
    l2_count_sub_l1 = sum(1 for p in sample_l2_root.belief_particles if p.models["i"][0].level == 1)
    
    assert 220 <= l2_count_sub_l0 <= 380
    assert 220 <= l2_count_sub_l1 <= 380
    
    # 4. Verify nested Level-1 belief structure inside Level-2 node
    sample_sub_l1_particle = [p for p in sample_l2_root.belief_particles if p.models["i"][0].level == 1][0]
    sample_l1_root = sample_sub_l1_particle.models["i"][1]
    assert sample_l1_root is not None
    
    # Level-1 models opponent at Level 0 with 100%
    for p in sample_l1_root.belief_particles:
        assert p.models["j"][0].level == 0
        assert p.models["j"][1] is None
        
    # 5. Verify recursive MCTS planning execution across all 3 levels
    action = l3_planner.get_action()
    assert action in env.get_all_actions("i")


def test_recursive_bootstrapper_level4_and_level5():
    bank = SolverBank()
    env = TigerModel()
    boot = I_POMDP_Bootstrapper(bank)
    
    cfg = IPOMCPConfig(mcts=MCTSConfig(n_sims=50, max_depth=3, node_capacity=500))
    
    # Bootstrap Level 5 agent (automatically resolves L0, L1, L2, L3, L4 for both agents)
    l5_planner = boot.create_solver(
        agent_id="i",
        level=5,
        model=env,
        other_agent_ids=["j"],
        n_particles=200,
        config=cfg
    )
    
    # Verify all solvers for agent i (0..5) and opponent j (0..4) are registered
    for level in range(6):
        assert bank.has_solver(SolverKey("i", level))
    for level in range(5):
        assert bank.has_solver(SolverKey("j", level))
        
    assert l5_planner is not None
    assert len(l5_planner.root.belief_particles) == 200
    
    # Verify action selection executes without error through 5 nested tiers
    action = l5_planner.get_action()
    assert action in env.get_all_actions("i")


def test_circular_import_safety():
    """Confirms that InteractiveParticle and AgentFrame import cleanly without circular dependency."""
    from ipomdp.belief import InteractiveParticle, AgentFrame
    assert InteractiveParticle is not None
    assert AgentFrame is not None


def test_solver_bank_isolation():
    """Confirms that two separate agents maintain fully isolated SolverBanks without cross-contamination."""
    env = TigerModel()
    cfg = IPOMCPConfig(mcts=MCTSConfig(n_sims=30, max_depth=3))

    # Bank for Agent J (Real-World Opponent)
    bank_j = SolverBank()
    boot_j = I_POMDP_Bootstrapper(bank_j)
    boot_j.create_level0_solver("i", env)
    planner_j = boot_j.create_level1_solver("j", env, ["i"], n_particles=50, config=cfg)

    # Bank for Agent I (Protagonist)
    bank_i = SolverBank()
    boot_i = I_POMDP_Bootstrapper(bank_i)
    boot_i.create_level0_solver("j", env)
    boot_i.create_level1_solver("j", env, ["i"], n_particles=50, config=cfg)
    planner_i = boot_i.create_level2_solver("i", env, ["j"], l1_probability=1.0, n_particles=50, config=cfg)

    # Initial state of Agent J's root visit count
    assert planner_j.root.visit_count == 0

    # Agent I performs planning (simulates Agent J internally inside bank_i)
    _ = planner_i.get_action()

    # Agent J's real-world planner root in bank_j MUST NOT have been mutated!
    assert planner_j.root.visit_count == 0



