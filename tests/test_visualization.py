import os
import pytest
from examples.tiger.model.tiger_model import TigerModel
from solvers.solver_bank import SolverBank
from utils.bootstrapper import I_POMDP_Bootstrapper
from utils.generic_batch_runner import extract_nested_belief_hierarchy
from utils.plotting import plot_nested_belief_sunburst, plot_episode_sunburst_slider
from utils.paper_plots import generate_nested_sunburst_pdf
from core.config import IPOMCPConfig, MCTSConfig


def test_extract_nested_belief_hierarchy_level3(tmp_path):
    bank = SolverBank()
    env = TigerModel()
    boot = I_POMDP_Bootstrapper(bank)
    
    cfg = IPOMCPConfig(mcts=MCTSConfig(n_sims=50, max_depth=3, node_capacity=1000))
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
    
    data = extract_nested_belief_hierarchy(l3_planner, agent_id="i", agent_level=3)
    assert data
    assert "ids" in data
    assert "labels" in data
    assert "parents" in data
    assert "values" in data
    
    # Root checks
    assert data["ids"][0] == "i_L3"
    assert data["parents"][0] == ""
    assert data["values"][0] == 1.0
    
    # Ring 1 children (j_L0, j_L1, j_L2)
    r1_children = [i for i, p in enumerate(data["parents"]) if p == "i_L3"]
    assert len(r1_children) == 3
    r1_sum = sum(data["values"][i] for i in r1_children)
    assert abs(r1_sum - 1.0) < 1e-5
    
    # Test Plotly Sunburst Generation
    fig_sb = plot_nested_belief_sunburst(data, title="Test Sunburst", save_dir=str(tmp_path), filename="test_sb")
    assert os.path.exists(tmp_path / "test_sb.html")
    
    # Test Animated Sunburst Slider Generation
    snapshots = {0: data, 1: data}
    fig_anim = plot_episode_sunburst_slider(snapshots, title_prefix="Test Animated", save_dir=str(tmp_path), filename="test_anim")
    assert os.path.exists(tmp_path / "test_anim.html")
    
    # Test Publication PDF Concentric Donut Generation
    pdf_path = str(tmp_path / "test_sunburst.pdf")
    generate_nested_sunburst_pdf(data, pdf_path, title="Level-3 Mental Model Hierarchy")
    assert os.path.exists(pdf_path)
    assert os.path.getsize(pdf_path) > 0


def test_sunburst_slice_position_stability(tmp_path):
    """
    Verifies that animated sunburst slider maintains identical canonical slice positions
    across all frames (sort=False, canonical node IDs, zero-padding for pruned levels).
    """
    # Simulate step 0 with full hierarchy, and step 1 where j_L2 dropped to 0 particles
    step0_data = {
        "ids": ["i_L3", "i_L3/j_L0", "i_L3/j_L1", "i_L3/j_L2"],
        "labels": ["i (Level-3)", "j (L0)", "j (L1)", "j (L2)"],
        "parents": ["", "i_L3", "i_L3", "i_L3"],
        "values": [1.0, 0.1, 0.1, 0.8],
        "levels": [3, 0, 1, 2],
        "hover_texts": ["i (L3)", "j (L0): 10%", "j (L1): 10%", "j (L2): 80%"]
    }
    step1_data = {
        "ids": ["i_L3", "i_L3/j_L0", "i_L3/j_L1"],  # j_L2 is absent (pruned)
        "labels": ["i (Level-3)", "j (L0)", "j (L1)"],
        "parents": ["", "i_L3", "i_L3"],
        "values": [1.0, 0.6, 0.4],
        "levels": [3, 0, 1],
        "hover_texts": ["i (L3)", "j (L0): 60%", "j (L1): 40%"]
    }

    snapshots = {0: step0_data, 1: step1_data}
    fig = plot_episode_sunburst_slider(snapshots, title_prefix="Stability Test", save_dir=str(tmp_path), filename="stability_test")

    # Base figure trace must enforce sort=False
    assert fig.data[0].sort is False
    assert fig.data[0].ids == ("i_L3", "i_L3/j_L0", "i_L3/j_L1", "i_L3/j_L2")
    assert fig.data[0].values == (1.0, 0.1, 0.1, 0.8)

    # Frames must maintain the identical canonical ID order and sort=False
    assert len(fig.frames) == 2
    frame0 = fig.frames[0].data[0]
    frame1 = fig.frames[1].data[0]

    assert frame0.sort is False
    assert frame1.sort is False
    assert frame0.ids == ("i_L3", "i_L3/j_L0", "i_L3/j_L1", "i_L3/j_L2")
    assert frame1.ids == ("i_L3", "i_L3/j_L0", "i_L3/j_L1", "i_L3/j_L2")

    # In frame 1, j_L2 was absent in raw data; it must be zero-padded in the canonical position
    assert frame1.values == (1.0, 0.6, 0.4, 0.0)
    assert "Belief: 0.0%" in frame1.hovertext[3]

