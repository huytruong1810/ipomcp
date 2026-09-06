import pytest
from solvers.node import POMCPNode
from ipomdp.belief import InteractiveParticle
from solvers.solver_types import AgentFrame
from core.pomdp_model import POMDPModel


class DummyModel(POMDPModel):
    def get_initial_state(self): return "S0"
    def sample_transition(self, state, joint_action): return state
    def sample_observation(self, state, joint_action, agent_id): return "O0"
    def get_observation_prob(self, observation, state, joint_action, agent_id): return 1.0
    def get_reward(self, state, joint_action, next_state, agent_id): return 0.0
    def is_terminal(self, state): return False
    def get_all_actions(self, agent_id): return ["A1", "A2"]
    def get_all_observations(self, agent_id): return ["O0"]


def test_node_creation_and_linking():
    root = POMCPNode(capacity=10)
    assert root.visit_count == 0
    assert len(root.belief_particles) == 0
    
    child = root.create_child("A1", "O0")
    assert child.parent is root
    assert root.get_child("A1", "O0") is child
    assert root.get_child("A2", "O0") is None


def test_node_reservoir_sampling_capacity():
    node = POMCPNode(capacity=5)
    model = DummyModel()
    frame = AgentFrame("j", 0, model)
    
    # Add 100 particles to a node with capacity 5
    for i in range(100):
        p = InteractiveParticle(state=f"S_{i}", models={"j": (frame, None)})
        node.add_particle(p)
        
    assert len(node.belief_particles) == 5
    assert node._total_particles_routed == 100


def test_node_serialization():
    root = POMCPNode(capacity=10)
    root.visit_count = 10
    root.action_counts["A1"] = 6
    root.action_values["A1"] = 2.5
    child = root.create_child("A1", "O0")
    child.visit_count = 5
    
    data = root.to_dict(max_depth=2)
    assert data["visit_count"] == 10
    assert "A1" in data["children"]
    assert "O0" in data["children"]["A1"]
