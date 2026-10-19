"""The I-NTMCP baseline adapter: the authors' planner on this package's domains."""

import json

import pytest

pytest.importorskip("intmcp", reason="third_party/i-ntmcp is not on the path")

from ipomdp.baselines.intmcp import INTMCPAgent, POSGAdapter  # noqa: E402
from ipomdp.domains.tiger import LISTEN, OPEN_LEFT, OPEN_RIGHT, TigerModel  # noqa: E402
from ipomdp.domains.wumpus.model import WumpusModel  # noqa: E402


def test_adapter_presents_the_domain_in_its_own_order():
    physics = WumpusModel(3, 3, 1)
    model = POSGAdapter(physics, ("i", "j"), seed=0)
    for index, agent in enumerate(("i", "j")):
        assert [a.value for a in model.get_agent_action_space(index)] == physics.get_all_actions(
            agent
        )
    assert model.r_min <= min(physics.reward_bounds("i")[0], physics.reward_bounds("j")[0])
    assert len(model.state_space) == len(physics.initial_distribution())


def test_step_follows_the_kernels():
    physics = TigerModel()
    model = POSGAdapter(physics, ("i", "j"), seed=0)
    state = model.sample_initial_state()
    action = model.action_space[0]
    following, observation, rewards, done = model.step(state, action)
    joint = {"i": action.get_agent_action(0).value, "j": action.get_agent_action(1).value}
    for index, agent in enumerate(("i", "j")):
        o = observation.get_agent_obs(index).value
        assert physics.observation_probability(o, following.value, joint, agent) > 0
        assert rewards[index] == physics.get_reward(state.value, joint, following.value, agent)
    assert done == physics.is_terminal(following.value)


def test_one_step_search_listens_at_the_tiger_prior():
    agent = INTMCPAgent(TigerModel(), level=1, simulations=2000, horizon=1, gamma=0.95, seed=0)
    assert agent.search() == LISTEN
    # One step left: listening's value is its certain reward.
    assert agent.values[LISTEN] == pytest.approx(-1.0)
    assert set(agent.values) == {LISTEN, OPEN_LEFT, OPEN_RIGHT}


def test_searches_are_reproducible_from_the_seed():
    def run():
        agent = INTMCPAgent(TigerModel(), level=2, simulations=300, horizon=2, gamma=0.95, seed=4)
        agent.search()
        agent.update_root(LISTEN, ("GL", "S"))
        agent.search()
        return agent.values

    assert run() == run()


def test_invalid_use_is_refused():
    with pytest.raises(ValueError, match="level"):
        INTMCPAgent(TigerModel(), level=0, simulations=10, horizon=1, gamma=0.95, seed=0)
    agent = INTMCPAgent(TigerModel(), level=1, simulations=10, horizon=1, gamma=0.95, seed=0)
    with pytest.raises(RuntimeError, match="search first"):
        agent.update_root(LISTEN, ("GL", "S"))


def test_tuned_variant_keeps_beliefs_consistent_with_deterministic_senses():
    physics = WumpusModel(3, 3, 1)
    agent = INTMCPAgent(
        physics, 1, simulations=2000, horizon=1, gamma=0.95, seed=17, variant="tuned"
    )
    agent.search()
    for action, observation in [("TurnLeft", ()), ("Forward", ("Breeze",))]:
        agent.update_root(action, observation)
        agent.search()
    assert agent.forced_shares == [0.0, 0.0]


def test_variants_are_validated():
    with pytest.raises(ValueError, match="variant"):
        INTMCPAgent(TigerModel(), 1, 10, 1, 0.95, 0, variant="other")


def _uav_spec(variant="tuned", **i_options):
    from ipomdp.experiments.study import AgentSpec, StudySpec
    from ipomdp.experiments.suite import hierarchy_agent, ipomcp

    planner = {"kind": "intmcp", "simulations": 60, "depth": 2, "variant": variant}
    return StudySpec(
        name="uav-intmcp",
        domain="uav",
        trials=1,
        steps=3,
        agents={
            "i": AgentSpec(level=2, planner=planner, **i_options),
            "j": hierarchy_agent(1, ipomcp(30, 2), None, 16),
        },
    )


def test_an_intmcp_agent_plays_uav_episodes_reproducibly():
    from ipomdp.experiments.episodes import run_episode

    def play():
        return [
            {
                k: v
                for k, v in row.items()
                if not k.endswith(("_cpu_i", "_wall_i", "_cpu_j", "_wall_j", "rss_mb"))
            }
            for row in run_episode(_uav_spec(), 3)
        ]

    rows = play()
    assert rows == play()
    decisions = [row for row in rows[1:] if "forced_share_i" in row]
    assert decisions and all(0 <= row["forced_share_i"] <= 1 for row in decisions)
    assert decisions[0]["forced_share_i"] == 0.0  # the prior
    assert all(json.loads(row["action_values_i"]) for row in decisions)
    assert all(row["prediction_i"] == "null" for row in decisions)


def test_the_episode_planner_sees_only_its_own_steps():
    from ipomdp.baselines.intmcp import INTMCPPlanner
    from ipomdp.domains.uav import UAVModel
    from ipomdp.experiments.study import build_agents

    physics, planners = build_agents(_uav_spec(), 0)
    planner = planners["i"]
    assert isinstance(planner, INTMCPPlanner)
    # Agent index 0 of the adapter is the planner's own agent.
    assert planner.agent.model.agents == ("i", "j")
    assert isinstance(physics, UAVModel)


@pytest.mark.parametrize(
    "options, message",
    [
        ({"modeled": "100"}, "modeled"),
        ({"particles": 64}, "particles"),
        ({"particle_filter": "sampled"}, "particle_filter"),
        ({"level_weights": {0: 0.5, 1: 0.5}}, "one opponent level"),
    ],
)
def test_intmcp_specs_outside_its_model_class_are_refused(options, message):
    with pytest.raises(ValueError, match=message):
        _uav_spec(**options)


def test_a_point_mass_on_the_level_below_is_accepted():
    _uav_spec(level_weights={1: 1.0})
