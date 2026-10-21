"""Runner-Chaser as kernels agrees with the authors' RCModel on every state and joint action."""

import itertools

import pytest

from ipomdp.domains.runner_chaser import ACTIONS, CHASER, LAYOUTS, RUNNER, RunnerChaserModel

rc = pytest.importorskip("intmcp.envs.rc.model", reason="third_party/i-ntmcp is not on the path")
from intmcp.envs.rc import action as rc_action  # noqa: E402
from intmcp.envs.rc import obs as rc_obs  # noqa: E402
from intmcp.envs.rc import state as rc_state  # noqa: E402

TOKENS = {rc_obs.OPPONENT: "X", rc_obs.WALL: "#", rc_obs.EMPTY: "0"}


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
def test_kernels_match_rcmodel(layout):
    ours, theirs = RunnerChaserModel(layout), rc.RCModel(layout)
    grid = theirs.grid
    assert ours.start == (grid.init_runner_locs[0], grid.init_chaser_locs[0])
    assert (len(grid.init_runner_locs), len(grid.init_chaser_locs)) == (1, 1)
    assert ours.blocked == grid.block_locs and ours.safe == grid.safe_locs
    joint_actions = {
        (ACTIONS[j.get_agent_action(0).action_num], ACTIONS[j.get_agent_action(1).action_num]): j
        for j in rc_action.get_action_space(2)
    }
    assert len(joint_actions) == 16
    free = [loc for loc in range(grid.size) if loc not in grid.block_locs]
    for runner, chaser in itertools.product(free, free):
        state = (runner, chaser)
        source = rc_state.RCState(runner, chaser, grid)
        assert ours.is_terminal(state) == theirs.is_terminal(source)
        for (a_r, a_c), joint in joint_actions.items():
            action = {RUNNER: a_r, CHASER: a_c}
            ((following, p),) = ours.transition_distribution(state, action)
            assert p == 1.0
            expected = theirs._get_next_state(source, joint)
            assert following == (expected.runner_loc, expected.chaser_loc)
            observed = theirs._get_obs(expected)
            for k, agent in enumerate((RUNNER, CHASER)):
                ((obs, q),) = ours.observation_distribution(following, action, agent)
                assert q == 1.0
                assert obs == tuple(TOKENS[t] for t in observed[k].adj_obs)
                assert obs in ours.get_all_observations(agent)
                assert ours.get_reward(state, action, following, agent) == theirs._get_reward(expected)[k]
            assert ours.is_terminal(following) == theirs.is_terminal(expected)


def test_rewards_within_bounds():
    model = RunnerChaserModel("cap4")
    for agent in (RUNNER, CHASER):
        low, high = model.reward_bounds(agent)
        assert (low, high) == (-100.0, 100.0)
