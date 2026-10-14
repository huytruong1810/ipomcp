"""Analytic properties of the exact L2 Tiger problem (exact L1 opponent).

The problem is solved by the unified best-response reference with an exact,
fixed-depth L1 opponent registered in the bank. The hand calculations below are
independent of that code.
"""

import random

import pytest

from ipomdp.belief.finite_belief import FiniteBelief, InteractiveState, MentalModel
from ipomdp.belief.frame import AgentFrame
from ipomdp.domains.tiger import LISTEN, OPEN_LEFT, OPEN_RIGHT, TIGER_LEFT, TIGER_RIGHT, TigerModel
from ipomdp.experiments.problems import chain_model, tiger_chain
from ipomdp.planning.random_planner import RandomPlanner
from ipomdp.planning.solver_bank import SolverBank
from ipomdp.planning.solver_types import SolverKey
from ipomdp.reference.best_response import BestResponseReference
from ipomdp.reference.exact_agent import ExactIPOMDPPlanner


def exact_l2(physics=None, opponent_depth=2, gamma=0.95):
    physics = physics or TigerModel()
    bank = SolverBank(0)
    reference = BestResponseReference(physics, bank, gamma)
    bank.register_solver(
        SolverKey("i", 0), RandomPlanner(physics.get_all_actions("i"), random.Random(0))
    )
    opponent = ExactIPOMDPPlanner(SolverKey("j", 1), physics, bank, reference, opponent_depth)
    bank.register_solver(SolverKey("j", 1), opponent)

    def q(p, p_j=0.5, horizon=2):
        model = chain_model(physics, "i", 2, tiger_chain(p, p_j))
        return reference.q_values(model, horizon)

    return q, opponent, physics


def private_model(physics, p):
    l0 = MentalModel(AgentFrame("i", 0, physics))
    belief = FiniteBelief(
        ((InteractiveState(TIGER_LEFT, l0), p), (InteractiveState(TIGER_RIGHT, l0), 1 - p))
    )
    return MentalModel(AgentFrame("j", 1, physics), belief)


def test_left_right_symmetry():
    q, _, _ = exact_l2()
    for p in [0.05, 0.2, 0.5]:
        left, right = q(p), q(1 - p)
        assert left[LISTEN] == pytest.approx(right[LISTEN], abs=1e-9)
        assert left[OPEN_LEFT] == pytest.approx(right[OPEN_RIGHT], abs=1e-9)


def test_confident_beliefs_open_and_ambiguous_beliefs_listen():
    q, _, _ = exact_l2()
    for p, best in [(0.02, OPEN_LEFT), (0.98, OPEN_RIGHT), (0.5, LISTEN)]:
        values = q(p)
        assert max(values, key=values.get) == best


def test_two_step_value_has_no_clairvoyant_continuation():
    """One 85% growl cannot make a final opening profitable: V2 = -1 - .95.

    Listening pays -1 on the last decision; the better opening pays
    .85 * 10 - .15 * 100 = -6.5. A statewise maximization would give +8.5.
    """
    q, _, _ = exact_l2()
    assert q(0.5)[LISTEN] == pytest.approx(-1.95, abs=1e-12)


def test_uninformative_sensors_cannot_reveal_hidden_state():
    q, _, _ = exact_l2(TigerModel(growl_accuracy={"i": 0.5, "j": 0.5}))
    assert max(q(0.5, horizon=3).values()) == pytest.approx(-1 - 0.95 - 0.95**2)


def test_one_step_reward_does_not_depend_on_the_opponents_private_belief():
    q, _, _ = exact_l2()
    for p_j in [0.0123456789, 0.4, 0.987654321]:
        assert q(0.5, p_j, horizon=1) == pytest.approx(
            {LISTEN: -1, OPEN_LEFT: -45, OPEN_RIGHT: -45}
        )


def test_fixed_depth_opponent_acts_on_its_own_belief_at_its_own_depth():
    # At p = .085 a one-step L1 opens left, a two-step L1 listens.
    for depth, expected in [(1, {OPEN_LEFT: 1.0}), (2, {LISTEN: 1.0})]:
        _, opponent, physics = exact_l2(opponent_depth=depth)
        assert opponent.policy_for(private_model(physics, 0.085), modeled=True) == expected
    _, opponent, physics = exact_l2()
    assert opponent.policy_for(private_model(physics, 0.01), modeled=True) == {OPEN_LEFT: 1.0}
    assert opponent.policy_for(private_model(physics, 0.99), modeled=True) == {OPEN_RIGHT: 1.0}
