"""Independent small enumerations for a finite-computation opponent contract."""

import itertools

import pytest

from core.config import IPOMCPConfig, MCTSConfig, OpponentPolicyConfig
from examples.experiments.planner_oracle_experiment import evaluate_case, run_oracle_comparison
from examples.tiger.model.tiger_model import TIGER_LEFT, TIGER_RIGHT, TigerModel
from ipomdp.finite_belief import FiniteBelief, InteractiveState, MentalModel
from ipomdp.finite_filter import InferenceBudgetExceeded
from ipomdp.frame import AgentFrame
from solvers.exact.finite_policy_l2 import FinitePolicyL2Reference
from solvers.generative_model import InteractiveGenerativeModel
from solvers.i_pomcp import IPOMCPPlanner
from solvers.solver_bank import SolverBank
from solvers.solver_types import SolverKey
from tests.reference_tiger import ACTIONS, STATES, transition_probability


def setup(seed=4, p=0.5, p_j=0.085):
    physics = TigerModel()
    bank = SolverBank(seed=seed)
    provider = IPOMCPPlanner(
        SolverKey("j", 1),
        physics,
        list(ACTIONS),
        bank,
        config=IPOMCPConfig(
            mcts=MCTSConfig(max_depth=3, n_sims=1000),
            opponent=OpponentPolicyConfig(n_sims=25),
        ),
    )
    bank.register_solver(SolverKey("j", 1), provider)
    l0 = MentalModel(AgentFrame("i", 0, physics))
    private = FiniteBelief(
        (
            (InteractiveState(TIGER_LEFT, l0), p_j),
            (InteractiveState(TIGER_RIGHT, l0), 1 - p_j),
        )
    )
    opponent = MentalModel(AgentFrame("j", 1, physics), private)
    model = MentalModel(
        AgentFrame("i", 2, physics),
        FiniteBelief(
            (
                (InteractiveState(TIGER_LEFT, opponent), p),
                (InteractiveState(TIGER_RIGHT, opponent), 1 - p),
            )
        ),
    )
    return physics, bank, provider, model


def test_equal_private_beliefs_have_same_policy_after_cache_eviction():
    # Before canonical sampling, seed4/b_j=.085 produced OL in one insertion
    # order and L in the other despite equal immutable models and identical seeds.
    physics, bank, provider, model = setup()
    original = model.belief.mass[0][0].opponent
    reversed_model = MentalModel(
        original.frame, FiniteBelief(tuple(reversed(original.belief.mass)))
    )
    assert original == reversed_model
    assert bank.ordered_mass(original.belief) == bank.ordered_mass(reversed_model.belief)
    first = provider.policy_for(original, modeled=True)
    bank.clear_caches()
    assert provider.policy_for(reversed_model, modeled=True) == first
    assert bank.policy(reversed_model) == first
    bank.clear_caches()
    assert bank.policy(original) == first


def independent_h2(physics, bank, model, gamma):
    """Integrate physical states directly; no reference branches/filter calls.

    With one decision left reward depends only on physical state, so the final
    maximization needs the physical posterior, not j's recursively updated model.
    Opponent policy at the initial private model is the declared environment law.
    """
    values = {}
    for action in ACTIONS:
        reward = 0.0
        branches = {}
        for atom, prior in model.belief.mass:
            for other, policy_mass in bank.policy(atom.opponent).items():
                for following in STATES:
                    weight = (
                        prior
                        * policy_mass
                        * transition_probability(atom.state, following, action, other)
                    )
                    own_reward = (
                        -1
                        if action == "L"
                        else (10 if (action == "OL") == (atom.state == TIGER_RIGHT) else -100)
                    )
                    reward += weight * own_reward
                    for obs in physics.get_all_observations("i"):
                        mass = weight * physics.get_observation_prob(
                            obs, following, {"i": action, "j": other}, "i"
                        )
                        if mass:
                            entry = branches.setdefault(obs, {s: 0.0 for s in STATES})
                            entry[following] += mass
        continuation = 0.0
        for weights in branches.values():
            total = sum(weights.values())
            # Maximize one expected reward AFTER integrating the hidden state.
            continuation += max(
                -total,
                10 * weights[TIGER_RIGHT] - 100 * weights[TIGER_LEFT],
                10 * weights[TIGER_LEFT] - 100 * weights[TIGER_RIGHT],
            )
        values[action] = reward + gamma * continuation
    return values


@pytest.mark.parametrize("p,p_j", [(0.5, 0.5), (0.05, 0.085), (0.8, 0.915)])
def test_exhaustive_h2_matches_independent_physical_enumeration(p, p_j):
    physics, bank, _, model = setup(p=p, p_j=p_j)
    reference = FinitePolicyL2Reference(physics, bank)
    assert reference.q_values(model, 2) == pytest.approx(
        independent_h2(physics, bank, model, 0.95), abs=1e-11
    )
    assert reference.q_values(model, 0) == dict.fromkeys(ACTIONS, 0)
    if p == p_j == 0.5:
        assert reference.q_values(model, 2)["L"] == pytest.approx(-1.95)


def scalar_private_update(physics, p, action, observation):
    weights = dict.fromkeys(STATES, 0.0)
    for previous, following, other in itertools.product(STATES, STATES, ACTIONS):
        prior = p if previous == TIGER_LEFT else 1 - p
        weights[following] += (
            prior
            / 3
            * transition_probability(previous, following, action, other)
            * physics.get_observation_prob(observation, following, {"j": action, "i": other}, "j")
        )
    return weights[TIGER_LEFT] / sum(weights.values())


def test_joint_branches_preserve_model_identity_and_independent_moments(monkeypatch):
    physics, bank, _, model = setup(p=0.2, p_j=0.085)
    # Retain the actual private posterior objects produced by the declared
    # kernel; the exhaustive reference must not reconstruct scalar copies.
    update = bank.filter.update
    returned = set()

    def traced(*args, **kwargs):
        result = update(*args, **kwargs)
        if args[0].frame.level == 1:
            returned.add(id(result.belief))
        return result

    monkeypatch.setattr(bank.filter, "update", traced)
    reference = FinitePolicyL2Reference(physics, bank)
    for action in ACTIONS:
        _, branches = reference.branches(model, action)
        for observation, evidence, successor in branches:
            # Independent event enumeration: scalar subjective Bayesian update
            # plus physical/private likelihoods, without the production filter.
            totals = [0.0] * 4
            for atom, prior in model.belief.mass:
                p_j = sum(w for state, w in atom.opponent.belief.mass if state.state == TIGER_LEFT)
                for other, policy_mass in bank.policy(atom.opponent).items():
                    for following, private in itertools.product(
                        STATES, physics.get_all_observations("j")
                    ):
                        joint = {"i": action, "j": other}
                        mass = (
                            prior
                            * policy_mass
                            * transition_probability(atom.state, following, action, other)
                            * physics.get_observation_prob(observation, following, joint, "i")
                            * physics.get_observation_prob(private, following, joint, "j")
                        )
                        if not mass:
                            continue
                        p_next = scalar_private_update(physics, p_j, other, private)
                        left = float(following == TIGER_LEFT)
                        for i, v in enumerate((1, left, p_next, left * p_next)):
                            totals[i] += mass * v
            actual = [0.0] * 3
            for atom, mass in successor.belief.mass:
                assert id(atom.opponent.belief) in returned
                private_p = sum(
                    w for state, w in atom.opponent.belief.mass if state.state == TIGER_LEFT
                )
                left = float(atom.state == TIGER_LEFT)
                for i, v in enumerate((left, private_p, left * private_p)):
                    actual[i] += mass * v
            assert evidence == pytest.approx(totals[0], abs=1e-12)
            assert actual == pytest.approx([v / evidence for v in totals[1:]], abs=1e-12)
    # Generative search receives the exact same initial model, not a scalar
    # projection of it; query logging verifies the production policy call.
    seen = []
    distribution = bank.filter.action_distribution

    def capture(opponent, state):
        seen.append(opponent)
        return distribution(opponent, state)

    monkeypatch.setattr(bank.filter, "action_distribution", capture)
    particle = model.belief.mass[0][0]
    InteractiveGenerativeModel(bank).sample_event(particle, "L", "i", physics)
    assert seen[0] is particle.opponent


def test_reference_budget_failure_is_explicit():
    physics, bank, _, model = setup()
    with pytest.raises(InferenceBudgetExceeded):
        FinitePolicyL2Reference(physics, bank, max_branches=1).q_values(model, 2)


def test_runner_records_distinct_real_and_modeled_configuration():
    row = evaluate_case(
        "mcts",
        2,
        100,
        11,
        0.5,
        0.95,
        "bounded",
        1,
        True,
        "empirical_bellman",
        level=2,
        opponent_depth=2,
        opponent_budget=25,
    )[0]
    assert row["opponent_model"]["policy"] == "finite_mcts_l1"
    assert row["opponent_model"]["bank_seed"] == row["bank_seed"] == 11
    assert row["opponent_model"]["config"]["opponent"]["n_sims"] == 25
    assert row["opponent_model"]["config"]["mcts"]["max_depth"] == 2
    assert row["opponent_model"]["config"]["mcts"]["backup"] == "sampled"
    assert row["planner_config"]["mcts"]["n_sims"] == 100
    assert row["planner_config"]["mcts"]["backup"] == "empirical_bellman"
    assert row["oracle_q"]["L"] == pytest.approx(-1.95)


def test_ambiguous_modeled_options_rejected_before_output(tmp_path):
    with pytest.raises(ValueError, match="explicit opponent budget"):
        run_oracle_comparison(
            tmp_path / "invalid",
            planners=("mcts",),
            level=2,
            opponent_depth=2,
            opponent_backup="empirical_bellman",
        )
    with pytest.raises(ValueError, match="requires L2"):
        run_oracle_comparison(tmp_path / "l1", opponent_budget=25)
    assert not list(tmp_path.iterdir())
