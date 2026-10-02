"""L3/L2 Tiger hierarchies: declared priors, exact level posteriors, mixed-level references."""

import itertools
import math

import pytest

from ipomdp.analysis.hierarchy import level_posterior, nested_hierarchy
from ipomdp.belief.finite_belief import FiniteBelief, InteractiveState, MentalModel
from ipomdp.belief.frame import AgentFrame
from ipomdp.domains.tiger import TIGER_LEFT, TIGER_RIGHT, TigerModel
from ipomdp.experiments.bootstrapper import HierarchyBuilder
from ipomdp.planning.config import PlannerConfig
from ipomdp.planning.solver_bank import SolverBank
from ipomdp.reference.best_response import BestResponseReference
from ipomdp.reference.pomdp_exact_vi import ExactPOMDPSolver

ACTIONS = ("L", "OL", "OR")
STATES = (TIGER_LEFT, TIGER_RIGHT)
# The declared design of studies/tiger_l3_vs_l2.toml.
PRIOR_I = {1: 0.8, 2: 0.1, 0: 0.1}
PRIOR_J = {1: 0.5, 0: 0.5}
NESTED = {"i": {3: PRIOR_I, 2: PRIOR_J, 1: {0: 1.0}}, "j": {2: PRIOR_J, 1: {0: 1.0}}}


def tiger():
    return TigerModel()


def build_agent(agent_id, level, simulations, depth, modeled):
    """An exact-belief agent with the study's declared priors."""
    other = "j" if agent_id == "i" else "i"
    config = PlannerConfig(simulations=simulations, depth=depth, modeled_simulations=modeled)
    return HierarchyBuilder(SolverBank(0)).build(
        agent_id,
        level,
        tiger(),
        other,
        config=config,
        level_weights=NESTED[agent_id][level],
        nested_level_weights=NESTED[agent_id],
    )


def physical_belief(planner):
    return math.fsum(w for atom, w in planner.belief.mass if atom.state == TIGER_LEFT)


def test_bank_prior_is_the_domains_exact_prior():
    physics, bank = tiger(), SolverBank(1)
    assert bank.physical_prior(physics) == ((TIGER_LEFT, 0.5), (TIGER_RIGHT, 0.5))
    assert bank.physical_prior(physics) is bank.physical_prior(tiger())


@pytest.mark.parametrize("agent_id,level", [("i", 3), ("j", 2)])
def test_declared_priors_and_hierarchies(agent_id, level):
    planner = build_agent(agent_id, level, 100, 2, 5)
    expected = PRIOR_I if agent_id == "i" else PRIOR_J
    levels = level_posterior(planner.belief)
    for k, mass in expected.items():
        assert levels[k] == pytest.approx(mass, abs=1e-15)
    assert physical_belief(planner) == 0.5
    tree = {node.path: node.mass for node in nested_hierarchy(planner.belief, agent_id, level)}
    if agent_id == "i":
        assert tree["i_L3/j_L2/i_L1"] == pytest.approx(0.05)
        assert tree["i_L3/j_L1/i_L0"] == pytest.approx(0.8)
    else:
        assert tree["j_L2/i_L1/j_L0"] == pytest.approx(0.5)


@pytest.mark.parametrize("agent", ["i", "j"])
@pytest.mark.parametrize("p", [0.5, 0.2, 0.93])
def test_general_reference_matches_alpha_vectors_against_uniform_l0(agent, p):
    physics = TigerModel(growl_accuracy={"i": 0.85, "j": 0.85}, creak_accuracy=0.9)
    other = "j" if agent == "i" else "i"
    l0 = MentalModel(AgentFrame(other, 0, physics))
    model = MentalModel(
        AgentFrame(agent, 1, physics),
        FiniteBelief(
            ((InteractiveState(TIGER_LEFT, l0), p), (InteractiveState(TIGER_RIGHT, l0), 1 - p))
        ),
    )
    reference = BestResponseReference(physics, SolverBank(3), 0.95)
    exact = ExactPOMDPSolver(physics, horizon=4, gamma=0.95, agent_id=agent, opponent_id=other)
    exact.solve()
    for horizon in (1, 2, 3, 4):
        assert reference.q_values(model, horizon) == pytest.approx(
            exact.q_values(p, horizon), abs=1e-9
        )


def independent_level_posterior(planner, action, observation):
    """One-step Bayes over opponent level from the bank's policy law, no filter."""
    me = planner.key.agent_id
    physics = planner.pomdp_model
    weights = {}
    for atom, prior in planner.belief.mass:
        opponent = atom.opponent
        other = opponent.frame.agent_id
        law = (
            dict.fromkeys(ACTIONS, 1 / 3)
            if opponent.frame.level == 0
            else planner.solver_bank.policy(opponent)
        )
        for other_action, mass in law.items():
            joint = {me: action, other: other_action}
            for following, transition in physics.transition_distribution(atom.state, joint):
                weights[opponent.frame.level] = weights.get(opponent.frame.level, 0.0) + (
                    prior
                    * mass
                    * transition
                    * physics.observation_probability(observation, following, joint, me)
                )
    total = math.fsum(weights.values())
    return {level: w / total for level, w in weights.items()} if total else None


@pytest.mark.parametrize("agent_id,level", [("i", 3), ("j", 2)])
def test_filter_level_posterior_matches_independent_bayes(agent_id, level):
    planner = build_agent(agent_id, level, 200, 3, 10)
    other = "j" if agent_id == "i" else "i"
    physics = planner.pomdp_model
    checked = 0
    for action, observation in itertools.product(ACTIONS, physics.get_all_observations(agent_id)):
        expected = independent_level_posterior(planner, action, observation)
        if expected is None:
            continue
        posterior = planner.solver_bank.filter.update(
            planner.model(), action, observation, terminal=False
        )
        actual = {}
        for atom, weight in posterior.belief.mass:
            actual[atom.opponent.frame.level] = actual.get(atom.opponent.frame.level, 0.0) + weight
        for k, mass in expected.items():
            assert actual.get(k, 0.0) == pytest.approx(mass, abs=1e-12), (action, observation, k)
        assert other in {a.opponent.frame.agent_id for a, _ in posterior.belief.mass}
        checked += 1
    # Listening supports 2 growls x 3 creak tokens; an opener observes all six uniformly.
    assert checked == 6 + 2 * 6


def test_general_reference_handles_mixed_levels_and_conserves_probability():
    planner = build_agent("j", 2, 200, 2, 10)
    reference = BestResponseReference(planner.pomdp_model, planner.solver_bank, 0.95)
    model = planner.model()
    assert reference.q_values(model, 1) == pytest.approx(
        dict(planner.solver_bank.expected_rewards(model)), abs=1e-12
    )
    for action in ACTIONS:
        _, branches = reference.branches(model, action)
        assert math.fsum(b[1] for b in branches) == pytest.approx(1, abs=1e-12)
        assert {a.opponent.frame.level for _, _, m in branches for a, _ in m.belief.mass} == {0, 1}
    values = reference.q_values(model, 2)
    assert max(values, key=values.get) == "L"
