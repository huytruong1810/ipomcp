"""Brute-force I-POMDP arithmetic for tests, written without the package's filters, references or planner.

The package's exact references (``ipomdp.reference``) are what every planner
result is measured against, so they need a check of their own that shares no
code with them. Everything here works on plain dictionaries and reads a domain
only through its kernels (``transition_distribution``,
``observation_distribution``, ``get_reward``, ``is_terminal``), which
``tests/test_domain_exhaustive.py`` checks separately. The Tiger closed form
does not even do that: it is written from the numbers of Table 1 of
Gmytrasiewicz and Doshi (2005).

Conventions, the same as the package's:
- level 0 acts uniformly at random over its full action set;
- an intentional agent plays uniformly over the maximizers of its own values,
  values within a relative ``TIE`` of the maximum counting as tied;
- beliefs are conditioned on the episode continuing; a step that ends the
  episode pays its reward and nothing after it;
- a modeled level-1 agent replans with the full lookahead at every step
  (``lookahead``), or, under the shared-horizon convention, with as many steps
  as the modeling agent has left.

A level-1 belief is ``{state: probability}``. A level-2 belief is a tuple of
rows ``(state, opponent's level-1 belief as a frozenset of items, mass)``; a
row whose opponent is the random level 0 carries ``None`` for that belief.
"""

import math

from ipomdp.belief.finite_belief import FiniteBelief, InteractiveState, MentalModel
from ipomdp.belief.frame import AgentFrame

TIE = 1e-9

# --------------------------------------------------------------------------- #
# Package models built by hand (no hierarchy builder, no solver bank)
# --------------------------------------------------------------------------- #


def nested_model(physics, agent, other, level, prior=None):
    """``agent`` at ``level`` whose opponents are all at ``level - 1``, every belief the domain prior."""
    frame = AgentFrame(agent, level, physics)
    if level == 0:
        return MentalModel(frame)
    opponent = nested_model(physics, other, agent, level - 1)
    rows = physics.initial_distribution() if prior is None else prior
    return MentalModel(frame, FiniteBelief(tuple((InteractiveState(s, opponent), p) for s, p in rows)))


def physical_marginal(belief):
    """{state: probability} of a package ``FiniteBelief``."""
    marginal = {}
    for atom, mass in belief.mass:
        marginal[atom.state] = marginal.get(atom.state, 0.0) + mass
    return marginal


def signature(marginal):
    """A belief over physical states as a hashable key, rounded so two summation orders agree.

    A frozenset of (state, probability) pairs: lossless in the state (a state's ``repr`` need not be) and
    free of any ordering of the states.
    """
    return frozenset((state, round(p, 9)) for state, p in marginal.items())


# --------------------------------------------------------------------------- #
# Tiger, level 1 against the uniformly random level 0: a closed form from Table 1
# --------------------------------------------------------------------------- #

GROWL, CREAK = 0.85, 0.9
TRUE_CREAK = {"L": "S", "OL": "CL", "OR": "CR"}
TIGER_ACTIONS = ("L", "OL", "OR")
TIGER_OBSERVATIONS = [(g, c) for g in ("GL", "GR") for c in ("CL", "CR", "S")]


def tiger_closed_form(p_left, action, observation):
    """(P(TL | action, observation), evidence) for prior ``p_left``, from Table 1.

    The other agent listens with probability 1/3 (the tiger stays) and opens a door with 2/3 (the tiger is
    placed uniformly). A listener hears the correct growl with 0.85 and the creak that reports the other
    agent's action with 0.9 (each other creak 0.05); an opener hears each of the six pairs with 1/6.
    """
    if action != "L":
        return 0.5, 1 / 6
    growl, creak = observation
    weights = {}
    for state, prior in (("TL", p_left), ("TR", 1 - p_left)):
        hears = GROWL if growl == ("GL" if state == "TL" else "GR") else 1 - GROWL
        stays = (1 / 3) * prior * (CREAK if creak == TRUE_CREAK["L"] else (1 - CREAK) / 2)
        resets = sum(
            (1 / 3) * 0.5 * (CREAK if creak == TRUE_CREAK[door] else (1 - CREAK) / 2)
            for door in ("OL", "OR")
        )
        weights[state] = hears * (stays + resets)
    evidence = weights["TL"] + weights["TR"]
    return weights["TL"] / evidence, evidence


def tiger_closed_form_q(p_left, horizon, gamma):
    """{action: Q_horizon(p_left, action)} by recursion over the closed-form belief tree.

    Rewards from Table 1: listening -1; opening the tiger's door -100, the other door 10.
    """
    if horizon == 0:
        return dict.fromkeys(TIGER_ACTIONS, 0.0)
    immediate = {"L": -1.0, "OL": -100.0 * p_left + 10.0 * (1 - p_left), "OR": 10.0 * p_left - 100.0 * (1 - p_left)}
    if horizon == 1:
        return immediate
    values = {}
    for action in TIGER_ACTIONS:
        continuation = 0.0
        for observation in TIGER_OBSERVATIONS:
            posterior, evidence = tiger_closed_form(p_left, action, observation)
            continuation += evidence * max(tiger_closed_form_q(posterior, horizon - 1, gamma).values())
        values[action] = immediate[action] + gamma * continuation
    return values


# --------------------------------------------------------------------------- #
# Any domain: level 1 against the uniformly random level 0
# --------------------------------------------------------------------------- #


def level_one_update(physics, agent, other, prior, action, observation):
    """The level-1 posterior {state: p} given the agent's own action and observation, or None at zero evidence."""
    other_actions = physics.get_all_actions(other)
    weights = {}
    for state, mass in prior.items():
        for other_action in other_actions:
            joint = {agent: action, other: other_action}
            for successor, p_next in physics.transition_distribution(state, joint):
                if physics.is_terminal(successor):
                    continue
                likelihood = dict(physics.observation_distribution(successor, joint, agent)).get(observation, 0.0)
                if likelihood:
                    weights[successor] = weights.get(successor, 0.0) + mass * p_next * likelihood / len(other_actions)
    total = math.fsum(weights.values())
    return {state: weight / total for state, weight in weights.items()} if total else None


def level_one_q(physics, agent, other, prior, horizon, gamma, cache):
    """{action: Q_horizon} of a level-1 agent at ``prior`` ({state: p}); ``cache`` is a dict the caller keeps."""
    key = (agent, frozenset(prior.items()), horizon)
    if key in cache:
        return cache[key]
    actions, other_actions = physics.get_all_actions(agent), physics.get_all_actions(other)
    values = {}
    for action in actions:
        reward, by_observation = 0.0, {}
        for state, mass in prior.items():
            for other_action in other_actions:
                joint = {agent: action, other: other_action}
                for successor, p_next in physics.transition_distribution(state, joint):
                    event = mass * p_next / len(other_actions)
                    reward += event * physics.get_reward(state, joint, successor, agent)
                    if horizon == 1 or physics.is_terminal(successor):
                        continue
                    for observation, likelihood in physics.observation_distribution(successor, joint, agent):
                        table = by_observation.setdefault(observation, {})
                        table[successor] = table.get(successor, 0.0) + event * likelihood
        continuation = 0.0
        for table in by_observation.values():
            evidence = math.fsum(table.values())
            posterior = {state: weight / evidence for state, weight in table.items()}
            continuation += evidence * max(level_one_q(physics, agent, other, posterior, horizon - 1, gamma, cache).values())
        values[action] = reward + gamma * continuation
    cache[key] = values
    return values


def greedy(values):
    """Uniform over the maximizers, values within a relative ``TIE`` of the maximum counting as tied."""
    best = max(values.values())
    tied = [action for action, value in values.items() if value >= best - TIE * max(1.0, abs(best))]
    return {action: 1.0 / len(tied) for action in tied}


# --------------------------------------------------------------------------- #
# Any domain: level 2 against an exactly optimizing level 1
# --------------------------------------------------------------------------- #


def prior_rows(physics, level_one=1.0):
    """The level-2 prior: every state, the opponent at level 1 (holding the domain prior) with probability
    ``level_one`` and at level 0 otherwise, independently of the state."""
    nested = frozenset(dict(physics.initial_distribution()).items())
    rows = [(state, nested, p * level_one) for state, p in physics.initial_distribution()]
    if level_one < 1.0:
        rows += [(state, None, p * (1.0 - level_one)) for state, p in physics.initial_distribution()]
    return tuple(rows)


def level_two_q(physics, agent, other, rows, horizon, gamma, lookahead, shared, cache):
    """{action: Q_horizon} of a level-2 agent at ``rows``.

    A level-1 opponent at belief b plays ``greedy`` over its own level-1 values at ``lookahead`` steps, or,
    with ``shared``, at as many steps as this agent has left (``horizon``). It then updates b on its own
    action and private observation by ``level_one_update``. A level-0 opponent (belief ``None``) acts
    uniformly and has nothing to update.
    """
    key = (frozenset(rows), horizon)
    if key in cache.setdefault("level-two", {}):
        return cache["level-two"][key]
    opponent_steps = horizon if shared else lookahead
    uniform = {a: 1.0 / len(physics.get_all_actions(other)) for a in physics.get_all_actions(other)}
    values = {}
    for action in physics.get_all_actions(agent):
        reward, by_observation = 0.0, {}
        for state, nested, mass in rows:
            if nested is None:
                law = uniform
            else:
                law = greedy(level_one_q(physics, other, agent, dict(nested), opponent_steps, gamma, cache))
            for other_action, p_action in law.items():
                joint = {agent: action, other: other_action}
                for successor, p_next in physics.transition_distribution(state, joint):
                    event = mass * p_action * p_next
                    reward += event * physics.get_reward(state, joint, successor, agent)
                    if horizon == 1 or physics.is_terminal(successor):
                        continue
                    privates = ((None, 1.0),) if nested is None else physics.observation_distribution(successor, joint, other)
                    for private, p_private in privates:
                        advanced = None
                        if nested is not None:
                            advanced = level_one_update(physics, other, agent, dict(nested), other_action, private)
                            assert advanced is not None, "the opponent's own belief must explain its observation"
                            advanced = frozenset(advanced.items())
                        for observation, likelihood in physics.observation_distribution(successor, joint, agent):
                            table = by_observation.setdefault(observation, {})
                            row = (successor, advanced)
                            table[row] = table.get(row, 0.0) + event * p_private * likelihood
        continuation = 0.0
        for table in by_observation.values():
            evidence = math.fsum(table.values())
            successors = tuple((state, nested, weight / evidence) for (state, nested), weight in table.items())
            continuation += evidence * max(
                level_two_q(physics, agent, other, successors, horizon - 1, gamma, lookahead, shared, cache).values()
            )
        values[action] = reward + gamma * continuation
    cache["level-two"][key] = values
    return values
