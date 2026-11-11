"""Post-search chance scoring: exact branch probabilities on the searched tree.

Why
---
Search backs Q(h, a) up with the *empirical* chance law, C(h, a, o) / N(h, a)
(``node.POMCPNode.bellman_backup``). The child values are the right quantities
to average, but the weights are sampling frequencies, and at small counts a
lucky or unlucky frequency moves Q across a decision threshold. The weights
are not unknown, though: the planner holds the agent's posterior b_h at every
history, so the exact law P(o, alive | b_h, a) is available
(``SolverBank.continuation_law``). After search, the tree is therefore
rescored bottom-up with that law:

    Q*(h, a) = E[R | b_h, a] + gamma * sum_o P(o, alive | b_h, a) * V*(h a o)
    V*(h)    = max over evaluated actions of Q*(h, a)

At the last decision (one step remaining) V* is the exact
max_a E[R | b_h, a], as during search. Terminal mass contributes zero
continuation and is never renormalized away.

Two steps left: every branch is exact
-------------------------------------
With two steps remaining, every child of (h, a) is the last decision, whose
exact value max_a' E[R | b_hao, a'] needs only the child's posterior. Such an
action is therefore scored over *all* its positive branches, sampled or not,
and is exact given b_h. Evidence (results/w3_variance_20261005, a0b): j-L1 at
L2 H4 near ties (gaps 0.2-1.0, 30 seeds each), where trees keep ~500
whole-action fallbacks at 1,600 simulations, almost all with two steps left.
Valuing those branches cut the spread of the scored gap 0.25 -> 0.03 at
1,600 (wrong decisions 18 -> 0 of 240; optimal-action Q error 0.15 -> 0.00),
and made every scored root exact at 6,400. Valuing rollout frontiers exactly
instead changed nothing at 1,600+ (a0): the near-tie floor of 12.6 was these
fallbacks' search Q, not rollout noise.

Certified bounds (optional)
---------------------------
With ``bounds=True`` the same pass also returns, for every root action, an
interval [lo, hi] that contains the exact finite-horizon value Q*(root, a) of
the searched model (the exact best response to the modeled opponents at the
searched belief), whatever the budget. Exact pieces enter exactly; whatever
search left unknown is bounded by the declared reward bounds
(``POMDPModel.reward_bounds``), with V_r in [lo_r, hi_r]:

    lo_r = r_min + gamma * min(0, lo_{r-1}),  hi_r = r_max + gamma * max(0, hi_{r-1})

(zero included because an episode may end): a rollout frontier with r steps
left is [lo_r, hi_r]; an unsampled branch of probability p adds p [lo, hi] of
its remaining horizon; an action search never tried at a history is
E[R | b_h, a] + gamma [min(0, lo_{r-1}), max(0, hi_{r-1})]. V*'s interval
maximizes over every legal action, tried or not, so it bounds the true optimum,
not the searched one. Width measures what the tree leaves unknown (contribution
2's error bound); it is zero for an exact root. Bounds need the subtrees of
fallback actions too, so they cost more, and real decisions request them while
modeled solves, scored at every stopping checkpoint, do not.

Belief-dependent bounds and probability-weighted scoring (optional)
-------------------------------------------------------------------
``frontier`` (a ``frontier_bounds.FrontierBounds``) replaces the scalar
[lo_r, hi_r] wherever a belief is at hand: an unsampled branch is bounded from
its parent's belief, a rollout frontier and an untried action from their own
(the I-FIB upper and blind lower bounds; intersected with the scalar ones).
Values are unchanged; only intervals tighten.

``scoring="pw"`` changes values: every branch of an evaluated action enters
with its exact probability; an unsampled branch at three or more steps to go
enters as a point of its bounded interval (``unsampled``: its lower end by
default, the value of a default policy), and each value is clipped into its
certified interval. It needs bounds. The default ``"legacy"`` keeps the
behaviour described next.

What stays approximate
----------------------
- **Missing branches, three or more steps left.** If a positive-probability
  observation of (h, a) was never sampled, there is no child value to weight,
  so the action keeps its search Q as a whole. Mixing exact weights for
  sampled branches with nothing for unsampled ones would bias Q toward zero
  (and renormalizing over the sampled ones biased it upward, RESULTS.md 6).
- **Rollout frontiers.** A child reached once has only its rollout estimate.
- **Unevaluated actions.** V* maximizes over the actions search tried; one
  never tried is not given a fictitious value.

Exactness is a conditional theorem, tracked per subtree: when every legal
action is evaluated, every positive branch has a child and every leaf is the
exact last step, backward induction gives the finite-horizon Bellman value of
the searched belief model under the opponents' modeled policies (to rounding).
That is the same quantity the exact reference computes. Local completeness at
one history does not certify its subtree, so ``exact`` is propagated, never
inferred.

Why a separate pass, not a different backup
-------------------------------------------
Search statistics, UCB scores and random streams are untouched: the pass
reads the finished tree and returns new values. Using exact weights *inside*
search would change which actions UCB explores, with no evidence that it
helps exploration; scoring afterwards changes only the recommendation.
Earlier evidence (legacy L2 Tiger, H1-H8, 3,645 paired decisions, finite
modeled opponents): 0 strict errors against 4 for empirical weights, at +3%
CPU; fully exact roots matched the reference to 5e-15. Rescoring only the root
corrected some errors and caused others, because the child values it weights
still carry their own frequency errors; hence the recursion.
"""

import math
from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class ChanceScores:
    """Rescored root action values and how much of the tree was rescored.

    Attributes:
        values: Q*(root, a) for every action search evaluated at the root.
        exact: True when ``values`` are exact finite-horizon Bellman values of
            the searched model (see the module docstring).
        fallback_actions: (history, action) pairs that kept their search Q
            because a positive-probability branch was never sampled.
        frontiers: Rollout frontiers whose estimate entered a rescored value.
        bounds: {a: (lo, hi)} certified to contain the exact Q*(root, a) of the
            searched model, when requested; else None.
        heuristic_branches: Unsampled branches valued by a bounded stand-in
            (``scoring="pw"`` only).
        heuristic_mass: Sum of those branches' probabilities over the tree.
    """

    values: dict
    exact: bool
    fallback_actions: int
    frontiers: int
    bounds: dict | None = None
    heuristic_branches: int = 0
    heuristic_mass: float = 0.0


@dataclass
class _Tally:
    fallback_actions: int = 0
    frontiers: int = 0
    heuristic_branches: int = 0
    heuristic_mass: list = field(default_factory=list)


@dataclass(frozen=True)
class _Ctx:
    bank: object
    successor: object
    gamma: float
    tally: _Tally
    span: object
    frontier: object = None
    scoring: str = "legacy"
    unsampled: str = "lower"

    def node_bounds(self, model, remaining):
        if self.frontier is None:
            return None
        return self.frontier.node_bounds(model, remaining, self.bank)


class _Span:
    """Bounds [lo_r, hi_r] on any value with r decisions left (module docstring)."""

    def __init__(self, r_min, r_max, gamma, depth):
        self.lo, self.hi = [0.0], [0.0]
        for _ in range(depth):
            self.lo.append(r_min + gamma * min(0.0, self.lo[-1]))
            self.hi.append(r_max + gamma * max(0.0, self.hi[-1]))


SCORING = ("legacy", "pw")
UNSAMPLED = ("lower", "mid", "upper")


def score_tree(
    root, model, depth, bank, successor, gamma, bounds=False, *,
    frontier=None, scoring="legacy", unsampled="lower",
):
    """Rescore a searched tree with exact chance probabilities (see module docstring).

    Args:
        root: The searched ``POMCPNode``; it is read, never modified.
        model: The agent's mental model at the root, as the search used it.
        depth: Remaining decisions at the root (the planner's horizon).
        bank: The ``SolverBank`` supplying E[R | b, a] and P(o, alive | b, a).
        successor: ``(model, action, observation) -> model``, the same belief
            update the search used, so scored and searched beliefs agree.
        gamma: Discount factor.
        bounds: Also return certified intervals for the root actions.
        frontier: ``FrontierBounds`` for belief-dependent intervals, or None
            for the scalar ones. Used only with ``bounds``.
        scoring: ``"legacy"`` or ``"pw"`` (module docstring); ``"pw"`` needs ``bounds``.
        unsampled: Stand-in of an unsampled branch under ``"pw"``.
    """
    if scoring not in SCORING or unsampled not in UNSAMPLED:
        raise ValueError(f"scoring must be in {SCORING} and unsampled in {UNSAMPLED}")
    if scoring == "pw" and not bounds:
        raise ValueError("Probability-weighted scoring needs bounds")
    tally = _Tally()
    span = None
    if bounds:
        r_min, r_max = model.frame.pomdp_model.reward_bounds(model.frame.agent_id)
        span = _Span(r_min, r_max, gamma, depth)
    ctx = _Ctx(bank, successor, gamma, tally, span, frontier if bounds else None, scoring, unsampled)
    values, exact, intervals = _action_values(root, model, depth, ctx)
    if span is not None and depth > 1:
        # Actions certified search pruned (or never tried) still get intervals,
        # so a containment check covers every legal action.
        for action, reward in bank.expected_rewards(model):
            if action not in intervals:
                intervals[action] = _untried_interval(root, model, depth, action, reward, ctx)
    return ChanceScores(
        values,
        exact,
        tally.fallback_actions,
        tally.frontiers,
        intervals,
        tally.heuristic_branches,
        math.fsum(tally.heuristic_mass),
    )


def _intersect(first, second):
    return (max(first[0], second[0]), min(first[1], second[1]))


def exact_two_step(model, action, bank, successor, gamma):
    """Q^2(b, a) = ER(b, a) + gamma * sum over positive branches of p * max_a' ER(b_ao, a').

    The arithmetic of the two-steps-left case of ``_action_values``, term for
    term, so certified search and scoring agree bit for bit.
    """
    rewards = dict(bank.expected_rewards(model))
    if gamma == 0:
        return rewards[action]
    terms = [
        probability * max(dict(bank.expected_rewards(successor(model, action, observation))).values())
        for observation, probability in bank.continuation_law(model, action)
    ]
    return rewards[action] + gamma * math.fsum(terms)


def _untried_interval(node, model, remaining, action, reward, ctx):
    """Interval of an action search did not evaluate (or pruned) at an expanded history."""
    span, gamma = ctx.span, ctx.gamma
    interval = (
        reward + gamma * min(0.0, span.lo[remaining - 1]),
        reward + gamma * max(0.0, span.hi[remaining - 1]),
    )
    nb = ctx.node_bounds(model, remaining)
    if nb is not None:
        interval = _intersect(interval, (nb.q_lo[action], nb.q_hi[action]))
    cert = node.cert
    if cert is not None and cert.expanded and action in cert.q_hi:
        interval = _intersect(interval, (cert.q_lo[action], cert.q_hi[action]))
    return interval


def _frontier_child(node, remaining, ctx):
    """(value, False, interval) of a certified-search frontier, without its belief."""
    ctx.tally.frontiers += 1
    cert = node.cert
    interval = None
    if ctx.span is not None:
        interval = _intersect((ctx.span.lo[remaining], ctx.span.hi[remaining]), (cert.lo, cert.hi))
    value = node.rollout_value
    if ctx.scoring == "pw":
        value = min(max(value, interval[0]), interval[1])
    return value, False, interval


def _value(node, model, remaining, ctx):
    """(V*(h), exact, interval or None) at a child history."""
    bank, gamma, span = ctx.bank, ctx.gamma, ctx.span
    if remaining == 1:
        value = max(dict(bank.expected_rewards(model)).values())
        return value, True, (value, value)
    if node.cert is not None and node.cert.solved:
        # Certified search closed this history's bounds: its value is exact.
        return node.cert.lo, True, (node.cert.lo, node.cert.hi)
    if not node.action_values:
        if node.rollout_value is None:
            raise ValueError("History node has neither action nor rollout estimates")
        ctx.tally.frontiers += 1
        interval = None if span is None else (span.lo[remaining], span.hi[remaining])
        nb = ctx.node_bounds(model, remaining)
        if nb is not None:
            interval = _intersect(interval, (max(nb.q_lo.values()), max(nb.q_hi.values())))
        if interval is not None and node.cert is not None and node.cert.lo is not None:
            interval = _intersect(interval, (node.cert.lo, node.cert.hi))
        value = node.rollout_value
        if ctx.scoring == "pw":
            value = min(max(value, interval[0]), interval[1])
        return value, False, interval
    values, exact, intervals = _action_values(node, model, remaining, ctx)
    interval = None
    if span is not None:
        lows, highs = [lo for lo, _ in intervals.values()], [hi for _, hi in intervals.values()]
        for action, reward in bank.expected_rewards(model):
            if action not in intervals:
                untried = _untried_interval(node, model, remaining, action, reward, ctx)
                lows.append(untried[0])
                highs.append(untried[1])
        interval = (max(lows), max(highs))
        if node.cert is not None and node.cert.lo is not None:
            interval = _intersect(interval, (node.cert.lo, node.cert.hi))
    return max(values.values()), exact, interval


def _action_values(node, model, remaining, ctx):
    """({a: Q*(h, a)} over evaluated actions, exact, {a: interval} or None) at an expanded history."""
    bank, successor, gamma, span, tally = ctx.bank, ctx.successor, ctx.gamma, ctx.span, ctx.tally
    rewards = dict(bank.expected_rewards(model))
    if remaining == 1:
        return rewards, True, None if span is None else {a: (r, r) for a, r in rewards.items()}
    if not set(node.action_values) <= set(rewards):
        raise ValueError("Search evaluated an action the model does not declare")
    values, intervals = {}, None if span is None else {}
    exact = set(node.action_values) == set(rewards)
    nb = ctx.node_bounds(model, remaining) if remaining > 2 else None
    for action, search_q in node.action_values.items():
        if gamma == 0:
            values[action] = rewards[action]
            if span is not None:
                intervals[action] = (rewards[action], rewards[action])
            continue
        law = bank.continuation_law(model, action)
        children = node.children.get(action, {})
        if not set(children) <= {o for o, _ in law}:
            raise ValueError("Search reached an observation with zero probability under b_h")
        missing = remaining > 2 and any(o not in children for o, _ in law)
        fallback = missing and ctx.scoring == "legacy"
        if fallback:
            tally.fallback_actions += 1
            values[action] = search_q
            exact = False
            if span is None:
                continue
        terms, lows, highs = [], [], []
        for observation, probability in law:
            if missing and observation not in children:
                if nb is None:
                    lo = probability * span.lo[remaining - 1]
                    hi = probability * span.hi[remaining - 1]
                else:
                    lo = max(probability * span.lo[remaining - 1], nb.branch_lo[action][observation])
                    hi = min(probability * span.hi[remaining - 1], nb.branch_hi[action][observation])
                lows.append(lo)
                highs.append(hi)
                if not fallback:
                    terms.append({"lower": lo, "upper": hi, "mid": 0.5 * (lo + hi)}[ctx.unsampled])
                    tally.heuristic_branches += 1
                    tally.heuristic_mass.append(probability)
                    exact = False
                continue
            child = children.get(observation)
            if remaining > 2 and child is not None and child.cert is not None and not child.cert.expanded:
                # A certified-search frontier: its interval came from this
                # history's belief, so its own belief is never computed.
                child_value, child_exact, child_interval = _frontier_child(child, remaining - 1, ctx)
                terms.append(probability * child_value)
                exact = False
                if span is not None:
                    lows.append(probability * child_interval[0])
                    highs.append(probability * child_interval[1])
                continue
            following = successor(model, action, observation)
            if remaining == 2:
                # The child is the last decision: exact whether sampled or not.
                child_value = max(dict(bank.expected_rewards(following)).values())
                child_exact, child_interval = True, (child_value, child_value)
            else:
                # A fallback's subtrees feed only its bounds, never the tally.
                child_value, child_exact, child_interval = _value(
                    children[observation],
                    following,
                    remaining - 1,
                    replace(ctx, tally=_Tally()) if fallback else ctx,
                )
            terms.append(probability * child_value)
            exact = exact and child_exact
            if span is not None:
                lows.append(probability * child_interval[0])
                highs.append(probability * child_interval[1])
        if span is not None:
            intervals[action] = (
                rewards[action] + gamma * math.fsum(lows),
                rewards[action] + gamma * math.fsum(highs),
            )
        if not fallback:
            value = rewards[action] + gamma * math.fsum(terms)
            if ctx.scoring == "pw":
                lo, hi = intervals[action]
                value = min(max(value, lo), hi)
            values[action] = value
    return values, exact, intervals
