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
from dataclasses import dataclass


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
    """

    values: dict
    exact: bool
    fallback_actions: int
    frontiers: int
    bounds: dict | None = None


@dataclass
class _Tally:
    fallback_actions: int = 0
    frontiers: int = 0


class _Span:
    """Bounds [lo_r, hi_r] on any value with r decisions left (module docstring)."""

    def __init__(self, r_min, r_max, gamma, depth):
        self.lo, self.hi = [0.0], [0.0]
        for _ in range(depth):
            self.lo.append(r_min + gamma * min(0.0, self.lo[-1]))
            self.hi.append(r_max + gamma * max(0.0, self.hi[-1]))


def score_tree(root, model, depth, bank, successor, gamma, bounds=False):
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
    """
    tally = _Tally()
    span = None
    if bounds:
        r_min, r_max = model.frame.pomdp_model.reward_bounds(model.frame.agent_id)
        span = _Span(r_min, r_max, gamma, depth)
    values, exact, intervals = _action_values(
        root, model, depth, bank, successor, gamma, tally, span
    )
    return ChanceScores(values, exact, tally.fallback_actions, tally.frontiers, intervals)


def _value(node, model, remaining, bank, successor, gamma, tally, span):
    """(V*(h), exact, interval or None) at a child history."""
    if remaining == 1:
        value = max(dict(bank.expected_rewards(model)).values())
        return value, True, (value, value)
    if not node.action_values:
        if node.rollout_value is None:
            raise ValueError("History node has neither action nor rollout estimates")
        tally.frontiers += 1
        interval = None if span is None else (span.lo[remaining], span.hi[remaining])
        return node.rollout_value, False, interval
    values, exact, intervals = _action_values(
        node, model, remaining, bank, successor, gamma, tally, span
    )
    interval = None
    if span is not None:
        lows, highs = [lo for lo, _ in intervals.values()], [hi for _, hi in intervals.values()]
        for action, reward in bank.expected_rewards(model):
            if action not in intervals:
                lows.append(reward + gamma * min(0.0, span.lo[remaining - 1]))
                highs.append(reward + gamma * max(0.0, span.hi[remaining - 1]))
        interval = (max(lows), max(highs))
    return max(values.values()), exact, interval


def _action_values(node, model, remaining, bank, successor, gamma, tally, span):
    """({a: Q*(h, a)} over evaluated actions, exact, {a: interval} or None) at an expanded history."""
    rewards = dict(bank.expected_rewards(model))
    if remaining == 1:
        return rewards, True, None if span is None else {a: (r, r) for a, r in rewards.items()}
    if not set(node.action_values) <= set(rewards):
        raise ValueError("Search evaluated an action the model does not declare")
    values, intervals = {}, None if span is None else {}
    exact = set(node.action_values) == set(rewards)
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
        fallback = remaining > 2 and any(o not in children for o, _ in law)
        if fallback:
            tally.fallback_actions += 1
            values[action] = search_q
            exact = False
            if span is None:
                continue
        terms, lows, highs = [], [], []
        for observation, probability in law:
            if fallback and observation not in children:
                lows.append(probability * span.lo[remaining - 1])
                highs.append(probability * span.hi[remaining - 1])
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
                    bank,
                    successor,
                    gamma,
                    _Tally() if fallback else tally,
                    span,
                )
            terms.append(probability * child_value)
            exact = exact and child_exact
            if span is not None:
                lows.append(probability * child_interval[0])
                highs.append(probability * child_interval[1])
        if not fallback:
            values[action] = rewards[action] + gamma * math.fsum(terms)
        if span is not None:
            intervals[action] = (
                rewards[action] + gamma * math.fsum(lows),
                rewards[action] + gamma * math.fsum(highs),
            )
    return values, exact, intervals
