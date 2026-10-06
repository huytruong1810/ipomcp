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
    """

    values: dict
    exact: bool
    fallback_actions: int
    frontiers: int


@dataclass
class _Tally:
    fallback_actions: int = 0
    frontiers: int = 0


def score_tree(root, model, depth, bank, successor, gamma):
    """Rescore a searched tree with exact chance probabilities (see module docstring).

    Args:
        root: The searched ``POMCPNode``; it is read, never modified.
        model: The agent's mental model at the root, as the search used it.
        depth: Remaining decisions at the root (the planner's horizon).
        bank: The ``SolverBank`` supplying E[R | b, a] and P(o, alive | b, a).
        successor: ``(model, action, observation) -> model``, the same belief
            update the search used, so scored and searched beliefs agree.
        gamma: Discount factor.
    """
    tally = _Tally()
    values, exact = _action_values(root, model, depth, bank, successor, gamma, tally)
    return ChanceScores(values, exact, tally.fallback_actions, tally.frontiers)


def _value(node, model, remaining, bank, successor, gamma, tally):
    """(V*(h), exact) at a child history."""
    if remaining == 1:
        return max(dict(bank.expected_rewards(model)).values()), True
    if not node.action_values:
        if node.rollout_value is None:
            raise ValueError("History node has neither action nor rollout estimates")
        tally.frontiers += 1
        return node.rollout_value, False
    values, exact = _action_values(node, model, remaining, bank, successor, gamma, tally)
    return max(values.values()), exact


def _action_values(node, model, remaining, bank, successor, gamma, tally):
    """({a: Q*(h, a)} over evaluated actions, exact) at an expanded history."""
    rewards = dict(bank.expected_rewards(model))
    if remaining == 1:
        return rewards, True
    if not set(node.action_values) <= set(rewards):
        raise ValueError("Search evaluated an action the model does not declare")
    values = {}
    exact = set(node.action_values) == set(rewards)
    for action, search_q in node.action_values.items():
        if gamma == 0:
            values[action] = rewards[action]
            continue
        law = bank.continuation_law(model, action)
        children = node.children.get(action, {})
        if not set(children) <= {o for o, _ in law}:
            raise ValueError("Search reached an observation with zero probability under b_h")
        if remaining > 2 and any(o not in children for o, _ in law):
            tally.fallback_actions += 1
            values[action] = search_q
            exact = False
            continue
        terms = []
        for observation, probability in law:
            following = successor(model, action, observation)
            if remaining == 2:
                # The child is the last decision: exact whether sampled or not.
                child_value, child_exact = (
                    max(dict(bank.expected_rewards(following)).values()),
                    True,
                )
            else:
                child_value, child_exact = _value(
                    children[observation], following, remaining - 1, bank, successor, gamma, tally
                )
            terms.append(probability * child_value)
            exact = exact and child_exact
        values[action] = rewards[action] + gamma * math.fsum(terms)
    return values, exact
