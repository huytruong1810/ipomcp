# Theory: what is computed, what is estimated, what "exact" means

## Interactive states and beliefs (Gmytrasiewicz & Doshi, 2005)

An agent i of level l > 0 holds a belief b_i over interactive states
(s, m_j). Here s is the physical state, and m_j = (frame_j, b_j) is a model of the opponent j at a level below l.

Level 0 is uniformly random and has no belief. Beliefs are joint: correlations between the physical state and the opponent's private belief are kept, never factorised.

## Belief update

For own action a_i and observation o_i:

    b_i'(s', m_j') ∝ Σ_{s, m_j} b_i(s, m_j) Σ_{a_j} π_j(a_j | m_j) T(s' | s, a_i, a_j)
                       O_i(o_i | s', a_i, a_j) Σ_{o_j} O_j(o_j | s', a_i, a_j) 1[m_j' = U_j(m_j, a_j, o_j)]

Here U_j is j's own update. It receives only j's action and observation and marginalises everything j cannot see with j's own models.

- **Exact update** (`belief.finite_filter`): enumerates every term. Supports grow geometrically with time and nesting. For example, L2 Tiger reaches 18,152 atoms after 8 steps.
- **Compressed filter** (`belief.compressed_filter`; the planner's default bounded belief): the exact update of a belief with at most N atoms, followed, only when the posterior has more than N distinct atoms, by N systematic draws. Nested opponents are updated by the same filter, so supports are at most N at every level.
  - Small beliefs (fewer than N atoms) stay exact. The only approximation is compression, with per-atom mass error below 1/N.
  - The evidence P(o_i | b, a_i) is exact under the compressed prior.
  - Cost per update: at most N × |A_j| × |S'| × |O_j| branches per level.
- **Interactive particle filter** (`belief.particle_filter`; Doshi & Gmytrasiewicz, 2009; the IPF + RTS baseline's belief):
  1. draws N particles;
  2. samples a_j, s' and o_j for each;
  3. advances m_j by the same particle update, recursively;
  4. weights each particle by O_i.

  Supports are at most N at every level, and the mean weight is an unbiased estimate of P(o_i | b, a_i).

  It holds every belief, however small, as N equal-weight samples, so nested probabilities move in steps of 1/N. Because modeled opponents act greedily, a step across a decision threshold flips their predicted action. It also redraws every term of every update.

  Measured errors of both filters against the exact update are in [RESULTS.md](RESULTS.md).

## The planner's estimator

The planner estimates, at the agent's belief b and fixed lookahead H,

    Q_H(b, a) = E[R | b, a] + γ Σ_{o} P(o | b, a) max_{a'} Q_{H-1}(b_{a o}, a'),

where opponents act by their modeled policies.

**Root sampling.** Each simulation draws (s, m_j) from b and descends the tree of the agent's private histories h.

**Backups (empirical chance-weighted Bellman).**

    Q(h, a) = R̄(h, a) + γ Σ_o C(h, a, o) / N(h, a) · V(h a o),   V(h) = max_{evaluated a} Q(h, a)

Terminal outcomes stay in N(h, a) with zero continuation. Child values are re-read on every backup.

**Exact pieces.** The planner carries the agent's own posterior b_h down the tree, using the same belief update as for real observations. R̄(h, a) is then the exact E[R | b_h, a], and the last decision uses Q_1(b_h, a) = E[R | b_h, a].

None of these exact pieces use the sampled hidden state or the opponent's actual action. Action selection never inspects the sampled state, so replacing sampled rewards by their expectations leaves the backup's target unchanged and removes their variance.

**Post-search chance scoring** (`planning.chance_scoring`). After the simulations, the tree is rescored bottom-up with the exact chance law instead of the sampled frequencies C(h, a, o) / N(h, a):

    Q*(h, a) = E[R | b_h, a] + γ Σ_o P(o, alive | b_h, a) V*(h a o),   V*(h) = max_{evaluated a} Q*(h, a)

with V* at the last decision equal to max_a E[R | b_h, a]. Terminal mass contributes zero and is not renormalized away.

With two steps left, every child of (h, a) is the last decision, whose exact value needs only the child's posterior. Such an action is therefore scored over all its positive branches, sampled or not, and is exact given b_h. These whole-action fallbacks, not rollout noise, set the near-tie floor of modeled solves (RESULTS.md 12.14).

**Certified bounds** (real decisions). The same pass returns, for each root action, an interval that contains the exact Q of the searched model. Exact pieces enter exactly. A rollout frontier with r steps left is bounded by [lo_r, hi_r], where lo_r = r_min + γ min(0, lo_{r−1}) and hi_r = r_max + γ max(0, hi_{r−1}), from the declared reward bounds. An unsampled branch of probability p adds p times its horizon's interval. An untried action is E[R | b_h, a] plus γ times its horizon's interval, with zero included. The width is the search error that the tree leaves unresolved. It does not cover opponent-model or belief error (RESULTS.md 12.17).

What stays approximate:
- with three or more steps left, an action with a positive-probability observation that search never sampled keeps its search Q as a whole (weighting only the sampled branches would bias it toward zero);
- a child reached once enters with its rollout estimate;
- actions search never tried are not maximized over.

If every legal action is evaluated, every positive branch has a child, and every leaf is the exact last step, backward induction makes the scored root equal to the exact best response to the opponents' modeled policies, relative to the searched belief (tests check this to 1e-9). The planner records whether this held (`scores.exact`); local completeness at one history does not certify its subtree, so exactness is propagated from the leaves.

The scored values decide, both for the planner's real decisions and when it is queried as a modeled opponent, so an opponent is predicted to act by the same rule. Search itself (UCB, backups, random streams) is unchanged; the search values are kept as `search_values` for paired comparison.

**Exploration.** UCB with width (r_max − r_min) Σ_{t<H} γ^t from the declared reward bounds. Ties are broken uniformly.

**What is not claimed.** There is no finite-budget optimality guarantee. Tree statistics are non-stationary, and the maximum over noisy children is optimistic at small counts. Convergence is established empirically against exact references.

## Three meanings of "exact"

Every error report states which one it uses.

| Term | Opponent policies | Module | Measures |
|---|---|---|---|
| Exact best response | as declared in the bank (e.g. finite I-POMCP) | `reference.best_response` | the planner's own search error |
| Exact I-POMDP | every level solved exactly, recursively | `reference.exact_agent` + best response | search error plus opponent-model error |
| Exact POMDP (L1) | fixed L0 | `reference.pomdp_exact_vi` | independent cross-check by alpha vectors |

## Error measures

- **First-action loss:** V*(b) − Σ_a π(a|b) Q*(b, a). This is the value lost by the planner's root decision when every later decision is optimal. It handles ties exactly and is not an episode regret.
- **Maximum Q error:** max_a |Q̂(b, a) − Q*(b, a)|.
- **Optimal-action Q error:** the same maximum, taken only over the exactly optimal actions. It is the error of the value the agent acts on, and the measure for error bounds. The all-action maximum is dominated by clearly wrong actions, which search rightly samples little (RESULTS.md Section 12.2).
- **Belief error:**
  - total variation between level posteriors;
  - total variation between nested path masses at a given depth (what a sunburst ring shows);
  - the absolute error in the physical marginal.

## Cost model

Searching futures costs root simulations (`simulations`). Simulating minds costs one search of `modeled_simulations` for every distinct nested belief that a search reaches.

Measured on an L3 Tiger decision, nested solves take 80–89% of CPU time. With exact beliefs a decision took 888 s and 1,779 nested solves; with the IPF (64 particles) it took 7.7 s and 228.

Modeled policies and belief updates are pure functions, so their caches are kept across real steps. How to divide a fixed budget between the two costs is measured by `experiments.panel` against the exact I-POMDP.
