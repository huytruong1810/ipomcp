# Theory: what is computed, what is bounded, what "exact" means

Each statement below names the module that implements it and the tests that check it on small problems against an independent computation. Proofs are sketched where they are short; the full proofs are in the paper's supplement.

## Interactive states and beliefs

An agent i of level l > 0 holds a belief b_i over interactive states (s, m_j) (Gmytrasiewicz & Doshi, 2005). Here s is the physical state, and m_j = (frame_j, b_j) is a model of the other agent j at a level below l.

Level 0 is uniformly random over its legal actions and has no belief. Beliefs are joint: correlations between the physical state and the other agent's private belief are kept, never factorised.

## Belief update

For own action a_i and observation o_i:

    b_i'(s', m_j') ∝ Σ_{s, m_j} b_i(s, m_j) Σ_{a_j} π_j(a_j | m_j) T(s' | s, a_i, a_j)
                       O_i(o_i | s', a_i, a_j) Σ_{o_j} O_j(o_j | s', a_i, a_j) 1[m_j' = U_j(m_j, a_j, o_j)]

U_j is j's own update. It receives only j's action and observation and marginalises everything j cannot see, with j's own models.

- **Exact update** (`belief.finite_filter`): enumerates every term. Supports grow geometrically with time and nesting.
- **Compressed filter** (`belief.compressed_filter`): the exact update of a belief with at most N atoms; when the posterior has more than N distinct atoms, it is compressed. Nested models are updated by the same filter, so supports are at most N at every level. The evidence P(o_i | b, a_i) is exact under the compressed prior. Two compressions:
  - **stratified** (default): N stratified draws in canonical order, identical draws merged. Per-atom mass error is below 2/N. The replaced mass is not recorded, so no value bound follows (see "Compression slack").
  - **head-tail**: the N − n_t heaviest atoms keep their exact masses, and n_t stratified draws replace the rest (default n_t = N/4). The replaced mass ε is recorded as the loss of the compression.
- **Interactive particle filter** (`belief.particle_filter`; Doshi & Gmytrasiewicz, 2009; the belief of the IPF + RTS baseline): draws N particles, samples a_j, s' and o_j for each, advances m_j by the same particle update recursively, and weights each particle by O_i. Small beliefs are held as N equal-weight samples, so nested probabilities move in steps of 1/N, and a modeled agent near a decision threshold can flip its predicted action.

## Search

I-POMCP estimates, at i's belief b and a fixed lookahead H,

    Q_H(b, a) = E[R | b, a] + γ Σ_o P(o, alive | b, a) max_{a'} Q_{H-1}(b_{ao}, a'),

where the other agent acts by its modeled policy. Each simulation draws (s, m_j) from b and descends the tree of i's private histories h. Every node carries i's posterior b_h, computed with the same update as for real observations, so the expected reward E[R | b_h, a] and the observation law P(o, alive | b_h, a) are exact quantities of the searched model. Action selection never reads the sampled state.

**The searched model.** Every modeled solve is seeded from a digest of its model, so each model j that the planner queries receives a fixed action law, and every belief is a fixed function of its parent, the action and the observation. i's problem with these laws and this update is a finite-horizon POMDP over the reachable interactive states, called the searched model. When every modeled level is solved exactly and no belief is compressed, it is the exact I-POMDP.

**Chance scoring** (`planning.chance_scoring`). After search, the tree is rescored bottom-up with the exact observation law instead of the sampled frequencies:

    Q*(h, a) = E[R | b_h, a] + γ Σ_o P(o, alive | b_h, a) V*(h a o),   V*(h) = max_{evaluated a} Q*(h, a),

and the last decision uses max_a E[R | b_h, a]. Terminal mass contributes zero and is not renormalised away. Under `scoring="pw"`, a branch the tree never sampled enters with a point of its certified interval (`unsampled_value`, default its lower end), and each scored value is clipped into its interval. Under `scoring="legacy"`, an action with an unsampled branch at three or more steps to go keeps its search value as a whole.

## Certified search

Certified search (`i_pomcp._simulate_certified`, enabled by the certified options of `PlannerConfig`; the profile `experiments.profile` "certified" sets all of them) keeps an interval [q⁻(h, a), q⁺(h, a)] for every action at every expanded history.

- **Frontier bounds** (`planning.frontier_bounds`): `"scalar"` uses only the horizon's return range [ℓ_r, u_r], with ℓ_r = R_min + γ min(0, ℓ_{r−1}) and u_r = R_max + γ max(0, u_{r−1}); `"ifib"` uses the belief-dependent interactive fast-informed bound and blind-policy lower bound of "Frontier bounds" below; `"itib"` adds the tighter informed bound on one-step beliefs.
- **Exact two-step tail** (`exact_tail`): a history with two steps to go is solved exactly when it is expanded; every child is the last decision, whose value needs only its posterior.
- **Pruning** (`prune`): bounds are backed up along every descent and only tighten; an action ruled out by its bounds is never selected again; the search stops when the root decision is settled.
- **Leaves** (`leaf_eval`, `lazy_expansion`): a new frontier is valued by a point of its interval (`"bound"`) or by a rollout clipped into it, and with lazy expansion its belief is computed only when it is visited again (its interval comes from the parent's belief meanwhile).
- **Action selection** (`selection`): UCB with the horizon's return width (`"ucb"`), PUCT with a softmax prior of the upper bounds (`"puct"`), or the largest upper bound first, ties by the PUCT score (`"iemax"`, the action rule of bound-directed solvers such as HSVI).
- **Observation rule** (`outcome`): `"sample"` draws the next observation with its probability; `"gap"` takes the open branch with the largest probability-weighted gap between its bounds, and `"gap_sample"` draws an open branch in proportion to that gap. Under both gap rules, the first history on the path that is not yet expanded is expanded and the simulation ends.
- **Root rule** (`root_rule`, gap rules): `"upper"` takes the largest upper bound among unpruned actions; `"open"` takes the largest upper bound among the plausible actions whose interval is still open; `"challenger"` takes, of the leading action and its strongest challenger, the one with the wider interval.

**Plausible and certain sets** (`planning.policy.plausible_sets`). With M⁻ the largest lower bound and M⁺ the largest upper bound at a history, an action is plausible if q⁺ ≥ M⁻ − τ⁺ and certain if q⁻ ≥ M⁺ − τ⁻, with τ⁺ = 10⁻⁹ max(1, |M⁻|, |M⁺|) and τ⁻ = 10⁻⁹ max(1, min(|M⁻|, |M⁺|)), or 10⁻⁹ when [M⁻, M⁺] contains 0. Then certain ⊆ exactly optimal (within the relative tie tolerance 10⁻⁹) ⊆ plausible.

**Tolerances.** Every comparison that prunes or certifies under the sampled rule uses the outward slack 10⁻⁹ max(1, |x|). The gap rules use one absolute tolerance per search, ε = 10⁻⁹ (u_H − ℓ_H): an interval is closed when its width is at most ε. A relative slack does not compose across levels of the tree; an absolute one does.

**Stop test** (`i_pomcp._resolved`). The root decision is settled when every unpruned action's interval is closed and the root is solved, or when the certain set equals the plausible set. Under a gap rule, pruning removes exactly the actions that are not plausible.

## Modeled levels and bi-level soundness

A model of j at level ≥ 1 is solved by a nested I-POMCP search. With `modeled_resolution` (`planning.stopping.ModeledResolution`), one tree is grown by doubling from a pilot size to the cap `modeled_simulations`; after each doubling, its root intervals, widened by the solve's own deviation bound, give the plausible and certain sets, and the solve stops when they agree (resolved) or at the cap. With `modeled_prediction="plausible"`, the solve predicts j uniform over its plausible set, and records δ̂ = 1 − max(1, |certain|)/|plausible|, a bound on the total variation between this prediction and the exact law. A model whose own belief was formed with laws of unresolved lower levels is not grounded and gets δ̂ = 1 (`SolverBank.deviation`; tracked per level).

## Guarantees

All hold in exact arithmetic. The implementation sums with compensated summation, and the tests check containment to 10⁻⁹.

### Exact scoring

If at every history with at least two steps to go every action has been evaluated and every observation of positive probability has a child, the scored root values equal the exact best response to the modeled laws (backward induction). The planner records whether this held (`scores.exact`), propagated from the leaves. Tests: `test_chance_scoring.py`, `test_planner_exactness.py`.

### Soundness

For every budget and seed, the certified root intervals contain the values Q_s^H(b_i, a) of the searched model, provided i's belief is not compressed below the root. Proof: by induction on the steps to go, each kind of node enters with an interval that contains its value (an exact leaf; an evaluated action as a sum of branch intervals; an untried action as E[R] plus γ times the continuation range, intersected with the frontier bounds; an expanded history as the maximum over its actions). Consequently an action whose lower bound reaches every other upper bound is optimal in the searched model, and the first-action loss of any action a is at most max_a' q⁺(a') − q⁻(a). With exact modeled levels and exact beliefs, the intervals contain the exact I-POMDP values. Tests: `test_certified_search.py`, `test_planner_exactness.py` (every expanded history and action, levels 1 to 3, every observation and root rule).

### Frontier bounds

With Q̄^r(s, a) the partner-optimistic vectors (each later action of j maximized), α̲^r_a the partner-pessimistic blind-policy vectors, and μ_{b,a,o}(s') the unnormalized physical marginal of the posterior, for every belief b, action a and r ≥ 1, in the model that does not compress i's belief below b:

    ℓ_r ≤ L^r(b, a) ≤ Q_s^r(b, a) ≤ Q^r_ITIB(b, a) ≤ Q^r_IFIB(b, a) ≤ Σ_s b(s) Q̄^r(s, a) ≤ u_r,

and each observation branch is bounded from the parent's belief, without computing the child's: max_{a'} Σ μ α̲^r_{a'} ≤ P(o, alive) U_s^r(b_ao) ≤ max_{a'} Σ μ Q̄^r(·, a'). When T_i is deterministic, I-TIB equals I-FIB. The tables have size |S||A_i||A_j|, not the size of the reachable model space. Tests: `test_frontier_bounds.py` (the chain at random beliefs against the exact solver).

### Compression slack

For head-tail compression, ‖b̂ − b‖₁ ≤ 2ε at every draw, the draws are unbiased, and with n_t = 0 the truncation attains the minimum L1 error over beliefs with N atoms. The value of a plan is linear in the belief with coefficients in [w̲^r, max_a Q̄^r], so the value at a compressed belief is within W̄_r ε of the value at the exact posterior (the L1 Lipschitz bound of Hsu, Lee & Rong, 2007). Two statements follow:

- **Over every reachable history:** |Q̃^H − Q^H| ≤ c_H, where c charges every reachable branch W̄_{r−1} ε plus the slack below it; c_H ≤ Σ_k γ^k W̄_{H−k} ε̄_k.
- **Over the tree** (`chance_scoring.compression_bound`): with c charging only the branches whose value scoring computes from their own belief (expanded histories, and every branch with two steps to go), the root intervals widened by c_H contain the value of the model that does not compress i's belief below the root. Branches bounded from the parent's belief need no charge, because the frontier bounds hold for the uncompressed model. The code uses the scalar range W_r ≥ W̄_r.

Stratified compression records no loss, so no slack is computed for it and its intervals are not certified (`compression_bound` raises). Compression of a nested belief is outside this bound: it enters through j's predicted action, which can switch discontinuously near a tie; bi-level soundness handles that case. Tests: `test_compression.py`.

### Bi-level soundness

With no compression of beliefs, at a grounded model node of j whose nested intervals contain j's exact values:

1. certain ⊆ OPT_τ ⊆ plausible, and the predicted law is within total variation δ̂ of the exact law Unif(OPT_τ);
2. a resolved node predicts the exact law, and a node whose belief was formed only through resolved nodes is grounded;
3. with d(b) = Σ b(s, m) δ̂(m), e^0 = 0, e^r(h) = min{W_r, W_r d(b_h) + γ max_a Σ_o P(o, alive | b_h, a) e^{r−1}(hao)} (and e^r = W_r at an unexpanded history), and Δ(a) = min{W_H, W_H d(b_i) + γ Σ_o P(o, alive | b_i, a) e^{H−1}(ao)}, the root intervals widened by Δ(a) contain the exact I-POMDP value (`chance_scoring.deviation_bound`);
4. the widened intervals satisfy the hypothesis one level up, so by induction from level 1 (whose level-0 models are exact) the guarantee holds at every level, budget and seed; when every reachable model node is resolved, Δ ≡ 0 and the searched model is the exact I-POMDP.

Tests: `test_resolution.py` (the set inclusions on random cases; the plausible set contains the exact optimal set at every checkpoint; unresolved solves widen the caller's intervals soundly).

### Termination of the gap-directed rules

Let T be the set of i's histories with at least two steps to go. Under `outcome="gap"` or `"gap_sample"` with the root rule `"open"` (or `"challenger"`), and with pruning aligned with the plausible set:

- every simulation started while the stop test is false expands exactly one history, so the search settles within |T| + 1 simulations, with no probabilistic argument. Lemma: at an expanded history, U − L ≤ q⁺(a*) − q⁻(a*) ≤ γ Σ_o P(o | h, a*) (U(ha*o) − L(ha*o)) for the action a* with the largest upper bound, so an open history has an open action with an open branch;
- the plausible set only shrinks, because τ⁺ = 10⁻⁹ max(1, M⁺, −M⁻) is non-increasing as the bounds tighten, so a pruned action is never needed again;
- under `scoring="pw"` with the lower stand-in and `leaf_eval="bound"`, every scored value lies in [q⁻(a), Q*(a)], so at any stop the decision's loss is at most M⁺ − m + τ, with m the largest lower bound among scored actions. When the agent's rewards satisfy R_min ≤ 0 ≤ R_max and u_H − ℓ_H ≥ 1 (true for both agents of every packaged domain), τ⁻ ≤ ε, so a settled root has every plausible action closed and recorded, and the loss is at most ε + τ.

Tests: `test_theorem_statements.py` (expanded histories equal simulations run plus one under arms B, C and D), `test_planner_exactness.py` (every small decision settles within its budget and plays exactly the exact maximizers).

### Consistency of UCB selection

With UCB selection and the sampled observation rule, as the budget grows the probability that the tree satisfies the condition of "Exact scoring" tends to 1; on that event every interval is a point and the decision is optimal in the searched model. With every modeled level solved by flat searches whose budgets also grow, and exact beliefs, the decision is optimal in the exact I-POMDP with probability tending to 1. This is an asymptotic statement; the gap-directed rules above have a finite bound.

## Three meanings of "exact"

Every error report states which one it uses.

| Term | Other agent's policies | Module | Measures |
|---|---|---|---|
| Exact best response | as declared in the bank (for example, finite I-POMCP solves) | `reference.best_response` | the planner's own search error |
| Exact I-POMDP | every level solved exactly, recursively | `reference.exact_agent` and the best response | search error plus model error |
| Exact POMDP (level 1) | fixed level 0 | `reference.pomdp_exact_vi` | independent cross-check by alpha vectors |

## Tie convention

In the exact I-POMDP, a modeled agent with several exactly optimal actions plays them uniformly; values within a relative 10⁻⁹ are tied (`planning.policy.TIE_TOLERANCE`), because algebraically equal values computed along different paths differ in the last bits. I-POMCP uses the same rule whenever its chance scores are exact, for its own decision and in every modeled solve. When the scores are not exact it keeps strict equality: search noise is not a tie, and an interval test cannot certify one. The real other agent in episodes draws uniformly among its exact ties too.

## Error measures

- **First-action loss:** V*(b) − Σ_a π(a|b) Q*(b, a), the value lost by the root decision when every later decision is optimal. It handles ties exactly and is not an episode regret.
- **Maximum Q error:** max_a |Q̂(b, a) − Q*(b, a)|.
- **Optimal-action Q error:** the same maximum over the exactly optimal actions only, the error of the value the agent acts on. The all-action maximum is dominated by clearly wrong actions, which search rightly samples little.
- **Belief error:** total variation between level posteriors; total variation between nested path masses at a given depth; the absolute error in the physical marginal.

A decision is **ε-correct** when its first-action loss is at most ε = 0.05 reward units (`experiments.oracle.CERTIFICATION_EPSILON`). Over a grid of n decisions with k losses above ε, δ is the upper end of the exact 95% Clopper–Pearson interval on k/n, and a configuration is certified (ε, δ) when k = 0. "Exactly optimal" (loss at most 10⁻⁹) is the stronger claim and is always reported beside it.

## The value of depth

Fix the stepping agent i, the lookahead H and uniform nesting: the agent at level l holds a uniform prior over j's levels 0, …, l − 1, its support. V_m(h, σ) is the true expected H-step reward of i's conditional plan σ from history h against j at level m, and σ_l(h) is the receding plan of the exact agent at level l. Module: `analysis.exact_curves`; tests: `test_exact_curves.py`.

**Correct prediction is enough.** If i's model gives, at every history of the next H steps, the same joint law of j's actions and i's observations as the real j, then i's exact best response maximizes the true value at every decision, and no other model does better there. Against an opponent whose actions do not depend on any belief about i (level 0 is the extreme case), a level-1 model is therefore the exact best response, and a deeper level can only tie or lose at a decision.

**Bayes dominance on the support.** If the agent at level k + 1 is exact, then at every history h it reaches, for every plan σ, Σ_{m' ≤ k} P_{k+1}(m' | h) [V_{m'}(h, σ_{k+1}) − V_{m'}(h, σ)] ≥ 0. Its belief is the posterior mixture of the exact conditional beliefs given each level, and plan values are linear in the belief. Corollaries: if the posterior of the true level m ≤ k is 1 − ε, the loss against it is at most ε/(1 − ε) (u_H − ℓ_H); a loss against one level is repaid, posterior-weighted, by gains against the others; and the prior-weighted expected gain at every step is nonnegative.

**Dominance at the boundary level.** With exact agents at levels k and k + 1 and nested priors (μ_{k+1}(m') = (1 − ρ) μ_k(m') for m' < k and μ_{k+1}(k) = ρ > 0; uniform and normalized-Poisson priors both qualify), when the true level is k, b_{k+1}(h) = (1 − λ_h) b_k(h) + λ_h b^k(h) with λ_h = P_{k+1}(m = k | h), and V_k(h, σ_{k+1}) − V_k(h, σ_k) ≥ ((1 − λ_h)/λ_h)[α_{σ_k}·b_k(h) − α_{σ_{k+1}}·b_k(h)] ≥ 0 at every history that both agents reach.

**Receding-horizon decomposition.** The cumulative gain of level k + 1 over level k in episodes equals the sum of the plan gains along the deeper agent's histories plus three drift terms (replanning drift of each level and the cross drift), so whole-episode gains and per-decision plan gains can have opposite signs.

**Level-0 mass keeps nested updates defined.** With deterministic percepts, a modeled j whose model of i is a point mass on one level can receive an observation that this model gives zero probability, and its Bayes update is then undefined. Any positive weight on level 0 restores it whenever the observation is possible under some action of i at some state in j's support, because level 0 gives every action probability 1/|A_i|. Every multi-level prior in the episode studies keeps level-0 mass.

## Cost model

Searching futures costs root simulations (`simulations`). Simulating minds costs one search of `modeled_simulations` for every distinct nested belief that a search reaches, and with exact beliefs these nested solves dominate the CPU of a level-3 decision. Modeled policies and belief updates are pure functions of their canonical digests, so their caches are kept across real steps. How to divide a fixed budget between the two costs is measured by `experiments.panel` against the exact I-POMDP.
