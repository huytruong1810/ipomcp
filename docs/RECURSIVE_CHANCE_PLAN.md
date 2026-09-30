# Next candidate: recursive post-search chance integration

Implemented opt-in candidate, September 29. Operational instructions and the
frozen 23-case, three-arm pilot are docs/RECURSIVE_CHANCE_PROTOCOL.md. The pilot
is complete and reviewed: 23/23 optimal with incomplete deeper coverage; see
latest docs/REVIEW.md. Broader observed evaluation is still required.

## Evidence and hypothesis

The audited five nonzero candidate-versus-oracle traces span 408 histories with
no unresolved terms. Their empirical margin errors decompose into chance terms
(up to numerical roundoff); traced local action and frontier losses are zero.
For P11/C556 the largest net remaining-horizon contribution is H3 +.046310383,
followed by H6 +.024069131. Root-only weighting removes an offsetting root error
without correcting those child estimates. The two corrected cases have zero
comparison weights and no expanded paths, so they do not establish exactness of
their descendants. This motivates testing integrated probabilities recursively,
not blending estimates using a margin threshold selected from the failures.

## Proposed estimator

Keep search and all defaults unchanged. Add a separately named opt-in FINAL
post-search evaluator over the completed observed history tree. At history h:

    Q_rec(h,a) = E[R | b_h,a]
                 + gamma * sum_o P(o,alive | b_h,a) * V_rec(child(h,a,o))
    V_rec(h) = max over evaluated legal actions Q_rec(h,a)

Use the production expected_rewards, continuation_law and recursive Bayesian
private-history update, never reference Q or branch probabilities. Every model
is conditioned only on the agent's own action, private observation and public
survival. Keep terminal mass unnormalized and all inference limits explicit.
Evaluate descendants before parents and cache by immutable history/model and
remaining horizon; avoid hidden-state maximization and belief rounding.

Reuse already exact tail nodes. Use a declared rollout estimate at an observed
frontier without inventing child actions; report it as approximate. For any action
with missing positive-probability child support or no child estimate, retain that
action's original empirical Q as a whole-action fallback. Propagate enumeration
or unsupported-observation errors; never turn exceptions into silent fallback.
Record evaluated versus legal action coverage at every node. Local complete
branch support does not mean the entire subtree is exact.

Preserve every search statistic, original Q vector, reservoir and random stream.
The scoring option must be omitted from search seed derivation but present in
candidate metadata. Preserve the root-only option and its historical contracts;
make scoring modes unambiguous and reject conflicting options. Modeled opponents
remain at their existing configuration. Store final scores and policies separately
from search Q, along with counts/mass for local fallbacks, propagated approximate
subtrees, rollout frontiers, unevaluated actions and exact boundaries.

## Theoretical scope and tests

If every legal action is evaluated, every positive continuation branch is
available, and every leaf is an exact remaining-horizon boundary, backward
induction gives the finite-model Bellman solution (up to floating-point error).
This is a conditional theorem, not a guarantee that a finite sampled tree has
that coverage. Incomplete branches, frontier estimates or missing actions retain
approximation; maximizing rescored actions may expose branches outside the
previous greedy trace and create new local decision errors.

Tests must cover a hand-computable multi-level tree where root-only integration
fails but recursive integration removes a descendant frequency error; terminal
mass; asymmetric branch coverage; whole-action fallback; frontier propagation;
unevaluated legal actions; ties; exact-boundary agreement; no oracle dependency;
unchanged full tree and seed/RNG identity; unchanged modeled policies; cache and
inference-failure behavior. A small paired L2 case should verify evaluator metadata
and scored policy. Do not infer asymptotic error rates from this fixed-budget test.

After implementation, freeze a small panel containing all seven diagnosed cases
and previously correct neighboring controls before inspecting candidate outcomes.
Compare empirical, root-only and recursive recommendations from identical search
settings, preserve all raw vectors and report resource costs and coverage. A
later broader observed comparison must precede any fresh qualification. Keep all
historical failures and the .020 +1e-8 tolerance unchanged.
