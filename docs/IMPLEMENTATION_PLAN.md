# Plan: correct inference before full-suite experiments

## Objective and current boundary

Make the existing WSL codebase suitable for a reproducible full experiment suite,
with mathematically specified inference, explicit approximation limits, and measured
resource requirements. Correctness takes precedence over reproducing a favorable
reward. Do not retain a known error merely because it improves the benchmark.

The original repository is `/home/andyj1810/projects/ipomcp`; the existing review
worktree is `/home/andyj1810/projects/ipomcp-review-20260916`. Its reviewed baseline
is commit `7d5402c`. No new clone is needed. The original checkout has uncommitted
Antigravity changes, so integration must preserve and reconcile those changes.
Existing source snapshots and benchmark artifacts provide the regression baseline.

This document preserves the accepted implementation order. Current status:
phases 2–3 are integrated; phase 4 has bounded independent references; phase 5 has
measured Tiger and L4 evidence; phase 6 cleanup/supervision/integration remains
partial. The controlled RTS support mismatch is corrected under a declared matched-computation
model; phase 7 still lacks full condition coverage and reward equivalence. All 122
tests pass and all 39 shallow conditions
pass. Neither result closes the outstanding scientific and execution gates.

## Modeling decisions

Preserve the current Tiger physical dynamics, observation timing, sensor accuracy,
discount, and always-listen leaf rollout for the first corrected comparison.
Opponent types remain fixed unless an explicit type-transition model is introduced.
A physical reset does not reset an opponent's type or automatically erase its
private history.

**Decision accepted:** retain uniform-random L0 for the corrected implementation.
The recommendation is to retain the current random-L0 convention initially, to
separate inference corrections from a change to the experiment's model class.
An optimizing L0 would instead require a physical belief and its own planner at
the base of the hierarchy. This choice must be recorded in results and reference
tests; the two hierarchies must never be presented as the same experiment.

For intentional agents, use a single policy-distribution interface. The intended
rational policy is a distribution over maximizing actions; a Monte Carlo planner
provides an approximation to the corresponding action values. Specify tie handling,
remaining planning horizon, preferences, and computational settings explicitly.
Real action selection and modeled action sampling must consume that interface.
If inner and outer solves use different budgets, report the resulting approximation
rather than claiming their distributions are identical. Do not retain the current
unexplained combination of maximum visits, normalized Q-softmax, and raw-Q entropy.

The mathematical reference is recursive subjective belief propagation, not direct
access to the outer simulator's hidden joint action. See Doshi and Gmytrasiewicz
(2009), Figures 4–5: https://arxiv.org/pdf/1401.3455. Their theoretical results do
not automatically apply to a modified particle representation or policy solver.

## Phase 1 — Reconcile source and lock the specification

1. Compare the authoritative checkout with the captured Antigravity source and
   review commit, including untracked scripts and notes. Recheck before integration
   to detect concurrent edits.
2. Preserve the original local changes in Git-addressable recovery evidence before
   applying anything to the authoritative checkout. Do not reset or silently
   overwrite Antigravity's work, and do not touch existing result directories.
3. Record the chosen hierarchy, observation/reward information available to each
   agent, policy rule, horizon convention, and particle/solver budgets.
4. Inventory all suite conditions and entry points. Define a manifest containing
   actual effective settings, source identity, seed streams, and environment data.

**Exit condition:** a concrete model specification, a reproducible baseline, and
an integration diff that accounts for every existing local source change.

## Phase 2 — Separate beliefs, frames, policies, and search state

Primary modules: `ipomdp/belief.py`, `solvers/node.py`, `solver_types.py`,
`solver_bank.py`, `planner.py`, and `utils/bootstrapper.py`.

- Represent an intentional model as an immutable frame plus immutable subjective
  belief. An interactive particle keeps the physical state and complete opponent
  models together. It must not use a mutable MCTS node as its latent belief state.
- Make frames identify the modeled dynamics, preferences, policy assumptions, and
  horizon convention. Agent ID and level alone do not identify a complete model.
- Keep visits, Q estimates, rollout state, and search children outside beliefs.
  Queries for one model must not mutate another model's belief.
- Share only equivalent immutable beliefs. Do not merge different private histories
  because they happen to have the same level. Cache keys must account for the belief,
  frame, horizon, and relevant solver settings; cache eviction may affect runtime
  but must not secretly change the model being solved.
- Make recursion terminate at the selected L0 convention and enforce decreasing
  nesting levels. Keep interfaces focused on the two-agent domains actually supported.

**Exit condition:** aliasing, mutation, serialization, and policy-contract tests
pass. Identical queries are reproducible under the specified computation model.

## Phase 3 — Implement the joint recursive Bayesian update

Primary modules: a dedicated belief-filter module, `generative_model.py`,
`i_pomcp.py`, `rts_planner.py`, and domain transition/observation interfaces.

For each prior interactive state, integrate or sample the modeled opponent action,
propagate the physical state, anticipate the opponent's private observation, update
that opponent's subjective belief recursively, and weight by the acting agent's
observation likelihood. Resample complete interactive states; do not independently
sample the state and opponent level.

The opponent's subjective update receives its own action and observation. Unknown
actions are marginalized under its own models. The outer simulator's hidden state
and action must not leak into that update. If private observations are sampled from
their model, avoid multiplying their likelihood a second time; if enumerated,
include their probability explicitly.

- Use the same filter in MCTS rerooting and sampled reachability-tree planning.
  Separate belief estimation from search traversal statistics.
- Remove epoch-preserving type overrides, uniform type floors, arbitrary pseudocount
  branches, and selection of a single representative private model.
- Specify finite transition support for the current finite domains, where practical,
  so reference calculations and likelihood accounting can be checked independently.
- Distinguish a proven impossible observation from zero weight in an approximate
  particle population. The latter is not proof of impossibility in the full model.
- Any replenishment must target the specified posterior with justified weighting.
  If the computation cannot recover supported mass within its explicit budget,
  raise a diagnostic failure and checkpoint it. Do not preserve the old belief,
  draw an unrelated initial prior, or silently fabricate uniform weights.

**Exit condition:** both existing theory counterexamples pass without xfail, and
tests establish state/type correlation preservation, diagnostic creak conditioning,
private-history advancement, and absence of hidden-information leakage.

## Phase 4 — Validate against independent exact references

- Extend the current fixed-policy Tiger reference to small intentional L1/L2 cases
  under the chosen L0 convention and short, explicitly bounded horizons.
- Enumerate states, hidden actions, private observations, and reachable model updates
  in the reference; do not call the production filter from that oracle.
- Compare the unresampled finite-support update and Bellman expectations to the
  reference numerically. Cover quiet listening sequences, conflicting evidence,
  own and opponent openings, simultaneous openings, and known reset versus persistent
  dynamics. Evaluate every reachable observation in the small fixtures.
- Test particle approximations across independent repetitions and particle budgets.
  Report distribution/value error and uncertainty; do not require every stochastic
  realization to improve monotonically as the budget increases.
- Revalidate UAV and Wumpus information restrictions, transition events, absorbing
  states, and observation-kernel normalization. Tiger results cannot certify them.
- Test cache order/eviction and serialization so they cannot alter the inference
  target or make a prediction depend on which unrelated query happened first.

**Exit condition:** the exact small cases agree, stochastic errors are characterized,
and no known theory defect is hidden by an expected-failure marker or weakened test.

## Phase 5 — Optimize only the validated implementation

- Profile recursive updates, policy solves, belief allocation, and cache growth.
  The previous profile identifies nested simulation as the dominant cost.
- Reuse immutable structures and memoize validated equivalent computations. Do not
  recover speed by freezing deeper beliefs, leaking hidden information, rounding
  belief keys without an explicit approximation, or changing policy semantics.
- Set retained particle budgets explicitly rather than silently clipping a nominal
  schedule. Record both configured budgets and actual populations.
- Measure CPU time, wall time, peak RSS, steady-state memory growth, and cache hit
  rates. Exercise long sequences, not only one-step startup behavior.
- Choose worker counts from observed per-worker peak memory with headroom. Make
  resource-limit failures visible and resumable; a watchdog must enforce limits
  if it is advertised as enforcing them.

**Exit condition:** the highest intended levels fit the declared memory/runtime
budget, and any accuracy/resource tradeoff is explicit and reproducible. No general
speedup claim based solely on overlapping wall-clock runs.

## Phase 6 — Validate experiment execution and clean integration

- Consolidate remaining duplicated demo execution paths around the common episode
  engine, preserving explicit differences in their experiment configurations.
- Use per-step, per-purpose environment random streams for stronger paired
  comparisons after policies change action-dependent draw counts.
- Require complete trial panels and matching manifests; test interrupted/resumed
  execution, worker exceptions, corrupt checkpoints, and failures during final output.
  Record failed trials rather than dropping them from statistical analysis.
- Remove obsolete compatibility paths, synthetic generators, and superseded APIs
  after replacement. Update imports, documentation, examples, and plotting consumers.
  Keep one current implementation and one current data contract.
- Integrate the validated changes into the authoritative checkout after the source
  reconciliation check. Preserve raw historical evidence with its source provenance;
  do not rewrite old results to appear generated by the corrected solver.

**Exit condition:** a clean integration with no unexplained local-change loss,
passing unit/integration/package checks, and demonstrated resume/failure behavior.

## Phase 7 — Qualify the full suite

1. Run every actual suite condition with short episodes and small trial counts.
   Check return bounds, normalized beliefs/policies, finite metrics, artifacts,
   complete output, and manifest identity. Reduced-budget smoke runs are explicitly
   distinct from final scientific experiments.
2. Run high-level and RTS stress cases with the intended effective budgets to qualify
   peak memory and runtime. Small-budget smoke success is insufficient here.
3. Repeat the 30-seed, 20-step, depth-five L3-vs-L2 Tiger comparison. Where the
   corrected model changes an assumption or budget, report that difference instead
   of presenting it as an identical-design comparison.
4. Analyze paired trial returns and resource measurements. Investigate regressions;
   do not restore a known inference bug to recover a reward target. Do not claim
   equivalence without a scientifically chosen margin and adequate power.
5. Launch the long full suite only after the preceding gates pass. Its source and
   settings remain fixed for that run, with checkpointing and enforced resource limits.

**Ready means:** no unresolved demonstrated inference defect in the supported
model; agreement with bounded exact references; every suite condition exercised;
stress-tested resource limits; trustworthy resume and failure handling; and honest
documentation of remaining finite-particle and finite-search approximations.

## Execution order and checkpoints

Keep changes reviewable in coherent commits: source/specification, belief/policy
representation, recursive filter, planner integration, exact references, optimization,
runner cleanup, and suite qualification. Update this plan and the handoff with actual
evidence as phases complete. Tests precede expensive experiment runs.

Do not publish an arbitrary completion estimate before measuring the corrected
nested filter: its computational cost is a material design constraint. If the
requested high-level configuration proves infeasible, report that limit and the
measured alternatives rather than silently reducing budgets or approximating away
the very belief dynamics being tested.
