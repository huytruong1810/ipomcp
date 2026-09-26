# Open work and acceptance criteria

## P0: accuracy and resource qualification

- Level-1 accuracy gate PASSED: The 2,000-case fresh held-out validation suite
  (H1–H5, 20 unseen beliefs, seeds 1000–1019, 1M traversals, empirical Bellman, exact
  final step, bounded $c=1$) completed with 0/2,000 primary gate violations
  ($\text{loss} \le 0.020$) and 0/2,000 strict errors (100.0% exact oracle agreement;
  mean loss 0.000000). Accuracy is scoped to the tested grid/configuration, not all horizons.
  External elapsed time (43,392.04 s) contradicts the worker-sum lower bound
  (45,027.127 s); investigate clock/provenance consistency before runtime claims.
  Peak monitored RSS was 113.42 MiB. Previous 50k and 200k validation failures remain historical
  failures. Global defaults remain unchanged: their budgets, depth, exploration and modeled
  opponents differ from the validated configuration. Resolve instrumentation
  and matched L2 semantics before deeper qualification.
- Level-2 fixed-depth exact-opponent development comparison COMPLETED: The 900-case
  development suite (depths 1 and 2, b_j in {.085, .5, .915}, H1-H3, budgets 1k/10k,
  seeds 300-304, own beliefs .05/.2/.5/.8/.95) completed with 899/900 strictly optimal
  decisions (99.89%) and 0/450 errors under d=2. Exactly 1 error occurred (d=1, b_j=.915,
  H3, 1k budget, seed 301, loss 0.3377; resolved at 10k). Concurrency bounds passed on
  all 6 panels, with parent monotonic elapsed (~52-55 s) matching GNU time within fractions
  of a second. Depth comparison confirmed 8 oracle action flips between d=1 and d=2.
- Level-2 finite-budget modeled-opponent development comparison COMPLETED: The 900-case
  development suite (modeled budgets 25 and 100, b_j in {.085, .5, .915}, H1-H3, budgets 1k/10k,
  seeds 400-404, own beliefs .05/.2/.5/.8/.95) completed with 900/900 strictly optimal decisions
  (100.0% strict agreement) and zero loss across all 6 development panels. Concurrency bounds passed
  on all panels (parent monotonic elapsed ~53-57s matching external GNU time within seconds).
  Opponent budget comparison confirmed 9 oracle action flips and 21 Q-value shifts between 25 and 100
  simulations, demonstrating that modeled computation alters the policy law and induced decision task.
  Codex verified all 900 cases and recomputed 450 oracle configurations. Monotonic
  timing passes concurrency checks but differs from external timers by 1.32–3.06s.
- Level-2 depth-20 H1-H3 development extension independently AUDITED: 360/360
  first actions agree. Codex verified every row and recomputed 180 distinct
  reference problems. Against depth 3, opponent budget 25 has 9/90 action changes
  and 26/90 Q/value changes; budget 100 has 0/90 action changes but 5/90 Q/value
  changes (maximum 2.264711). Root-budget copies are not independent problems.
  Parent monotonic 140.804341s versus external 135.67s remains unresolved.
- Level-2 H4/H5 development study independently AUDITED: verified 360 rows and
  recomputed 120 distinct reference problems. Six errors, all above .020,
  occur at root 1k/10k with modeled budget 25. Root 50k has 0/120 errors, but maximum
  Q error 4.200977. This is development evidence, not a convergence or validation
  certificate. At H4, errors increase 1 to 2 from 1k to 10k before reaching 0 at 50k.
  Modeled budget 25 versus 100 changes 6/60 optimal action sets and 42/60 Q vectors.
  All concurrency bounds pass; external 454.09s versus monotonic 478.198078s
  remains unresolved. Next resource protocol is in HANDOFF.md.
- Level-2 H6/H8 development independently AUDITED: all 120 references recomputed,
  source hashes verified against e63db30, zero first-action loss. Every optimum
  is listen, minimum gap .640013; opening and near-tie coverage are absent.
  Maximum Q error 8.523833 remains. Six pilot repeats reproduce exactly.
  All concurrency checks pass; external 555.49s versus monotonic 588.865233s
  is unresolved.
- Level-2 fresh held-out validation suite (1,680 cases: H1–H6, H8, 10 fresh physical beliefs,
  seeds 6000–6003, root 50k, empirical Bellman, exact tail, bounded c=1) COMPLETED: Primary
  gate FAILED under frozen zero-violation criterion (1,657/1,680 passes = 98.63%; 23/1,680
  violations = 1.37% exceeding the frozen .020 bound; mean loss 0.002198, max loss 0.741558).
  Strict optimal choices: 1,654/1,680 (98.45%). All 960 cases across H1–H4 passed with zero
  violations and zero strict errors (100.0%). All 23 violations were concentrated at H>=5,
  with 22/23 localized at the critical transition frontier beliefs b_i in {.0325, .9675}
  (12 at .0325, 10 at .9675, 1 at .0625). Extreme and moderate beliefs achieved 100.0%
  pass rate (1,176/1,176). Concurrency bounds passed on all 6 panels (raw parent monotonic 4,426.169385s,
  external GNU 4,152.20s). Codex independently recomputed all references; the
  original report timing table did not match raw files and has been corrected.
- Canonical MCTS belief sampling fixes a reproduced cache/model identity bug:
  equal beliefs in different insertion orders could return different policies.
  Preserve old source-bound evidence; qualify the corrected finite policy on
  declared modeled budgets/depths rather than assuming old trajectories persist.
- Parent/worker monotonic timing is recorded and internally checked. External
  elapsed still differs in recent L2 panels and the earlier long L1 study.
  Instrument both clock endpoints and execution provenance before speed claims;
  do not replace historical measurements.
- Qualify corrected L4 and all 39 production conditions at intended resources.
  Old uniform-matrix runs with frozen rollout beliefs are not this qualification.


## P1: architecture and approximation limits

- Reject conflicting repeated bootstrap configuration, not just physics identity.
- Quantify RTS omitted observation mass for production configurations using top-k;
  the matched oracle suite already retains every observation token.
- Characterize finite empirical prior error separately from search error.
- Record per-attempt effective execution limits when limits change on resume;
  preserving only the final manifest loses earlier resource settings.
- Consolidate remaining domain/demo episode loops around the common runner.
- Strengthen common-random-number comparisons with per-purpose RNG streams.
- Bound-aware exploration and exact final-step integration are implemented as
  explicit options. Qualify their cost/accuracy beyond L1 before any production
  change. Empirical Bellman backups are also opt-in; no general convergence claim
  follows from finite action agreement. See docs/THEORY.md and docs/BENCHMARK.md.

## P2: broader evidence and presentation

- Substantive UAV and Wumpus studies and optional native Graphviz rendering.
- Figure QA and uncertainty for complete matrix panels. No equilibrium proof
  follows from a finite noisy payoff table.
- Complete semantic review of remaining demo/visualization entry points listed
  in docs/REVIEW.md; structural import checks alone are insufficient.

Completed in this pass: removal of L2 hidden-state maximization, unrounded joint
belief recursion, removal of nearest-belief fallback, correction of zero-horizon
queries and probability validation, absolute slope tolerance in alpha pruning,
visible LP failures, removal of unsupported tree action pruning, restoration of
private-model propagation in rollouts, persistent/noisy-sensor rollout memory,
and consolidation of four misleading oracle/report scripts into one matched,
resource-supervised budget sweep. Historical results remain source-bound evidence.

Worker startup now uses spawn to avoid inheriting numerical-library thread locks. Fault-injection, resume, and domain integration checks cover the change.


Current next step: matched control/candidate development for the implemented
opt-in conditional immediate-reward estimator, as specified in HANDOFF.md. The fresh 1,680-case gate failed permanently;
the observed grid may now support development only.
No global default promotion or L4/all39 launch is authorized by these results.


Codex independently audited f0b98c5 and reproduced two failing solves. Root
chance-frequency error is small relative to child-value error in both diagnoses.
Implemented opt-in exact conditional immediate rewards at internal tree histories.
The 12-pair smoke has 1 control violation and 0 candidate violations; this does not
repair the historical gate. Next: full matched development comparison, preserving
the same new-source modeled-policy law between arms. No global promotion.
