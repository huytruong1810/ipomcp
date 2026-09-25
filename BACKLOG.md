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
  Codex independently verified all cases, values and timing intervals. The depth
  comparison has 8 action flips out of 45 configurations. Opening resets rather
  than terminates Tiger. Results remain development evidence; production finite-budget
  modeled opponents remain unmatched. HANDOFF.md specifies the next implementation
  plan, including exact private-belief identity and deterministic policy semantics.
- Parent/worker monotonic timing is now recorded and checked. In the L2 study, monotonic
  and external GNU elapsed matched closely across all panels. Older long-duration L1 timing
  discrepancies remain historically unresolved without retroactive replacement.
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
