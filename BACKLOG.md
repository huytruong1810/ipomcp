# Open work and acceptance criteria

## P0: accuracy and resource qualification

- Level-1 accuracy gate PASSED: The 2,000-case fresh held-out validation suite
  (H1–H5, 20 unseen beliefs, seeds 1000–1019, 1M traversals, empirical Bellman, exact
  final step, bounded $c=1$) completed with 0/2,000 primary gate violations
  ($\text{loss} \le 0.020$) and 0/2,000 strict errors (100.0% exact oracle agreement;
  mean loss 0.000000). Direct elapsed panel wall time was 43,392.04 s (12.05 h) and
  peak RSS was 113.42 MB. Previous 50k and 200k validation failures remain historical
  failures. Production defaults remain unchanged pending promotion review.
- Match the intentional policy model before comparing L2 with an exact oracle.
  The L2 reference currently has a common decreasing finite horizon, whereas
  production modeled MCTS policies replan at their configured fixed depth and
  simulation budget. Those are different opponent models.
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
