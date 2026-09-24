# Open work and acceptance criteria

## P0: accuracy and resource qualification

- Characterize I-POMCP finite-budget error against the matched L1 reference.
  Horizon-three action errors persist through 1,000,000 simulations in the
  reviewed panel (six of nine belief/seed cases at that budget). See
  docs/BENCHMARK.md for the source-bound measurements.
  Use budget/error/cost curves and held-out beliefs/seeds; do not reinstate
  confidence-based action pruning or freeze intentional beliefs to pass a test.
- Match the intentional policy model before comparing L2 with an exact oracle.
  The L2 reference currently has a common decreasing finite horizon, whereas
  production modeled MCTS policies replan at their configured fixed depth and
  simulation budget. Those are different opponent models.
- Qualify corrected L4 and all 39 production conditions at intended resources.
  Old uniform-matrix runs with frozen rollout beliefs are not this qualification.
- Establish meaningful reward equivalence margins and independent validation
  seeds. Reused 30-seed comparisons are exploratory, not equivalence proofs.


## P1: architecture and approximation limits

- Reject conflicting repeated bootstrap configuration, not just physics identity.
- Quantify RTS omitted observation mass for production configurations using top-k;
  the matched oracle suite already retains every observation token.
- Characterize finite empirical prior error separately from search error.
- Record per-attempt effective execution limits when limits change on resume;
  preserving only the final manifest loses earlier resource settings.
- Consolidate remaining domain/demo episode loops around the common runner.
- Strengthen common-random-number comparisons with per-purpose RNG streams.
- Investigate optional bound-aware exploration or exact conditional reward
  integration as separately specified algorithms, not unreported benchmark fixes.
  The September 24 bound-aware exploration ablation is implemented; all tested
  settings still have positive action loss. See docs/BENCHMARK.md. Mean backups
  remain unchanged; no new production coefficient is qualified.

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
