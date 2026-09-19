# Open work and acceptance criteria

## P0 — Full-suite qualification

- Qualify the declared matched-computation comparison across conditions and seeds.
  The previous RTS mismatch is addressed by separate modeled-planner configuration,
  matched budgets and common initial computation. Other misspecified point-prior
  conditions may still fail; do not silently repair those posteriors.
- Reward equivalence is not established. The corrected thirty-seed Tiger result
  is I=-21.10, J=8.23; the earlier heuristic review was I=15.93, J=5.30.
  Investigate policy-budget sensitivity using held-out seeds before research claims.
- Qualify every condition at intended depth and budgets. All 39 shallow smoke
  cases pass; one twenty-step L4 probe passes, and the revised declared-computation RTS panel passes ten twenty-step seeds. These are
  not interchangeable forms of evidence.
- Provide ordinary suite workers with enforced time/RSS limits and failure records
  equivalent to the isolated audit driver before unattended full-suite execution.

## P1 — Architecture and approximation evidence

- Enforce fixed bank configuration on repeated bootstrap registration and frame
  lookup; currently callers must honor that invariant.
- Extend modeling beyond the controlled comparison if unknown solver seeds, priors
  or budgets are research variables; represent that uncertainty explicitly.
- Quantify RTS omitted observation mass and its value-error impact.
- Characterize value and distribution error over independent seeds and budgets;
  current exact-reference tests establish bounded cases only.
- Consolidate remaining demo episode loops around the common runner.
- Use per-step/per-purpose environment streams for stronger common-random-number
  comparisons when action-dependent random consumption differs.
- Reconcile and integrate the review into the original WSL checkout while preserving
  concurrent Antigravity changes. No reset, automatic overwrite or push.

## P2 — Domain and presentation evidence

- Run substantive UAV and Wumpus studies. Wumpus integration tests pass but are
  expensive; Tiger evidence alone does not qualify those domains.
- Verify optional tree rendering once native Graphviz dot is installed.
- Expand figure QA and payoff uncertainty analysis. No equilibrium theorem follows
  from a noisy finite payoff matrix.
- Select a scientific equivalence margin and sufficiently powered held-out study
  before asserting equal or better expected reward.

Completed: immutable joint beliefs; recursive private-history filtering; common
online update in both planners; explicit zero-support failures; unified greedy
policy contract; exact level weights; stable private solves; independent bounded
references; removal of reinvigoration/JIT compatibility paths; root reward control
variate. Tests: 122 passed, no xfails. See docs/REVIEW.md and docs/BENCHMARK.md.
