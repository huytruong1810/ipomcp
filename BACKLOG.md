# Open work and acceptance criteria

## P0 — Scientific full-suite gate

- Resolve the 5x5 matrix's prior design. Full-budget twenty-step qualification
  exposes unsupported observations under capped point priors. The user is choosing
  between explicit uniform lower-level priors including L0 and retaining undefined
  cells under strict point priors. Do not fabricate posteriors or numeric payoffs.
- Bound the expensive L4 matrix computations without freezing private beliefs or silently lowering budgets. Profile evidence must guide that work.
- Requalify any changed matrix model over full episodes and multiple seeds. Full
  condition coverage has now been executed; coverage alone does not mean success.
- Reward equivalence remains unestablished: the corrected thirty-seed L3-vs-L2
  result is I=-21.10, J=8.23, versus the earlier heuristic review I=15.93, J=5.30.
  Use held-out seeds and a scientific equivalence margin for claims of equal reward.

## P1 — Remaining architecture and approximation work

- Reject conflicting repeated bootstrap configuration, beyond the implemented
  physics-frame identity guard.
- Represent unknown opponent solver seeds, priors or budgets explicitly if they
  are research variables; the controlled comparison declares shared initialization.
- Quantify RTS omitted observation mass and resulting value error.
- Characterize finite-prior and finite-search approximation error across budgets.
- Consolidate remaining demo episode loops around the common runner.
- Strengthen common-random-number alignment with per-step/per-purpose streams.

## P2 — Broader domain evidence

- Run substantive UAV and Wumpus studies; passing integration tests is insufficient
  for domain-specific scientific conclusions.
- Verify optional tree rendering when native Graphviz dot is available.
- Expand figure QA and payoff uncertainty analysis. No equilibrium proof follows
  from a noisy finite payoff matrix.

Completed: immutable joint inference, recursive private-history updates, shared
planner filtering, strict failures, reproducible policies, matched controlled
comparison, root reward control variate, frame binding, bounded worker execution
including snapshots, failure-preserving resume, and full-depth condition coverage.
Tests: 127 passed, no xfails. See docs/BENCHMARK.md for the failed conditions and
measured limits; the original matrix design is not ready for an unattended full suite.
