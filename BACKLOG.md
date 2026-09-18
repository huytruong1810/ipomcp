# Open work and acceptance criteria

These items are verified limitations, not a maturity scorecard. Passing engineering
tests does not close the mathematical items. See [THEORY.md](docs/THEORY.md).

## P0 — Research-validity blockers

- Replace epoch-preserving and factorized root reconstruction with a joint
  history-conditioned filter. Close the diagnostic-creak counterexample without
  forcing posterior floors or changing the fixture's known policies.
- Separate immutable subjective beliefs from MCTS nodes; recursively transition
  all nested private models. Close the history-advancement counterexample and
  compare an enumerated small L2 case across all reachable observations.
- Specify one opponent policy contract, including the intentionality assumption,
  tie handling, and whether real and modeled agents differ. JIT uncertainty must
  correspond to that same distribution. Establish the limits of any convergence
  argument for the resulting simulator.
- Replace unsupported deprivation behavior: old-belief retention and initial-prior
  proposals cannot silently count as Bayesian updates. Distinguish impossible
  observations from finite-sample impoverishment and test both.

## P1 — Experimental and architectural completeness

- Separate belief storage from particle routing and count observation likelihoods
  independently of capped reservoirs. Preserve state/type/history correlations.
- Add exact small intentional-agent references, then evaluate approximation error
  as particles, simulation count, planning horizon, and model depth vary separately.
- Define matched budgets/opponents for RTS comparisons. Top-k observation omission
  is an approximation; quantify lost mass and reward error. Remove remaining RTS
  belief-update duplication when implementing the common subjective filter.
- Report retained versus requested particle budgets. The principal Tiger setup
  requests more particles than its capacity; changing capacity needs its own study.
- Use counter-based per-step environment randomness if strict CRN alignment is
  required after policies cause differing numbers of random draws. Current streams
  are separated from planning but consumption still depends on actions.
- Add process isolation to file logging if concurrent shared-file logs are needed;
  source/manifest/trial records remain the authoritative evidence.
- Replace old ad hoc interactive demo runners with the batch episode engine once
  their differing sensor configurations and visualization contracts are explicit.
  Those entry points have been reviewed but not all coalesced in this patch.

## P2 — Scope-specific validation

- Rebenchmark corrected UAV and Wumpus semantics; Tiger evidence does not validate
  those domains. Verify self-observation and action-mask information assumptions.
- Expand figure QA and finite-policy equilibrium uncertainty analysis. A pure
  best-response cell in a noisy estimated payoff matrix is not a proven equilibrium.
- Introduce a user-selected statistical equivalence margin and a powered study
  before asserting equal or better expected performance. Thirty paired trials are
  a regression experiment, not a universal guarantee.

Completed fixes and per-file dispositions are recorded in docs/REVIEW.md and tests.
No schema migrations or backward-compatibility shims are required.
