# Current review handoff

## Ownership and current decision

Antigravity runs experiments; Codex owns code/math changes. Use only
/home/andyj1810/projects/ipomcp on fix/tiger-policy-inversion.
No source copies or source edits during runs. Raw evidence remains local and
Git-ignored; a documentation commit is not a raw-data archive.

The frozen 2000-case L1 validation accuracy gate PASSED. This qualifies the
tested first-action decisions and configuration, not all beliefs, all horizons,
whole-episode policies or deeper agents. Do not call policy inversion permanently
solved or claim a general Bayes-optimality certificate.

Do not promote global MCTSConfig defaults yet. The validated combination includes
bounded UCB c=1 and 1M traversals as well as empirical Bellman and exact tail.
Changing only backup/tail defaults would not reproduce it. Current generic
defaults use n_sims=1000, max_depth=20; modeled policies default to 25 simulations.
Production schedules and other domains are outside this validation.

## Independent audit of 1b3527a

Source checkpoint a866e69 froze epsilon_loss=.020 before the run.
All 2000 raw cases have complete/unique requested coverage and settings matching
the frozen H1-H5, twenty-belief, seeds1000-1019 protocol. Source fingerprints
match a866e69. Recomputed exact L1 reference Q values for all 100 horizon/belief
pairs, every policy loss and maximum absolute Q error. This independently
checks the report against the existing exact reference, not a new proof of it.

Results:
- 0 violations of loss <= .020 + 1e-8;
- 0 strict first-action errors at numerical tolerance 1e-8;
- maximum and mean first-action loss 0;
- all 2000 cases complete.

This is first-action loss with optimal continuation afterward. At H5 mean max
absolute Q error is 3.668401 and the largest is 6.476666 reward units; accurate
selected actions do not imply exact estimated values. Earlier development at
1M still had four nonzero-loss cases at H4/H5, disproving universal agreement.
Old 50k/200k validation gates stay historical failures. These newly inspected
cases must not be represented as untouched validation for later tuned solvers.

Evidence:
- results/oracle/bellman_validation_20260924/
- results/oracle/bellman_validation_20260924.log
- results/oracle/bellman_validation_20260924.time
- results/oracle/bellman_validation_review_20260925.json (independent audit)

## Timing discrepancy: unresolved

The GNU time file really contains elapsed_seconds=43392.04. However, the
verified sum of worker monotonic durations is 90054.254809 s. With two workers,
a comparable panel elapsed duration must be at least 45027.127405 s (12.51 h).
The reported 43392.04 s (12.05 h) is 1635.087405 s below that lower bound.

Do not replace the raw timer, invent a corrected actual duration, or claim
runtime certification. Clock-domain changes, environment behavior and timing
provenance need investigation; the cause is not established. The numerical
accuracy gate is preserved, while elapsed-runtime evidence is unresolved.
Peak monitored process-tree RSS is 113.421875 MiB (sampled, not a hard peak bound).

## Next phase: instrumentation and matched L2 semantics

Codex should first make panel/worker timing comparable: inspect the supervisor
and runner timing boundaries, record parent monotonic elapsed alongside external
elapsed, and verify worker-sum/concurrency consistency in a short supervised
multi-job run. Preserve both measurements if they disagree. No need to rerun
the 12-hour accuracy panel merely to overwrite its timing record.

For the L2 phase, review these coherent modules together:
- src/solvers/exact/ipomdp_exact_vi.py and pomdp_exact_vi.py;
- src/solvers/i_pomcp.py and solver_bank.py;
- src/ipomdp/frame.py, finite_belief.py and the finite filtering machinery;
- bootstrap configuration, oracle runner and corresponding exact/filter tests.

Current mismatch is explicit in code: ExactIPOMDPSolver uses a common decreasing
remaining horizon for both agents, whereas SolverBank requests a modeled policy
whose MCTS uses its fixed max_depth and opponent.n_sims at every private belief.
An exact L1 opponent and a finite-budget modeled MCTS opponent can also choose
different actions. Merely giving both solvers the same initial depth is not
sufficient to match the decision problem.

Before implementation, document one explicit comparison contract: physical
dynamics, private observations, posterior update, opponent policy law (including
horizon, computation and tie convention), protagonist horizon and discount.
Keep countdown finite-horizon and fixed-depth replanning models distinct;
neither is automatically a bug. Do not silently change production semantics to
fit a reference. Validate tiny analytically tractable cases, then bounded L2
comparisons with matched opponent policies before any long higher-level suite.

Antigravity should retain current evidence and await the matched L2 protocol.
Do not start L4/all39 qualification from this L1 pass alone. Further gates include
long/deep Tiger before/after performance, intended modeled-policy resources,
finite-prior approximation and remaining demo/visualization semantic review.

## Engineering state

This evidence review changes documentation only; solver behavior is unchanged.
The previous 204 passing tests and lint/format results remain the latest
engineering verification; no claim of a fresh test run is made.
Defaults remain sampled backups, sampled final steps, empirical-range UCB.

Keep README.md, HANDOFF.md, BACKLOG.md and docs/BENCHMARK.md consistent.
Historical active protocols belong in Git. Read older METADATA_ERRATUM.md files
when interpreting historical modeled-policy budgets.
