# Review handoff

The authoritative source is the Ubuntu WSL checkout at
`/home/andyj1810/projects/ipomcp`. Its uncommitted Antigravity work is preserved.
Review edits are isolated in `/home/andyj1810/projects/ipomcp-review-20260916`, on
`review/rigor-20260916`. Nothing has been pushed or merged into the original tree.

The pre-review snapshot commit is `18f666f`, based on original HEAD `e64ef8d` plus
the original dirty files and three untracked figure generators. The September 17
Antigravity source is copied to `../ipomcp-antigravity-baseline-20260917`, including
its local patch. Its new always-listen Tiger rollout is retained in the review;
the hook has one explicit contract and illegal actions raise errors.

Old handoff process IDs are not live execution evidence. WSL restarted before the
September 17 continuation, and none of the previously described experiment workers
were running when inspected. Check live processes and trial checkpoints directly.

The review compares distinct source snapshots using 30 seeds, 20 environment
decisions, depth 5, L3/L2 budgets 20,000/15,000, and the 80% L2 prior condition.
Raw audit data and the final comparison report are exported to the Codex task's
`outputs` directory. The earlier Windows clone experiment is excluded: it did not
contain the authoritative WSL local changes. Scientific provenance uses Git and
source hashes, not schema-version branches inside the application.

Read docs/REVIEW.md for phases and file coverage, docs/THEORY.md for equations and
counterexamples, and BACKLOG.md for remaining acceptance criteria. Two strict
expected-failure tests deliberately keep unresolved theory gaps visible. Do not
claim this is an exact Bayesian I-POMDP implementation or that RTS is an oracle.


Final measured result: 30/30 complete trials in each of the three source snapshots.
Agent I mean undiscounted return is -30.63 initially, -10.10 after Antigravity's
rollout change, and 15.93 after the review. The paired review-minus-Antigravity
95% interval is [-6.83, 58.90]; improvement is not statistically established.
Full tests: 76 passed, two documented strict xfails. Short CPU probes are 17.3%
slower despite better observed full-run wall times; do not claim a general speedup.
See docs/BENCHMARK.md for the complete measurements and their limits.

## September 18 implementation checkpoint

The user approved keeping uniform-random L0. Read docs/MODEL_SPECIFICATION.md and
docs/IMPLEMENTATION_PLAN.md before continuing. The new finite_belief.py and
finite_filter.py provide an immutable enumerated recursive-conditioning kernel.
They deliberately do not accept mutable search nodes. Tests include an independent
small L2 oracle and a substantive L3 nested-belief update, rather than merely
checking that a pointer changes. All current domains expose finite transition laws.

The running MCTS/RTS stack still uses its prior representation and update. This
checkpoint does NOT close its two expected-failure counterexamples or qualify
long experiments. Next migrate policy/belief consumers together, remove the old
representation and heuristic reconstruction, validate sampled updates against the
finite kernel, then run the specified before/after and suite stress experiments.
Preserve the original WSL checkout's uncommitted Antigravity changes throughout.

Checkpoint validation: 109 passed and 2 strict expected failures in 94.78 seconds;
the 33 finite-filter tests pass. Ruff lint/format, git diff whitespace validation,
and wheel build pass. The original tracked and untracked source files still match
the captured September 17 Antigravity snapshot byte-for-byte. No new long-run
reward comparison is claimed for this not-yet-integrated kernel.
