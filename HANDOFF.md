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
