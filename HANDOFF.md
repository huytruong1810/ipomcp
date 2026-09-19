# Review handoff

Authoritative WSL checkout: /home/andyj1810/projects/ipomcp.
Review worktree: /home/andyj1810/projects/ipomcp-review-20260916.
The pre-integration tracked and untracked Antigravity source is preserved in
backup/antigravity-before-supervision-20260919 (commit 6a9393648ad1f6d9d3c8c3731e7b05250d689956).
The original index/worktree were unchanged when that recovery snapshot was made.
Engineering source is integrated into main at 02d814b; the final
integration record documents the retained stash and recovery branch. The original checkout also passed all 123 non-integration tests after dependency synchronization. Nothing is pushed.

## Supervised full-depth qualification — September 19

Every batch trial now uses an isolated Linux process session, including runs with
one worker. Defaults are two workers, 900 seconds per trial and 3,072 MiB sampled
process-tree RSS. An explicit run_batch(max_workers=...) overrides the default
worker count. RSS sampling can miss short spikes; it is not an allocation ceiling.
The supervisor kills timed-out/over-budget sessions and reaps workers. Snapshot
episodes preceding suite batches use the same limits. Each attempt retains logs,
status, wall/RSS measurements and Python traceback where available. Successful
trial checkpoints survive failures; retries keep historical failure records.

The full-depth qualifier uses the actual condition tables with no reduced search
budgets. One seed per condition, twenty steps, depth five for prior/matrix and
depth three for comparison produced:

| Suite | Complete | Failed | Max wall seconds | Max sampled RSS MiB |
|---|---:|---:|---:|---:|
| Prior | 7 | 0 | 50.48 | 153.96 |
| Comparison | 7 | 0 | 76.34 | 134.58 |
| Matrix | 12 | 13 | 900.04 | 696.13 |

All 127 unit/integration tests pass, including fault injection for time, RSS,
process crashes, descendant cleanup and checkpoint recovery. The failed matrix
trials remain failures, not zero returns or omitted observations. Its so-called
exact prior is a point mass on a capped lower level, not a complete exact model
of the actual opponent policy. Strict inference cannot define a posterior on
zero-probability evidence. Choosing explicit uniform lower-level priors including
L0 is a separate experimental design and awaits user selection.

The prior and comparison families have a full-budget twenty-step execution check;
one seed per condition is not a powered validation of expected payoff. The full
suite remains unqualified while matrix conditions fail. Existing payoff/value
limitations and the unestablished L3-vs-L2 reward equivalence remain in force.
Runtime results are source-bound in results/full-depth-qualification-20260919.

Next scientific decision: matrix point priors versus explicit uniform lower-level
priors. Keep strict inference until the user selects a changed model. Continue
with BACKLOG.md; do not describe the full suite or all planning as optimal.
