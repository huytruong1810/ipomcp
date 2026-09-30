# Process supervisor review — September 30

Claude reviewed the project as process supervisor while panels 1–11 of
docs/RECURSIVE_CHANCE_DEVELOPMENT.md ran on `main`. Nothing on `main` was
edited. All work is on branch `claude/snapshot-and-hardening`, in the separate
worktree `../ipomcp-claude`.

## Branch contents

1. `3578b00` is a verbatim snapshot of the uncommitted `main` working tree.
   - All 78 `src/**/*.py` files and the launcher match master contract
     `7a1bb7bd…9c63`.
   - It preserves the bytes that the source hashes only verify. Before this
     commit, the candidate solver existed only as uncommitted files.
2. `f9e5e0e` enforces the source lock inside every supervised trial.
   - Workers are spawned, so each trial re-imports `src` from disk. A hash taken
     at launch therefore did not bind a multi-hour panel.
   - Replay equality would have caught edits to shared search code. It would
     not have caught an edit made only to `solvers/recursive_chance.py`.
   - `supervise_jobs(..., source_lock=(root, hashes))` now verifies the lock in
     the worker before and after each job.
   - This changes `src`, so every existing frozen lock is expected to reject
     this branch. Do not use the branch to replay frozen contracts.
3. `cf257eb` adds `benchmarks/audit_recursive_estimator_mixing.py`.
   - It is a read-only audit of whole-action fallback. That fallback keeps the
     empirical search Q beside sibling actions that are rescored with
     probabilities.
   - The script sits outside `src` and does not affect any lock.

Merge only after Antigravity's sweep and Codex's audit are complete. New
contracts then need a fresh source lock.

Testing caveat: the shared `.venv` has the package installed pointing at the
`main` tree's `src`. To test the worktree, set `PYTHONPATH=$PWD/src`.
Supervisor and lock tests: 12 passed. Recursive and root-chance focused tests:
42 passed. Ruff passes. The full suite was not run, to avoid competing for CPU
with the running panels.

## Panel 0 estimator-mixing audit (descriptive)

- **Root:** in all 240 of 240 triples, every root action was rescored. No root
  decision compared a fallback action with a rescored one.
- **Internal histories:** 34,680 of 426,531 action histories (8.1%) mix the two
  estimators, and 7,559 use fallback for every action.
- **Mixed-class comparisons:** in 34,669 histories, the greedy action and its
  closest competitor use different estimators. These internal maxima feed the
  root values.
- **What this does not show:** it is a count, not a measured harm. Whether mixing
  affects root decisions is untested. Run the audit on panels 1–11 when they
  complete.

## Open issues for Codex and the user

- **No rule for acting on the result.** Decide which result from panels 1–11
  leads to a fresh gate, to dropping the candidate, or to more development.
  Write that down before the audit, not after.
- **Gate power and seed spending.** The last fresh gate failed on 1 violation
  out of 3,360 cases, and 0 versus 1 at that size can't be told apart
  statistically. Each failure has produced a new mechanism designed on the
  failing case, and fresh seed sets keep turning into development data.
  - Consider a rate criterion with a confidence bound for future gates, together
    with a budget of remaining fresh gates.
  - Historical criteria stay frozen.
- **Generality.** Full coverage of a root makes recursive scoring equal to exact
  finite-tree value iteration. That is feasible for Tiger L2. Its value in
  domains where enumeration isn't possible (L4, UAV, Wumpus) is untested.
- **Bias that remains.** Taking the max over thinly visited actions still causes
  optimizer's-curse bias (see HANDOFF case 5). Chance integration alone can't
  remove it. Maximum Q error keeps getting worse (panel 0: 4.40 → 4.98).
- **Raw evidence has no backup.** `results/` holds 5.5 GB, is git-ignored, and
  exists only on the WSL disk.
- **Clock discrepancies.** Differences between external, Unix and monotonic
  timings recur and have never been explained.
- **Governance.** The runner created and froze the master contract before
  review. The mathematician should own contract design.
