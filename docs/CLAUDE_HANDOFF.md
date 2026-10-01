# Claude handoff — September 30, 2026 (21:45 CDT)

Read this first in a fresh session. From now on Claude is the primary agent for
experimentation and improvement of this codebase. Codex (mathematics, audits) and
Antigravity (experiment runner) will wrap up their current work. Their note is
`docs/NOTE_TO_CODEX_ANTIGRAVITY.md`.

## Where everything is

- **Do not edit `main` yet.**
  - `/home/andyj1810/projects/ipomcp` is `main` at 62d6cd8 with a large uncommitted working tree. That tree belongs to Codex and Antigravity.
  - Antigravity's recursive-chance sweep reloads `main/src` on every case, so any edit to `main/src` changes the running experiment.
  - Sweep status at 21:45: panels 0–2 complete, panel 3 at 57 of 240 cases, panels 4–11 still to run (6–11 are 560 triples each).
- **All Claude work lives in the worktree `/home/andyj1810/projects/ipomcp-claude`.**
  - Branch `claude/snapshot-and-hardening`: a verbatim snapshot of `main`'s working tree (3578b00), per-trial source locks, the estimator-mixing audit, and the supervisor review.
  - Branch `claude/l3-l2-tiger-experiment`: everything below. It is checked out in the worktree; HEAD 9ec85fb.
- **Environment.**
  - Run with `PYTHONPATH=$PWD/src ../ipomcp/.venv/bin/python …` from the worktree. The shared `.venv` otherwise imports `main`'s `src`.
  - Results live under `results/` in the worktree and are git-ignored.
- **Reviewer report.** A Claude Doc, "I-POMCP vs Exact I-POMDP: L1–L3 Tiger Results": https://claude.ai/code/artifact/96fa514f-dc7c-477c-8f97-b31eb1b8fca7
  - The review period is over. Treat the doc as a snapshot, not the source of truth.
- **Long-form write-ups.**
  - `docs/L3_L2_TIGER_STUDY.md`: the study's design, validation and problems.
  - `docs/SUPERVISOR_REVIEW_20260930.md`: process findings on the Codex/Antigravity work.

## Goals

1. **Headline experiment.** Two-agent Tiger, 20 steps, at least 100 trials.
   - Physics: growl .85, creak .90. Fixed 5-step rolling lookahead, gamma .95.
   - Agent i is L3 with prior over j's level {L1 .8, L2 .1, L0 .1}. Agent j is L2 with prior over i's level {L1 .5, L0 .5}.
   - Tiger prior is exactly 50/50 at every nesting level.
   - Deliverables: sunbursts of each agent's nested opponent model over time, and both agents' cumulative rewards on one graph.
2. **Error quantification.** Measure how far I-POMCP is from the exact I-POMDP on rewards and on opponent modeling at every nesting level.
   - Some error is tolerated, but it must be recorded and shown to shrink as resources grow.
3. **Process integrity.** Keep provenance intact: source locks, a preserved source snapshot, and no edits under running experiments.

## What exists (branch `claude/l3-l2-tiger-experiment`)

| File | Purpose |
| --- | --- |
| `src/examples/experiments/tiger_l3_vs_l2_experiment.py` | Experiment runner: declared design, both agents' metrics, `build_agent`, `--max-support` |
| `src/solvers/solver_bank.py` | `set_physical_prior` (exact prior); `SolverBank(max_support=...)` |
| `src/ipomdp/finite_filter.py` | Opt-in bounded support: stratified deterministic resampling, `resample()` |
| `src/solvers/exact/finite_policy_reference.py` | Exact finite-horizon best response for any level and mixed opponent levels |
| `src/utils/source_lock.py`, `process_supervisor.py`, `generic_batch_runner.py` | Per-trial source lock, checked before and after every job |
| `benchmarks/exact_ipomdp_agent.py` | Exact I-POMDP agent, recursively exact down to L0 |
| `benchmarks/l3_l2_convergence.py` | Resource-ladder runner (`--rung r1..r4`) with capped exact shadow audits |
| `benchmarks/l3_l2_ladder_analysis.py` | On-history error, outcome convergence, coverage |
| `benchmarks/l3_l2_figures.py` | Sunburst grids (PNG/PDF), plotly slider, cumulative rewards, level posteriors |
| `benchmarks/validate_l3_l2_exact.py` | Planner vs exact reference at chosen horizons, replaying pilot histories |
| `benchmarks/l1_pomcp_vs_exact.py`, `l1_summary.py` | L1 I-POMCP vs alpha-vector value iteration |
| `benchmarks/run_l3_l2_ladder.sh`, `rebalance_*.sh` | Ladder orchestration (see Running now) |
| `benchmarks/audit_recursive_estimator_mixing.py` | Read-only audit for Codex's recursive-chance runs |
| `tests/test_l3_l2_experiment.py`, `test_l3_l2_figures.py`, `test_source_lock.py` | 22 + 3 + 6 tests |

Planner settings used throughout (the "validated" profile, named so in code):
- `backup="empirical_bellman"`
- `exact_final_step=True`
- `exact_history_rewards=True`
- `HorizonBoundUCB(-100, 10, c=1)`

## Results so far

- **Level 1, I-POMCP vs exact value iteration** (1,320 comparisons; `results/tiger/l1_vs_exact/`).
  - Validated profile: 100% exact-optimal from 1k simulations at every horizon 1–5.
  - Horizon-5 mean max Q error: 2.89, 1.01, 0.58, 0.43 at 100, 1k, 10k, 50k simulations.
  - The repository's default profile (sampled backups, NormalizedUCB c = √2) is 61–88% optimal at horizons 3–5 at every budget. Its Q error is 40–72 and does not shrink. The repo's global defaults still use this profile.
- **L2/L3 root decisions vs the finite-policy reference** (`results/tiger/validation/`).
  - Validated profile: 30 of 30 optimal at horizons 3–5.
  - Default profile: 3/5 and 4/5 optimal at horizons 3–4, loss up to 8.26.
- **Nested-belief growth is exponential.**
  - j's exact L2 support: 6, 144, 624, 2,428 atoms at steps 1, 8, 12, 16. Merging within 1e-3 barely helps; only a 1e-2 grid caps it (about 180 atoms).
  - Exact L3 decision time: 12, 19, 61, 160 s over steps 1–4.
- **Interim resource ladder (19:45; `results/tiger/ladder_20260930/interim_1945/`).**

  | Rung | Level-belief TV, i | Level-belief TV, j | Modeled-policy TV, i | Modeled-policy TV, j |
  | --- | --- | --- | --- | --- |
  | r1 | .164 | .159 | .125 | .093 |
  | r2 | .144 | .116 | .085 | .109 |
  | r3 | .025 | .022 | .012 | .053 |
  | r4 | .030 | .010 | .013 | .018 |

  - j's exact regret is .17, .16, .17 at r1–r3 and 0 at r4. Its search regret stays at or below .003, so j's loss is opponent modeling, not search.
  - Interim r3 sunbursts (22 trials): i settles on "j is L1" at 82–88%, with L2 rising to 18% late in the episode. j settles on "i is L1" at 85–99%.
- **Panel-0 estimator-mixing audit for Codex** (recursive chance).
  - No root decision mixes fallback and rescored estimators (0 of 240).
  - 8.1% of internal histories do mix them (34,680 of 426,531).

## Running now (detached; survives the session)

- Ladder runners in `results/tiger/ladder_20260930/{r1,r2,r3,r4}`. Each targets 100 trials; trials are written to `trials/*.csv`, failures to `attempts/*/trial-*.json`. Progress is in `progress.log`.

  | Rung | Workers | Per-trial limit |
  | --- | --- | --- |
  | r1 | 2 | 4 h |
  | r2 | 3 | 4 h |
  | r3 | 8 | 4 h |
  | r4 | 8 | 10 h |

- At 21:42: r1 44, r2 25, r3 23, r4 7 trials done.
- Six r4 trials hit the old 4-hour limit; they are being rerun with 10 hours. Dropping them would bias r4 toward easy episodes.
- Watcher `benchmarks/rebalance_after_r1.sh` restarts r2 with 5 workers once r1 completes.
- **Check:**

  ```bash
  cd /home/andyj1810/projects/ipomcp-claude/results/tiger/ladder_20260930
  for r in r1 r2 r3 r4; do echo $r $(ls $r/trials/*.csv | wc -l); done
  tail progress.log
  ```

- **Resume** after a stop: rerun the exact command in `run_l3_l2_ladder.sh` with the same `--output`.
  - `run_batch` resumes pending trials while source and parameters match.
  - Worker count, timeout and RSS limit may change between resumes.
- **Stop:** for each runner, kill the process group of each child worker (`pgrep -P <parent>`, then `kill -KILL -- -<child>`), then `kill -TERM <parent>`.
- **Analyze** (reads CSVs only; safe while running):

  ```bash
  cd benchmarks
  PYTHONPATH=../src ../../ipomcp/.venv/bin/python l3_l2_ladder_analysis.py ../results/tiger/ladder_20260930/r{1,2,3,4} --output <dir>
  PYTHONPATH=../src ../../ipomcp/.venv/bin/python l3_l2_figures.py ../results/tiger/ladder_20260930/r3 --output <dir>
  ```

## Failures and lessons (do not repeat)

**Methods**
1. **Unverified planner profile.** The default planner profile made wrong decisions against the exact solution and collapsed level beliefs to L0 after early door openings. Validate any planner setting against an exact reference before using it.
2. **Unbounded exact filtering.** Every 20-step L3 trial with exact nested beliefs ran out of memory at 3–4 GiB. Long episodes need `max_support`.
3. **Plain resampling.** At K = 32 it dropped every L0 atom, so the next unexpected opening had zero evidence (`UnsupportedObservation`). Stratify by (level, state), as now.
4. **Runtime estimates from 2-trial probes.** They were 3x too optimistic once 24 cores were contended. Measured: r4 averages about 2.3 h per trial and 6 of 12 trials took over 4 h; r3 averages about 1 h. Size runs from per-trial wall times under real load, and leave slack.
5. **Exact-audit coverage for L3.** The decision audit covers only about 3 steps per trial (the exact solve passes 120 s by step 3–4). The belief audit reaches about step 8. Never extrapolate past coverage.

**Process and shell**
6. **`pgrep`/`pkill -f` matching your own shell.** This killed my own shell three times, because the pattern text also appeared in my command line. Kill by PID, or put the kill logic in a script file.
7. **A trailing `&` on `cd X && a && b &`** backgrounds the whole chain; later commands then run in the original directory (here `main`). Use `(cd X && cmd &)` or separate calls.
8. **SIGINT is ignored by background jobs.** Processes started with `&` don't respond to it. Kill worker process groups, then send TERM to the parent.
9. **Adding or editing a `src/*.py` file while your own runs are active** fails those runs' per-trial source lock. Keep new code in `benchmarks/` until runs finish.
10. **The shared `.venv` imports `main`'s source.** Always set `PYTHONPATH` to the worktree's `src`.
11. **Artifact uploads** must come from the session scratchpad. Monitors expire every 30 minutes and must be re-armed.

**Theory and spec still open**
12. **Creak law.** This repo's creak law has no false creaks (a listening opponent is always silent). I did not verify the Gmytrasiewicz–Doshi specification. Check the paper before claiming equivalence.
13. **Level is confounded with computation.** Modeled opponents are finite MCTS. Level posteriors measure how well those finite models predict, not true reasoning depth.

## Next steps (priority order)

1. **Finish the ladder.**
   - Collect when complete (or by count): run the analysis and figures, fill the TBD Results in `docs/L3_L2_TIGER_STUDY.md`, and update the report if wanted.
   - Report r4 with its n.
   - Pair outcome convergence only on shared trial ids.
2. **Cut the level-3 cost.** The modeled-opponent solve cascade dominates, not the root budget. Profile it, then consider:
   - reusing modeled policies across real steps (the bank clears caches at every `update_root`);
   - a shallower modeled-opponent depth, declared in the exact counterpart too;
   - sharing nested model solves across tree nodes.
3. **Diagnose j's modeling regret** (.16–.17 at r1–r3). Separate the contributions of modeled-policy budget, K, and misspecification (the real i is L3).
4. **Extend i's exact coverage beyond step 3–4,** for example with a cheaper exact L3 solver using behavioral equivalence of nested models.
5. **Optional opt-in creak variant** with false creaks, if the Gmytrasiewicz–Doshi specification requires it.
6. **After Codex and Antigravity finish:**
   - Merge `claude/snapshot-and-hardening`, then `claude/l3-l2-tiger-experiment`, into `main`.
   - All existing frozen source locks will stop matching. That is expected; new contracts need new locks.
   - Rerun the full test suite (only focused suites were run).
   - Back up `results/` outside WSL.
7. **Propose changing the global planner defaults to the validated profile** (needs Codex's qualification protocol).
