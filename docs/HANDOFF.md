# Handoff

Read this first. Then read [ARCHITECTURE.md](ARCHITECTURE.md), [THEORY.md](THEORY.md) and [RESULTS.md](RESULTS.md).

## Where things are

- **Code.** Branch `claude/restructure`, in the worktree `../ipomcp-claude`.
  - It descends from a full review of the previous codebase; the findings are in [review/](review/).
  - The pre-restructure code, frozen validation gates and lab-notebook docs are preserved under the tag `archive/pre-restructure-20260930`.
- **`main` has not been touched.** It still holds Codex's and Antigravity's uncommitted working tree and a running recursive-chance sweep. Merge only after they finish (see below).
- **Run.** From the worktree, run `PYTHONPATH=$PWD/src ../ipomcp/.venv/bin/python …`. The shared venv otherwise imports `main`'s source.
- **Results.** Stored under `results/` (git-ignored). A full copy of `main/results` and the worktree results from September 30 is in `C:\Users\Andyj\ipomcp-results-backup-20260930`.

## Restructure status (plan in review/phase-11-synthesis.md, Section 5)

| Step | Status |
|---|---|
| R1: one `ipomdp` package; legacy drivers, replays and frozen gates removed | done |
| R2: the validated estimator is the only planner | done (L1 gate passes) |
| R3: one domain-agnostic exact reference | done (equal to the old exact L2 solver to 1e-15) |
| R4: kernel-only domain contract; published Tiger; revised UAV; exact Wumpus prior; explicit RNG | done |
| R5: interactive particle filter | done; systematic resampling; L2 gate in RESULTS.md Section 3 (mean shrinks with N; a worst-case floor of about 0.05 level TV remains at N = 4,096) |
| R6: compute balance | done: cross-step reuse; allocation rule in RESULTS.md Section 5 (modeled budget first) |
| R7: RTS baseline | done: `baselines.rts`; `experiments.l2_panel` compares I-POMCP and IPF + RTS against the exact I-POMDP (RESULTS.md Section 6) |
| R8: analysis layer | done: `analysis.hierarchy`, `metrics`, `figures` (sunbursts with content and error) |
| Runner | done: TOML `StudySpec`, symmetric `episodes`, supervised `runner` |
| R9: scale-up ladder (Tiger L1 → L4, then UAV, then Wumpus) | not started |

## Next steps, in order

1. **Contribution 2: where the IPF pays off.** Mid-episode results are in RESULTS.md Section 7. Wherever the exact I-POMDP is tractable (L2 H3, k ≤ 4), exact beliefs are faster or more accurate than the IPF; the IPF only wins where exact planning and the reference become intractable (k ≥ 6). Open questions:
   - lower the IPF's in-tree error (it resamples the opponent afresh at every in-tree update);
   - find a reference that scales past k = 4, so accuracy can be measured where the IPF is needed: for example exact filtering of i's belief, plus exact best response with a bounded nested enumeration;
   - L3, and H4 with 4,000 modeled simulations on IPF beliefs at N ≥ 1,024.
2. **Per-level modeled budgets.** The allocation rule says modeled agents must
   be accurate at the beliefs they are modeled at; lower levels are cheaper and
   more numerous, so a per-level budget may buy that accuracy more cheaply.
3. **Wumpus.** Add IPF reinvigoration from the exact prior conditioned on the
   agent's own history (deterministic sensors make particle beliefs lose the
   true world).
4. **R9.** Every rung passes its gate before the next starts. Report IPF
   hierarchy errors with N ≥ 1,024, mean and maximum.

## Merging into main

Only after the user confirms that Codex and Antigravity are finished:

1. Commit or archive their working tree on `main`.
2. Merge `claude/snapshot-and-hardening`, `claude/l3-l2-tiger-experiment` (already contained), and `claude/restructure`.
3. Run the full suite.
4. Old frozen source locks will no longer match. That is expected.

## Lessons (do not repeat)

- **Never `pkill -f` with a pattern** that also appears in your own command line. Kill by PID.
- **Do not edit `src/` while runs that re-import it are active.** That includes formatting.
- **Validate every planner setting against an exact reference** before using it in a study. The old default estimator passed every test while being 61–88% optimal.
- **Size runs from measured per-trial CPU** under realistic load, not from 2-trial probes.
- **Report results with their n** and the reference they were scored against (see THEORY.md, "Three meanings of exact").
