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
| R5: interactive particle filter | done (L2 gate passes; see RESULTS.md Section 3) |
| R6: compute balance | cross-step reuse done; allocation study written, results pending |
| R7: IPF + RTS baseline as published | not started |
| R8: analysis layer (metrics once, sunbursts with content and error) | not started |
| R9: scale-up ladder (Tiger L1 → L4, then UAV, then Wumpus) | not started |

## Next steps, in order

1. **Finish R6.**
   - Read the allocation study and write a rule for dividing a budget between root and modeled simulations.
   - Consider per-level modeled budgets, since lower levels are cheaper and more numerous.
2. **R7.** IPF + reachability-tree sampling as in Doshi & Gmytrasiewicz (2009):
   - sampled observation branching;
   - a matched budget unit (generative calls and CPU seconds);
   - the same reward integration for both planners;
   - add it as a `planner_kind` in `experiments.oracle`.
3. **Wumpus.** Add IPF reinvigoration (deterministic sensors make particle beliefs lose the true world).
4. **R8.** One metrics layer that computes everything once from run records:
   - sunbursts annotated with each modeled agent's belief, predicted action and error;
   - confidence-aware best responses in place of float-equality "equilibria".
5. **The experiment runner.**
   - Replace hand-written `_setup_domain` subclasses with one declarative specification per study.
   - Record both agents' CPU per component (already in the runner).
6. **R9.** Every rung passes its gate before the next starts.

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
