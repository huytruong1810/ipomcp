# Phase 0 — Baseline (worktree `claude/l3-l2-tiger-experiment`, September 30)

## Environment

- **Python:** 3.12.13 from the shared `../ipomcp/.venv`, with `PYTHONPATH=$PWD/src`. Without that override the venv imports `main`'s source.
- **Project:** `i-pomcp-py` 0.1.0, built with setuptools from the `src/` layout. Its packages are `core`, `ipomdp`, `solvers`, `utils` and `examples`, and they install as top-level packages, so generic names like `core` and `utils` can clash.

## Size

| Area | Files | Lines |
|---|---|---|
| `src/` | 76 `.py` (16 empty `__init__`) | about 11,600 |
| `tests/` | 35 `.py` | 4,610 |
| `benchmarks/` | 17 scripts and 3 shell scripts, plus about 30 frozen JSON contracts | 2,168 (code) |
| `docs/` | 22 Markdown files | — |

## Checks

- `ruff check src tests benchmarks`: clean.
- `ruff format --check`: clean (130 files).
- `pytest`: **329 passed in 437 s.**
  - `test_integration.py::test_wumpus_batch_runner_short` alone takes 361 s.
  - The next slowest test takes 13 s.
  - The ruleset is minimal (`E4, E7, E9, F, I`). It has no `B` (bugbear), `UP`, `SIM`, `D` (docstrings) or `ANN` rules, and no type checker is configured.

## What the passing suite does not show

- It never checks that the default planner profile converges. The L1 study shows it doesn't (Phase 3 T1).
- It never tests that bounded resampling is independent of insertion order, nor that the stated error bound holds. Both fail (Phase 2 B1–B2).
- It never runs Wumpus with a real planner long enough to hit zero evidence (Phase 5 B1).
