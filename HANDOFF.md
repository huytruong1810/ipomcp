# I-POMCP Multi-Level Reasoning: Project Status & Agent Handoff

**Handoff Date**: September 11, 2026  
**Repository Remote**: [https://github.com/huytruong1810/ipomcp](https://github.com/huytruong1810/ipomcp)  
**Branch**: `main` (Synced with origin/main at `e643686`; working tree clean)  
**Test Suite Status**: **44 / 44 Passing** (`uv run python -m pytest tests/ -q`, 81.95s execution)  
**Codebase Audit Status**: Complete report and prioritized action items compiled in [`BACKLOG.md`](BACKLOG.md).

---

## 1. Active Long-Running Benchmark (CRITICAL: DO NOT TERMINATE)

A master large-scale benchmark is actively executing in the background, resumed after the September 11 system reboot.

### Process Telemetry
- **Task ID**: `6df503eb-35ad-4600-a30e-981c5e0955db/task-305`
- **Command**:
  ```bash
  uv run python src/examples/experiments/run_all_large_scale_benchmarks_N200.py --trials 200 --steps 20 --suite all --resume-dir results/deep_prior/deep_prior_benchmark_20260909_163530_N200_T20
  ```
- **Process Hierarchy**:
  - Runner Process (`python3`): PID `34815`
  - Active Worker Processes: Dispatched via `ProcessPoolExecutor` (deadlock-free worker lifecycle)
- **Current Suite**: Suite 1/3 (Deep Hierarchy Prior Benchmark, 7 conditions $\times$ 200 trials, $T=20$, Lv1-Lv4)
- **Current Progress**:
  - Condition 1 (`L3 vs L2, 80% L2 Prior`): Actively executing.
- **Log Location**:
  `~/.gemini/antigravity-cli/brain/6df503eb-35ad-4600-a30e-981c5e0955db/.system_generated/tasks/task-305.log`
- **Results Directory**:
  `results/deep_prior/deep_prior_benchmark_20260909_163530_N200_T20/`

---

## 2. Key Architecture & Benchmark Hardening Accomplishments

1. **Pruned Level-5 Reasoning**:
   - Pruned Level-5 from all suites, focusing experimental statistical power on Levels 0 through 4 where asymptotic convergence is definitively demonstrated.
   - Deep Prior suite pruned to 7 conditions (Lv1-Lv4).
   - Payoff matrix standardized to 5x5 ($25\text{ cells} \times 200\text{ trials} = 5,000\text{ episodes}$, Lv0-Lv4).
2. **100% Deterministic Creak Observation (`creak_accuracy = 1.0`)**:
   - Fixed generative and evaluative creak observation calculations in [`TigerModel`](src/examples/tiger/model/tiger_model.py), eliminating negative probabilities (`-0.05`) and the 10% unobserved door resets that caused severe negative reward cascades.
3. **Escalating Particle Schedule ($2,500 \times \text{level}$)**:
   - Centralized in [`src/core/config.py`](src/core/config.py) to cover the combinatorial union of nested lower-level opponent models: L1: 2.5k, L2: 5k, L3: 7.5k, L4: 10k particles.
4. **Near-Optimal Simulation Schedule ($50\text{k} \times \text{level}$)**:
   - Re-introduced high simulation counts in [`src/core/config.py`](src/core/config.py): L1: 50k, L2: 100k, L3: 150k, L4: 200k simulations.
5. **Upgraded JIT Configuration**:
   - Centralized in `JITConfig`: `sims=50` (was 10), `visit_threshold=10` (was 5), `entropy_threshold=0.5` (was 0.6).
6. **Hardened Batch Pipeline & Atomic Resume Verification**:
   - In [`GenericBatchRunner`](src/utils/generic_batch_runner.py), incomplete batches save to `batch_results_partial.csv` and raise `RuntimeError` rather than claiming completion.
   - Atomic `is_batch_complete(csv_path, expected_trials)` replaces fragile file size checks across all runners.
   - Suite resume directories are strictly isolated, preventing cross-suite directory collisions.

---

## 3. Next Agent Action Plan

When you take over this workspace:

1. **Verify Background Task Progress**:
   - Check status using `manage_task(status, TaskId="cf2b6ec9-8616-487b-8fbd-2ad75abc06ec/task-1552")`.
   - Tail the log:
     ```bash
     tail -n 25 ~/.gemini/antigravity-cli/brain/cf2b6ec9-8616-487b-8fbd-2ad75abc06ec/.system_generated/tasks/task-1552.log
     ```
   - Monitor the 3 suites as they complete:
     1. Deep Prior Benchmark (7 conditions $\times$ 200 trials)
     2. Apples-to-Apples Oracle RTS Benchmark (7 conditions $\times$ 200 trials)
     3. 5x5 Payoff Matrix Benchmark (25 cells $\times$ 200 trials)

2. **Code Quality & Testing Contract**:
   - Full test suite:
     ```bash
     uv run python -m pytest tests/ -v
     ```
   - All 44 tests must remain passing.

3. **Preserved Artifact Directory**:
   - Active results are saved under `results/deep_prior/`, `results/oracle/`, and `results/payoff_matrix/`.

