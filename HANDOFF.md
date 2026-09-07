# I-POMCP Multi-Level Reasoning: Project Status & Agent Handoff

**Handoff Date**: September 6, 2026  
**Repository Remote**: [https://github.com/huytruong1810/ipomcp](https://github.com/huytruong1810/ipomcp)  
**Branch**: `main` (Clean working tree; no local code divergence)  
**Test Suite Status**: **38 / 38 Passing** (`uv run python -m pytest tests/ -v`, ~100s execution)  
**Codebase Audit Status**: Complete report and prioritized action items compiled in [`BACKLOG.md`](BACKLOG.md).

---

## 1. Active Long-Running Benchmark (CRITICAL: DO NOT TERMINATE)

A master large-scale benchmark is actively executing in the background. The user has explicitly requested to keep this session and task running.

### Process Telemetry
- **Command**:
  ```bash
  uv run python src/examples/experiments/run_all_large_scale_benchmarks_N200.py --trials 200 --steps 20 --suite all --resume-dir results/deep_prior/deep_prior_benchmark_20260904_010926_N200_T20
  ```
- **Process Hierarchy**:
  - Main Wrapper (`uv`): PID `150519`
  - Runner Process (`python3`): PID `150522`
  - Active Worker Processes: PIDs `155492`, `155493` (running at 93–97% CPU utilization)
- **Current Suite**: Suite 1/3 (Deep Hierarchy Prior Benchmark, 10 conditions $\times$ 200 trials, $T=20$)
- **Current Progress**:
  - Condition 1 (`L3 vs L2, 80% L2 Prior`): Completed & verified.
  - Conditions 2, 3, 4 (`L3 vs L1` variations): Completed & skipped via checkpoint.
  - **Condition 5 (`L4 vs L3, 80% L3 Prior`)**: **Currently running at 80 / 200 trials completed**.
- **Log Location**:
  `~/.gemini/antigravity-cli/brain/cf2b6ec9-8616-487b-8fbd-2ad75abc06ec/.system_generated/tasks/task-823.log`
- **Why It Must Not Be Interrupted**:
  Batch runners write `batch_results.csv` upon batch completion. If killed mid-condition, the 80 completed trials of Condition 5 would be lost and must restart from trial 0.

---

## 2. Recent Engineering Accomplishments

1. **$O(NL)$ Canonical Root Pooling**:
   - Resolved the exponential $O(N^L)$ particle duplication in [`src/utils/bootstrapper.py`](src/utils/bootstrapper.py#L259-L272) by sharing canonical root search trees across particles with identical reasoning levels.
   - Reduced Level-4/Level-5 RAM footprint from **14.8 GB** down to **< 200 MB** per process.
2. **OS Telemetry & Memory Watchdog Suite**:
   - Built [`src/core/telemetry.py`](src/core/telemetry.py) (`SystemMonitor`, `MemoryWatchdog`, `TelemetryLogger`) with real-time RSS, VMS, and host RAM detection.
   - Integrated `process_rss_mb` tracking directly into batch result CSVs.
   - Added 5 unit tests in [`tests/test_telemetry.py`](tests/test_telemetry.py); all passing.
3. **CPython Allocator Arena Creep Mitigation**:
   - Added `max_tasks_per_child=20` to `ProcessPoolExecutor` in [`src/utils/generic_batch_runner.py`](src/utils/generic_batch_runner.py#L359), ensuring worker processes are recycled periodically to prevent small-object memory fragmentation.
4. **Resumption & Checkpoint Architecture**:
   - Implemented `--resume-dir` and batch CSV inspection across all suites, enabling graceful recovery from interruptions.
5. **Git Version Control & Repository Hygiene**:
   - Initialized Git, linked to [huytruong1810/ipomcp](https://github.com/huytruong1810/ipomcp), and pushed initial commit.
   - Cleaned local repository (`git gc --prune=now`), reducing `.git/` to 348 KB with zero local file duplication.

---

## 3. Systematic Codebase Review ([`BACKLOG.md`](BACKLOG.md))

A line-by-line audit across all 7 architectural phases was completed and documented in [`BACKLOG.md`](BACKLOG.md). Key findings include:

| Priority | Issue | Affected File | Impact |
| :---: | :--- | :--- | :--- |
| **P0** | Division-by-Zero / NaN on unbounded ranges | [`src/solvers/exploration.py`](src/solvers/exploration.py#L86-L105) | Action selection in `NormalizedUCB` degrades to random if `q_max <= q_min`. |
| **P0** | Caller dict mutation & zero-weight crash | [`src/core/distribution.py`](src/core/distribution.py#L150-L165) | `DictDistribution` mutates caller argument; crashes if weight sum is 0.0. |
| **P0** | Opponent particle starvation in MCTS | [`src/solvers/generative_model.py`](src/solvers/generative_model.py#L103-L110) | Existing child nodes receive only 1 initial particle, collapsing opponent beliefs. |
| **P1** | Missing action masking in UAV domain | [`src/examples/uav/model/uav_model.py`](src/examples/uav/model/uav_model.py#L49-L54) | Out-of-bounds boundary moves are explored as valid MCTS branches. |
| **P1** | Missing RNG injection in UAV & Wumpus | `uav_model.py`, `wumpus_model.py` | Global `random` module bypasses Common Random Numbers (CRN). |
| **P1** | Coarse-grained condition-level checkpointing | [`src/utils/generic_batch_runner.py`](src/utils/generic_batch_runner.py#L349-L385) | Interrupted batches lose in-flight trials. |
| **P2** | `InteractiveParticle` immutability violation | [`src/ipomdp/belief.py`](src/ipomdp/belief.py#L42-L67) | Contains mutable dict; unhashable in sets/dicts. |

---

## 4. Next Agent Action Plan

When you take over this workspace:

1. **Verify Background Task Progress**:
   - Check process vitals: `ps -fp 150522` and check CPU usage: `top -p 155492,155493`.
   - Inspect the latest progress log:
     ```bash
     tail -n 25 ~/.gemini/antigravity-cli/brain/cf2b6ec9-8616-487b-8fbd-2ad75abc06ec/.system_generated/tasks/task-823.log
     ```
   - Monitor Condition 5 until it reaches `200/200` trials and progresses to Condition 6 (`L4 vs L1, 80% Over-estimated L3 Prior`).

2. **Review [`BACKLOG.md`](BACKLOG.md)**:
   - Discuss findings with the user/reviewers before modifying code.
   - Plan implementation of **P0 items** (NormalizedUCB guard, DictDistribution hardening, generative model particle replenishment).

3. **Code Quality & Testing Contract**:
   - **Never edit code without running the full test suite afterwards**:
     ```bash
     uv run python -m pytest tests/ -v
     ```
   - All 38 tests must remain passing.

4. **Preserved Artifact Directory**:
   - All completed results live under [`results/`](results/).
   - Prior benchmark runs from August 2026 are preserved in `results/deep_prior/`, `results/oracle/`, and `results/payoff_matrix/`.
