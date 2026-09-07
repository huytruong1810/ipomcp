# Codebase State Assessment & Engineering Backlog (BACKLOG.md)

**Date**: September 6, 2026  
**Auditor**: Antigravity Autonomous Systems Architecture Review  
**Target Repository**: [huytruong1810/ipomcp](https://github.com/huytruong1810/ipomcp)  
**Scope**: Complete System Scan — Core Math, Recursive Belief Engine, Solvers, Orchestration, Domains, Experiments, and Observability.

---

## Executive Summary & Candid Verdict

### Is the codebase in a "Prime State"?
**Verdict: Near-Production / High-Maturity Research Grade, but NOT yet in a Flawless "Prime State".**

While the codebase has undergone significant hardening (e.g., resolving the exponential $O(N^L)$ particle explosion, establishing Common Random Numbers, building OS-level telemetry, and maintaining a 38/38 passing test suite), a critical, line-by-line inspection reveals several **subtle theoretical assumptions, edge-case failure modes, and architectural trade-offs** that must be resolved before claiming true zero-defect prime status.

### Summary Scorecard

| Dimension | Grade | Status | Core Observations |
| :--- | :---: | :---: | :--- |
| **1. Theoretical Soundness** | **B+** | *Good with Gaps* | High mathematical fidelity in Tiger and RTS baselines; however, nested particle transition truncates opponent recursive beliefs past depth 1, and single-particle collapse occurs in nested MCTS subtrees. |
| **2. Architectural Cleanliness & DRY** | **A-** | *Strong* | Centralized `SolverBank`, unified `I_POMDP_Bootstrapper`, standardized `POMDPModel` base class, and clean separation between generative simulation and evaluative weighting. |
| **3. Execution Robustness & Memory** | **B+** | *Hardened* | Canonical root pooling resolved $O(N^L)$ RAM exhaustion ($14.8\text{ GB} \to <200\text{ MB}$); `max_tasks_per_child=20` addresses CPython allocator creep; minor edge cases in `NormalizedUCB` and `DictDistribution` remain. |
| **4. Logging & Observability** | **A** | *Industry Standard* | OS-level telemetry (`SystemMonitor`, `MemoryWatchdog`, RSS tracking in CSVs), Sunburst/Icicle nested belief visualizations, and structured multi-level logging. |

---

## Pillar 1: Theoretical Soundness & Mathematical Integrity

### 1.1 Verified Mathematical Strengths
1. **Generative-Evaluative Symmetry**:
   - In [`TigerModel`](file:///home/andyj1810/projects/ipomcp/src/examples/tiger/model/tiger_model.py#L86-L198) and [`WumpusModel`](file:///home/andyj1810/projects/ipomcp/src/examples/wumpus/model/wumpus_model.py#L155-L233), `sample_observation` (used for MCTS tree search) and `get_observation_prob` (used for Particle Filter weighting) are rigorously aligned. This eliminates the artificial particle divergence that historically plagued early I-POMDP implementations.
2. **Doshi & Gmytrasiewicz (JAIR 2009) RTS Baseline Fidelity**:
   - [`RTSPlanner`](file:///home/andyj1810/projects/ipomcp/src/solvers/rts_planner.py#L45-L336) faithfully reproduces Reachability Tree Sampling with exact backward induction over all observation branches, providing a verifiable ground-truth Oracle baseline for comparing Monte Carlo sampling convergence.
3. **Stochastic Universal Sampling (SUS)**:
   - [`ParticleDistribution.resample`](file:///home/andyj1810/projects/ipomcp/src/core/distribution.py#L78-L101) uses $O(N)$ low-variance systematic resampling rather than naive multinomial sampling, drastically reducing particle filter variance across long horizons.

### 1.2 Theoretical Shortcomings & Mathematical Edge Cases
1. **Opponent Mental Model Transition Truncation ($L \ge 2$)**:
   - **Location**: [`src/solvers/generative_model.py:106-109`](file:///home/andyj1810/projects/ipomcp/src/solvers/generative_model.py#L106-L109)
   - **Mechanism**: When transitioning physical state $s \to s'$, the opponent's child node is updated with:
     ```python
     p_sample = random.choice(node_ptr.belief_particles)
     child_j.add_particle(InteractiveParticle(state=s_next, models=p_sample.models))
     ```
   - **Theoretical Issue**: In an Interactive POMDP, the opponent $j$'s model of agent $i$ ($\theta_i$) must also transition based on agent $i$'s action and observation. By copying `p_sample.models` verbatim without transitioning the nested opponent models, nested beliefs are frozen in time relative to the physical state.
   - **Impact**: For $L \ge 2$, deeper MCTS rollouts evaluate opponents whose nested beliefs do not evolve dynamically during tree rollouts.

2. **Single-Particle Collapse in MCTS Opponent Trees**:
   - **Location**: [`src/solvers/generative_model.py:103-109`](file:///home/andyj1810/projects/ipomcp/src/solvers/generative_model.py#L103-L109)
   - **Mechanism**: When a child node `child_j` already exists (`child_j is not None`), no new particle is added. When created, only **one** particle is sampled from `node_ptr.belief_particles`.
   - **Theoretical Issue**: If this action-observation branch is traversed 1,000 times by MCTS, `child_j` retains only that single historical particle. When JIT expansion triggers `opp_solver.extend_search(child_j, ...)`, the opponent plans from a degenerate 1-particle Dirac-delta belief rather than a diverse belief distribution.
   - **Impact**: Opponent policy estimates in deeper branches suffer from severe particle deprivation.

3. **Leaf Node Particle Drop at Horizon Limit**:
   - **Location**: [`src/solvers/i_pomcp.py:77-115`](file:///home/andyj1810/projects/ipomcp/src/solvers/i_pomcp.py#L77-L115)
   - **Mechanism**: `node.add_particle(particle)` occurs at line 110 at the conclusion of `_simulate()`. If `depth >= max_depth` or `is_terminal`, `_simulate` returns `0.0` at line 78 before reaching line 110.
   - **Theoretical Issue**: Leaf nodes reached at max depth never accumulate particles through reservoir sampling. If the root advances toward a former leaf node via `update_root`, the node may suffer instant particle deprivation.

---

## Pillar 2: Architectural Cleanliness & Code Design

### 2.1 Strengths
1. **Centralized Dependency Injection via `SolverBank`**:
   - [`SolverBank`](file:///home/andyj1810/projects/ipomcp/src/solvers/solver_bank.py) cleanly isolates agent registries using immutable [`SolverKey(agent_id, level)`](file:///home/andyj1810/projects/ipomcp/src/solvers/solver_types.py#L18-L36) pairs, preventing solver duplication and enabling cross-agent lookup during rollouts.
2. **Unified Factory Pipeline**:
   - [`I_POMDP_Bootstrapper`](file:///home/andyj1810/projects/ipomcp/src/utils/bootstrapper.py#L26-L299) cleanly delegates all convenience constructors (`create_level1_solver`, `create_level2_solver`) to the general recursive `create_solver` method, upholding strict DRY principles.
3. **Formal Base Interfaces**:
   - [`POMDPModel`](file:///home/andyj1810/projects/ipomcp/src/core/pomdp_model.py) enforces a strict abstract contract for physics, transitions, rewards, and legal action queries.

### 2.2 Shortcomings & Design Smells
1. **`InteractiveParticle` Immutability Violation**:
   - **Location**: [`src/ipomdp/belief.py:42-67`](file:///home/andyj1810/projects/ipomcp/src/ipomdp/belief.py#L42-L67)
   - **Issue**: Defined as `@dataclass(frozen=True, slots=True)`, but contains `models: Dict[...]`. In Python, dictionaries are mutable and unhashable. As a result, `InteractiveParticle` cannot be used in sets or as dictionary keys (`TypeError: unhashable type: 'dict'`), and internal models can be mutated from outside.
2. **Unenforced `get_legal_actions` in Domains**:
   - **Location**: [`src/examples/uav/model/uav_model.py:49-54`](file:///home/andyj1810/projects/ipomcp/src/examples/uav/model/uav_model.py#L49-L54)
   - **Issue**: `UAVModel` fails to implement `get_legal_actions`, falling back to `get_all_actions`. MCTS evaluates out-of-bounds moves (e.g. moving North when at row 0) rather than pruning the branch, violating the design contract described in `pomdp_model.py`.
3. **Global `import graphviz` in Visualizer**:
   - **Location**: [`src/utils/visualizer.py:1`](file:///home/andyj1810/projects/ipomcp/src/utils/visualizer.py#L1)
   - **Issue**: Top-level unconditional import. If run in a headless environment lacking Graphviz binaries, the module fails on import rather than failing gracefully during graph rendering.

---

## Pillar 3: Execution Robustness & Memory Safety

### 3.1 Strengths
1. **$O(NL)$ Canonical Root Pooling**:
   - Refactored [`bootstrapper.py`](file:///home/andyj1810/projects/ipomcp/src/utils/bootstrapper.py#L259-L272) to share canonical root trees across particles of the same reasoning level. This eliminated the exponential $O(N^L)$ particle tree duplication, reducing memory consumption from **14.8 GB** down to **< 200 MB** per process.
2. **Worker Process Recycling (`max_tasks_per_child=20`)**:
   - Added `max_tasks_per_child=20` to [`ProcessPoolExecutor`](file:///home/andyj1810/projects/ipomcp/src/utils/generic_batch_runner.py#L359), ensuring worker processes are regularly terminated and spawned afresh, eliminating CPython small-object allocator (`pymalloc`) memory leakage.
3. **Resumption & Checkpointing**:
   - Implemented `--resume-dir` and batch CSV inspection across all experiment suites, preventing duplicate computation upon re-runs.

### 3.2 Shortcomings & Execution Bugs
1. **`NormalizedUCB` Floating-Point NaN on Unbounded Range**:
   - **Location**: [`src/solvers/exploration.py:86-105`](file:///home/andyj1810/projects/ipomcp/src/solvers/exploration.py#L86-L105)
   - **Issue**: When `bounds` is passed as `{'q_min': float('inf'), 'q_max': -float('inf')}` on a node with pre-visited actions, `q_range = (q_max - q_min) + self.epsilon` evaluates to `-inf`. Normalizing $Q$ yields `(q_raw - inf) / -inf = nan`. All action comparisons fail, causing action selection to fall back to uniform random.
   - **Remediation**: Guard with `if q_max <= q_min: q_range = 1.0; q_norm = 0.5`.

2. **`DictDistribution` Caller Mutation & Zero-Weight Crash**:
   - **Location**: [`src/core/distribution.py:150-165`](file:///home/andyj1810/projects/ipomcp/src/core/distribution.py#L150-L165)
   - **Issue**: Line 159 directly mutates the passed-in dictionary (`self._probs[item] = ...`). If `total == 0` (or dictionary is empty), normalization is bypassed, and `sample()` triggers `random.choices(..., weights=[0.0, ...])`, raising `ValueError: Total of weights must be greater than zero`.
   - **Remediation**: Copy input dictionary (`self._probs = dict(probabilities)`), and handle `total <= 0` by falling back to uniform weights.

3. **`ParticleDistribution.get_support()` Includes Zero-Probability Items**:
   - **Location**: [`src/core/distribution.py:114-121`](file:///home/andyj1810/projects/ipomcp/src/core/distribution.py#L114-L121)
   - **Issue**: `_lookup_cache` adds all particles regardless of whether their weight is `0.0`. `get_support()` returns all cached keys, violating its docstring contract.

---

## Pillar 4: Observability, Logging & Problem Detection

### 4.1 Strengths
1. **OS Telemetry Engine**:
   - [`src/core/telemetry.py`](file:///home/andyj1810/projects/ipomcp/src/core/telemetry.py) tracks `process_rss_mb`, `host_available_ram_mb`, and classifies OS regimes (`DRAM_BOUND_NORMAL`, `MEMORY_PRESSURE`, `SWAP_THRASHING`).
2. **Telemetry Integration in Experiment Results**:
   - `process_rss_mb` is captured at every step in `batch_results.csv`, providing direct visibility into memory growth over time.
3. **Advanced Visualizations**:
   - Full support for Plotly Sunburst diagrams, animated timestep scrubber sliders, and publication-ready vector PDFs for nested belief hierarchies.

### 4.2 Shortcomings & Observability Gaps
1. **String-Serialized Dicts in CSVs**:
   - In [`generic_batch_runner.py`](file:///home/andyj1810/projects/ipomcp/src/utils/generic_batch_runner.py#L305-L306), `action_values_i` is serialized as a string representation of a Python dict into the CSV. Reading large CSVs requires slow `ast.literal_eval` parsing in [`plotting.py:358-363`](file:///home/andyj1810/projects/ipomcp/src/utils/plotting.py#L358-L363).
2. **Coarse-Grained Checkpointing**:
   - Checkpoints occur at the **condition level** rather than the **trial level**. If a 200-trial condition crashes at trial 190, all previous 189 trials must be recomputed.
3. **Hardcoded Parameter Schedules Across Experiments**:
   - `SIM_SCHEDULE` and `PARTICLE_SCHEDULE` are hardcoded independently in `deep_hierarchy_prior_experiment.py` and `level_convergence_matrix_experiment.py` rather than centralized in `src/core/config.py`.

---

## Prioritized Remediation Backlog

### Priority 0: Critical (Theoretical Integrity & Runtime Safety)
- [ ] **P0-1**: Fix `NormalizedUCB` division-by-zero / NaN calculation when `q_max <= q_min` ([`exploration.py:86-105`](file:///home/andyj1810/projects/ipomcp/src/solvers/exploration.py#L86-L105)).
- [ ] **P0-2**: Prevent caller dictionary mutation and zero-weight crash in `DictDistribution` ([`distribution.py:150-165`](file:///home/andyj1810/projects/ipomcp/src/core/distribution.py#L150-L165)).
- [ ] **P0-3**: Resolve opponent particle starvation in `InteractiveGenerativeModel.tree_step` by replenishing `child_j.belief_particles` across multiple simulations ([`generative_model.py:103-110`](file:///home/andyj1810/projects/ipomcp/src/solvers/generative_model.py#L103-L110)).

### Priority 1: High (Convergence & Domain Accuracy)
- [ ] **P1-1**: Implement `get_legal_actions` in `UAVModel` to prune impossible boundary moves from MCTS trees ([`uav_model.py:49-54`](file:///home/andyj1810/projects/ipomcp/src/examples/uav/model/uav_model.py#L49-L54)).
- [ ] **P1-2**: Inject RNG instances into `UAVModel` and `WumpusModel` methods to ensure Common Random Numbers (CRN) provide true variance reduction across parallel workers ([`uav_model.py:42-89`](file:///home/andyj1810/projects/ipomcp/src/examples/uav/model/uav_model.py#L42-L89)).
- [ ] **P1-3**: Implement trial-level checkpointing in `GenericBatchRunner` to allow resuming partially completed batches ([`generic_batch_runner.py:349-385`](file:///home/andyj1810/projects/ipomcp/src/utils/generic_batch_runner.py#L349-L385)).
- [ ] **P1-4**: Fix `ParticleDistribution.get_support()` to filter out particles with `weight == 0.0` ([`distribution.py:114-121`](file:///home/andyj1810/projects/ipomcp/src/core/distribution.py#L114-L121)).

### Priority 2: Medium (Architecture & Code Cleanliness)
- [ ] **P2-1**: Centralize `SIM_SCHEDULE` and `PARTICLE_SCHEDULE` into `src/core/config.py` to eliminate drift across runner scripts.
- [ ] **P2-2**: Wrap `import graphviz` in `try-except ImportError` inside `src/utils/visualizer.py` with informative fallback warnings.
- [ ] **P2-3**: Make `InteractiveParticle` truly immutable by storing `models` as an immutable mapping (e.g. `tuple` of pairs or `MappingProxyType`) and implementing `__hash__` / `__eq__`.

### Priority 3: Low (Optimization & Tech Debt)
- [ ] **P3-1**: Transition `batch_results.csv` export to Apache Parquet format to eliminate string-serialized dictionaries and accelerate post-hoc analysis.
- [ ] **P3-2**: Expand test coverage in `tests/test_distribution.py` and `tests/test_solvers.py` for edge-case degenerate inputs (empty particle sets, zero weights, out-of-bound Q-ranges).

---

## Active Benchmark Telemetry Status (Background Task)

- **Task**: Master Large-Scale Benchmark Suite ($N=200, T=20$)
- **Command**: `uv run python src/examples/experiments/run_all_large_scale_benchmarks_N200.py --trials 200 --steps 20 --suite all --resume-dir results/deep_prior/deep_prior_benchmark_20260904_010926_N200_T20`
- **Active Phase**: Suite 1 (Deep Hierarchy Prior Benchmark), Condition 5/10 (`L4 vs L3, 80% L3 Prior`).
- **Progress**: 80 / 200 trials completed (Conditions 1-4 completed/cached).
- **Resource Status**: 2 worker processes operating stably at 93–97% CPU utilization; OS memory healthy (6.8 GB available RAM).
