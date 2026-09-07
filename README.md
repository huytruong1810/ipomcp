# Interactive POMCP (I-POMCP) Framework for Finitely Nested Multi-Agent Systems

[![Tests](https://img.shields.io/badge/pytest-38%20passing-brightgreen)](https://github.com/huytruong1810/ipomcp)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![Repository](https://img.shields.io/badge/github-huytruong1810%2Fipomcp-blue)](https://github.com/huytruong1810/ipomcp)

An enterprise-grade, high-performance implementation of **Finitely Nested Interactive Partially Observable Markov Decision Processes (I-POMDPs)** in Python. The framework provides online Monte Carlo Tree Search (I-POMCP) with Just-In-Time (JIT) mental model expansion, alongside an exact Reachability Tree Sampling (RTS) Oracle baseline reproducing Doshi & Gmytrasiewicz (*JAIR* 2009).

---

## 1. Mathematical Foundations of Finitely Nested I-POMDPs

In multi-agent systems with partial observability and strategic interactions, agents must maintain nested recursive beliefs about other agents' intentional models. An agent $i$ operating at strategic reasoning level $l \ge 1$ formalizes its environment as:

$$\text{I-POMDP}_{i,l} = \langle IS_{i,l}, A, T_i, \Omega_i, O_i, R_i, \gamma \rangle$$

### 1.1 Finitely Nested Interactive State Space
The interactive state space $IS_{i,l}$ is defined inductively over physical environment states $S$ and the union of all lower-level opponent intentional models $\Theta_{-i}^{<l}$:

$$IS_{i,0} = S$$
$$IS_{i,l} = S \times \Theta_{-i}^{<l} \quad \text{for } l \ge 1$$

where the opponent model union space is formally expressed as:

$$\Theta_{-i}^{<l} = \bigcup_{k=0}^{l-1} \Theta_{-i}^k$$

For each strategic reasoning level $k \ge 1$, an intentional opponent model is a pair:

$$\theta_{-i}^k = \langle b_{-i}^k, \widehat{\theta}_{-i} \rangle$$

where $b_{-i}^k \in \Delta(IS_{-i,k})$ is the opponent's subjective nested belief over its own interactive state space:

$$IS_{-i,k} = S \times \Theta_i^{<k} = S \times \left( \bigcup_{j=0}^{k-1} \Theta_i^j \right)$$

At level 0 ($k = 0$), $\Theta^0$ represents the non-strategic baseline governed by a uniform random policy $\pi^0(a) = \frac{1}{|A|}$.

### 1.2 Interactive Belief Update & Private Observation Marginalization
When agent $i$ executes action $a_i^{t-1}$ and receives private observation $o_i^t$, it updates its interactive belief $b_{i,l}^t(s^t, \boldsymbol{\theta}_{-i}^t)$ via Bayesian filtering:

$$b_{i,l}^t(s^t, \boldsymbol{\theta}_{-i}^t) = \eta \cdot P(o_i^t \mid s^t, \mathbf{a}^{t-1}) \sum_{s^{t-1}, \boldsymbol{\theta}_{-i}^{t-1}} b_{i,l}^{t-1}(s^{t-1}, \boldsymbol{\theta}_{-i}^{t-1}) \left( \prod_{j \neq i} P(a_j^{t-1} \mid \theta_j^{t-1}) \right) T(s^t \mid s^{t-1}, \mathbf{a}^{t-1}) \prod_{j \neq i} P(\theta_j^t \mid \theta_j^{t-1}, a_j^{t-1}, s^t, \mathbf{a}^{t-1})$$

Because opponent observations $o_j^t$ are strictly private, updating the opponent's intentional model requires marginalizing across all possible private observation signals $o_j^t \in \Omega_j$:

$$P(\theta_j^t \mid \theta_j^{t-1}, a_j^{t-1}, s^t, \mathbf{a}^{t-1}) = \sum_{o_j^t \in \Omega_j} O_j(o_j^t \mid s^t, \mathbf{a}^{t-1}) \cdot \delta_{\operatorname{SE}(\theta_j^{t-1}, a_j^{t-1}, o_j^t)}(\theta_j^t)$$

---

## 2. Core Architectural Components

```
src/
├── core/
│   ├── config.py              # Centralized dataclass configurations (MCTS, JIT, RTS)
│   ├── distribution.py        # ParticleDistribution (SUS resampling) & DictDistribution
│   ├── logger.py              # Structured logging with per-trial file routing
│   ├── paths.py               # Centralized path resolvers for results
│   ├── pomdp_model.py         # Abstract POMDPModel base interface
│   └── telemetry.py           # OS-level resource monitoring & MemoryWatchdog
├── ipomdp/
│   └── belief.py              # InteractiveParticle and AgentFrame definitions
├── solvers/
│   ├── exploration.py         # StandardUCB and scale-invariant NormalizedUCB
│   ├── generative_model.py    # InteractiveGenerativeModel with variance-driven JIT expansion
│   ├── i_pomcp.py             # IPOMCPPlanner with reservoir sampling
│   ├── node.py                # POMCPNode AND-OR search tree node
│   ├── planner.py             # Abstract Planner base class
│   ├── random_planner.py      # Sub-intentional Level-0 baseline
│   ├── rts_planner.py         # Exact Reachability Tree Sampling (RTS) Oracle
│   ├── solver_bank.py         # Centralized SolverKey -> Planner registry
│   └── solver_types.py        # SolverKey and AgentFrame definitions
├── utils/
│   ├── bootstrapper.py        # Topological I_POMDP_Bootstrapper with O(NL) root pooling
│   ├── generic_batch_runner.py# Multiprocessing runner with Common Random Numbers (CRN)
│   ├── paper_plots.py         # High-DPI publication vector graphics generator (PDF)
│   ├── plotting.py            # Interactive Plotly Sunburst and progression visualizers
│   └── visualizer.py          # Graphviz MCTS forest exporter
└── examples/
    ├── tiger/                 # Multi-Agent Tiger Domain (persistent & reset modes)
    ├── uav/                   # 2D Grid UAV Target Pursuit-Evasion Domain
    ├── wumpus/                # Multi-Agent Wumpus World Domain
    └── experiments/           # Master benchmark orchestration scripts
```

---

## 3. Engineering & Production Guardrails

1. **$O(NL)$ Canonical Root Pooling**:
   - Eliminates exponential $O(N^L)$ particle duplication in [`src/utils/bootstrapper.py`](src/utils/bootstrapper.py) by sharing canonical root trees across particles of identical reasoning levels. Reduces process RAM from **14.8 GB** down to **< 200 MB** for Level-4/Level-5 agents.
2. **OS Telemetry & Memory Watchdog**:
   - Embedded [`src/core/telemetry.py`](src/core/telemetry.py) tracks `process_rss_mb` and `host_available_ram_mb`, classifying operating regimes (`DRAM_BOUND_NORMAL`, `MEMORY_PRESSURE`, `SWAP_THRASHING`).
3. **CPython Allocator Arena Recycling**:
   - [`src/utils/generic_batch_runner.py`](src/utils/generic_batch_runner.py) configures `max_tasks_per_child=20` inside `ProcessPoolExecutor`, periodically recycling worker processes to completely prevent `pymalloc` memory fragmentation.
4. **Common Random Numbers (CRN)**:
   - Strict stream isolation (`env_rng_trans`, `env_rng_obs_i`, `env_rng_obs_j`) guarantees variance-reduced experimental comparisons under identical environmental noise.
5. **Resumption & Checkpointing**:
   - All benchmark runners support `--resume-dir` and verify completed condition CSVs, enabling long-running runs to resume without recomputing finished conditions.

---

## 4. Verification & Testing

The test suite validates theoretical invariants, numerical stability, distribution properties, domain models, and OS telemetry.

```bash
# Run the complete test suite (38 passing tests)
uv run python -m pytest tests/ -v
```

### Test Suite Breakdown
- `tests/test_distribution.py`: Particle distribution resampling (SUS), lazy cache invalidation, and CDF boundary conditions.
- `tests/test_exploration.py`: Standard UCB1 and Normalized UCB scale-invariance.
- `tests/test_models.py`: Tiger generative-evaluative symmetry, UAV mechanics, and Wumpus dynamics.
- `tests/test_node.py`: Reservoir sampling (Algorithm R) capacity bounds and tree serialization.
- `tests/test_oracle_rts.py`: RTS exact lookahead tree construction, action values, and batch execution.
- `tests/test_solvers.py`: Arbitrary level recursive bootstrapping, mixture priors, and solver bank decoupling.
- `tests/test_telemetry.py`: Process RSS, host memory detection, watchdog enforcement, and JSONL logging.
- `tests/test_visualization.py`: Nested belief extraction, Plotly Sunburst generation, and vector PDF exports.
- `tests/test_integration.py`: End-to-end multi-agent batch runners across Tiger, UAV, and Wumpus.

---

## 5. Running Large-Scale Experiments

All experiment results, telemetry logs, and publication plots are automatically routed to `<project_root>/results/`.

```bash
# Execute the Master Benchmark Suite (N=200 trials, T=20 steps)
# 1. Deep Hierarchy Prior Benchmark (10 conditions)
# 2. Apples-to-Apples Oracle RTS vs I-POMCP (7 conditions)
# 3. Full 6x6 Strategy Level Payoff Matrix (36 conditions)
uv run python src/examples/experiments/run_all_large_scale_benchmarks_N200.py --trials 200 --steps 20 --suite all

# Resume an interrupted benchmark run:
uv run python src/examples/experiments/run_all_large_scale_benchmarks_N200.py --trials 200 --steps 20 --suite all --resume-dir results/deep_prior/deep_prior_benchmark_20260904_010926_N200_T20
```

---

## 6. Codebase Review & Quality Roadmap

For reviewers and incoming engineers, see:
- [`HANDOFF.md`](HANDOFF.md): Immediate runtime state, active process telemetry, and reviewer guidance.
- [`BACKLOG.md`](BACKLOG.md): Comprehensive architectural, theoretical, execution, and observability audit report cataloging open P0/P1/P2/P3 action items.
