# Interactive POMCP (I-POMCP) Framework for Finitely Nested Multi-Agent Systems

## 1. Theoretical Foundations of Finitely Nested I-POMDPs

The computational formalization of multi-agent planning under partial observability requires representing an agent's recursive reasoning about other operating agents. The **Interactive Partially Observable Markov Decision Process (I-POMDP)** framework generalizes single-agent POMDPs to multi-agent settings by replacing flat state distributions with hierarchical interactive belief systems.

### 1.1 Finitely Nested Interactive State Space
For a subject agent $i$ interacting with opponent agents $-i$ at strategy reasoning level $l \ge 1$:

$$\text{I-POMDP}_{i,l} = \langle IS_{i,l}, A, T_i, \Omega_i, O_i, R_i, \gamma \rangle$$

The interactive state space $IS_{i,l}$ is strategically nested and defined inductively over the union of all lower-level opponent intentional models:

$$IS_{i,0} = S$$
$$IS_{i,l} = S \times \Theta_{-i}^{<l} \quad \text{for } l \ge 1$$

where the opponent model union space is formally expressed as:

$$\Theta_{-i}^{<l} = \bigcup_{k=0}^{l-1} \Theta_{-i}^k$$

For each strategic reasoning level $k \ge 1$, an intentional opponent model is a pair:

$$\theta_{-i}^k = \langle b_{-i}^k, \widehat{\theta}_{-i} \rangle$$

where $b_{-i}^k \in \Delta(IS_{-i,k})$ is the opponent's subjective nested belief over its own interactive state space:

$$IS_{-i,k} = S \times \Theta_i^{<k} = S \times \left( \bigcup_{j=0}^{k-1} \Theta_i^j \right)$$

At the base case ($k = 0$), $\Theta^0$ represents the non-strategic, zero-intelligence baseline model governed by a uniform stochastic policy:

$$\pi^0(a) = \frac{1}{|A|}$$

---

### 1.2 Interactive Belief Update & Private Observation Marginalization
When agent $i$ at level $l$ executes action $a_i^{t-1}$ and receives private observation $o_i^t$, it updates its interactive belief $b_{i,l}^t(s^t, \boldsymbol{\theta}_{-i}^t)$ according to the exact Bayesian filter:

$$b_{i,l}^t(s^t, \boldsymbol{\theta}_{-i}^t) = \eta \cdot P(o_i^t \mid s^t, a_i^{t-1}, \mathbf{a}_{-i}^{t-1}) \sum_{s^{t-1}} \sum_{\boldsymbol{\theta}_{-i}^{t-1}} b_{i,l}^{t-1}(s^{t-1}, \boldsymbol{\theta}_{-i}^{t-1}) \sum_{\mathbf{a}_{-i}^{t-1}} \left( \prod_{j \neq i} P(a_j^{t-1} \mid \theta_j^{t-1}) \right) T(s^t \mid s^{t-1}, a_i^{t-1}, \mathbf{a}_{-i}^{t-1}) \cdot \prod_{j \neq i} P(\theta_j^t \mid \theta_j^{t-1}, a_j^{t-1}, s^t, \mathbf{a}^{t-1})$$

Because opponent local observations $o_j^t$ are strictly private and unobservable to agent $i$, the transition dynamics of the opponent's intentional model must explicitly marginalize across all possible private observation signals $o_j^t \in \Omega_j$:

$$P(\theta_j^t \mid \theta_j^{t-1}, a_j^{t-1}, s^t, \mathbf{a}^{t-1}) = \sum_{o_j^t \in \Omega_j} O_j(o_j^t \mid s^t, a_i^{t-1}, \mathbf{a}_{-i}^{t-1}) \cdot \delta_{\operatorname{SE}(\theta_j^{t-1}, a_j^{t-1}, o_j^t)}(\theta_j^t)$$

where $\operatorname{SE}(\cdot)$ is the opponent's internal belief transition update function and $\delta$ is the Dirac/Kronecker indicator.

---

## 2. Algorithmic Invariants of the I-POMCP Architecture

This codebase implements sample-based online planning via Interactive POMCP, enforcing six core invariants:

1. **Invariant 1 (Joint Action Integrity)**: All physical environment transitions $T(s, \mathbf{a})$, observation likelihoods $P(o_i \mid s', \mathbf{a})$, and reward evaluations $R_i(s, \mathbf{a}, s')$ strictly consume immutable `JointAction` vectors $\mathbf{a} = \langle a_i, \mathbf{a}_{-i} \rangle$.
2. **Invariant 2 (Union Model Space Recursion)**: Level-$l$ agents maintain explicit belief distributions across the full union $\Theta_{-i}^{<l} = \bigcup_{k=0}^{l-1} \Theta_{-i}^k$.
3. **Invariant 3 (Slotted Particle Memory & Sub-Tree Caching)**: Particles are frozen dataclasses with `__slots__` mapping opponent frames to their respective MCTS sub-trees. JIT expansions update sub-tree search statistics, amortizing simulation work across sibling particles.
4. **Invariant 4 (Observational Symmetry)**: The evaluative probability density $P(o_i \mid s', \mathbf{a})$ computed by `get_observation_prob` strictly matches the generative density of `sample_observation`.
5. **Invariant 5 (Normalized Policy Entropy Standard)**: Opponent action selection uncertainty is evaluated via normalized policy entropy $H_{\text{norm}}(\pi) = \frac{-\sum p \ln p}{\ln |A|} \in [0, 1]$ to dynamically gate JIT MCTS expansions.
6. **Invariant 6 (Scale-Invariant Normalized UCB)**: MCTS tree action selection dynamically scales local Q-values using adaptive empirical bounds $Q_{\text{norm}} = \frac{Q - q_{\min}}{(q_{\max} - q_{\min}) + \epsilon}$, keeping exploration parameter $c = \sqrt{2}$ effective across disparate reward scales.

---

## 3. Directory Layout

```
ipomcp/
├── pyproject.toml
├── README.md
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── distribution.py
│   │   ├── logger.py
│   │   ├── paths.py             # Centralized results routing
│   │   ├── telemetry.py         # Zero-overhead OS telemetry & watchdog
│   │   └── pomdp_model.py
│   ├── ipomdp/
│   │   ├── __init__.py
│   │   └── belief.py
│   ├── solvers/
│   │   ├── __init__.py
│   │   ├── exploration.py
│   │   ├── generative_model.py
│   │   ├── i_pomcp.py
│   │   ├── node.py
│   │   ├── planner.py
│   │   ├── random_planner.py
│   │   ├── rts_planner.py
│   │   ├── solver_bank.py
│   │   └── solver_types.py
│   ├── examples/
│   │   ├── tiger/
│   │   │   ├── model/
│   │   │   │   └── tiger_model.py
│   │   │   └── runners/
│   │   │       ├── persistent_tiger_runner.py
│   │   │       ├── tiger_baseline_runner.py
│   │   │       ├── tiger_batch_runner.py
│   │   │       ├── tiger_level3_experiment.py
│   │   │       ├── tiger_mixture_experiment.py
│   │   │       └── apples_to_apples_oracle_benchmark.py
│   │   ├── uav/
│   │   │   ├── model/
│   │   │   │   ├── uav_model.py
│   │   │   │   └── uav_viz.py
│   │   │   └── runners/
│   │   │       ├── run_rts.py
│   │   │       ├── run_uav.py
│   │   │       ├── uav_baseline_runner.py
│   │   │       └── uav_batch_runner.py
│   │   ├── wumpus/
│   │   │   ├── model/
│   │   │   │   ├── constants.py
│   │   │   │   ├── wumpus_model.py
│   │   │   │   ├── wumpus_state.py
│   │   │   │   └── wumpus_viz.py
│   │   │   └── runners/
│   │   │       ├── run_wumpus.py
│   │   │       ├── wumpus_baseline_runner.py
│   │   │       └── wumpus_batch_runner.py
│   │   └── experiments/
│   │       ├── deep_hierarchy_prior_experiment.py
│   │       ├── level_convergence_matrix_experiment.py
│   │       ├── master_nested_ipomdp_benchmark.py
│   │       └── run_all_large_scale_benchmarks_N200.py
│   └── utils/
│       ├── __init__.py
│       ├── bootstrapper.py
│       ├── generic_batch_runner.py
│       ├── paper_plots.py
│       ├── plotting.py
│       ├── regenerate_all_experiment_plots.py
│       └── visualizer.py
├── results/                     # Centralized experimental artifact repository
│   ├── deep_prior/
│   ├── oracle/
│   ├── payoff_matrix/
│   ├── tiger/
│   ├── uav/
│   ├── wumpus/
│   └── benchmarks/
├── benchmarks/
│   └── tiger_pre_post_benchmark.py
└── tests/
    ├── test_distribution.py
    ├── test_exploration.py
    ├── test_integration.py
    ├── test_models.py
    ├── test_node.py
    ├── test_oracle_rts.py
    ├── test_solvers.py
    ├── test_telemetry.py
    └── test_visualization.py
```

---

## 4. Verification and Benchmarking

### Running the Test Suite
```bash
uv run python -m pytest tests/ -v
```

### Running the Tiger Pre/Post Verification Benchmark
```bash
uv run python benchmarks/tiger_pre_post_benchmark.py post_edits
```

### Running the Large-Scale Master Benchmark Suite
```bash
# Run all three suites sequentially (Deep Prior -> Oracle RTS -> Payoff Matrix)
uv run python src/examples/experiments/run_all_large_scale_benchmarks_N200.py --trials 200 --steps 20 --suite all

# Or run individual benchmark suites
uv run python src/examples/experiments/run_all_large_scale_benchmarks_N200.py --trials 200 --steps 20 --suite prior
uv run python src/examples/experiments/run_all_large_scale_benchmarks_N200.py --trials 200 --steps 20 --suite oracle
uv run python src/examples/experiments/run_all_large_scale_benchmarks_N200.py --trials 200 --steps 20 --suite matrix
```
All outputs are automatically routed into the centralized `<project_root>/results/` directory.