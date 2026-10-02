# Architecture

The package is organised as bounded contexts with one-way dependencies:

```
model  <-  domains
model  <-  belief  <-  planning  <-  reference
                        ^   ^            ^
                        |   experiments -+---- analysis
                       infra
```

## Contexts

### `model`: the domain contract

`POMDPModel` declares the physics through **finite kernels**:
- `initial_distribution`, `transition_distribution` and `observation_distribution`;
- `get_reward`, with `reward_bounds`;
- `is_terminal`, `get_all_actions` and `get_all_observations`.

Sampling and observation likelihoods are *derived* from the kernels. That is why a sampler can never disagree with the law used by exact belief updates and exact references.

Two information rules are encoded in the signatures, not checked after the fact:
- **Actions depend only on the agent.** An action set that varied with the hidden state would let a planner branch on information it lacks.
- **Rollout policies see only private memory.** `rollout_action` and `update_rollout_memory` never receive the sampled state.

### `domains`

Tiger, UAV and Wumpus implement the contract. Their specifications, with sources, are in [DOMAINS.md](DOMAINS.md).

### `belief`: interactive states and updates

- `MentalModel(frame, belief)` is an opponent hypothesis.
  - `AgentFrame` holds agent, level and physics.
  - The belief is `None` for uniform-random L0.
- `InteractiveState(state, opponent)` is one joint hypothesis.
- `FiniteBelief` is a normalized measure over interactive states, with duplicates merged exactly.
- `canonical` gives every belief an order-independent, process-independent digest. Search seeding, sampling order and resampling depend only on *what* a belief is, never on how it was built.
- **Belief updates** are subclasses of `InteractiveFilter`; both are pure and memoized:
  - `FiniteInteractiveFilter`: the exact I-POMDP update, enumerated. Optionally it compresses posteriors to K atoms.
  - `ParticleInteractiveFilter`: the interactive particle filter, at most N particles at every level, seeded per update.

### `planning`: the I-POMCP planner

- **`IPOMCPPlanner`** is root-sampling tree search over the agent's private histories. It uses:
  - empirical chance-weighted Bellman backups (`node.POMCPNode.bellman_backup`);
  - exact expected immediate rewards at every history, plus exact last-decision values, integrated over the agent's own posterior at that history;
  - horizon-bounded UCB scaled by declared reward bounds (`exploration.HorizonBoundUCB`).

  These components are fixed rather than configurable. They are the estimator shown to converge against exact references; the alternatives that did not converge were removed.
- **`SolverBank`** is one agent's registry of nested models. It holds:
  - the domain prior;
  - the belief update (exact or particle);
  - a cache of modeled policies;
  - the exact reward integrator.

  A modeled opponent's policy is the greedy policy of that opponent's own planner, solved at the opponent's belief with the *modeled* simulation budget.
- **Purity.** Every solve is seeded from (bank seed, agent, level, canonical belief digest, configuration). A modeled policy is therefore a pure function of the modeled belief, and caches remain valid across real steps.
- **`PlannerConfig`** holds the only tunables:
  - `simulations` (searching futures);
  - `modeled_simulations` (simulating minds);
  - `depth`, `gamma` and the UCB constant.

### `reference`: exact solutions for measuring error

- `BestResponseReference` is the exact finite-horizon best response to the opponents declared in a bank, for any finite domain and level.
- `ExactIPOMDPPlanner` / `build_exact_agent` make every level exact recursively, so the same reference then solves the exact I-POMDP.
- `ExactPOMDPSolver` is alpha-vector value iteration for the two-state L1 problem. It is an independent algorithm used to cross-check the others.

### `experiments`

- `HierarchyBuilder` builds an agent and all lower-level models in one bank, under declared level priors.
- `GenericBatchRunner` runs supervised, checkpointed episodes. It records, per step and per agent:
  - rewards;
  - planning and belief-update CPU time;
  - belief support and level posteriors.
- The qualification and study drivers are `l1_qualification`, `ipf_qualification`, `oracle` and `allocation_study`.

### `analysis`

- Paired statistics.
- Additive attribution of search-tree error (`error_attribution`).
- Figures.

### `infra`

- Process supervision with per-job wall-time and process-tree memory limits.
- Source locks verified inside every worker.
- Resource measurement.
- Repository paths.

## Design decisions worth knowing

- **Determinism by content, not by call order.** All randomness flows from explicit `random.Random` streams. No code uses the global `random` module. Per-solve and per-update streams are seeded from canonical digests.
- **Exactness where it buys convergence, sampling where it buys scale.** The planner integrates immediate rewards and the last step exactly. The IPF replaces exact enumeration of beliefs at every level. Both are measured against exact references (see [RESULTS.md](RESULTS.md)).
- **Failures are explicit.**
  - Zero evidence raises `UnsupportedObservation`.
  - Enumeration budgets raise `InferenceBudgetExceeded`.
  - Nothing resets a belief to its prior or adds probability floors.
- **Levels.** L0 is uniformly random over the full action set, the convention of the IPF and nested-MCTS literature. Gmytrasiewicz & Doshi (2005) use an optimizing L0. Studies state which they use.
