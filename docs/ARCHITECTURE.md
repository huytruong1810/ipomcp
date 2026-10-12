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
  - `FiniteInteractiveFilter`: the exact I-POMDP update, enumerated.
  - `CompressedInteractiveFilter`: the exact update, with posteriors compressed to N atoms by systematic resampling only when they exceed N. This is the planner's default bounded belief; small beliefs stay exact.
  - `ParticleInteractiveFilter`: the published interactive particle filter, which samples every term of the update; it is the belief of the IPF + RTS baseline.
  - `SolverBank(seed, particles=N, particle_filter="compressed" | "sampled")` selects one; without `particles`, beliefs are exact.

### `planning`: the I-POMCP planner

- **`IPOMCPPlanner`** is root-sampling tree search over the agent's private histories. It uses:
  - empirical chance-weighted Bellman backups (`node.POMCPNode.bellman_backup`);
  - exact expected immediate rewards at every history, plus exact last-decision values, integrated over the agent's own posterior at that history;
  - horizon-bounded UCB scaled by declared reward bounds (`exploration.HorizonBoundUCB`);
  - post-search chance scoring (`chance_scoring.score_tree`): the finished tree is rescored bottom-up with the exact chance law P(o, alive | b_h, a) in place of sampled frequencies. The scored values decide, for real decisions and modeled opponents alike; the search values are kept beside them (`search_values`).

  These components are fixed rather than configurable. They are the estimator shown to converge against exact references; the alternatives that did not converge were removed.

  One budget policy is configurable: `PlannerConfig.modeled_stopping` (`stopping.ModeledStopping`). When set, a *modeled* policy is solved by one main tree, seeded as the flat search of the cap (`modeled_simulations`) and grown by doubling from a pilot size. It stops once its leading gap exceeds a t critical value times the spread that 3 small pilot trees measure, scaled by sqrt(pilot / n). The decision always comes from the main tree, so at the cap the rule is the flat search, and it is consistent. It replaced averaging 200-simulation replicates, which converged to a shallow tree's biased value (RESULTS.md 12.10–12.11).
- **`SolverBank`** is one agent's registry of nested models. It holds:
  - the domain prior;
  - the belief update (exact or particle);
  - a cache of modeled policies;
  - the exact reward integrator (`expected_rewards`) and chance law (`continuation_law`).

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

- **Studies are declarative.** A `StudySpec` gives the domain, each agent's level, planner (`ipomcp` or `rts`), belief representation (exact or IPF particles) and level priors, and the trial design. Specifications load from TOML; see `studies/`.
- `HierarchyBuilder` builds an agent and all lower-level models in one bank, under declared level priors.
- `episodes.run_episode` runs one trial, symmetric in the two agents. Each row records, per agent:
  - actions, observations and rewards;
  - planning and belief-update CPU and wall time;
  - belief support, the level posterior and the physical marginal;
  - optionally the nested hierarchy.
- `runner.run_study` executes trials under supervision. It writes atomic per-trial checkpoints and a manifest of the spec plus source hashes, resumes only when both match, and writes a completion marker bound to the CSV bytes.
- **Suites** (`experiments.suite`): a `SuiteSpec` is a set of named `StudySpec` cells on shared trial ids, so cells pair by trial. `run_suite` schedules trials round-robin over cells, one supervised process per episode, checkpoints `trials/<cell>/<trial>.csv`, and resumes against a manifest in which only `trials` may grow. Builders, generic over the domain: `level_matrix` (every level pair, level 0 included, uniform priors over lower levels at every nesting), `planner_comparison` (I-POMCP against IPF + RTS at level 2), `prior_sensitivity` (correct, over-, under-estimated and uniform priors over the opponent's level). Their endpoints are in `analysis.endpoints`.
- **Qualification drivers:** `l1_qualification`, `belief_qualification`, `oracle` (one decision problem vs exact best response or exact I-POMDP; levels 1–4, the published Tiger or a payoff variant; planners `mcts` and `rts`; exact or particle beliefs) and `panel` (grids of planners, levels 2–4, budgets, modeled budgets and particle counts vs the exact I-POMDP: the allocation and baseline studies), and `modeled_qualification` (the modeled level below a level-2 or level-3 agent, scored against its exact reference at the beliefs it is queried at, over a ladder of modeled stacks).
- **Per-level modeled stacks** (`experiments.modeled_levels`): the oracle, panel and qualification drivers name how every intentional level below the agent is solved with one label per level, top-down: `400` (flat), `6400s` (stopping, cap 6,400), `exact`; `6400s/400` at level 3 solves j-L2 with stopping and the i-L1 it models with a flat 400. Each level's `PlannerConfig` carries its own budget and rule; `register_levels` builds the hierarchy from the stack for every driver.

### `baselines`

- `RTSPlanner`: reachability-tree sampling over interactive beliefs (Doshi & Gmytrasiewicz, 2009). With an IPF bank it is the IPF + RTS baseline. It uses the same reward integrator and declared opponents as I-POMCP, so comparisons vary only the lookahead.

### `analysis`

- `hierarchy`: nested belief hierarchies as path nodes carrying joint mass and the physical belief held at each node, and their error against an exact hierarchy per depth.
- `metrics`: long-format rewards, costs and level posteriors computed once from episode records, with trial-clustered bootstrap intervals.
- `figures`: static figures (matplotlib) and interactive sunbursts (plotly) of any depth, with content and error in the hover text.
- `statistics`: paired comparisons, Holm correction and Clopper–Pearson intervals.
- `endpoints`: the suites' pre-registered endpoints (paired differences by trial, the one-level endpoint of a level matrix, arm differences, per-cell tables with prediction misses).
- `error_attribution`: an additive decomposition of search-tree error.

### `infra`

- Process supervision with per-job wall-time and process-tree memory limits.
- Source locks verified inside every worker.
- Resource measurement.
- Repository paths.

## Design decisions worth knowing

- **Determinism by content, not by call order.** All randomness flows from explicit `random.Random` streams. No code uses the global `random` module. Per-solve and per-update streams are seeded from canonical digests.
- **Exactness where it buys convergence, bounds where they buy scale.** The planner integrates immediate rewards and the last step exactly. Bounded beliefs cap supports at N atoms at every level, keeping updates exact until a posterior exceeds N. Both are measured against exact references (see [RESULTS.md](RESULTS.md)).
- **Failures are explicit.**
  - Zero evidence raises `UnsupportedObservation`.
  - Enumeration budgets raise `InferenceBudgetExceeded`.
  - Nothing resets a belief to its prior or adds probability floors.
- **Levels.** L0 is uniformly random over the full action set, the convention of the IPF and nested-MCTS literature. Gmytrasiewicz & Doshi (2005) use an optimizing L0. Studies state which they use.
