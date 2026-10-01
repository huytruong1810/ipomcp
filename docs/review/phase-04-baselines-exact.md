# Phase 4 — Exact references and baseline planners

Files reviewed in full (1,177 lines):
- `solvers/rts_planner.py`, `solvers/random_planner.py` (the latter in Phase 3)
- `solvers/exact/__init__.py`, `alpha_vector.py`, `pomdp_exact_vi.py`
- `solvers/exact/ipomdp_exact_vi.py`, `finite_policy_l2.py`, `finite_policy_reference.py`, `fixed_tiger_opponent.py`

## What exists vs what contribution 2 claims

| Claimed comparator | What the code has | Gap |
|---|---|---|
| Exact alpha-vector iteration for I-POMDPs | `ExactPOMDPSolver`: alpha-vector incremental pruning for **2-state Tiger at L1** against a fixed opponent policy (a POMDP, since the opponent is a fixed L0 or a fixed mixture) | No alpha-vector solver over *interactive* states (L ≥ 2). `AlphaVectorND`/`prune_nd` exist but are used only by tests. |
| Exact I-POMDP at L ≥ 2 | Three exhaustive belief-tree enumerators: `ExactIPOMDPSolver` (Tiger L2, exact L1 opponent), `FinitePolicyL2Reference` (Tiger L2 vs bank policies), `FinitePolicyReference` (Tiger, any level, vs bank policies). Plus `benchmarks/exact_ipomdp_agent.py` (recursively exact agent, outside `src`). | Valid exact references, but they are exhaustive search and not alpha vectors. Three overlap. All are Tiger-only. |
| IPF + Reachability Tree Search (Doshi) | `RTSPlanner`: weighted-particle lookahead, expanding the top-k most likely observations | Not Doshi's RTS (details in T1). Shares I-POMCP's opponent policies and exact nested filter. |
| — | `RandomPlanner` (L0) | Fine. |

## Findings

### Correctness and fairness

- **B1 — RTS uses top-k most-likely observations.** Doshi & Gmytrasiewicz (2009) sample N_o observations per action from the propagated particles.
  - Deterministic top-k by probability is a biased lookahead: the omitted mass is not renormalised. Continuation is `Σ_{top-k} p·V`, so omitted mass is treated as value 0 (`rts_planner.py:109-122`).
  - In a domain with negative rewards that is optimistic; with positive rewards, pessimistic.
  - The module docstring calls omission "an explicit approximation", but its bias is not reported anywhere.
- **B2 — RTS inherits I-POMCP's opponent machinery.**
  - `tree_step` calls the exact nested filter and the bank's modeled policies. The modeled opponents are whatever planners are registered, typically MCTS.
  - A comparison "I-POMCP vs RTS" therefore varies only the protagonist's lookahead. The IPF half of the baseline (sampled nested particle propagation) doesn't exist.
  - The comparison is still fair as a lookahead ablation, but it is not a comparison against the published IPF+RTS method.
- **B3 — Unequal estimators.**
  - I-POMCP always replaces the root immediate reward with its exact expectation (Phase 3 T5).
  - RTS uses sampled reward means.
  - Any "sample efficiency" comparison must give both the same root control variate, or neither.
- **B4 — Argmax tie bias.** `ExactPOMDPSolver.policy` and `ExactIPOMDPSolver.policy` return the first maximum (`pomdp_exact_vi.py:301`, `ipomdp_exact_vi.py:177`). Callers that check optimality should compare against the set of maximisers. The finite-policy references already return Q dicts.
- **B5 — RTS budget is not comparable to MCTS budgets.**
  - RTS cost is `num_particles × Σ_{d} (|A| · k)^d` generative steps, plus every opponent solve.
  - Nothing in the config expresses this cost in simulations or wall time.
  - Contribution 2's sample-efficiency claim needs one common budget unit: generative-model calls, or wall-clock under a fixed core.

### Theory

- **T1 — "Exact" means three different things.**
  1. *Exact I-POMDP*: every intentional model is exactly solved, as in `ExactIPOMDPSolver` and `exact_ipomdp_agent.py`.
  2. *Exact best response to declared finite policies*: modeled opponents are the bank's MCTS (`FinitePolicyReference`, `FinitePolicyL2Reference`). This measures search regret only.
  3. *Exact POMDP*: the L1 alpha-vector solver.

  Error reports must say which, and the API should make the distinction impossible to blur. Today they are interchangeable classes with similar names.
- **T2 — Horizon conventions differ across references.**
  - `ExactPOMDPSolver` and `ExactIPOMDPSolver` use a decreasing horizon (`opponent_horizon` optional).
  - I-POMCP uses a fixed receding depth (`max_depth`) at every real step.
  - `FixedTigerL1Policy` uses a fixed depth.
  - A comparison is only valid when the reference uses the same convention as the planner; some tests do this explicitly. The convention should be a typed parameter shared by planner and reference.
- **T3 — Exact references scale only to Tiger.**
  - Each checks `isinstance(physics, TigerModel)` and Tiger state names.
  - `FinitePolicyReference` uses nothing Tiger-specific beyond the check. It works for any model with finite `transition_distribution` and `observation_distribution`, which UAV and Wumpus provide (Phase 5 confirms).
  - Domain-agnostic exact references at small horizons would let UAV and Wumpus get error measurements too.

### Structure

- **S1 — Three overlapping L2+ references.**
  - `FinitePolicyReference` strictly generalises `FinitePolicyL2Reference`: same algorithm, any level.
  - `ExactIPOMDPSolver` is the special case where opponents are themselves exact, which `exact_ipomdp_agent.py` also implements recursively.
  - Keep one recursive exact reference that is parameterised by how opponents are solved (exact or declared finite).
- **S2 — Immediate-reward integration exists in four places:** `SolverBank._expected_rewards`, `FinitePolicyReference._immediate`, `FinitePolicyL2Reference._q` (horizon 1) and `ExactPOMDPSolver._precompute_immediate_rewards`. The reference must stay independent of the planner's code to remain an independent check. Even so, three copies inside the reference layer are too many.
- **S3 — `FixedTigerL1Policy`** is an opponent provider with a `policy_for` method, duck-typed as a planner. Policy providers deserve one explicit interface.
- **S4 — Imports inside functions** (`pomdp_exact_vi.py:123,139,152`).
- **S5 — The sort comment in `prune_2d_with_intervals` is wrong** (it says "intercept descending", `alpha_vector.py:87`, but the sort is ascending). Correctness doesn't depend on it, because the parallel filter keeps the max.
- **S6 — Dead code:** `AlphaVectorND`, `prune_nd` (tests only).
