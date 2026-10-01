# Phase 3 — Search core (`src/solvers/`, the I-NTMCP side)

Files reviewed in full:
- `__init__.py`, `solver_types.py`, `planner.py`, `generative_model.py`, `policy.py`
- `random_planner.py`, `exploration.py`, `node.py`, `i_pomcp.py`, `solver_bank.py`, `recursive_chance.py`

Total: 1,365 lines.

## How the planner actually works (needed to judge contribution 1)

1. **Root.** The root belief is an exact `FiniteBelief`. Each simulation draws a root particle by mass, using the canonical order from `SolverBank._ordered_mass` (`i_pomcp.py:82,98`).
2. **Tree step** (`generative_model.tree_step`).
   - Sample a_j from the opponent's policy: `filter.action_distribution`, which calls `SolverBank.policy`, which runs a **full nested MCTS solve** of the opponent's planner with the modeled budget.
   - Sample s' and o_j.
   - Advance the opponent's nested belief with the **exact enumerated filter**.
3. **Optional exact pieces along every tree path:**
   - `exact_history_rewards` and `exact_final_step` carry an exact posterior of the agent's own private history, through `_history_successor`, which calls `filter.update`.
   - `exact_tail_depth=2`, `exact_root_chance` and `recursive_chance` re-score with exact chance laws.
4. **Real step.** `update_root` runs the exact filter and **clears every bank cache**: modeled policies, filter, rewards and laws (`i_pomcp.py:392-403`, `solver_bank.py:218-228`).

So each simulated opponent mind is a fresh MCTS. Each nested belief is exact Bayes. Nothing is reused across real steps.

Compare **I-NTMCP** (Schwartz, Zhou & Kurniawati, 2022). It keeps one persistent nested tree per (agent, level). Opponent particles point at nodes of those trees, and nested beliefs are the particle sets at those nodes, updated by tree traversal and reused across real steps.

The two contributing ideas are therefore not "mechanistically combined" as claimed:
- the IPF side is exact enumeration, not particle filtering (Phase 2 T1);
- the I-NTMCP side is replaced by on-demand, memoised, independent solves.

The cost balance between "simulating futures" and "simulating other minds" in the claim exists only as two fixed budgets: `mcts.n_sims` and `opponent.n_sims`. Nothing allocates computation adaptively between them, and there is no theory for doing so.

## Findings

### Correctness

- **B1 — The root reservoir is always empty, so the UAV runner's reservoir path never runs.**
  - `add_particle` is called only on children (`i_pomcp.py:271`).
  - `update_root` replaces the root with a fresh node (`:398`).
  - `uav_batch_runner.py:92-93` reads `planner_i.root.belief_particles`, so that branch never fires.
  - Node reservoirs are otherwise used only by a diagnostic (`root_chance_experiment.py:32-35`). They cost `node_capacity` references per node for no estimator. Candidate for deletion (Phase 6 confirms).
- **B2 — Legal actions inside the tree come from one sampled hidden state** (`i_pomcp.py:247`).
  - The root checks that the legal set is constant over the belief (`:73-79`); internal nodes do not.
  - In a domain where legality depends on hidden state, the tree would branch on hidden information. The same applies to rollout legality (`:366`).
  - Phase 5 checks whether UAV or Wumpus have this property. Root cause: Phase 1 T1.
- **B3 — Tie-breaking is biased toward action-list order.**
  - `StandardUCB._select` keeps the first maximum (`exploration.py:58`).
  - The unreachable fallback `random.choice` (`:62`) hides this.
  - `greedy_policy` splits ties uniformly, but tree selection does not. This is harmless asymptotically, but it is a systematic bias at small budgets. It is also dead fallback code.

### Theory

- **T1 — The default profile is not a consistent estimator.**
  - `backup="sampled"` with `NormalizedUCB` is the default in both `MCTSConfig` and `IPOMCPPlanner` (`i_pomcp.py:37`).
  - `NormalizedUCB` scales by empirical extrema pooled across depths (`exploration.py:65-73`, documented as heuristic).
  - The L1 study measured non-convergence. The only validated profile is empirical Bellman + exact final step + exact history rewards + `HorizonBoundUCB(c=1)`.
- **T2 — The "sampling-based" planner is exact Bayes on every tree path.**
  - With the validated profile, every simulated step calls `filter.update` on the agent's own history model (`:275-280`) and on each opponent's nested belief (`generative_model.py:37-40`).
  - Per-node cost is therefore the cost of exact interactive filtering, which grows exponentially with depth and nesting.
  - This is the measured L3 bottleneck: 12, 19, 61 and 160 s per decision at steps 1–4 for the exact solver. I-POMCP L3 decisions take minutes.
  - A sampling-based solver should replace the exact filter on these paths with particle propagation. A tree node then carries a weighted particle set, as in POMCP and I-NTMCP.
- **T3 — Recursive chance is exact value iteration on the visited tree.**
  - With full coverage it equals exact finite-tree value iteration (`recursive_chance.py`). It enumerates every observation branch with exact posteriors.
  - It is feasible for L2 Tiger, where |O| = 6 and the horizon is short. In UAV and Wumpus it is not.
  - It adds the whole-action fallback (`:308-310`): a deliberate mix of empirical and rescored estimators (the earlier mixing audit counted 8.1% of internal histories).
  - Evaluation under the agreed policy (merit) is deferred to Phase 11, together with Codex's sweep outcome.
- **T4 — Clearing all caches at every real step discards valid work** (`solver_bank.py:218-228`).
  - Modeled policies are pure functions of (frame, belief, settings, seed). A nested belief reached again after a real step has the same policy.
  - Clearing is a memory-bound workaround, not a semantic requirement.
  - This is the "reuse modeled policies across real steps" item from the old handoff. It only matters for the design that Phase 11 settles.
- **T5 — The root control variate is always on** (`i_pomcp.py:325-326`).
  - The immediate reward at depth 0 is always replaced by its exact expectation, even in the "sampled" profile.
  - The default profile is therefore not textbook POMCP either, and the RTS and IPF baselines don't get the same correction. This is a fairness concern for contribution 2 (Phase 4).

### Structure

- **S1 — Flag-driven control flow in `_simulate`.** Six configuration flags interleave in one 113-line method (`i_pomcp.py:229-342`). Leaf evaluation, backup operator, reward estimator and final recommendation should be separate strategy objects (Phase 1 S1).
- **S2 — The planner reaches into bank internals.** `recursive_chance` calls the private `planner._final_action_values` and `_history_successor`. `root_chance_experiment` calls `solver_bank._belief_row`.
- **S3 — `SolverBank` has seven responsibilities:**
  - the registry;
  - physical priors;
  - the policy cache;
  - canonical encoding and digests;
  - seeding;
  - the reward integrator;
  - the continuation-law integrator.

  The canonical encoding (`_belief_row`, `_encode_belief`, `_ordered_mass`) also duplicates the job of `FiniteBelief`'s own canonical key.
- **S4 — Global RNG reseeding.**
  - `search_randomness` reseeds the process-global `random` and restores it afterwards (`policy.py:61-69`). It is correct but fragile, and it is why threads are unsupported.
  - Every sampling call (`random.choices`, `sample_transition`, `add_particle`) uses the global RNG.
  - Pass an explicit `random.Random` per solve.
- **S5 — Imports inside functions** (`solver_bank.py:137-139,171-174,226`; `i_pomcp.py:124`; `planner.py:19-21`).
- **S6 — `Planner.visualize`** renders `get_detailed_stats()` into a PNG through graphviz on the base class. Visualisation belongs in the analysis layer.
- **S7 — Duplicated logic:**
  - `_root_recommendation` (`i_pomcp.py:128-167`) is a depth-1 special case of `recursive_recommendation`;
  - the "legal set constant over belief" check appears five times (`i_pomcp.py:73,194`, `finite_filter.py:127`, `rts_planner.py:55,116`);
  - "one root action set" is computed separately in each place.
- **S8 — `SolverKey` duplicates `AgentFrame`** (Phase 2 S1).

### Docs

- The package docstring (`solvers/__init__.py:8-9`) says I-POMCP uses "interactive particle filtering"; it does not (T2, Phase 2 T1).
- `node.py`'s reservoir documentation describes an unused feature.
