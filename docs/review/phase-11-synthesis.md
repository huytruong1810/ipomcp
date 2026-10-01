# Phase 11 — Holistic review and restructure plan

This synthesises Phases 0–10 (`docs/review/phase-*.md`). Every claim below points to a finding there.

## 1. Verdict per research contribution

### Contribution 1 — an online sampling I-POMDP solver combining I-NTMCP and IPF

**What exists.** The planner is *root-sampling MCTS over an exact, enumerated interactive belief* (Phase 3).

| Element of the claim | Literature mechanism | What the code does | Gap |
|---|---|---|---|
| IPF (Doshi & Gmytrasiewicz 2009) | Sampled nested particle sets. Propagate by sampling a_j, s′, o_j; weight by O_i; resample per level. Cost is linear in particles per level. | Exact enumeration of every a_j × s′ × o_j at every level (`FiniteInteractiveFilter`), plus my opt-in deterministic stratified resampling | **No IPF exists** (Phase 2 T1). Exact support grows exponentially: 2,428 atoms by step 16 for L2 Tiger. Wumpus priors can't be represented (Phase 5 B1). |
| I-NTMCP (Schwartz, Zhou & Kurniawati 2022) | Persistent nested trees per (agent, level), reused across steps. Opponent beliefs are particle sets at nested tree nodes. | A fresh, memoised MCTS solve per distinct nested belief. All caches cleared at every real step (`solver_bank.clear_caches`). | **No persistent nested trees and no reuse** (Phase 3 T4). Each L3 decision runs a cascade of nested solves. |
| "Balancing simulating futures vs simulating other minds" | — | Two fixed budgets, `mcts.n_sims` and `opponent.n_sims` | **No allocation mechanism and no theory** (Phase 3 overview). The ladder showed the modeled budget dominates opponent-model error. |
| Convergence and optimality | — | Only the *validated profile* converges against exact references at L1 (all H ≤ 5) and L2 (H ≤ 5, Tiger). The *default profile* does not (Phase 3 T1). | The repository default is the non-convergent one, and every episode suite uses it (Phase 6 E1). |

**Soundness of what exists.** The interactive belief update is a correct, test-verified implementation of the I-POMDP update (Phase 2, and Phase 9's independent oracle). The validated search profile is consistent where it has been tested. The design is semantically right but computationally the wrong shape for the stated contribution. Its cost is the cost of exact I-POMDP filtering on every search path.

### Contribution 2 — benchmarking against existing solvers

| Comparator | Status |
|---|---|
| Exact alpha-vector I-POMDP | Only 2-state Tiger L1, which is a POMDP (Phase 4). There is no interactive-state alpha-vector solver. L2+ "exact" means three overlapping exhaustive tree enumerators, all Tiger-only. |
| IPF + RTS (Doshi) | `RTSPlanner` is a top-k weighted-particle lookahead over **this codebase's** exact nested filter and MCTS opponent models (Phase 4 B1–B2). It is not the published method. |
| Runtime, sample efficiency | Only i's wall-clock decision time is recorded (Phase 6 B3). There is no common budget unit across planners (Phase 4 B5). The control variate is applied to I-POMCP only (Phase 4 B3). The L3 ladder timings include audit time (Phase 8 B1). |
| Error bounds | Exact error is measured only while the exact reference is tractable: L1 all H ≤ 5; L2/L3 ≈ the first 3–8 steps. Bounded-support error had a false bound, now corrected (Phase 2 B2). Nothing propagates error through time or nesting (Phase 2 T2). |

### Contribution 3 — interpretability and equilibria

| Element | Status |
|---|---|
| Sunbursts at high levels | The plotly version shows level masses at every depth. The PDF shows two rings only (Phase 7 B3). Wedges merge all nested models of a level, so *mental content* is invisible (Phase 7 I1). No error overlay (Phase 7 I2). |
| Equilibrium finding | A "Nash" check over a 5×5 level matrix, by exact float equality on noisy default-profile means (Phase 6 E4). It is not a statistical or game-theoretic result. |
| Domains | Tiger's creak law and opener observation are unverified against G&D 2005 (Phase 5 T1–T2). The UAV can't observe its own position (Phase 5 T5). Wumpus priors miss the true state in 20/20 probes (Phase 5 B1). |

## 2. Cross-component soundness issues

1. **Information leaks are prevented by runtime checks, not by the types.** `get_legal_actions(state)` and `get_rollout_action(state)` accept hidden state. Five call sites check after the fact that the result is constant over the belief (Phases 1, 3, 5).
2. **Randomness is global.** Planners reseed and restore the process-global `random` module. Domains mix that with optional `rng` arguments (Phase 1 T3, Phase 3 S4). This is why threads are unsupported and why reproducibility depends on call order.
3. **"Exact" is overloaded.** It means three different things, in classes with similar names (Phase 4 T1). Tie rules (four variants, Phase 8 B5) and horizon conventions (decreasing vs receding, Phase 4 T2) differ across references.
4. **Identity is split.** `AgentFrame` vs `SolverKey`, and physics compared by object identity, force the separate canonical encoding (Phase 2 S1–S2). That encoding now lives in `ipomdp/canonical.py` (fixed).
5. **Behaviour is selected by flags.** Six interacting `MCTSConfig` booleans drive a 113-line `_simulate` (Phase 1 S1, Phase 3 S1). The 22-parameter `evaluate_case` follows from it (Phase 6 S6).
6. **Experiments are hand-written.** 23 entry points, 11 near-identical `_setup_domain` bodies, and literal budgets (Phase 6). Nothing declares a single experiment specification.
7. **Machinery exists to keep the past runnable.** It includes:
   - frozen gates;
   - byte-exact replays;
   - `unittest.mock` in production code;
   - a backward-compatibility proxy;
   - config serialisation that hides default fields "for historical seeds".

   Together these are about 1,300 lines plus 35 MB of tracked JSON (Phases 6, 8). The target removes all of it.

## 3. Confirmed bugs

| Phase | Finding | Status |
|---|---|---|
| 2 | B1–B3: resampling depended on order, its bound was false, and strata used `repr` keys | **Fixed** (+ regression tests) |
| 2 | B4: `FiniteBelief` normalization not idempotent | Restructure (no live path affected) |
| 3 | B1: the root reservoir is always empty, so the UAV metric is never recorded | Restructure (reservoirs deleted) |
| 3 | B2: internal-node legality comes from one sampled hidden state | Restructure (type-level fix, item 1 above) |
| 5 | B1: the Wumpus prior misses the true state | Restructure (factored exact prior, or IPF) |
| 6 | B1: payoff-matrix snapshots never fire; B2: palette mismatch; B4, B6, B7: dead code; B8: snapshot run skips the source lock | Restructure (the suites are being rewritten) |
| 6 | B3: only i's wall time is recorded | Restructure (CPU time per agent per component) |
| 7 | B1: the belief-health plot is always empty; B2: j's reward is drawn on i's clock; B3: the PDF sunburst has 2 rings; B4: level posteriors condition on survival | Restructure (the analysis layer is being rewritten) |
| 8 | B1: ladder planning time includes audit time; B2: pooled intervals ignore within-trial correlation | Documented. The L3 study is deprioritised, and its timing must not be quoted. |

These bugs sit in code that the plan below replaces. Patching them first would mean writing every fix twice, so they are documented and fixed by construction.

**Exception:** if you want the current suites usable *before* the restructure, Phase 6 B1/B2 and Phase 7 B1/B2 are each a few lines. I can patch them now.

## 4. Target architecture (DDD bounded contexts)

```
src/ipomdp/                         # one installable package; no generic top-level names
  model/        domain contract     POMDPModel (history-based legality, rollout on private info only),
                                    finite kernels with sampling derived from them, explicit RNG streams
  belief/       interactive beliefs FiniteBelief + canonical identity (done), exact update,
                                    particle update (IPF), resampling with stated bounds
  planning/     solvers             search tree, strategies (exploration, backup, leaf evaluator,
                                    final scorer) as typed objects; NestedModelBank with persistent
                                    nested trees and cross-step reuse; compute allocator
  reference/    exact solvers       one recursive exact reference, parameterised by how opponents are solved
                                    (exact | declared finite policy), domain-agnostic; L1 alpha vectors;
                                    shared tie rule and horizon convention
  baselines/    comparators         IPF + RTS as published; random L0
  domains/      tiger, uav, wumpus  each with a declared spec document; sampling derived from kernels
  experiments/  one runner          declarative ExperimentSpec (TOML or dataclass) → episodes → long-format
                                    records (both agents' CPU time per component, beliefs, predictions)
  analysis/     metrics + figures   one metrics layer (error vs exact, convergence, cluster-bootstrap CIs);
                                    renderers (sunburst with mental content and error overlay)
  infra/        execution           process supervisor, provenance (git commit + dirty diff), logging
tests/  contracts/ references/ unit/ qualification/ integration/     (Phase 9 target)
docs/   one current document per context + RESULTS.md; history in git tags
```

**Principles applied:**
- **SOLID.** Strategies replace flags (open/closed). One responsibility per class: the bank splits into registry, policy cache and integrators.
- **DRY.** One canonical encoding, one reward integrator per layer, one statistics module, one runner.
- **KISS.** No compatibility paths, no versioned serialisation, no in-production monkeypatching.

## 5. Plan: small scale first

Each step ends with tests green, and with a qualification run at small scale against exact references before anything larger is attempted.

| Step | Work | Small-scale gate before moving on |
|---|---|---|
| **R0** | Archive: tag the frozen gates, the replay harness and the historical docs; back up `results/`; delete the default-profile results. *Needs your decisions (§6).* | — |
| **R1** | Package layout and renames. `core`/`utils` become `ipomdp.*`. Delete dead code (Phase 1 S3, Phase 4 S6, Phase 6 S3). Explicit RNG streams. History-based legality. | Full suite green. Seeded trajectories reproducible run to run. |
| **R2** | Planning strategies. Make the validated profile *the* planner, with no flags for the non-convergent options. Typed `PlannerSpec`. | L1 qualification: H 1–5, budgets 100–10k, 11 beliefs × 5 seeds, against alpha vectors. 100% optimal from 1k, and Q error shrinking with budget. |
| **R3** | One domain-agnostic exact reference. Shared tie rule and horizon convention. | Matches L1 alpha vectors and the existing independent L2 oracle. Reproduces the old references' Q values on their test cases. |
| **R4** | Domain contract suite. Tiger spec checked against G&D 2005. UAV and Wumpus specs decided (§6). Factored exact Wumpus prior. | Kernel = sampler (statistical test) for all domains. Zero-evidence rate is 0 under random play. |
| **R5** | **IPF belief propagation** (sampled nested particles, with reinvigoration) as a belief strategy beside exact enumeration. | Against the exact filter on Tiger L1–L3 (≤ 8 steps): level-posterior and nested TV error shrinking as K grows, with a fitted rate. |
| **R6** | **Persistent nested trees** (I-NTMCP-style), with cross-step reuse and a compute allocator between root and nested simulations. | Matches R2's accuracy at L2 with lower CPU. Per-component CPU accounting proves where the savings come from. |
| **R7** | IPF + RTS baseline as published. One budget unit: generative calls and CPU seconds. | Tiger L1/L2 against exact. Same accuracy table as R2. |
| **R8** | Analysis and interpretability: one metrics layer, sunbursts with content and error, confidence-aware equilibrium analysis. | Figures regenerate from records alone. Metric unit tests. |
| **R9** | Scale-up ladder: Tiger L1 → L4, then UAV, then Wumpus. Each rung runs only after the previous one passes its qualification gate. | Preregistered budgets and n per rung. |

R1–R4 are refactors that keep behaviour, plus spec fixes. R5–R7 are the research contribution itself. Today's code doesn't contain it yet.

## 6. Decisions I need from you

1. **Codex/Antigravity work.** Recursive chance is exact finite-tree value iteration on the visited tree (Phase 3 T3). It helps at Tiger L2 and can't scale to UAV/Wumpus.
   - My recommendation: archive it and its gates once Codex's sweep audit is recorded, and keep the validated profile as the planner.
   - Do you agree, and are Codex and Antigravity done?
2. **Results.**
   - Where should the 6.5 GB `results/` backup go?
   - May I delete `payoff_matrix` (1.3 GB) and `deep_prior` (369 MB)? They are default-profile and not interpretable.
3. **Domain specs.** Before R4 I need:
   - (a) the Tiger spec you want: strict G&D 2005 (false creaks, opener's observation), or this repository's variant, declared;
   - (b) the intended UAV spec. Should the UAV know its own position? Should it observe the target's relative direction?
   - (c) the Wumpus grid and pit count to keep (4×4/2 pits gives a 94,500-state prior).
4. **Package name.** `ipomdp` as the single top-level package (the project is `i-pomcp-py`), or something else?
5. **Scope of R5–R7.** These implement the claimed contribution: IPF beliefs, persistent nested trees, a compute allocator. Do you want them designed and built here, or does Codex own the mathematics while I implement?
