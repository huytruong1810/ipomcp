# Handoff

Read this first. Then read, in order:
- [RESULTS.md](RESULTS.md): every number below, with its n and its reference;
- [THEORY.md](THEORY.md): what is computed exactly, what is estimated, and the three meanings of "exact";
- [ARCHITECTURE.md](ARCHITECTURE.md): how the package fits together.

Last updated October 7, 2026, after the reviewer meeting. The reviewers liked the episode-study page. The October 5 plan (PLAN.md, W1–W8) is mostly done: RESULTS.md 12.12–12.21. `main` holds the restructured package (merge `f40c525`); Claude is the sole owner, and there are no other agents.

## 00. October 7 evening session (newest; read before Section 0)

The October 7 (evening) phase is in PLAN.md (top). Done so far:
- **P1** `JointTigerModel` (`domains/tiger.py`, study domain `tiger-joint`); `oracle` and `panel` take a `physics` and levels up to 4; `nested_beliefs` (one point belief per level below j) replaced `nested_belief`. 309 tests pass.
- **P2** exact payoff screen (RESULTS 12.22): the published game gives level 3 nothing; **`together_safe = 40` (the coordination Tiger)** makes each level count (2→3 worth 2.10 at H3, 3→4 0.81 at H2).
- **P3** gates on the coordination Tiger (12.23): 1,244/1,244 optimal, including the first L4 gate (486/486 at H2).
- **P5** search-error attribution (12.24): at H3, 98% chance error (removed by scoring); at L2 H4, 22% deeper greedy choices and 9% frontiers remain.
- **P4 (12.25, provisional):** levels 1–3 at N = 32, n = 100: level 2 beats level 1 by 25–29 against a level-1 opponent (Holm p ≤ 0.001); no 2 → 3 step survives. The level-4 cells reached n = 14–15 at N = 32 and were stopped (memory peaked near the limit with 16 workers; the watchdog never had to act).
- **Correct-prior check** (`results/p4b_correct_prior_20261007`): level 3 with a correct prior did not beat level 2 at N = 32, but predicted j *worse*. Diagnosis (**12.26**): N = 32 compression drops the opponent's realized branch (zero-probability predictions in 3–20% of trials above level 1). N = 64 removes it in a 10-trial probe; N = 128 adds nothing at 4.3× the CPU.
- **N = 64 reruns, done (12.27):** grid levels 1–3 (`results/p4c_joint_level_grid_n64_20261007`) and the correct-prior check (`results/p4b_correct_prior_20261007/trials_n64`), n = 100, same seeds. Level 1 → 2 gains 27–30 against a level-1 opponent and about 15 against level 3 (Holm p ≤ 0.03); level 3 predicts best (log-loss 0.41 vs 0.50) but its reward gain at H2 is +5 to +7, none surviving correction. Compression misses fell to 0–6% of trials per cell.
- **P6 done:** reviewer page republished with a "coordination Tiger" section (version 18, https://claude.ai/artifact/PqpRuTLcuSq5afJdSSxoCB; generator `results/w4_report_20261007`, now reading `p4c` and `p2`); Claude Doc report has a coordination paragraph, a bounded-belief caveat, and the error-attribution row marked measured.
- **Nothing is running.**
- **Next, in order of value per CPU-hour:**
  1. **Belief gate redesign** (contribution 1): the 12.16 gate missed dropped branches. Add a gate on the probability a bounded belief gives to realized opponent actions (zero-probability rate), and consider a compression that keeps every opponent-observation branch above a mass floor. Cheap to test on the existing trial seeds.
  2. **Depth at H3** (contribution 3): level 3 vs level 2 on the coordination Tiger at H3, N = 64, where the screen puts the value of a third level (2.10). About 100 CPU-h for n = 100 in two conditions; PLAN's gate for it (an H2 effect against j at level 2) was not met (Holm p = 0.10), so it needs the user's call or a cheaper pilot first.
  3. Level 4 on the coordination Tiger at N = 64 (about 7× N = 32 per step; the N = 32 cells reached n = 14–15 and are not reported).
  4. HANDOFF 0.2 items 3–4 (Wumpus beyond 3 beliefs; root adaptive budgets, which 12.24 shows would act on the 22% deeper-choice error at H4).

## 0. Start here (fresh agent)

- **State.**
  - `main` is clean; 292 tests pass in about 90 s; ruff is clean.
  - Nothing has been pushed, and nothing is running.
  - The last commits are the RESULTS 12.18–12.21 write-ups and this file.
- **Back to the normal schedule.** The October 7 reviewer deadline is over. Resume the mandate in Section 1: rigorous, gated progress on the three contributions, with redesign over scaling and A/B gates with statistics. The user wants autonomy: take the recommended option, state it in one line, and keep going.
- **Before running anything:**
  - Read Section 6 (what not to repeat) and the standards in Section 1.
  - The machine is shared: `~/projects/jepa_ipomdp` runs multi-core jobs using up to about 12 GB. Never touch its processes.
  - Compare CPU only within paired, same-process runs.
- **Results** go in a new dated folder under `results/`, with the script copied inside. Numbers go into `docs/RESULTS.md`; the next free section is 12.22. Update this file, and the Claude Doc report (Section 4), when results land.
- **First task for the new session:** write the next plan, appended to PLAN.md as a new phase, from Section 0.2 below. Order it by value per CPU-hour, then start executing.

### 0.1 What happened October 5–7 (PLAN.md W1–W8)

| Work | Result | RESULTS |
|---|---|---|
| W1 speed up simulating minds | 2.2× faster in situ at L3 H4 with **bit-identical** outputs (O(1) model validation, cached hashes, shared successor models, atoms grouped as built) | 12.12 |
| W2 harm of near-tie j mistakes | Flipping j's exact near-tie decisions (gaps 0.15–0.3) never changed i's exact decision. No gap below 0.05 existed to test | 12.13 |
| W3 the near-tie floor | It was the chance-scoring **fallback**, not rollout noise. Actions with two steps left are now scored exactly over every branch. L3 H4 residual misses fell from 14 to 4 of 80 (p = 0.013); i's Q error at L3 H3 fell from 0.06 to 0.0006 | 12.14 |
| Stopping rule | Rescaled to the measured rate 1/n (was 1/√n): same decisions on 2,016 replayed solves, 14–23% fewer simulations | 12.15 |
| Certified bounds (W6) | Every real decision returns per-action intervals that contain the exact Q of the searched model | 12.17 |
| Episode study (W4) | L3 i vs L2 j, plus L2 and L1 controls, n = 100 each, paired. Modeling j cuts prediction log-loss from 1.10 to 0.45; **L3 adds nothing over L2**; no robust reward effect | 12.16, 12.18 |
| Wumpus L1 (W7) | The pilot's losses were the simulation budget. Qualified at search N = 1,024 with 3,000 simulations (9/9, **3 distinct beliefs only**) | 12.19 |
| I-POMCP vs IPF + RTS (W5) | I-POMCP is 380/380 at L2 H3 and H4; IPF + RTS is 95–99.5%, and 3–5× slower at H4 | 12.20 |
| Level grid | i and j at levels 1–4 each (16 cells, n = 39–42, H2, paired seeds). Prediction separates L1 from L≥2; identifying the opponent's level weakens as levels rise; no consistent reward effect | 12.21 |
| Equilibria (W8) | Design note `docs/EQUILIBRIA.md` (subjective equilibrium, i.e. prediction merging). Episodes record each agent's predicted opponent action | — |

**Reviewer page (presented October 7):** https://claude.ai/artifact/PqpRuTLcuSq5afJdSSxoCB. It is linked from the Claude Doc report.

- To rebuild: `cd results/w4_report_20261007 && ../../.venv/bin/python analyze.py && ../../.venv/bin/python build_report.py`, then republish `report.html` with the Artifact tool (pass the `url` from a new session).
- Grid data comes from `results/w9_level_grid_20261007/analyze_grid.py`.
- The page text, lead included, is generated from the data. Edit `template.html` and `build_report.py`, not `report.html`.

### 0.2 Goals and next steps (input to the next plan)

The science so far:
- The solver is accurate wherever exact references exist.
- It beats the published baseline.
- Its stopping and scoring are consistent.

The weak point the reviewers saw is that **Tiger does not reward deeper reasoning**: an agent's payoff depends only on its own door, so level 3 cannot beat level 2 on reward (12.18, 12.21). The next phase should show where higher levels pay off, and finish what the plan left open.

Recommended order (re-check costs before committing):

1. **A domain where opponent modeling pays (contributions 1 and 3).**
   - **Option A:** a Tiger variant with joint-action payoffs, e.g. a bonus for opening the same safe door together, or a penalty for opening when the other listens. Cheap, and every gate and tool works unchanged.
   - **Option B:** UAV pursuit (`domains/uav.py`), where the opponent's moves are the point. Needs an oracle and panel path (both are Tiger-specific: `_two_state_belief`, `_point_model`).
   - **Recommendation:** start with A and a small L1–L3 grid, using `results/w9_level_grid_20261007/grid.py` and `studies/`. Gate decisions at H2 and H3 against the exact I-POMDP first.
2. **Level 4 in the drivers (W8):**
   - `oracle.py:359` rejects level > 3;
   - `_two_state_belief` supplies only 3 nested beliefs;
   - `panel.py` allows levels 2–3.

   `modeled_levels` and `HierarchyBuilder` are already generic. Exact L4 is tractable at H2 (0.2 s per solve) but not at H3 (more than 10 minutes, killed).
3. **Wumpus beyond 3 beliefs:** the three seeds of `wumpus_qualification` produced the same observations. Vary the scripted history (or sample it per seed) to get distinct beliefs, then attempt L2 Wumpus.
4. **Root budget allocation (C, contribution 1):** adaptive stopping for real decisions, using the certified bounds (12.17) or the 1/n-calibrated test (12.15). Real decisions also need Q accuracy for error bounds.
5. **Error attribution at L2 and L3:** `analysis.error_attribution.trace_margin`, the last "Not yet measured" row in the Claude Doc report besides equilibria.
6. **L3 H4 residual:** 4 of 80 misses, all at 0.03-gap beliefs. A PAC-style certification standard (12.13) is the user's decision and has not been asked. Exact integration deeper than two steps is not recommended (Section 6).
7. **Cost at high levels:**
   - Level-4 planning grows with belief size: at N = 64 it reached 65 s per step by step 8, against 1–2 s at N = 16.
   - Nested belief solving dominates in-situ CPU (12.12, 12.15). The 1/n stopping rule saved simulations but not in-situ CPU (0.88×, p = 0.09).

Housekeeping, still the user's call (Section 1):
- about 12 GB of legacy `results/` and stale branches;
- `results/` is git-ignored and not backed up since September 30. It includes the reviewer-page generator (`results/w4_report_20261007`) and every October dataset, so copy it off the machine.

### 0.3 Files touched October 5–7

- **`src/ipomdp/belief/`:**
  - `frame.py`: cached hash, explicit pickle state;
  - `finite_belief.py`: `opponents` summary, O(1) `MentalModel` validation, cached model hash, one-pass validation, identity shortcut in `__eq__`, `FiniteBelief.from_groups`;
  - `finite_filter.py`: `advance` (memoized successor models), rows grouped as built.
- **`src/ipomdp/planning/`:**
  - `chance_scoring.py`: exact two-steps-left scoring, `bounds=True` certified intervals, `_Span`;
  - `i_pomcp.py`: `_search(..., bounds)`, `grow(..., bounds)`, bounds in stats, `advance` in `_history_successor`;
  - `generative_model.py`: `advance`;
  - `stopping.py`: 1/n scaling.
- **`src/ipomdp/reference/best_response.py`:** uses `advance`.
- **`src/ipomdp/experiments/`:**
  - `bootstrapper.py`: `modeled_configs` per level, replacing `modeled_config`;
  - `study.py`: `AgentSpec.modeled` stack label, replacing `modeled_planner`;
  - `episodes.py`: decision values, bounds and predictions recorded before `update_root`; `_prediction`;
  - `oracle.py`: `bound_width`, `bounds_contain_reference`;
  - `belief_qualification.py`: exact belief once per history, `--steps`;
  - new `wumpus_qualification.py`.
- **Tests:**
  - new `test_study_spec.py`, `test_wumpus_qualification.py`;
  - bound and two-steps-left tests in `test_chance_scoring.py`;
  - `test_stopping.py` (depth 4; 1/n thresholds);
  - `test_rts.py` (`modeled_configs`).
- **Docs:**
  - new `PLAN.md`, `EQUILIBRIA.md`;
  - `THEORY.md` (two-steps-left scoring, bounds);
  - `RESULTS.md` 12.12–12.21;
  - `studies/tiger_l3_vs_l2.toml` (H3, root 1,000, compressed 64, stacks `6400s/100` and `400`, 6 workers).

**Measured costs** (CPU per unit; 24 cores; the other project uses up to about 12 GB at times):

| Unit | Cost | Source |
|---|---|---|
| j-L1 modeled solve, L2 H4, cold bank | about 0.37 s `6400s` (fewer simulations since 12.15) | 12.11, 12.15 |
| i decision, L2 H4, root 1,000, `6400s` | about 11–17 s | 12.18, 12.20 |
| j-L2 query, L3 H3, cold bank, `6400s/100` | about 4–5 s (1.18× after 12.14) | 12.14 |
| j-L2 query, L3 H4, in situ, `1600/exact` | 0.45× of 12.11's cost (12.12); first 40 queries about 16 s each | 12.12, 12.15 |
| Episode, L3 i vs L2 j, H3, 20 steps | about 30 CPU-min (i plans about 91 s per step); L2 i about 5 s per step | 12.18 |
| Episode at H2, N = 32 | 0 s (L1 vs L1) to 62 s per step (L4 vs L4); about 20–30 min for an L4 episode | 12.21 |
| Exact best response, Wumpus L1 H3 | 1,034 s at the prior; about 190 s after 2–4 steps | 12.19 |
| Exact I-POMDP, Tiger, H2 | under 2 s up to level 5 | probe, October 7 |
| Exact I-POMDP, Tiger, L4 H3 | over 10 minutes for one solve (killed) | probe, October 7 |

**Reusable tools and data** (each folder holds its script):
- `results/stopping_v2_20261006`: 2,016 solves with checkpointed main and pilot trees under the current scoring; `replay.py` replays any stopping rule offline.
- `results/w1_profile_20261005`: in-situ cProfile and the bit-identity bench (`bench.py`, `gate_w1.py`). Use these for any refactor that must keep outputs identical.
- `results/w3_variance_20261005`: fixed-tree scoring diagnostics (`a0_frontiers.py`, `a0b_fallbacks.py`): score the same trees several ways for exactly paired comparisons.
- `results/w2_near_tie_harm_20261005/harm.py`: flips modeled decisions inside the exact hierarchy (wrapping a level via `register_levels(wrap=...)`).
- `results/w9_level_grid_20261007/grid.py`: any level × level episode grid, round-robin, paired seeds, resumable.
- `results/w4_report_20261007`: the reviewer page generator.
- Older: the L3 H4 offline replay (`l3_h4_tree_trajectories_20261005`, scores from before 12.14), the residual set (`stopping_redesign_gates_20261005/residual_l3h4.py`), and the gate scripts in `stopping_redesign_gate(s)_20261005`.

**Run hygiene learned the hard way** (details in Section 6):
- Jobs append a row per unit of work (JSON lines) and skip finished rows on restart. A job that writes only at the end ran 5 hours with no visible progress.
- To keep editing `src/` while a long run is going, launch it on a frozen copy: `cp -r src/ipomdp <results>/src_snapshot/` and `PYTHONPATH=<results>/src_snapshot`. The editable install is a plain `.pth` path, so `PYTHONPATH` takes precedence.
- Size runs from a profile of a whole representative problem and of the hardest one. The first queries of a problem are the cheap ones.
- `ProcessPoolExecutor(max_tasks_per_child=1)` uses spawn, which frees memory per job; guard launchers with `if __name__ == "__main__":`. When workers write their own outputs, the parent can be stopped (by PID) without losing running jobs; otherwise stopping it loses them.

## 1. Goals

### The user's mandate (since September 30)

The main aim is **not** the L3-vs-L2 Tiger study. The main aim is a rigorous review and clean restructure of the codebase, so that it supports three research contributions:

1. **An online I-POMDP solver** (nested I-POMCP with bounded interactive beliefs). It balances simulating futures (root simulations) against simulating minds (modeled-opponent simulations).
2. **Benchmarks** against the exact alpha-vector I-POMDP and against IPF + RTS: runtime, sample efficiency and error bounds.
3. **Interpretability.** Sunbursts of nested beliefs at high strategy levels, with quantified error, plus equilibria, in Tiger, UAV and Wumpus.

### Standards the user set

- **Progressive scaling.** Never run large before small-scale behaviour is verified optimal against an exact reference.
- **Engineering quality.** Big-Tech-level code: DDD, DRY, KISS, SOLID.
- **Free to restructure.** Restructure, rename, delete and coalesce freely, including results.
- **No legacy.** No fallbacks, deprecation shims, backward compatibility or schema versions. The code should read as if it was written once.
- **Documented decisions.** Verbose docstrings and comments that record design decisions.
- **Autonomy.** Take the recommended option on decisions without asking. Ask only when genuinely unclear.
- **Redesign on diminishing returns** (October 3). When more budget gives diminishing returns short of optimal, first rule out a bug, then redesign the algorithm from state-of-the-art MCTS/UCB research. Gate every redesign by A/B tests with logged statistics (paired seeds, exact references, intervals), not intuition. Use the literature and informed intuition to skip experiments that cannot succeed (12.6 skipped Sequential Halving this way).

### Ground rules

- Work in `/home/andyj1810/projects/ipomcp` on `main`; do not clone.
- Run code with `.venv/bin/python` from the repository root. The editable install points at `src/`, so no `PYTHONPATH` is needed.
- **Do not push to `origin`** unless the user asks.
- Back up before deleting evidence. Kill processes by PID only.
- **Housekeeping.** The worktree and the legacy caches were removed on October 2. Two optional items remain; the permission classifier blocks Claude from deleting them, so the user decides:
  - **Legacy raw outputs, about 12 GB in `results/`:** `oracle/`, `archive/`, `deep_prior/`, `payoff_matrix/`, `review-20260923/`, `tiger/`, and the `*-202609*` folders (`bellman-backup`, `exact-tail`, `exploration`, `history-rewards`, `l2-contract`, `l2-finite`). They are evidence for the retired code only. Material up to September 30 is backed up on `C:\`. The recursive-chance panels of September 30–October 1 and Claude's audit (`results/oracle/l2_recursive_chance_development_claude_audit_20261002/`) are not backed up yet. Every other folder in `results/` belongs to the current package.
  - **Stale branches:**
    - Already in `main`'s history, so safe to delete with `git branch -d`: `claude/restructure`, `claude/l3-l2-tiger-experiment`, `claude/snapshot-and-hardening`, `codex/final-20261002` (also tagged), `refactor/bigtech-hardening` and `review/rigor-20260916`.
    - **Not** in `main` and not tagged (checked October 2): `archive/review-snapshot-notes-20260923`, `archive/tiger-baseline-20260923` and `backup/antigravity-before-supervision-20260919`. Deleting them loses their commits, so keep them or tag them first.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Report results with chosen methods only, label interim n, and define jargon (the user's report style).

## 2. Current state

- **Tests.** 292 pass in about 90 s on `main`; `ruff check` and `ruff format --check` are clean.
- **Running jobs.** None.
- **Archives.**
  - Tag `archive/pre-restructure-20260930`: the pre-restructure code.
  - Tag `archive/codex-final-20261002` (branch `codex/final-20261002`): Codex's and Antigravity's final working tree, preserved unchanged. It holds the legacy recursive-chance candidate (`src/solvers/recursive_chance.py`), its protocols (`docs/RECURSIVE_CHANCE_*.md`), the frozen contracts (`benchmarks/`) and the launchers. `main`'s merge has it as a parent, but `main`'s tree is the restructure.
- **Results.** Stored under `results/` (git-ignored). A backup of everything up to September 30 is in `C:\Users\Andyj\ipomcp-results-backup-20260930`. Results from October 1–3 are not backed up yet; copy `results/` off the machine.

### What the package is now

```
src/ipomdp/
  model/        kernel-only domain contract (sampling derived from the kernels)
  domains/      tiger (published spec), uav, wumpus (exact 94,500-state prior)
  belief/       FiniteBelief, canonical identity, and three filters:
                  finite_filter       exact enumeration
                  compressed_filter   exact updates, supports compressed to N  <- planner default
                  particle_filter     published sampled IPF                   <- IPF + RTS baseline
                resampling (systematic draws, shared)
  planning/     IPOMCPPlanner, HorizonBoundUCB, POMCPNode (Bellman backups), SolverBank,
                chance_scoring (post-search exact chance weights),
                stopping (opt-in sequential stopping for modeled policies: one growing tree)
  reference/    BestResponseReference (exact, any level/domain), ExactIPOMDPPlanner, alpha vectors (L1)
  baselines/    RTSPlanner (reachability-tree sampling)
  experiments/  study (TOML StudySpec), episodes, runner, bootstrapper,
                oracle (one problem vs the exact reference; levels 1-3),
                panel (grids of cells vs the exact I-POMDP; levels 2-3),
                modeled_qualification (a modeled level scored at the beliefs it is queried at),
                l1_qualification, belief_qualification
  analysis/     hierarchy (sunburst nodes with content and error), metrics, figures, statistics
  infra/        process supervision, source locks, resources, paths
studies/        tiger_l3_vs_l2.toml
```

### Estimator, as validated (do not change without re-running the gates)

- **Search.** I-POMCP over private histories, with empirical chance-weighted Bellman backups and horizon-bounded UCB from the declared reward bounds.
- **Recommendation.** After search, the tree is rescored with exact chance probabilities (`planning.chance_scoring`); actions with two steps left are scored exactly over every branch, sampled or not (12.14). The scored values decide, for real and modeled agents.
- **Exact pieces.** E[R | b_h, a] at every history, and Q_1(b_h, a) at the last step, integrated over the agent's posterior b_h.
- **State sampling.** Since `f793712`, each tree and rollout step draws its state from b_h, instead of carrying one root particle. The two are equal in distribution when b_h is exact; with bounded b_h only the new form is consistent. It was re-validated in RESULTS.md Section 9.
- **Beliefs.** `SolverBank(seed, particles=N, particle_filter="compressed" | "sampled", search_particles=M)`:
  - `particles` bounds the beliefs kept between real steps (unset: exact);
  - `search_particles` bounds the beliefs each search works on.

  Exact real beliefs with a compressed search is the Wumpus configuration.
- **Exact reference budget.** The oracle's exact I-POMDP uses `oracle.REFERENCE_BRANCHES = 50M`. At history 6 it completes in 146 s for one problem.
- **Modeled solves.** One search of `modeled_simulations` by default. `PlannerConfig(modeled_stopping=ModeledStopping())` switches to sequential stopping capped at `modeled_simulations` (`planning.stopping`, 12.11):
  - one main tree, seeded as the flat search of the cap and grown by doubling from 200, so at the cap it is the flat search;
  - 3 pilot trees of 200 calibrate a t-test on the main tree's leading gap, scaled by 200/n (the measured rate since 12.15; it was sqrt(200/n)), with α = 0.01 over all checkpoints.

  Search seeds exclude the rule, so flat results are unchanged. It replaced 12.8's replicate averaging, which was biased (12.10). The `6400s` results in 12.8–12.10 are the old rule.
- **Per-level modeled stacks** (`experiments.modeled_levels`, October 3). The drivers name each modeled level's solve, top-down: `400` flat, `6400s` stopping with cap 6,400, `exact`; `6400s/400` at level 3 is j-L2 with stopping over i-L1 flat 400. One label applies to every level. This replaced `opponent_budget`/`opponent_stopping` (oracle) and the panel-wide `modeled_stopping`; the panel's `modeled` cell is now this label (`exact` instead of 0).
- **Certified bounds.** Real decisions get per-action intervals containing the exact Q of the searched model (`chance_scoring(..., bounds=True)`, 12.17); modeled solves skip them.
- **Metrics.** Rows record `first_action_loss`, `max_abs_q_error` and `optimal_q_error`, for both search and scored values. Use `optimal_q_error` for error bounds (12.2), and Clopper–Pearson intervals (`analysis.statistics.clopper_pearson`) for optimal rates.

### Validated configurations (Tiger; root 1,000 simulations unless noted)

| Problem | Modeled opponent | Beliefs | Evidence |
|---|---|---|---|
| L1, H1–H5 | uniform L0 | exact | 220/220 at H5 (12.1) |
| L2, H3, k ≤ 4 | exact, or I-POMCP 400 (zero-wrong certified, 12.5) | exact or compressed 64 | 373–380/380 (12.3) |
| L2, H4 | I-POMCP 1,600 flat, or `6400s` (stopping, cap 6,400; 0.85× flat 1,600's CPU) | exact | 380/380 (12.4, 12.11) |
| L3, H3, k = 0 | I-POMCP 400 flat (i's decisions = exact); certified zero-wrong stack `6400s/100` | exact | 162/162 (12.7, 12.11) |
| L3, H4, k = 0 | **none zero-wrong.** `6400s/exact`: residual set 4/80 wrong after 12.14, all at 0.03-gap beliefs (loss ≤ 0.03) | exact | 12.10, 12.11, 12.14 |
| L3 vs L2 episodes, H3 | `6400s/100` for i, `400` for j, mixed level priors (gated 120/120) | compressed 64 (level TV ≤ 0.061 to step 5) | 12.16, 12.18 |
| Levels 1–4 × 1–4, H2 | decisions exact for the searched model at any budget (two steps left) | compressed 32 | 12.21 |
| Wumpus L1, H3 | search N = 1,024, 3,000 simulations: 9/9 (3 distinct beliefs) | exact real, compressed search | 12.19 |

## 3. Key results (details and n in RESULTS.md)

| Section | Result |
|---|---|
| 1 | L1 Tiger: 100% optimal against alpha vectors at H1–5 from 1,000 simulations (n = 33 per cell) |
| 5 | **Allocation:** the modeled-opponent budget decides correctness; the root budget only reduces Q error. (H4 L2's need was later refined to 1,600 with root 1,000; see 12.4) |
| 6 | I-POMCP vs IPF + RTS, L2 H3–H4: I-POMCP is more accurate at matched CPU. RTS cost grows 11–24× from H3 to H4 |
| 7 | With the **sampled IPF**, exact beliefs dominate wherever the exact reference is tractable |
| 8 | **The compressed filter** reverses Section 7. At N = 64 it is about as accurate as exact beliefs: k = 4 mid-episode, 0.18 vs 0.11 Q error at 1.9 s vs 26 s. At k = 6, where exact planning takes over 30 minutes, it is 100% optimal (n = 30). Belief errors are 4–40× smaller than the IPF's at equal N |
| 9 | Re-validation after per-step belief sampling: everything reproduces except H4 at modeled 100 (93% → 80%), which is opponent-model luck. Modeled 4,000 stays at 100% |
| 10 | Wumpus pilot (exact real beliefs, compressed search): runs without failure, but is **not accurate yet**: losses of 1.8–7.7 in 13 of 18 decisions |
| 11 | **Chance scoring:** decisions neutral (1 corrected and 2 regressed of 747 paired decisions); Q error lower at H2–H3 (L2 H3 k = 0: 0.13 → 0.05). Kept as the default |
| 12.1 | L1 H5, paired, n = 220: **no regression** from per-step belief sampling (the 0.88 → 1.46 was 3-seed noise). Chance scoring adds +0.11 Q error at 5,000 simulations, on clearly wrong actions only |
| 12.2 | **The long-horizon Q plateau is the metric.** Max Q error over all actions is set by clearly wrong actions UCB rightly neglects. Optimal-action Q error converges: L1 H5 0.40 → 0.12 from 1,000 to 5,000 simulations (all-action max 1.57 → 1.35). Now recorded as `optimal_q_error` |
| 12.3 | **L2 at n = 380 per cell** with Clopper–Pearson intervals: H3 98–100% optimal; H4 modeled 4,000 380/380. H4 modeled 100: 364/364 off the corners, 6/16 on the four corner problems (both agents confident), so the old 80% was the grid's mix |
| 12.4 | Wrong j-L1 decisions at the beliefs i's search queries cause every H4 miss; **modeled 1,600 gives 380/380** at 2.6× less CPU than 4,000 |
| 12.5 | `modeled_qualification` over 190 problems per horizon: mistakes cluster on a few beliefs. j's own loss does not separate harmful from harmless ones, so the level-local rule is **zero wrong**: H3 400, H4 6,400 (conservative) |
| 12.6 | Near-tie modeled decisions: **no bug, no allocation problem** (UCB splits the contenders evenly); the floor is return variance (about 6.5 per simulation). Sequential Halving would not help and was skipped |
| 12.7 | **L3 vs the exact L3 I-POMDP** (reference cross-checked against `build_exact_agent`): H3 k = 0 162/162 for exact, 400 and 100 modeled. Exact mid-episode L3 checks are intractable beyond H2 k ≤ 2 |
| 12.8 | **Sequential replicate searches with stopping** for modeled policies: equal accuracy to flat 1,600 at L2 H4 (380/380) at **0.47× the CPU** (paired, n = 40). At L3 H3, accurate (162/162) but 1.3× the CPU of flat 400, which already matches exact. Opt-in, chosen per level |
| 12.9 | **Per-level modeled stacks** and the first **L3 modeled qualification** (j-L2 at the beliefs i-L3 queries; 3,294 queries). At H3, flat 400 j-L2 is wrong on 4 near-tie beliefs (harmless to i: 162/162); i-L1's budget inside j-L2 changes no policy (100 = 400). Certified: `6400s/100`, 0 wrong, 0.75× the CPU of flat 1,600. Stopping floor: 4 or 5 replicates cost 29–57% more and fix nothing; keep 3 |
| 12.10 | **L3 H4 qualification** (27 problems, 2,631 queries): no stack qualifies. The misses are confident j beliefs. The replicate stopping rule is **biased** there: on 0.19-gap beliefs it is 16–20/30 wrong even at a 25,600 cap, while one 6,400 search is 0/30. i-L1 at 100 vs exact makes no difference |
| 12.11 | **Consistent stopping:** one growing tree (seeded as the flat cap search) with a pilot-calibrated t-test. L3 H4, 666 beliefs: the same 6 wrong as flat 6,400 at 0.52× the in-situ CPU; residual beliefs 14/80 wrong (old rule 57/80, flat 6,400 14/80). Re-gated: L2 H4 380/380, L3 H3 162/162 and 0/3,294 queries. A single tree's spread falls as about n^-0.8; replicate averaging only as k^-0.5 |
| 12.12 | Simulating minds **2.2× faster in situ**, bit-identical (40/40 L3 H4 queries, 183/183 L3 H3 beliefs) |
| 12.13 | Flipping j's exact near-tie decisions (gaps 0.15–0.3; 110 beliefs, 74 problems) never changed i's exact decision |
| 12.14 | **The near-tie floor was the fallback:** exact scoring with two steps left; L3 H4 residual 14 → 4/80; i's L3 H3 Q error 0.06 → 0.0006; 1.18× CPU |
| 12.15 | Stopping scaled at the measured 1/n: same decisions, 14–23% fewer simulations; in situ CPU 0.88× (not significant) |
| 12.16 | Episode-study gates: mixed priors 120/120; compressed 64 at L3 level TV ≤ 0.061 to step 5 |
| 12.17 | Certified bounds on every real decision (containment tested in 15 cases) |
| 12.18 | **Episodes, L3 vs L2 with L2 and L1 controls (n = 100 each):** prediction log-loss 0.46/0.45 vs 1.10; L3 = L2; no robust reward effect for i or j |
| 12.19 | Wumpus L1 qualified at N = 1,024, 3,000 simulations (9/9; 3 distinct beliefs) |
| 12.20 | I-POMCP 380/380 at L2 H3 and H4 vs IPF + RTS 95.3–99.5%; RTS 3–5× slower at H4 |
| 12.21 | **Level grid 1–4 × 1–4 (n = 39–42):** prediction separates L1 from L≥2; identification weakens with level; 2/24 own-level reward effects exclude zero (about chance) |

**Implication the user has been told about.** Contribution 1's bounded belief is now the compressed filter, not the published IPF; the IPF remains the baseline's belief. If the user wants the published IPF inside the proposed method too, that changes the story; ask before assuming.

## 4. Where Tiger stands, and next steps

### Assessment and next steps

Superseded by Sections 0.1–0.2 (October 7). Section 2's validated configurations and Section 3 hold the evidence.

### Reviewer report

The Tiger report for reviewers is a Claude Docs document, rewritten October 2 around the current package: "I-POMCP vs Exact I-POMDP: Tiger Performance and Interpretability", https://claude.ai/code/artifact/96fa514f-dc7c-477c-8f97-b31eb1b8fca7.

- **Update it in place;** never create a new one. Edit it with the Claude Docs connector (load the docs skill first).
- **Sections:**
  - Setup with a metric-definitions table;
  - L1; L2; root vs modeled budget;
  - bounded beliefs; benchmark vs IPF + RTS;
  - cost; interpretability sunbursts with per-depth error;
  - a "Not yet measured" table.
- **When results land,** update the matching section and remove the row from "Not yet measured".
- **Charts** are hand-drawn widgets with inline rows.
- **Sunburst and belief-growth data** come from `results/report_hierarchies_20261002/hierarchies.py` (one history, n = 1).
- **Allocation, belief and baseline sections** predate the last two estimator revisions. The doc says so; their key cells reproduced in Sections 9 and 11.
- **Updated October 2 with Section 12:** lead, an optimal-action Q error metric row, the L1 plateau finding, a 380-problem L2 table (with the modeled 400/1,600 sweep) and the allocation rule (1,600 at depth 4). The IPF + RTS comparison (97% vs 80–83%) is still from the 15-problem grid; next step 6 re-runs it.
- **Updated October 4–5:** the Level 3 section has the H3 qualification table and an H4 paragraph (no certified stack; the adaptive rule's bias); the adaptive-budget bullet says per-level solves.
- **Updated October 5, evening (12.11):** the adaptive-budget bullet describes the growing-tree rule, and the L3 H4 paragraph gives its result (flat 6,400's decisions at half the CPU; near-tie misses remain).
- **Updated October 6–7:** an L3 H4 update paragraph (12.14); an "Episode study" paragraph and a "Level grid" paragraph, both linking the reviewer page https://claude.ai/artifact/PqpRuTLcuSq5afJdSSxoCB; the episode-metric rows of "Not yet measured" are marked measured. Still open there: error attribution at L2/L3 and equilibria beyond the design note.
- **Updated October 3:** a "Level 3 vs the exact I-POMDP" section (12.7), the adaptive-modeled-budget bullet in the allocation section (12.8), the lead's level-3 sentence; the level-3 row was removed from "Not yet measured".

## 5. What changed, and where (since `archive/pre-restructure-20260930`)

Everything under `src/ipomdp`, `tests`, `docs` and `studies` was rewritten or reviewed; `git log archive/pre-restructure-20260930..HEAD` has the full record.

**October 6–7** (full list in Section 0.3): W1 speedups in `belief/`; exact two-steps-left scoring and certified bounds in `planning/chance_scoring.py`; 1/n stopping; per-level `modeled_configs` and `AgentSpec.modeled`; episode rows record values, bounds and predictions; `wumpus_qualification`; `belief_qualification --steps`. Results folders: `w1_profile_20261005`, `w2_near_tie_harm_20261005`, `w3_variance_20261005` (A0, A0b and gates g1–g4), `stopping_v2_20261006`, `w4_mixed_prior_gate_20261006`, `w4_belief_l3_20261006`, `w4_episode_pilot_20261006`, `w4_episodes_20261006`, `w4_controls_20261006`, `w4_report_20261007`, `w5_rts_baseline_20261006` (pilot and full), `w7_wumpus_20261006`, `w9_level_grid_20261007` (with probes).

**October 5 (evening):**
- `planning/replicates.py` → `planning/stopping.py`: `ModeledStopping(pilot, trees, alpha)` is the growing-tree rule; labels `6400s[pilot,trees,alpha]`;
- `planning/i_pomcp.py`: `_GrowingSearch` (one tree grown in stages, scored at each), `_tree`, `_stopped_values`; `_search` is a full growth, with unchanged seeds;
- tests: `test_replicates.py` → `test_stopping.py`.

**October 4:**
- new: `experiments/modeled_levels.py` (`ModeledSolve`, stack labels, `modeled_keys`, `planner_config`, `register_levels`); `oracle.planner_config` moved there;
- `oracle.evaluate_case`: `modeled` (stack label) replaces `opponent_budget` and `opponent_stopping`; `reference="none"` skips scoring;
- `panel`: the `modeled` cell is a stack label; the panel-wide `modeled_stopping` is gone;
- `modeled_qualification`: levels 2–3, stack rungs, cheapest zero-wrong rung, `score_decision`; two phases (search per problem → `queries.pkl`, reusable with `queries_from`; one scoring job per distinct belief, cold CPU);
- `belief/finite_belief.py`: exact unpickling (masses restored bit for bit);
- tests: new `test_modeled_levels.py`; level-3 and unscored qualification tests.

**October 3:**
- new: `experiments/modeled_qualification.py`, `planning/replicates.py`;
- renamed: `experiments/l2_panel.py` → `experiments/panel.py`, with `level` in each cell, a `nested_beliefs` grid and a panel-wide `modeled_stopping`;
- `experiments/oracle.py`:
  - levels 1–3, via `_modeled_keys` (the opponent stack) and `_point_model` (nested point beliefs);
  - `opponent_observer` and `opponent_stopping`;
  - public `planner_config`, `optimal_q_error` and `OPTIMALITY_TOLERANCE`;
- `planning/config.py` gained `modeled_stopping`, and `planning/i_pomcp.py` gained `_search` and `_replicate_values`;
- `analysis/statistics.py` gained `clopper_pearson`;
- tests: new `test_modeled_qualification.py`, `test_l3_oracle.py`, `test_replicates.py`; `test_l2_panel.py` became `test_panel.py`.

**October 1–2:**
- new: `belief/compressed_filter.py`, `belief/resampling.py`, `experiments/study.py`, `episodes.py` and `runner.py` (they replace `batch_runner`, `experiments/config.py` and `tiger_l3_vs_l2.py`), `studies/tiger_l3_vs_l2.toml`;
- renamed: `experiments/ipf_qualification.py` → `belief_qualification.py`, which now compares both filters;
- `planning/solver_bank.py`: `particle_filter`, `search_particles`, `search_filter`, `search_model`, `draw`;
- `planning/i_pomcp.py` (per-step draws from b_h) and `planning/chance_scoring.py`;
- `baselines/rts.py`, `belief/particle_filter.py` (systematic resampling), `belief/finite_filter.py` (exact only).

**Docs:** README, ARCHITECTURE, THEORY (including the optimal-action Q error), RESULTS (Sections 1–12) and this file. The review findings are in `docs/review/phase-00..11`.

**Results directories,** each holding the launcher or probe script that produced it:
- belief accuracy: `ipf_resampling_probe_20261002`, `belief_qualification_compressed_20261002`, `compressed_prototype_20261002`;
- L2 panels: `l2_baseline_20261002`, `l2_baseline_h4_modeled4000_20261002`, `l2_ipf_miss_20261002`, `l2_mid_episode_20261002`, `l2_compressed_20261002`, `l2_compressed_k6_20261002`, `l2_revalidate_belief_sampling_20261002`;
- gates and references: `l1_qualification_belief_sampling_20261002`, `reference_scaling_20261002`;
- October 5, evening (12.11): `single_tree_bias_20261005` (one tree's convergence on the residual beliefs), `shared_bank_cost_20261005` (first 24 queries in one bank), `l3_h4_tree_trajectories_20261005` (main trees to 6,400 and 3 pilots at 666 beliefs; `replay.py` replays any rule offline; `src_snapshot/` is the frozen code the pilots ran on), `stopping_redesign_gate_20261005` (L2 H4 and L3 H3 near-tie sets, paired CPU), `stopping_redesign_gates_20261005` (i's decisions at L2 H4 and L3 H3, L3 H3 qualification, paired CPU, the residual check);
- October 5: `l3_modeled_qualification_h4_20261004` (9 nested-0.5 problems, old driver), `l3_modeled_qualification_h4_confident_20261004` (18 problems, two-phase driver, with `queries.pkl` reusable via `queries_from`), `l3_modeled_qualification_h4_oom_20261004` (the run that exceeded memory), `qualification_dedup_equivalence_20261004` and `..._prefix_20261004` (before and after the pickling fix), `purity_diagnosis_20261004`, `l3_h4_residual_probe_20261005`; memory profiles in `l3_qualification_pilot_20261003`;
- October 4: `l3_modeled_qualification_h3_20261003` (first ladder; dated by launch), `l3_modeled_qualification_h3_ladder2_20261004`, `l3_near_tie_probe_20261003`, `stopping_floor_ab_20261004`, `l3_qualification_pilot_20261003` (H3 and H4 timing pilots; the first H4 pilot was killed at 2.4 CPU-h and 10 GB inside i's exact L3 reference);
- October 3: `modeled_qualification_20261003`, `modeled_qualification_h4_6400_20261003`, `near_tie_diagnosis_20261003`, `l3_panel_20261003`, `sequential_replicates_ab_20261003`, `l2_h4_stopping_gate_20261003` (with `cost_profile.py`, `paired_cpu.py`), `l3_stopping_gate_20261003`;
- wider n (Section 12): `gap2_belief_sampling_paired_20261002` (before/after/head arms; before and after ran from temporary worktrees, since removed), `q_plateau_diagnosis_20261002`, `l2_wide_n_20261002`, `modeled_budget_probe_20261002`, `l2_h4_modeled_sweep_20261002`;
- chance scoring: `l1_qualification_chance_scoring_20261002`, `l2_chance_scoring_20261002`;
- report figures: `report_hierarchies_20261002` (nested hierarchies, exact vs compressed 64, L2 to step 8 and L3 to step 5);
- Wumpus: `wumpus_exact_filtering_20261002`, `wumpus_split_pilot_20261002`.

## 6. Tried and did not work (do not repeat)

| Tried | What happened | What replaced it |
|---|---|---|
| Repository default estimator (sampled rewards, no exact tail) | Passed every test while 61–88% optimal | The estimator in Section 2 |
| Sampled IPF as the planner's belief | Quantizes small nested beliefs to 1/N steps and flips greedy modeled opponents (Q 3.3 → 6.3 at N = 64). Error accumulates with history. Slower than exact beliefs where exact is tractable | Compressed filter |
| Multinomial resampling | Noisier than systematic at every N (paired over 12 histories) | Systematic resampling |
| Judging IPF accuracy from 3 histories | They were easy ones; the error was understated 2–5× | At least 12 histories, reporting max and share of steps above 0.2 |
| Carrying one root particle down the tree with bounded beliefs | The particle can sit on an atom compression dropped; deterministic sensors then give zero evidence (Wumpus) | Draw each step's state from b_h |
| Compressed or sampled *real* beliefs on Wumpus | The 94,500-state prior compressed to N atoms loses the true world, and the next deterministic observation has zero evidence | Exact real updates (cheap: at most 2.9 s per step) + `search_particles` |
| Assuming exact Wumpus L1 is too big | Wrong for filtering: deterministic sensors prune 94,500 → 46 atoms in 8 steps | Exact real filtering |
| Exact reference with the default 200k-branch guard | Stops at history 6; mistaken for intractability | `REFERENCE_BRANCHES = 50M` (146 s at k = 6) |
| Raising root simulations to fix H4 L2 misses | No effect; the misses are opponent-model error | Raise the modeled budget (4,000 at depth 4) |
| L1 qualification as evidence for the modeled budget | It scores decisions at i's beliefs, not at the beliefs j is modeled at | Qualify modeled agents at their modeled beliefs |
| Launcher scripts without `if __name__ == "__main__":` | The supervisor uses spawn; workers re-run the launcher and crash on the existing output directory | Always guard launchers |
| Killing a `ProcessPoolExecutor` parent by PID | Its workers are reparented and keep running (10 orphans on October 5) | List with `ps -eo pid,ppid,cmd`, then kill the workers' PIDs too |
| `pkill -f <pattern>`, or `for p in $(pgrep -f <pattern>); do kill $p; done` | Killed our own shell, because the pattern also matches the command that runs it (happened again October 2) | Look up the PID with `ps`, then `kill <pid>` |
| `until ! pgrep -f <pattern>; do sleep 30; done` as a waiter | Never exits: the waiter's own command line matches the pattern (two waiters ran 4.7 h on October 3) | Wait on a PID (`until ! ps -p <pid>`) or on an output file |
| Judging a slow panel by elapsed time | The k = 6 cells looked hung; the cost is the exact reference (about 146 s per problem), not the planner | Time the planner alone with the reference stubbed out, and compare with the previous run's wall time |
| Scaling simulations to improve long-horizon Q error | All-action max Q error barely moves (L1 H5: 1.57 → 1.35) | It was the metric: report `optimal_q_error` (0.40 → 0.12) |
| Judging a regression from 3 seeds (L1 H5: 0.88 → 1.46) | Per-seed means range 0.56–2.15; paired n = 220 showed no change | Paired runs over ≥ 20 seeds |
| Comparing CPU across runs made at different machine load | Flat 1,600 measured 13.7 s on a quiet day and 20.6 s under another project's jobs; the stopping rule looked like no saving | Paired, same-process, alternating-order CPU measurements |
| Exact L3 reference inside an L3 H4 qualification job | 2.4 CPU-h and 10 GB for one problem before it was killed | `score_decision=False` (`reference="none"`): only the modeled level's reference is needed |
| One exact L2 reference (shared LRU cache) across an L3 H4 qualification job | Grew about 370 MB per query to 9.6 GB; all 27 jobs exceeded 2.5 GB | A fresh reference per query: 1.3 GB peak, about 30% more reference CPU |
| Passing beliefs between processes by pickle (before the October 4 fix) | `FiniteBelief.__reduce__` renormalized masses: 504 of 549 digests (and search seeds) changed, flipping 5 near-tie policies | Exact restore (`_restore_belief`), with a regression test; earlier results never pickled beliefs |
| One qualification job per problem at L3 H4 | Confident nested beliefs need 103 queries of up to 86 atoms; jobs exceeded 3 GB and repeated work | Two phases: search per problem, then one job per distinct belief (exactly equivalent: 1,098/1,098 rows) |
| Averaging replicate searches of 200 (12.8's rule) | Converges to a 200-simulation tree's value; biased for j-L2 at L3 H4 (16–20/30 wrong at 0.19 gaps even at a 25,600 cap) | One growing tree with a pilot-calibrated test (12.11) |
| k = 2–4 independent trees grown in lockstep (honest standard error) | Hard beliefs run to the cap and each tree is k times shallower: worse than one tree at equal budget | One main tree; pilots only calibrate (12.11) |
| Memoizing the hashes of `AgentFrame`, `MentalModel`, `InteractiveState` | Bit-identical, but 8% more CPU: most objects are hashed once or twice | Left as is |
| A larger exact-reference LRU cache | No change: a solve's 173k i-L1 reference nodes are all distinct | Left at 4,096 |
| Estimating in-situ cost from the first queries of a problem | The first 24 queries cost 16–20 s each; later ones 180–340 s at 6,400 | Measure over a whole problem |
| A pool job that writes its output only at the end (5-hour jobs) | No progress visible, nothing resumable | Append a row per query (JSON lines) and skip done rows on restart |
| Raising the stopping floor (4–5 replicates) | 29–57% more simulations; the remaining near-tie miss is at the cap | Floor 3 (12.9) |
| A loss tolerance on the modeled agent's own decisions | Harmful mistakes lose 0.66; harmless ones 0.2–1.0 | Zero wrong modeled decisions (12.5) |
| Replicate searches at an equal budget | Worse on near ties (17/20 vs 20/20 at 1,600): small trees lose depth | Replicates only with adaptive stopping and a higher cap (12.8) |
| Exact L3 references mid-episode at H3 | One k = 2 problem ran over 30 minutes at 5 GB | Exact L3 checks at H3 k = 0 and H2 k ≤ 2 only |
| Reading one optimal rate off a belief grid | H4 modeled 100 was 80% on 15 problems and 97% on 95; misses are all on the 4 corners | Report hard regions separately, with Clopper–Pearson intervals |
| Renormalizing exact chance weights over sampled branches | Same decisions, more upward bias (listen at L1 H4: +0.04 → +0.13) | Keep the whole-action fallback |
| Editing `src/` (including formatting) while supervised jobs run | The source lock fails, or workers import mixed code | Wait for runs to finish |
| `sleep N` chained in foreground tool calls | Blocked by the harness | Background job plus an `until …; do sleep 30; done` waiter |

| Assuming rollout noise sets the near-tie floor (12.6) | Valuing every frontier exactly left the spread unchanged from 1,600 simulations on | Fixed-tree diagnosis found the fallbacks (12.14) |
| A two-step exact tail inside search (archive) | Failed its zero-violation gate (1/3,360); the error came from chance error at depths 3–8 | Exact scoring only where two steps are left, at scoring time (12.14) |
| Choosing the stopping exponent from the replay data (β = 1.25) | It is the edge before mistakes appear: tuning on the test set | The measured decay exponent, 1/n (12.15) |
| Grouping rows inside the exact reference; memoizing Tiger's kernels | No measurable gain (a kernel lookup costs as much as the kernel) | Reverted; changes below 3% of CPU are skipped (12.12) |
| Reading an interim effect as a finding (j's reward at n = 77) | It vanished at n = 100 under a correction for 3 comparisons | Final n only, with multiple-comparison correction |
| Comparing conditions on different trial sets (prediction loss, n = 12 vs 51) | Looked like a 0.07 gap; paired on the same trials it was −0.006 | Paired, same-trial differences |
| Comparing i with j inside a cell of a seed-paired grid | Each agent's growl luck repeats across all cells (5 tiger hits counted 67 times) | Compare cells along a row or column; paired own-level differences |
| Level-5 episodes; exact L4 at H3 | L4 at H3 took over 10 minutes per exact solve; L3 → L4 planning about 18× per step; level 5 would take hours per step | H2 for level grids (exact for the searched model); levels ≤ 4 |
| N = 64 beliefs for level-4 episodes | Planning grew to 65 s per step by step 8 | N = 32 |
| Exact L3 belief qualification to step 8 | 3.3 GB and rising (stopped) | `belief_qualification --steps 5` |
| Episode rows built after `update_root` | `action_values` were always empty (it clears the decision's statistics) | Read `get_detailed_stats()` right after `get_action` |
| Running `analyze.py` from another directory | Relative paths found no trials and the report rendered empty | Paths anchored to the repository root |
| Monkeypatching `_GrowingSearch.grow` with a fixed signature | Broke when `grow` gained `bounds` (a gate crashed) | Accept `*args`/the full signature in harness wrappers |
| A panel run from a snapshot not laid out as `<run>/src/ipomdp` | The source lock found no files and refused to start | Snapshot to `<run>/src/ipomdp` and `PYTHONPATH=<run>/src` |
| Restarting a runner to add workers | Discards the trials in progress (up to 30 CPU-min each) | Restart only when it saves over an hour |
| `pgrep -f` inside a waiter loop (again, October 6) | Matched its own command line | Wait on a PID or an output file |

## 7. Merge record

On October 2, 2026, after the user confirmed that Codex and Antigravity had finished:
1. Codex's working tree was committed unchanged (`69097e6`, branch `codex/final-20261002`, tag `archive/codex-final-20261002`).
2. `claude/restructure` merged it with `-s ours` (`f40c525`), keeping the restructured tree and recording the ancestry.
3. `main` was fast-forwarded to `f40c525`. Nothing was pushed.

## 8. How to run

```bash
cd /home/andyj1810/projects/ipomcp
PY=.venv/bin/python
$PY -m pytest -q                                   # about 45 s (slower while other jobs load the machine)
.venv/bin/ruff check src tests && .venv/bin/ruff format --check src tests
# Gates
$PY -m ipomdp.experiments.l1_qualification --budgets 100 1000 5000 --seeds 0 1 2 --output results/<new>
$PY -m ipomdp.experiments.belief_qualification --levels 2 --particles 16 64 256 \
    --particle-filters compressed sampled --output results/<new>
# L2/L3 panels vs the exact I-POMDP (cells: planner, level, horizon, budget, modeled, particles, filter, history)
$PY -m ipomdp.experiments.panel --planners mcts --levels 2 --horizons 3 --budgets 1000 --modeled exact \
    --particles 0 64 --history 0 2 4 --output results/<new>
# L3 panel (i-L3 -> j-L2 -> i-L1): add --levels 3 and --nested-beliefs 0.085 0.5 0.915
# Modeled-agent qualification (j-(L-1) at the beliefs i's level-L search queries it at; rungs are stacks)
$PY -m ipomdp.experiments.modeled_qualification --horizons 4 --modeled 100 --ladder 100 400 1600 6400s \
    --seeds 0 1 --workers 12 --output results/<new>
# Level 3: add --level 3 --nested-beliefs 0.085 0.5 0.915 and stack rungs such as 100/400 1600s/400
# Studies
$PY -m ipomdp.experiments.runner studies/tiger_l3_vs_l2.toml --output results/<new>
```

- For custom grids, call `panel.run_panel(output, cells, seeds, workers, beliefs=..., opponent_beliefs=..., nested_beliefs=...)` from a guarded launcher script, and copy the launcher into the output directory. Cells are `(planner, level, horizon, budget, modeled, particles, filter, history)`, with `modeled` a stack label such as `exact`, `1600` or `6400s/400`. The wide L2 grid (19 × 5 beliefs) and the L3 grid (9 × 3 × 3) are in the launchers in `results/l2_wide_n_20261002` and `results/l3_panel_20261003`.
- Wait on background jobs by PID or output file, never with `pgrep -f` (Section 6).
- Size runs from measured CPU. One machine has 24 cores and about 19 GB free; 12 workers are safe for Tiger L2 panels.
