# Handoff

Read this first. Then read, in order:
- [RESULTS.md](RESULTS.md): every number below, with its n and its reference;
- [THEORY.md](THEORY.md): what is computed exactly, what is estimated, and the three meanings of "exact";
- [ARCHITECTURE.md](ARCHITECTURE.md): how the package fits together.

Last updated October 3, 2026 (modeled qualification driver, L3 oracle and panel, sequential modeled stopping; RESULTS.md Sections 12.5–12.8). `main` now holds the restructured package (merge `f40c525`); Claude is the sole owner. Codex and Antigravity have finished, and there are no other agents.

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

- **Tests.** 252 pass in about 45 s on a quiet machine (about 170 s while another project's jobs load it) on `main`; `ruff check` and `ruff format --check` are clean.
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
  planning/     IPOMCPPlanner, HorizonBoundUCB, POMCPNode (Bellman backups), SolverBank
  reference/    BestResponseReference (exact, any level/domain), ExactIPOMDPPlanner, alpha vectors (L1)
  baselines/    RTSPlanner (reachability-tree sampling)
  experiments/  study (TOML StudySpec), episodes, runner, bootstrapper, oracle, panel,
                l1_qualification, belief_qualification
  analysis/     hierarchy (sunburst nodes with content and error), metrics, figures, statistics
  infra/        process supervision, source locks, resources, paths
studies/        tiger_l3_vs_l2.toml
```

### Estimator, as validated (do not change without re-running the gates)

- **Search.** I-POMCP over private histories, with empirical chance-weighted Bellman backups and horizon-bounded UCB from the declared reward bounds.
- **Recommendation.** After search, the tree is rescored with exact chance probabilities (`planning.chance_scoring`). The scored values decide, for real and modeled agents.
- **Exact pieces.** E[R | b_h, a] at every history, and Q_1(b_h, a) at the last step, integrated over the agent's posterior b_h.
- **State sampling.** Since `f793712`, each tree and rollout step draws its state from b_h, instead of carrying one root particle. The two are equal in distribution when b_h is exact; with bounded b_h only the new form is consistent. It was re-validated in RESULTS.md Section 9.
- **Beliefs.** `SolverBank(seed, particles=N, particle_filter="compressed" | "sampled", search_particles=M)`:
  - `particles` bounds the beliefs kept between real steps (unset: exact);
  - `search_particles` bounds the beliefs each search works on.

  Exact real beliefs with a compressed search is the Wumpus configuration.
- **Exact reference budget.** The oracle's exact I-POMDP uses `oracle.REFERENCE_BRANCHES = 50M`. At history 6 it completes in 146 s for one problem.

## 3. Key results (details and n in RESULTS.md)

| Section | Result |
|---|---|
| 1 | L1 Tiger: 100% optimal against alpha vectors at H1–5 from 1,000 simulations (n = 33 per cell) |
| 5 | **Allocation:** the modeled-opponent budget decides correctness; the root budget only reduces Q error. H4 L2 needs about 4,000 modeled simulations to reach 100% |
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

**Implication the user has been told about.** Contribution 1's bounded belief is now the compressed filter, not the published IPF; the IPF remains the baseline's belief. If the user wants the published IPF inside the proposed method too, that changes the story; ask before assuming.

## 4. Where Tiger stands, and next steps

### Assessment (updated October 3)

**Every decision checked against an exact reference is optimal at the budgets below, now including level 3; the modeled-opponent cost is the remaining inefficiency, and the first redesign halves it at L2.**

- **Decisions** (95% Clopper–Pearson):
  - L1 H5: 220/220 [98.3, 100];
  - L2 H3: 373–380/380 up to k = 4;
  - L2 H4: 380/380 [99.0, 100] with modeled 1,600 flat, or with sequential stopping (cap 6,400) at half the CPU (paired);
  - L3 H3, k = 0: 162/162 [97.7, 100] with exact, 400 and 100 modeled.
- **Limits:**
  - Exact L3 references are tractable only at H3 k = 0 and H2 k ≤ 2 (12.7). Compressed-belief error at L3 must be measured as belief error.
  - Rates over a belief grid depend on the grid; report hard regions separately (12.3).
  - Chance-scoring fallbacks add Q error on clearly wrong actions (+0.11 at L1 H5), not on the optimal action or decisions.
  - No episode-level results on the current code.

### Next steps, in order

0. **Done October 3:** `experiments.modeled_qualification` (12.5); oracle and `panel` (was `l2_panel`) at level 3 (12.7); near-tie diagnosis (12.6); sequential modeled stopping (12.8).
1. **Per-level modeled solves.** The L3 gate (12.8) shows stopping pays only where the qualified flat budget is far above its 600-simulation floor. Today `oracle`/`panel` apply one rule to every modeled level.
   - Give each modeled level its own budget and rule, chosen by `modeled_qualification` for that level. This needs a stopping option on the driver's ladder, and an L3 version of the driver (j-L2 queried by i-L3, scored against the exact L2 agent).
   - Candidate refinement: a lower floor, for example 2 replicates or a batch tied to the cap. Gate it on the 12.6 near-tie set first; 12.8 shows small trees lose depth.
2. **Adaptive budgets for real decisions as well.** The same stopping test applies to i's root (stop when the decision is separated), but real decisions also need Q accuracy for contribution 2's error bounds. Design and gate it separately.
3. **Variance reduction** (12.6: the near-tie floor is return variance). Candidates from the literature, each A/B-gated on the 12.6 near-tie set and the 12.5 queried beliefs:
   - more exact integration below the root (chance scoring already cuts the spread 2.5× at 6,400);
   - common random numbers between sibling actions;
   - control variates on rollout returns.
4. **Chance-scoring fallback:** score sampled branches and bound the unsampled mass by the declared reward bounds; or score only complete subtrees. Gate on 12.1's paired design.
5. **Wumpus qualification** (Section 10). Run a grid over `search_particles` (64 → 4,096) × simulations (1k → 10k) at L1 H3 against the exact best response, separating the root-compression and simulation-budget causes. Pilot: `results/wumpus_split_pilot_20261002/wumpus_split.py`.
6. **Contribution 2 at scale.**
   - I-POMCP (compressed) vs IPF + RTS (sampled, as published) on the wide grid and at L3 H3, scored with `optimal_q_error` as well as decisions. Section 6's 97% vs 80–83% is from the 15-problem grid.
   - Report RTS with compressed beliefs as an ablation.
   - Compare CPU only within paired, same-process runs (12.8).
7. **R9 scale-up ladder:** Tiger L1 → L4, then UAV, then Wumpus. Each rung passes its gate first.
   - Then run the L3-vs-L2 episode study (`studies/tiger_l3_vs_l2.toml`, compressed beliefs, stopping for modeled solves) for the pending report metrics: episode rewards with bootstrap CIs, per-step costs, level posteriors and sunbursts over time.
8. **Equilibria** (contribution 3): no code yet; needs a design.

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
- **Updated October 2 with Section 12:** lead, an optimal-action Q error metric row, the L1 plateau finding, a 380-problem L2 table (with the modeled 400/1,600 sweep) and the allocation rule (1,600 at depth 4). The IPF + RTS comparison (97% vs 80–83%) is still from the 15-problem grid; next step 5 re-runs it.
- **Updated October 3:** a "Level 3 vs the exact I-POMDP" section (12.7), the adaptive-modeled-budget bullet in the allocation section (12.8), the lead's level-3 sentence; the level-3 row was removed from "Not yet measured".

## 5. Files touched this session (since `archive/pre-restructure-20260930`)

Everything under `src/ipomdp`, `tests`, `docs` and `studies` was rewritten or reviewed; see `git log archive/pre-restructure-20260930..HEAD`. Changes in the latest stretch (October 1–2):

**New modules:**
- `belief/compressed_filter.py`;
- `belief/resampling.py`;
- `experiments/panel.py` (replaces `allocation_study`; levels 2 and 3);
- `experiments/study.py`, `episodes.py` and `runner.py` (replace `batch_runner`, `experiments/config.py` and `tiger_l3_vs_l2.py`);
- `studies/tiger_l3_vs_l2.toml`.

**Renamed:** `experiments/ipf_qualification.py` → `belief_qualification.py`, which now compares both filters.

**Changed:**
- `planning/solver_bank.py`: `particle_filter`, `search_particles`, `search_filter`, `search_model`, `draw`;
- `planning/i_pomcp.py` (per-step draws from b_h; search on `search_model`) and `planning/generative_model.py` (in-tree work uses `search_filter`);
- `baselines/rts.py`: search on `search_model` and `search_filter`;
- `belief/particle_filter.py`: systematic resampling; documented as the baseline's filter;
- `belief/finite_filter.py`: exact only (the old `max_support`/`resample` compression was removed);
- `experiments/oracle.py`:
  - `particles`, `particle_filter` and `history_steps`, with `sample_history` and `HISTORY_ACTIONS`;
  - same-kind finite opponents;
  - an exact reference in its own bank, with `REFERENCE_BRANCHES`.

**Tests:**
- new: `test_bounded_beliefs.py` (renamed from `test_particle_filter.py`), `test_panel.py`, `test_hierarchies.py`, `study_fixtures.py`;
- updated: `test_rts.py`, `test_history_rewards.py`, `test_bellman_backups.py`, `test_exact_final_step.py`, `test_search_and_domain_contracts.py`, `test_online_filter.py`.

**Docs:** README, ARCHITECTURE, THEORY, RESULTS (Sections 1–10), and this file. The review findings are in `docs/review/phase-00..11`.

**Results directories,** each holding the launcher or probe script that produced it:
- belief accuracy: `ipf_resampling_probe_20261002`, `belief_qualification_compressed_20261002`, `compressed_prototype_20261002`;
- L2 panels: `l2_baseline_20261002`, `l2_baseline_h4_modeled4000_20261002`, `l2_ipf_miss_20261002`, `l2_mid_episode_20261002`, `l2_compressed_20261002`, `l2_compressed_k6_20261002`, `l2_revalidate_belief_sampling_20261002`;
- gates and references: `l1_qualification_belief_sampling_20261002`, `reference_scaling_20261002`;
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
| `pkill -f <pattern>`, or `for p in $(pgrep -f <pattern>); do kill $p; done` | Killed our own shell, because the pattern also matches the command that runs it (happened again October 2) | Look up the PID with `ps`, then `kill <pid>` |
| Judging a slow panel by elapsed time | The k = 6 cells looked hung; the cost is the exact reference (about 146 s per problem), not the planner | Time the planner alone with the reference stubbed out, and compare with the previous run's wall time |
| Scaling simulations to improve long-horizon Q error | All-action max Q error barely moves (L1 H5: 1.57 → 1.35) | It was the metric: report `optimal_q_error` (0.40 → 0.12) |
| Judging a regression from 3 seeds (L1 H5: 0.88 → 1.46) | Per-seed means range 0.56–2.15; paired n = 220 showed no change | Paired runs over ≥ 20 seeds |
| Comparing CPU across runs made at different machine load | Flat 1,600 measured 13.7 s on a quiet day and 20.6 s under another project's jobs; the stopping rule looked like no saving | Paired, same-process, alternating-order CPU measurements |
| A loss tolerance on the modeled agent's own decisions | Harmful mistakes lose 0.66; harmless ones 0.2–1.0 | Zero wrong modeled decisions (12.5) |
| Replicate searches at an equal budget | Worse on near ties (17/20 vs 20/20 at 1,600): small trees lose depth | Replicates only with adaptive stopping and a higher cap (12.8) |
| Exact L3 references mid-episode at H3 | One k = 2 problem ran over 30 minutes at 5 GB | Exact L3 checks at H3 k = 0 and H2 k ≤ 2 only |
| Reading one optimal rate off a belief grid | H4 modeled 100 was 80% on 15 problems and 97% on 95; misses are all on the 4 corners | Report hard regions separately, with Clopper–Pearson intervals |
| Renormalizing exact chance weights over sampled branches | Same decisions, more upward bias (listen at L1 H4: +0.04 → +0.13) | Keep the whole-action fallback |
| Editing `src/` (including formatting) while supervised jobs run | The source lock fails, or workers import mixed code | Wait for runs to finish |
| `sleep N` chained in foreground tool calls | Blocked by the harness | Background job plus an `until …; do sleep 30; done` waiter |

## 7. Merge record

On October 2, 2026, after the user confirmed that Codex and Antigravity had finished:
1. Codex's working tree was committed unchanged (`69097e6`, branch `codex/final-20261002`, tag `archive/codex-final-20261002`).
2. `claude/restructure` merged it with `-s ours` (`f40c525`), keeping the restructured tree and recording the ancestry.
3. `main` was fast-forwarded to `f40c525`. Nothing was pushed.

## 8. How to run

```bash
cd /home/andyj1810/projects/ipomcp
PY=.venv/bin/python
$PY -m pytest -q                                   # about 35 s
.venv/bin/ruff check src tests && .venv/bin/ruff format --check src tests
# Gates
$PY -m ipomdp.experiments.l1_qualification --budgets 100 1000 5000 --seeds 0 1 2 --output results/<new>
$PY -m ipomdp.experiments.belief_qualification --levels 2 --particles 16 64 256 \
    --particle-filters compressed sampled --output results/<new>
# L2/L3 panels vs the exact I-POMDP (cells: planner, level, horizon, budget, modeled, particles, filter, history)
$PY -m ipomdp.experiments.panel --planners mcts --levels 2 --horizons 3 --budgets 1000 --modeled 0 \
    --particles 0 64 --history 0 2 4 --output results/<new>
# Modeled-agent qualification (j-L1 at the beliefs i's L2 search queries it at)
$PY -m ipomdp.experiments.modeled_qualification --horizons 4 --modeled 100 --ladder 100 400 1600 6400 \
    --seeds 0 1 --workers 12 --output results/<new>
# Studies
$PY -m ipomdp.experiments.runner studies/tiger_l3_vs_l2.toml --output results/<new>
```

- For custom grids, call `panel.run_panel(output, cells, seeds, workers)` from a guarded launcher script, and copy the launcher into the output directory.
- Size runs from measured CPU. One machine has 24 cores and about 19 GB free; 12 workers are safe for Tiger L2 panels.
