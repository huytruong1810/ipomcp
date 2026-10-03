# Handoff

Read this first. Then read, in order:
- [RESULTS.md](RESULTS.md): every number below, with its n and its reference;
- [THEORY.md](THEORY.md): what is computed exactly, what is estimated, and the three meanings of "exact";
- [ARCHITECTURE.md](ARCHITECTURE.md): how the package fits together.

Last updated October 2, 2026 (wider-n Tiger evidence, Q plateau diagnosed, modeled-budget probe; RESULTS.md Section 12). `main` now holds the restructured package (merge `f40c525`); Claude is the sole owner. Codex and Antigravity have finished, and there are no other agents.

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

- **Tests.** 242 pass in about 35 s on `main`; `ruff check` and `ruff format --check` are clean.
- **Running jobs.** None.
- **Archives.**
  - Tag `archive/pre-restructure-20260930`: the pre-restructure code.
  - Tag `archive/codex-final-20261002` (branch `codex/final-20261002`): Codex's and Antigravity's final working tree, preserved unchanged. It holds the legacy recursive-chance candidate (`src/solvers/recursive_chance.py`), its protocols (`docs/RECURSIVE_CHANCE_*.md`), the frozen contracts (`benchmarks/`) and the launchers. `main`'s merge has it as a parent, but `main`'s tree is the restructure.
- **Results.** Stored under `results/` (git-ignored). A backup of everything up to September 30 is in `C:\Users\Andyj\ipomcp-results-backup-20260930`. Results from October 1–2 are not backed up yet; copy `results/` off the machine.

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
  experiments/  study (TOML StudySpec), episodes, runner, bootstrapper, oracle, l2_panel,
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

**Implication the user has been told about.** Contribution 1's bounded belief is now the compressed filter, not the published IPF; the IPF remains the baseline's belief. If the user wants the published IPF inside the proposed method too, that changes the story; ask before assuming.

## 4. Where Tiger stands, and next steps

### Assessment (updated October 2 with RESULTS.md Section 12)

**Decisions are optimal wherever they can be checked exactly, now at n = 380 per L2 cell; the decision-relevant Q value converges with budget.**

- **Decisions** (95% Clopper–Pearson in brackets):
  - L1 H5: 220/220 at 1,000 and 5,000 simulations [98.3, 100];
  - L2 H3: 373–380/380 up to k = 4 (compressed 64 and exact);
  - L2 H4: 380/380 with root 1,000 and modeled 1,600 [99.0, 100], at 13.7 s per decision.
- **Closed this session:**
  - Gap 2 (possible regression from `f793712`): none, paired n = 220 (12.1).
  - Gap 1 (Q plateau): mostly the metric. All-action max Q error is set by clearly wrong actions. Optimal-action Q error converges (L1 H5: 0.40 → 0.12). `optimal_q_error` is now recorded; use it for error bounds (12.2).
  - H4 misses: wrong j-L1 decisions at confident j beliefs. Modeled 1,600 removes them (12.4).
- **Still open:**
  - Chance scoring raises Q error on wrong actions at L1 H5 (+0.11) because every root falls back on some action. It is neutral on the optimal action and for decisions.
  - Problem sets are one Tiger belief grid and one action script for i (`oracle.HISTORY_ACTIONS`); exact-belief H3 at k ≥ 4 costs up to 279 s per problem and was skipped.
  - No L3 accuracy, no episode-level results on the current code.
  - Rates over a belief grid depend on the grid (12.3). Report hard regions (the corners) separately.

**Decision kept: do not scale resources to get better-looking results.** Scale evidence and fix causes; 12.4 shows targeted budgets beat blanket ones.

### Next steps, in order

0. **Done October 2:** chance scoring (Section 11); wider n (12.1, 12.3); Q plateau diagnosis (12.2); modeled-budget probe (12.4).
1. **Per-level modeled budgets as a package driver** (was step 2).
   - Turn `results/modeled_budget_probe_20261002/probe.py` into `experiments/modeled_qualification.py`: for a problem set, log the beliefs each modeled level is queried at, score that level's decisions against its exact agent at those beliefs over a budget ladder, and report the smallest budget with no wrong decision.
   - Then make it the rule for choosing modeled budgets per level (L1 under L2: 1,600 at H4, Tiger). Check H3 the same way.
2. **Chance-scoring fallback** (was the open question in step 0).
   - Where an action has an unsampled branch, it keeps its search Q. On wrong actions this costs +0.11 Q error at L1 H5.
   - Candidates: score sampled branches and bound the unsampled mass with the declared reward bounds; or score only actions whose subtrees are complete. Gate each on 12.1's paired design (n = 220), judged on both optimal-action and all-action error.
3. **L3 decisions against the exact L3 I-POMDP**, where tractable: Tiger, H2–H3, compressed vs exact beliefs.
   - The oracle supports levels 1–2 only; extend it with an exact j-L2 provider (`ExactIPOMDPPlanner` handles any level).
   - Exact L3 *filtering* reaches 4,368 atoms and 72 s per update by step 5; step 6 did not finish in 10 minutes. Keep exact L3 checks to k ≤ 5.
   - Qualify j-L2's modeled budget at its modeled beliefs first (step 1).
4. **Wumpus qualification** (RESULTS.md Section 10).
   - Run a grid over `search_particles` (64 → 4,096) × simulations (1k → 10k), at L1 H3, against the exact best response. The pilot script is `results/wumpus_split_pilot_20261002/wumpus_split.py`.
   - Separate the two candidate causes: compressing a 25k–94k-atom root to N atoms, and too few simulations for 5 actions × 32 observations.
   - Consider a package driver (like `l2_panel`, but domain-generic) once the design settles.
5. **Contribution 2 at scale.**
   - I-POMCP (compressed) vs IPF + RTS (sampled, as published) on longer histories and L3, scored with `optimal_q_error` as well as decisions.
   - Report RTS with compressed beliefs as an ablation; it shows that RTS's misses are search error.
   - Re-run Section 6's baseline cells on the wide grid; the 80–83% vs 97% comparison is from the 15-problem grid.
6. **R9 scale-up ladder:** Tiger L1 → L4, then UAV, then Wumpus. Each rung passes its gate first.
   - Then run the L3-vs-L2 episode study (`studies/tiger_l3_vs_l2.toml`, compressed beliefs) for the metrics the report lists as pending: episode rewards with bootstrap CIs, per-step costs, level posteriors over time, and sunbursts over time.
   - Its purpose is those metrics, not better accuracy numbers.
7. **Equilibria** (contribution 3): no code yet; needs a design.

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

## 5. Files touched this session (since `archive/pre-restructure-20260930`)

Everything under `src/ipomdp`, `tests`, `docs` and `studies` was rewritten or reviewed; see `git log archive/pre-restructure-20260930..HEAD`. Changes in the latest stretch (October 1–2):

**New modules:**
- `belief/compressed_filter.py`;
- `belief/resampling.py`;
- `experiments/l2_panel.py` (replaces `allocation_study`);
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
- new: `test_bounded_beliefs.py` (renamed from `test_particle_filter.py`), `test_l2_panel.py`, `test_hierarchies.py`, `study_fixtures.py`;
- updated: `test_rts.py`, `test_history_rewards.py`, `test_bellman_backups.py`, `test_exact_final_step.py`, `test_search_and_domain_contracts.py`, `test_online_filter.py`.

**Docs:** README, ARCHITECTURE, THEORY, RESULTS (Sections 1–10), and this file. The review findings are in `docs/review/phase-00..11`.

**Results directories,** each holding the launcher or probe script that produced it:
- belief accuracy: `ipf_resampling_probe_20261002`, `belief_qualification_compressed_20261002`, `compressed_prototype_20261002`;
- L2 panels: `l2_baseline_20261002`, `l2_baseline_h4_modeled4000_20261002`, `l2_ipf_miss_20261002`, `l2_mid_episode_20261002`, `l2_compressed_20261002`, `l2_compressed_k6_20261002`, `l2_revalidate_belief_sampling_20261002`;
- gates and references: `l1_qualification_belief_sampling_20261002`, `reference_scaling_20261002`;
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
# L2 panels vs the exact I-POMDP (cells: planner, horizon, budget, modeled, particles, filter, history)
$PY -m ipomdp.experiments.l2_panel --planners mcts --horizons 3 --budgets 1000 --modeled 0 \
    --particles 0 64 --history 0 2 4 --output results/<new>
# Studies
$PY -m ipomdp.experiments.runner studies/tiger_l3_vs_l2.toml --output results/<new>
```

- For custom grids, call `l2_panel.run_panel(output, cells, seeds, workers)` from a guarded launcher script, and copy the launcher into the output directory.
- Size runs from measured CPU. One machine has 24 cores and about 19 GB free; 12 workers are safe for Tiger L2 panels.
