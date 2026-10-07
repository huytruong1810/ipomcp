# Plan: the next phase (October 8, 2026)

## Context
The October 5 phase (W1–W8, below) is complete (RESULTS 12.12–12.21). The reviewers' weak point is that
**Tiger does not reward deeper reasoning**: an agent's payoff depends only on its own door, so level 3
cannot beat level 2 on reward (12.18, 12.21). This phase shows where higher levels pay off (HANDOFF 0.2,
Option A) and finishes level 4 in the drivers.

Every reward consumer is already joint-action aware (`reference/best_response.py`,
`planning/solver_bank.py`, `planning/generative_model.py`, `experiments/episodes.py`), and UCB and the
certified bounds read `reward_bounds`. Only the drivers hard-code the published game and levels ≤ 3:
`oracle.py` (`TigerModel()`, `_validate_problem`, `_point_model`'s three beliefs) and `panel.py` (levels 2–3).

## Summary, in order of value per CPU-hour

| Order | Workstream | Contribution | CPU | Kill criterion |
|---|---|---|---|---|
| P1 | Joint-payoff Tiger; oracle and panel generic in physics and in level (≤ 4) | 1, 3 | tests only | — |
| P2 | Exact design screen: which payoff makes one more level pay | 3 | ~1 CPU-h | no variant separates L3 from L2 → Option B (UAV) |
| P3 | Gate the chosen variant at L2 H3, L3 H3 k = 0, and L4 H2 against the exact I-POMDP | 1, 2 | ~3–6 CPU-h | any miss → `modeled_qualification` before episodes |
| P4 | Level grid 1–4 × 1–4 on the variant at H2; an L3-vs-L2 H3 study if H2 shows an effect | 3 | sized by a 2-trial pilot | pilot > 3× estimate → halve n |
| P5 | Error attribution at L2 and L3 (`trace_margin`), alongside P4 | 2, 3 | < 1 CPU-h | — |
| P6 | Write-up: RESULTS 12.22+, reviewer page section, Claude Doc, HANDOFF | — | — | — |

**Deferred, with reasons:**
- Wumpus beyond three beliefs (HANDOFF 0.2 item 3): new domain evidence, but no reviewer question waits on it.
- Root adaptive budgets (item 4): matter for cost only beyond H3.
- L3 H4 residual (item 6): needs the user's decision on a PAC-style standard (12.13), not yet asked.
- Cost at high levels (item 7): H2 grids are affordable at N = 32.

## Workstreams

**P1. Engineering (no CPU).**
- `domains/tiger.py`: one subclass of `TigerModel` with joint-payoff parameters, overriding `get_reward`
  and `reward_bounds`. The physics and observations stay the published spec. Registered in `study.DOMAINS`.
- Candidate payoffs, screened in P2:
  - a **coordination bonus** b to each agent when both open the same safe door in the same step;
  - a **shared prize**: the +10 is split when both open the safe door together (a race);
  - a **penalty c for opening while the other listens**.
- `experiments/oracle.py`: a `domain` argument; `_point_model` over a chain of any length; levels 1–4.
- `experiments/panel.py`: `domain` and levels 2–4.
- Tests: the variant's rewards and tight bounds; the level-4 oracle reference against `build_exact_agent`
  at H2; a panel cell on the variant. The published Tiger stays the default; no shims.

**P2. Exact design screen.** For each candidate (and the published Tiger as the control), over the oracle's
point-belief grid: exact policies at levels 1–4 (H2; H3 up to level 3) and the **value of one more level**,
the exact Q loss under the level-(k+1) model of taking level k's action. Report the share of beliefs with
a positive loss and its mean and max for k = 1→2, 2→3, 3→4. Choose the variant with the largest minimum
over 1→2 and 2→3.

**P3. Gates against the exact I-POMDP (chosen variant).**
- Panel L2 H3 (380 problems) and L3 H3 k = 0 (162), with the certified stacks (`400`, `6400s/100`),
  Clopper–Pearson intervals. Payoffs move the near ties, so any miss sends the variant through
  `modeled_qualification` first.
- L4 at H2 over a point-belief grid (exact L4 H2 costs about 0.2 s per solve).

**P4. Episodes.**
- The 12.21 grid launcher with a `domain` argument: same seeds, round-robin, resumable, on a source snapshot.
- **Primary endpoint (fixed before the run):** paired own-level reward differences along rows (i at level
  k + 1 vs level k against each fixed j), Holm-corrected. Secondary: prediction log-loss, level
  identification. The published-Tiger grid (12.21) is the control.
- If H2 shows level 3 beating level 2 against j at level 2: an H3 L3-vs-L2 study, n = 100, paired.

**P5. Error attribution.** `analysis.error_attribution.trace_margin` on L2 H3 and L3 H3 trees; it closes the
report's last open row besides equilibria.

**P6. Write-up.** RESULTS 12.22+; a reviewer-page section (edit `template.html` and `build_report.py`,
republish to the same URL); the Claude Doc; HANDOFF.

Run hygiene and verification as in the October 5 plan below.

# Plan: the next phase (October 5, 2026)

## Status (October 7, 2026): phase complete

All planned workstreams were run or designed (RESULTS 12.12–12.21). The next phase is planned from HANDOFF Section 0.2.

## Status (October 6, 2026)

| Workstream | Status | Result |
|---|---|---|
| W1 | done | bit-identical; 0.45× CPU in situ at L3 H4, 0.84× cold at L3 H3 (RESULTS 12.12) |
| W2 | done (evidence) | flips of j near ties (gaps 0.15–0.3) never changed i's exact decision; no gap below 0.05 testable (12.13). Standard: the user's call |
| W3 | A0, A0b and the scoring change done; A1–A3 not needed yet | the floor was whole-action fallbacks; exact scoring with two steps left: L3 H4 residual 14 → 4 of 80 (12.14) |
| W3 follow-up | done | the stopping rule now scales the pilot spread as 1/n, the measured rate: the same decisions with 14–23% fewer simulations; in situ, CPU 0.88× (not significant) (12.15) |
| W4 | done | 100 trials per condition, with level-2 and level-1 controls (12.18). Modeling j cuts prediction loss 1.10 → 0.45; level 3 adds nothing over level 2; no robust reward effect. Reviewer page: https://claude.ai/artifact/PqpRuTLcuSq5afJdSSxoCB |
| W5 | running | pilot done (10 problems per cell; RTS horizon-4 K = 8 takes 168 s per decision); the full 380-problem run is in `results/w5_rts_baseline_20261006/full` |
| W6 | bounds done | certified intervals on every real decision (12.17); error attribution at L2/L3 not yet |
| W7 | done (3 beliefs) | qualified Wumpus L1 H3 configuration: N = 1,024, 3,000 simulations (9/9); the pilot's losses were the budget (12.19) |
| W8b | done | level grid, i and j at levels 1–4 (12.21): prediction separates level 1 from levels ≥ 2; identification weakens with level; no consistent reward effect |
| W8 | equilibria design done | `docs/EQUILIBRIA.md`; predictions recorded per step. Level 4 and UAV not started |

## Context
HANDOFF Section 0.1 asks for a written, committed plan (`docs/PLAN.md`) before any experiment runs.

**Where things stand:**
- `main` is clean and its tests pass.
- Every Tiger decision checked against an exact reference is optimal: L1 H5, L2 H3–H4, and L3 H3.
- At L3 H4, j-L2 still gets 6 of 666 beliefs wrong. These are flat 6,400's misses: 5 near-exact ties with gaps of 0.03–0.04, and one gap of 0.14.
- One i-L3 H4 decision costs about 2.5 CPU-hours.

**The reviewer report's "Not yet measured" table** has seven rows:
- episode rewards;
- CPU per step;
- level posteriors;
- sunbursts over time;
- error attribution at L2 and L3;
- IPF + RTS at L3;
- equilibria.

The first four wait on the episode study, which cannot run as configured. Its spec is stale: depth 5, modeled 100, mixed-level priors that no gate covers. TOML also cannot express stopping or per-level stacks.

**Exploration this session found:**
- **Free variance reduction.** Rollouts already compute the exact posterior at every step (`i_pomcp.py:259`), so their sampled rewards can be replaced by E[R|b_h,a] at no extra cost.
- **Wasted work in `tree_step`.** It updates the opponent model and nothing reads the result (`generative_model.py:38-43`).
- **Redundant work in `best_response.py`** (`:90-93`, `:135-150`).
- **Uncached work in canonical and compress** (`canonical.py:86-116`, `compressed_filter.py:79-80`).
- **The two-step exact tail was not dropped for a bug.** It failed a zero-violation gate (1 of 3,360 cases). The error came from chance error at depths 3–8, above the tail (archive `TAIL2_FAILURE_DIAGNOSIS.md`).

## Summary

| Order | Workstream | Contribution | Gate CPU | Kill criterion |
|---|---|---|---|---|
| W1 | Bit-identical speedups of simulating minds (B) | 1, 2 | ~6 CPU-h | no removable hotspot ≥ 10% |
| W2 | Harm of near-tie modeled mistakes to i (certification standard) | 1, 2 | ~11 CPU-h | none (either outcome is a result) |
| W3 | Variance reduction at the near-tie floor (A) | 1 | ~1 h (A0, A1 at L2/L3 H3), then ~3–5 CPU-h per arm | < 1.2× gap-SD reduction |
| W4 | Episode study, L3 vs L2 at H3 | 3, 2 | gates ~3 CPU-h; run 15–25 CPU-h | pilot > 3× estimate |
| W5 | I-POMCP vs IPF + RTS at scale (D) | 2 | sized by a 10-problem pilot | RTS > 10× I-POMCP CPU, capped |
| W6 | Certified Q intervals; chance-scoring fallback (H); error attribution | 2, 3 | ~2 CPU-h | intervals miss the exact Q |
| W7 | Wumpus L1 qualification (E) | 1, 3 | ~12 CPU-h | — |
| W8 | Level 4, UAV, equilibria (F, G) | 1–3 | planning only | — |

Root adaptive budgets for real decisions (C) are deferred: they matter for cost only beyond H3, and real decisions also need Q accuracy for W6's bounds.

## Workstreams, in order of value per CPU-hour

**W1. Speed up simulating minds without changing any output (workstream B; contributions 1 and 2).**
- **Why first:** every later workstream pays this cost (L3 H4 datasets, residual gates, episodes). Bit identity makes its gate cheap and definitive.
- **Step 1, profile.** Run cProfile in situ in a results folder:
  - one confident-nested L3 H4 problem's query sequence, replayed in one bank from `results/l3_modeled_qualification_h4_confident_20261004/queries.pkl` (the pattern of `l3_h4_tree_trajectories_20261005/trajectories.py`), including late queries, since those are the expensive ones;
  - one L2 H4 j-L1 set.

  Record CPU and RSS per phase. About 2 CPU-h.
- **Step 2, changes that keep outputs bit-identical:**
  - drop the unused opponent update in `tree_step`, but keep its RNG draw so the streams do not change;
  - hoist the own-observation law out of the loop in `best_response.py:90-93`;
  - reuse `branches` in `_immediate`;
  - cache `physics_identity`;
  - make `compress` use the bank's cached `ordered_mass`;
  - check whether the `expected_rewards` LRU (1,024) thrashes;
  - add a `FiniteBelief` fast path only if it is bit-identical.

  Each change is driven by the profile; skip any below 3% of CPU.
- **Step 3, conditional on the profile:** a vectorized belief for an L1 facing L0, over finite physical states, for any domain (not 2-D alpha vectors). It is equal to 1e-12, not bitwise, so it gets its own gate.
- **Gate:**
  - the full test suite;
  - bit-identical policies, scores and digests on the 183 L3 H3 j-L2 beliefs (`6400s/100`, cold) and on 40 in-situ L3 H4 queries;
  - paired CPU in one process, alternating order: report the speedup with a Wilcoxon test.
- **Cost:** about 6 CPU-h; peak 1.5 GB per job.
- **Kill:** if no removable hotspot is at least 10% and step 3 is ruled out by generality, record the profile and stop.

**W2. Is a near-tie mistake harmless to i? The certification standard (contribution 1, error bounds for 2).**
- **Hypothesis:** j decisions wrong at gap < ε, for small ε, never change i's exact decision.
- **Test where i's reference is tractable:** L2 H4 (380 problems) and L3 H3 (162). In the exact hierarchy, flip every modeled decision with gap < ε to the second-best action, for ε in {0.05, 0.15, 0.3}. Recompute i's exact Q and record i's loss and decision changes.
- **Code:** a small hook in `reference/exact_agent.py` or `best_response.py` (a policy override by a gap threshold), with tests.
- **Cost:** about 10 CPU-h at L2 H4 (50–140 s per reference) and 1 CPU-h at L3 H3.
- **Outcome:** if harmless at ε = 0.05, propose a PAC standard: correct with probability at least 1 − δ when the gap is at least ε. The current `6400s/exact` then certifies L3 H4. This is the one decision flagged for the user, presented with the evidence. Otherwise zero-wrong stays, and W3 becomes mandatory.
- **Kill:** none; either outcome is a result.

**W3. Variance reduction at the near-tie floor (workstream A; contribution 1).** A redesign gated by A/B, following Veness, Lanctot and Bowling (2011), *Variance Reduction in Monte-Carlo Tree Search*.
- **A0, diagnosis, paired and done at scoring time on the same trees.** Split the variance of the scored gap into three sources: frontier rollouts, fallback actions and tree shape. Re-score each tree with its frontiers replaced by exact values. Run on the L2 H4 near-tie set (`stopping_redesign_gate_20261005/gate.py`) and the L3 H3 near ties. Under 1 CPU-h.
- **A1, Rao-Blackwellized rollout rewards** (E[R|b_h,a] in `_rollout`). The RNG streams are unchanged, so seeds pair exactly. Unbiased, and adds no cost.
- **A2, common random numbers across sibling actions:** a substream per simulation, `(tree seed, simulation, depth)`.
- **A3, a control variate on the continuation return,** or an exact frontier value at remaining = 2 used only when scoring, if A0 shows frontiers dominate.
- **Gates, in order:**
  1. L2 H4 near ties (n = 1,175; under 1 CPU-h per arm) and L3 H3 near ties (n = 40).
  2. The L3 H4 residual set: 8 beliefs × 10 seeds, `residual_l3h4.py`, about 3 CPU-h per arm, less after W1.
  3. The beliefs with gaps below 1 from the replay dataset: 110 beliefs, about 5 CPU-h per arm for main trees to 6,400.

  Metrics:
  - wrong decisions (McNemar on paired seeds);
  - SD of the scored gap at a fixed n;
  - mean simulations to stop under `6400s`.

  Pass:
  - no new wrong decision on any certified set (L2 H4 1,175; L3 H3 3,294 queries; i's decisions on 380 and 162);
  - and either fewer residual misses or at least a 1.5× smaller gap SD.
- **Kill:**
  - an arm that gives under 1.2× SD reduction on the L2 H4 near ties is dropped;
  - if every arm fails, accept W2's standard and stop scaling caps.
- A searched variant changes the trees, so a new replay dataset is built only for a winning arm, restricted to the 110 beliefs.

**W4. Episode study, L3 vs L2 (contributions 3 and 2; fills 4 of the report's 7 rows).**
- **Engineering:**
  - `StudySpec` accepts per-level modeled stack labels (`modeled = "6400s/100"`), parsed by `experiments.modeled_levels`;
  - remove `modeled_planner` duplication if the stack replaces it (no legacy);
  - fix the TOML: horizon 3, certified stacks, `particles = 64` compressed, a corrected comment.
- **Gate before the run:**
  - `belief_qualification` at level 3: compressed 64 against exact, to k = 4;
  - a mixed-level-prior check at L2 H3 and L3 H3 k = 0 against the exact I-POMDP (the panel may need a level-weights option);
  - a 2-trial pilot to measure CPU per step and RSS.
- **Run:** 100 trials × 20 steps. Estimated at 15–25 CPU-h (an i-L3 H3 decision about 18 s, j-L2 about 5 s, growing mid-episode), which is about 2 wall-hours on 12 workers. Refine from the pilot.
- **Outputs:** rewards with bootstrap CIs, CPU per step, level posteriors, `sunburst_over_time`. Root adaptive budgets (C) are deferred: H3 is affordable without them.
- **Kill:** if the pilot exceeds 3× the estimate, cut to 50 trials and record why.

**W5. I-POMCP vs IPF + RTS at scale (workstream D; contribution 2).**
- A new guarded launcher on `panel` replaces the broken one in `l2_baseline_20261002`.
- Run the 380-problem L2 grid at H3 and H4 and the 162-problem L3 H3 grid.
- Score decisions, `optimal_q_error`, and paired CPU in the same process. Add an RTS + compressed-beliefs ablation.
- Size from a 10-problem pilot; RTS grows 11–24× from H3 to H4.
- **Kill:** RTS cells beyond 10× I-POMCP's CPU are capped and reported as such.

**W6. Error bounds and the chance-scoring fallback (workstream H, plus the error-attribution row; contributions 2 and 3).**
- Bound each unsampled branch by the declared reward bounds, giving certified [lo, hi] values per scored Q. Decisions are unchanged and the intervals are new outputs.
- Run `error_attribution.trace_margin` on L2 and L3 trees.
- Gate: 12.1's paired L1 H5 design. Intervals must contain the exact Q in 100% of cases, with their width reported.
- Low CPU: about 2 h.

**W7. Wumpus L1 qualification (workstream E).**
- Grid: `search_particles` {64, 256, 1,024, 4,096} × simulations {1k, 3k, 10k} × the pilot's 18 decisions × 3 seeds.
- Compute the reference once per decision: about 1.5 CPU-h. The planner takes about 10 CPU-h.
- This becomes a module, replacing the ad-hoc pilot script.

**W8. Scale ladder and equilibria (workstreams F and G): planning only in this phase.**
- **Level 4:** generalize the `oracle` and `panel` levels and the point beliefs, then gate Tiger L4 at H2–H3.
- **UAV:** an L2 gate.
- **Equilibria:** a design note in `docs/` that reviews Doshi and Gmytrasiewicz (2006), *On the difficulty of achieving equilibrium in interactive POMDPs*, and chooses a measurable notion, such as the empirical convergence of level-k policies in Tiger. No CPU.

**Ordering rationale.**
- W1 multiplies the throughput of everything after it.
- W2 is cheap and may resolve L3 H4's certification outright.
- W3's first arms are nearly free.
- W4 then delivers the most report value.
- W5 and W6 complete contribution 2.
- W7 and W8 open new domains and levels only after Tiger's gates hold.

**Questions:**
- The certification standard needs the user after W2's evidence.
- How far Tiger goes: recommend an L4 H3 gate before UAV and Wumpus, decided after W4.
- Everything else follows the autonomy rule.

## Run hygiene (applies to every run)
- New dated folder in `results/` with its script copied in.
- Long runs on `src_snapshot` with `PYTHONPATH`; JSONL rows, resumable.
- Guarded launchers; `max_tasks_per_child=1`; at most 12 workers.
- Never touch `jepa_ipomdp`; kill by PID only.
- Paired CPU in one process only.
- After each workstream, update `docs/RESULTS.md` (12.12 onward), HANDOFF, and the reviewer report doc (in place).

## Verification
- `.venv/bin/python -m pytest -q`, `ruff check` and `ruff format --check` after each code change.
- W1: bit-identity diffs (policies, scores, digests) on the stated sets.
- W2 and W3: the paired-seed statistics above.
- Each workstream's commit records its gate result in RESULTS.md.
