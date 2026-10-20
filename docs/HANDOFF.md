# Handoff

Read this first: Section 0.00000 is enough to start. Then, as needed:
- [PLAN.md](PLAN.md): the phases, newest on top. The top one is "the definitive Tiger level grid" (N0–N5).
- [RESULTS.md](RESULTS.md): every number, with its n and its reference. The latest sections are 12.70–12.74, and the next free one is 12.75.
- [PAPER.md](PAPER.md) and [paper/draft.md](paper/draft.md): the claims map and draft 1. Draft 1 is written for AAMAS and will be reframed in N5.
- [THEORY.md](THEORY.md): what is computed exactly, what is estimated, the three meanings of "exact", and the ε = 0.05 standard.
- [ARCHITECTURE.md](ARCHITECTURE.md): how the package fits together.
- [RELATED_WORK.md](RELATED_WORK.md): the literature check.

Last updated October 19, 2026, evening, for a fresh session. Claude is the sole owner of `main`; there are no other agents.
- `main` is clean and pushed. There are 390 tests, and ruff is clean.
- **N1 is done (October 19; PLAN.md N1 status). Nothing is running.** N1(d) found no cache worth building (RESULTS 12.70).
- **N2 was priced from data on disk (RESULTS 12.71). The user chose design B at 10 steps (October 19).**
- **N3 is done (RESULTS 12.73; 231.7 CPU-h).** Figures: `docs/figures/grid_curves.png` and `grid_heatmap.png`. The (history) notes below describe the run: `results/n5_level_grid_20261019` (`ENDPOINTS.md`, `KILL_CRITERION.txt`: 420 CPU-h; xargs PID in `runner.pid`; `cpu_now.sh`; 6,810 chunk jobs, 14 workers, watchdog and memlog on the xargs PID).
  - n: levels ≤ 2 at 15,000 (coordination) and 2,500 (published); level-3 row 700; level-4 row 100; 10 steps.
  - Held 12.37/12.40 trials are **not** reused: today's code does not reproduce their compressed beliefs (`gate/`).
  - Projected 228 CPU-h; expected to finish around 09:00 on October 20. When done: `PYTHONPATH=<run>/src python analyze.py` (written and tested on interim data) → RESULTS 12.73.
- **N4(c) is done (RESULTS 12.72):** exact curves of the Bayes-optimal pairs, `results/n6_exact_curves_20261019/summary.txt`. On the published Tiger, step 3 is 2.663 for a pair at level 3 or above, and 2.148 at levels 1–2.
- **N4(a) is done (RESULTS 12.74).** Level 1 agrees on all 96,000 decisions; levels 2–4 on 97.7–100% per cell. All 148 diagnosed misses are compression at N = 64. History notes: `results/n7_exact_replay_20261019` (`ENDPOINTS.md`; 2 workers; kill line 60).
  - It replays N3 agents through the exact agent (`analysis.exact_replay`). Its jobs skip N3 trials not yet written, so **rerun `run.sh` once N3 is done** to fill the sample (finished trials are skipped).
  - Pilot: every replayed decision agrees with the exact maximizers (L2: 160/160; L3: 40/40; L4: 12/12), with level TV ≤ 0.044.
  - Result → RESULTS 12.74. Diagnosis of every disagreement: `diagnose_all.sh` (`diagnosis.jsonl`). As of 23:13, 9 of 9 are reproduced as configured and fixed by exact beliefs: belief compression at N = 64.
- **N4(b) is running:** `results/n8_convergence_20261019` (`ENDPOINTS.md`; the kill line was amended from 90 to 100 before it was reached).
  - It reruns cells with i at level 2–4 for the exact window at N = 256 (and N = 1,024 at level 2), then replays i through the exact agent.
  - At 02:37 it stopped when a hand kill of a `timeout` wrapper aborted xargs (Section 6). It was relaunched at 02:45 with `watchdog2.sh`; all three runs now use `watchdog2.sh`.
  - **Rerun `run.sh` at the end** to fill any skipped chunk. Result → RESULTS 12.75.
- **Draft figures** (on interim data): `results/n5_level_grid_20261019/figures.py` → `grid_curves.png`, `grid_heatmap.png`. Rerun without `--partial` when N3 is done.

## 0. Start here (fresh agent)

### 0.00000 Start here: the definitive Tiger level grid (user's direction, October 19, evening)

**The task.** Build the experiment suite for the paper around the two priorities the user named:
1. **Low statistical variance.** "Getting the number of trials up so we get less statistical variance (for example, for optimal policy in tiger game at time 3, the expected cumulative reward should be 2.7)."
2. **Correct modeling of the other agent against the Bayes-optimal benchmark,** "across the grid of agent i lv0->4 vs agent j lv0->4".

The user wants the experiments to be "the most optimal and explainable".

**Venue.** The user first said NeurIPS 2026, whose deadline passed on May 6, 2026. Asked, they answered "any track in AI/ML that is the closest".
- ICLR 2027, AISTATS 2027 and AAMAS 2027 have also passed.
- The closest are **IJCAI 2027 (about January 15–19)** and **ICML 2027 (about January 22)**. Neither call was published on October 19, so the dates are projections.
- Claude recommended IJCAI (earlier, and the better fit). The user has not confirmed. Check the official calls once they are out.

**What N0 established** (October 19, details in PLAN.md N0):
- **Exact Bayes-optimal agents cannot play whole episodes.**
  - Pilot: `results/n1_exact_grid_pilot_20261019/exact_episode.py` (exact agents, coordination Tiger, H2; logs `pilot_*.log`).
  - Exact beliefs grow about 3× per step. The window is about 8 steps at L2 (step 8: 13,800 atoms, 33 s per decision, 0.7 GB; step 9 hits the 100,000-branch budget), 4 at L3 (step 4: 109 s; then out of memory at 5 GB) and 3 at L4 (step 3: 154 s; then out of memory).
  - **So the benchmark is exact inside that window, and convergence in N beyond it.**
- **Rao-Blackwellized (RB) rewards:** `results/n2_rao_blackwell_20261019/rb.py`.
  - Each sampled reward is replaced by its exact expectation given both agents' actions and observations over the whole episode, by forward–backward over the tiger's position with the package's kernels.
  - The estimand is unchanged (tower property), and variance falls 2.3–3.8× on 12.48's 10,000 L1-vs-L1 trials.
  - The policies depend on the state only through observations, which is why conditioning on actions and observations leaves a plain HMM.
- **Exact expected curve of the pair actually played:** `results/n2_rao_blackwell_20261019/exact_pair.py`.
  - It enumerates (state, belief i, belief j) forward, merging equal nodes; ties are uniform and level 0 is uniform.
  - L1 vs L1, published Tiger: steps 3/4/5 are exactly 2.148 / 1.308 / 1.750. The RB estimates contain all three; the sampled ones miss at step 3.
  - Nodes grow about 10× per step (1.5M at step 6 for L1), so expect fewer steps at higher levels.
  - **2.72 is the single-agent ceiling (12.48), not this pair's expectation:** j's opens reset the tiger. Say this plainly to the user and in the paper.
- **Bug:** `src/ipomdp/reference/exact_agent.py`, `ExactIPOMDPPlanner.get_action` calls `sample_policy(policy)` without the required `rng`. The pilot worked around it with `sample_policy(planner.policy_for(planner.model()), random.Random(f"{trial}/exact/ties"))`.

**N1, what to build first** (each with tests, then commit and push):
- **(a) An `exact` planner kind for episodes.**
  - `experiments/study.py`, `_planner_config` and `build_agents`: build with `reference.exact_agent.build_exact_agent(agent_id, opponent_id, level, physics, nested, horizon, gamma)`, where `nested = {n: {k: 1/n for k in range(n)}}` as `suite.hierarchy_agent` uses.
  - Fix `get_action` to take an rng seeded from the trial, so ties are reproducible.
  - It must also report the episode columns `run_episode` reads: `belief`, `key.level`, `solver_bank.filter` for level posteriors, and predictions (`episodes._prediction`).
  - Exact agents must stop or raise cleanly at their window. The filter raises `InferenceBudgetExceeded`; memory needs a guard.
- **(b) RB reward columns** in `analysis/endpoints`, generalizing `rb.py`. It works for any domain with explicit kernels: forward–backward over physical states given both agents' actions and observations. Note: Tiger rewards ignore `next_state`, but other domains may not, so condition properly on (s_{t−1}, s_t).
- **(c) The exact pair evaluator** in the package (for example `analysis/exact_curves.py`), generalizing `exact_pair.py`, with a node cap and a test against a hand-computed Tiger value.
- **(d) A cross-trial cache, so large n is affordable.**
  - Today each episode is one process with a fresh `SolverBank` seeded per trial (`study.build_agents`), so modeled solves are recomputed in every trial. 12.40 measured 13–45 CPU-min per episode for level-4 cells.
  - First measure how much modeled-solve work repeats across trials (count distinct nested-belief digests over about 20 trials per cell).
  - If it is large, share the cache across trials in one process.
  - **Caveat:** at H2, chance scoring makes modeled decisions exact for any seed *except tie-breaking*, which depends on the bank's search stream. A shared cache can change tie choices. Gate it as identical rows on 50 trials per cell, or identical except documented tie choices, and decide which before the run.
- **N2:** price every cell at n = 20 after N1, then ask the user (the full grid surely exceeds 100 CPU-h).
  - The grid: both Tiger games, levels 0–4 × 0–4, H2, N = 64 compressed, uniform priors over lower levels at every nesting (12.37 and 12.40's configuration).
  - Both games are symmetric in the agents, so the 15 cells with i's level ≤ j's give all 25.
- **Then N3–N5** as PLAN.md describes: a tiered n with an RB half-width target of about 0.5 at step 3 and 1.0 at step 20 (n ≈ 1,500–2,000 where affordable); the Bayes-optimal benchmark; figures; the reframe for the venue.

**Existing data to reuse:**
- 12.37 (levels 0–3, both games, n = 100) and 12.40 (level 4 against 0–4, coordination, n = 40): `results/m1_level_matrix_20261011`, `results/m2_level4_20261011`.
- 12.48's large-n trials (L1/L2, published, up to 10,000 fresh): `results/c2_tiger_ceiling_20261014/grid_*`.
- 12.47's exact replay of j's level posterior: `results/c1_level_posterior_20261014/replay_j.py`.
- Trial rows are CSVs under `<run>/trials/<cell>/<trial>.csv`, with columns `action_*`, `observation_*`, `reward_*`, `prob_l{k}_{other}`, `prediction_*` and `physical_*`.

**Rules that bind this work:**
- Commit and push to `origin main` after every implementation, without asking. Never force-push.
- Ask the user before any run above 100 CPU-h, and before deleting anything. Do not remind the user to back up `results/`.
- For every run: write `ENDPOINTS.md` and `KILL_CRITERION.txt` before the main run; snapshot `src` into the run folder and check the import path; run one process per episode or job with `timeout` wraps and the watchdog on the **xargs** PID.
- **Track cumulative CPU, running jobs included, at every progress check** (`results/g6_rts_l3_20261019/cpu_now.sh`).
- Price under the load the run will see: pilots under-priced by 1.8× at load 20.
- Tiger panel rows carry `problem = None`. Pair them by (seed, belief_p, opponent_belief, nested_beliefs), and assert pair counts.
- Report style: chosen method only, n beside numbers, jargon defined. Reviewer patterns need exact, control or large-n backups, never "noise".
- Section 0.0 below has the machine rules (shared 24-core, 23 GB machine; `jepa_ipomdp` runs there; kill by PID only).

### 0.0000 Session of October 19: reviewer readiness done, paper draft 1 written (history)

Plan: `~/.claude/plans/pick-up-handoff-and-keen-cat.md` (0.000's items 1–5).

**State.** `main` is clean and pushed, with no code change, so 369 tests still pass. Nothing is running. **The next free RESULTS section is 12.70.**

**Done:**
- **Item 1, the Claude Doc** (now rev 100):
  - fixed the lead's stale sentence and old baseline numbers, the metrics table's "pending" row and the benchmark bullet;
  - added a level-3 paragraph after "In full episodes";
  - renamed "Not yet measured" to "Still open", which now has one row: equilibria.
- **Item 2, IPF + RTS at level 3** (RESULTS 12.68; `results/g6_rts_l3_20261019`):
  - Coordination Tiger L3 H2–H3, n = 162 per cell. I-POMCP is 162/162 at both horizons (it reproduces 12.23).
  - IPF + RTS is 91–98%, never right where I-POMCP is wrong. **By the fixed reading, C2 does not fully extend to L3 on Tiger:** the gap is significant only at K = 2 (Holm p ≤ 0.005); at K = 4–8, Holm p ≥ 0.14. I-POMCP is 5–30× cheaper at H3.
  - At H2 the misses are identical under both filters and do not shrink beyond K = 4. The cause is undiagnosed (candidates in 12.68).
  - **Cost: 25.0 CPU-h.** The pilot priced it at 14; the main run measured a 1.8× load factor. The kill line was amended from 21 to 40 *before* it was reached and recorded in `KILL_CRITERION.txt`. Cumulative CPU, running jobs included, was tracked every 10 minutes with `cpu_now.sh`.
- **Item 3:** the 12.67 weak spots are in PAPER.md (C1/C2 and Limitations).
- **Item 4:** equilibria are future work (**user's decision, October 19**), now in the PAPER outline and the draft.
- **Figure 3:** `docs/figures/depth_gain.png` (script `results/g5c_figure3_20261019/figure.py`, values copied from 12.37, 12.49 and 12.60).
- **Item 5, paper draft 1:** `docs/paper/draft.md` (Markdown; there is no LaTeX template or toolchain on this machine). Every claim cites RESULTS. Table 1 includes 12.68.

**Afternoon, after "go ahead":**
- **References checked** against their own abstracts (RELATED_WORK.md). Panella & Gmytrasiewicz (2017) is model learning, not a solver; the draft is fixed.
- **12.68 diagnosis:** the L3 H2 RTS misses are search error at K = 2 and particle error at K = 4–8. RTS on exact beliefs makes none at K ≥ 4, and IPF + RTS makes none at K = 32 (under 0.1 CPU-h).
- **Skeptical read: four objections answered in the draft.**
  - The failing budgets are now shown.
  - Selection on the grid is answered by **RESULTS 12.69, a held-out check:** 960/960 optimal on random beliefs with fresh seeds, at the qualified Tiger L2 H3 and L3 H3 configurations, both games; 4.6 CPU-h.
  - The screened coordination payoff is disclosed.
  - Proposition 1 is now Observation 1.
  - CPU comparability: all planners are pure Python and single-threaded.
- **Figure 1** now has the two held-out rows. The Claude Doc has a held-out paragraph (rev 101+).

**Next, in Claude's order:**
1. ~~Figure 1~~ done: `docs/figures/exactness.png` (script `results/g2_figure1_20261019/figure.py`).
2. ~~References~~ checked October 19 (RELATED_WORK.md); Panella & Gmytrasiewicz recast as model learning. Volume and page numbers remain for the JAIR, JAAMAS and IJCAI entries.
3. **A skeptical read of the draft** for reviewer objections not yet on record. (The Tiger L3 H2 RTS misses are diagnosed in 12.68: search error at K = 2, particle error at K = 4–8.)
4. **Optional:** the H5 uniform-prior episodes at root 64,000 (0.000, item 3).
5. **LaTeX conversion** (the AAMAS template) once the user confirms the venue.

**Lessons:**
- **Tiger panel rows carry `problem = None`.** Pair them by (seed, belief_p, opponent_belief, nested_beliefs), and assert the pair count. `analyze.py` first paired on `problem` and silently collapsed 162 pairs to 18.
- **Price CPU under the load the run will see.** A pilot of 14 concurrent jobs at load 14 under-priced 16 workers at load 20 (other users' jobs) by 1.8×.

### 0.000 Next session (written October 19, 09:25): reviewer readiness before the paper draft (history; done)

**The user's question (October 19):** before drafting, have the reviewer's questions and skepticism all been answered? The user asked to work through the answer below in this new session.

**State.**
- `main` is clean and pushed. There are 369 tests, and ruff is clean.
- Nothing is running. The next free RESULTS section is 12.68.
- **Do not remind the user to back up `results/`** (user, October 19).

**Every reviewer concern on record has an answer** (checked October 19):

| When | Concern | Answered by |
|---|---|---|
| Oct 7 (0.10 below) | Tiger does not reward deeper reasoning | Coordination Tiger 12.22–12.37; UAV 12.49; Wumpus 12.60 (fresh-trial confirmation and a control); prediction is not reward, 12.66 |
| Oct 11 (PLAN, UAV phase) | Every result is on Tiger (2 states) | UAV and Wumpus, 12.41–12.61 |
| Oct 14, C1 (0.1a) | j's belief that i is level 1 is one curve for every i level | 12.47 (exact fixed point of levels 2–5, level-0 control, level-3 j, exact replay); reviewer page v22 |
| Oct 14, C2 (0.1a) | Reward curves above the 2.72 ceiling at step 3 | 12.48 (exact ceiling, 15,600 fresh trials) |
| Oct 14, C3 (0.1a) | Level-grid plot flattened by level 0 | Split y-axis, page v22 |
| Oct 17, G1–G5 (Claude's anticipated objections) | Strawman baseline; no ablations; "depth pays" is obvious; tiny exact domains | 12.62, 12.65, 12.67; 12.63; 12.64, 12.66; 12.67 |

**What is still open, in Claude's recommended order:**
1. **Fix the stale Claude Doc table first (cheap, high value).**
   - The report (https://claude.ai/code/artifact/96fa514f-dc7c-477c-8f97-b31eb1b8fca7, tab body node `213f45dc-b495`) still has a section "Not yet measured" (heading block `met1kpn5tgz.28231`, doc rev 90 on October 19).
   - Its table lists episode rewards, CPU per step, level posteriors and sunbursts as unmeasured. All four were measured on October 7 (12.18, reviewer page). Error attribution is measured (12.24).
   - Only "IPF + RTS at level 3" and "equilibria" are really open.
   - Rewrite the table to what is open now, with RESULTS references. Load the docs skill first, guard whole-block edits with `ifHash`/`ifRev`, and follow the report style (chosen method only, n beside numbers, jargon defined).
   - Also skim the doc for other stale statements written before October 18 (the sections listed in memory `tiger-report-doc`).
2. **IPF + RTS at level 3: price it first, then run if it is cheap.**
   - The baseline claim (PAPER C2) covers level 2 only, and the doc lists level 3 as unmeasured.
   - Suggested cell: the coordination Tiger, L3 H2, against the exact I-POMDP (12.23's level-3 problems, where I-POMCP is exact). Arms: IPF + RTS, published and smoothed, K = 2, 4, 8, N = 64.
   - **Tooling:** `oracle.evaluate_case(..., planner_kind="rts", level=3, particle_filter=...)` and the panel drivers support level 3 (HANDOFF Section 8, "L3 panel").
   - Fix `ENDPOINTS.md` before the run. Ask the user above 100 CPU-h.
   - **Track cumulative CPU against the kill line at every progress check** (G3 overran to 173 CPU-h unnoticed).
3. **Say the 12.67 weak spots plainly in PAPER.md** (no new runs needed):
   - (a) At lookahead 5 with the uniform prior, I-POMCP below root 64,000 misses near-ties. This was diagnosed as root-search error and fixed at root 64,000 in single-decision replays, not in episodes.
   - (b) On UAV, tuned I-NTMCP ties I-POMCP at about 55 s per decision.
   - (c) Only the first 5 decisions (H4) or 2 decisions (H5) of each episode are exactly scored.
   - **Optional, cheap:** re-run the H5 uniform-prior episodes at root 64,000 (same CPU per decision as root 1,000; about 30 × 1 h CPU per arm at H5, so price it).
4. **Equilibria:** `docs/EQUILIBRIA.md` defines the measure, and the episodes record predictions. No RESULTS section reports it. Recommend stating it as future work in the paper rather than running anything, but ask the user.
5. **Then draft the paper from PAPER.md** (G2's outline; every claim has a RESULTS section; Figure 2 is `docs/figures/anytime_uav.png`).

**Where things are:**
- G3 run folder: `results/g3_uav_episodes_20261018` (`ENDPOINTS.md`, `analyze.py`, `analysis.txt`, `score.py`, `ref.py`, `diag_h5.py`, `figure.py`).
- Reviewer page v27: https://claude.ai/artifact/PqpRuTLcuSq5afJdSSxoCB. Rebuild with `results/w4_report_20261007/g3_section.py`, then `build_report.py`, and republish `report.html` by URL (backups `*_20261019.bak`).
- The October 14 reviewer concerns in full: 0.1a below. The October 7 weak point: 0.10 below.

### 0.00 Session of October 18–19: G3 done (RESULTS 12.67)

**State.**
- `main` is clean and pushed. There are 369 tests, and ruff is clean.
- Nothing of this project is running.
- **The next free RESULTS section is 12.68.**
- **Do not remind the user to back up `results/`** (user, October 19: "there is no need"). Older sections below still say to; ignore them.

**What happened (RESULTS 12.67; `results/g3_uav_episodes_20261018`, `ENDPOINTS.md` fixed before the main run, with one dated amendment).**
- **Episodes:** UAV, 20 steps, from the common prior, at H4 and H5, n = 30 per arm (IPF + RTS: H4 only, n = 10).
- **Arms:**
  - I-POMCP, uniform and point-mass priors, modeling the target as configured;
  - I-POMCP at its own budget (`ipomcp-pm-own-<n>`);
  - I-NTMCP, tuned and published, M = 1,000 to 64,000;
  - IPF + RTS, K = 2.
- **Matched CPU:** I-POMCP beats tuned I-NTMCP by +42.3 (M = 4,000, H4), +4.7 (M = 16,000, H4) and +3.5 (M = 16,000, H5). They tie at M = 64,000, where both earn 35 in every episode (about 55 s per decision).
- **Exact scoring at H4** (every decision of i at histories 0–4):
  - I-POMCP: 120/120 per arm from root 4,000 or own budget 1,000;
  - tuned I-NTMCP: 64–82% at M ≤ 16,000, 120/120 at 64,000;
  - IPF + RTS: 62–70%.
- **H5 with the uniform prior:** I-POMCP at root ≤ 16,000 misses near-ties (37–46/60, gap 0.67). The diagnosis (`diag_h5.py`) shows root-search error. Root 64,000 fixes it (19/19) at the same CPU, because the modeled target dominates the cost.
- **Published I-NTMCP force-fills 31–49% of its beliefs on UAV.**
- **Write-up:** PAPER Figure 2 (`docs/figures/anytime_uav.png`), and the C1 and C2 rows are filled. One paragraph went into the Claude Doc. The reviewer page is at version 27, with a new section "Beyond exact reach" built by `results/w4_report_20261007/g3_section.py`.
- **Code:**
  - `baselines/intmcp.INTMCPPlanner`;
  - the study kind `"intmcp"`;
  - `suite.anytime_comparison` (with the `ipomcp-pm-own-<n>` arms);
  - `oracle.episode_reference_q`;
  - `forced_share_<agent>` in episode rows.

**Cost: 173 CPU-h, over the user's 100 CPU-h ask line and the run's own 120 CPU-h kill line.**
- The fixed price was about 95 CPU-h plus about 20 for the pilots.
- The overrun came from the exact references (48 against about 25) and from IPF + RTS (32 against about 15; its decisions cost about twice 12.46's CPU under load).
- Cumulative CPU was not tracked during the run, so the kill line was never checked. Say so to the user.

**Next, Claude's order:**
1. Write the paper draft from PAPER.md: every claim now has a RESULTS section.
2. **Optional:** re-run the H5 uniform-prior I-POMCP arms at root 64,000, to show the qualified configuration in episodes (cheap: same CPU per decision).
3. **Optional:** the untested screen prediction (12.64) and Wumpus 3 × 3 at H3 (0.0a item 3).

**Lessons (also in Section 6):**
- **Track cumulative CPU against the kill line during a run.** Sum `planning_cpu` plus `update_cpu` across finished trials, and the reference CPU, at every progress check.
- **Price exact references by history length, from measured points.** The exact cost about triples per history step (H4: 10, 24, 73, 211, 615 and 1,989 s at histories 0–5), so cap scoring by history before the run.
- **On this machine, `date` jumped about 40 minutes** (a WSL clock re-sync). Judge durations by `ps -o etime`, not by differences of `date`.
- **GNU xargs did not lower its parallelism on SIGUSR2 here.** To throttle, kill the xargs (its running jobs finish), then relaunch the remaining list.
- **A spawned worker's main script needs `if __name__ == "__main__"`,** or every spawned worker exits at once (`run_suite` uses spawn).
- **A pairing rule written in haste ("the cheapest arm at no more CPU") was not a matched comparison.** I amended it before the main-run analysis and recorded the amendment. Write the rule, then test it on the pilot rows before fixing it.
- **The CPU per decision of every planner on UAV is dominated by its model of the target.** I-POMCP's root budget barely changes its cost, so budget-only anytime arms are flat.

### 0.0a Next session (written October 18, evening, at the user's request)

**State.**
- `main` is clean and pushed. The last commit is this HANDOFF commit.
- Nothing is running.
- There are 357 tests. `ruff check` and `ruff format --check` are clean; the `third_party` submodule is excluded from ruff.
- **The next free RESULTS section is 12.67.**

**First thing to tell the user:** back up `results/`. It has not been copied off the machine since September 30. It holds every October dataset, the reviewer-page generator, and today's run folders (12.62–12.66).

**New in the repository today:**
- **The submodule `third_party/i-ntmcp`,** pinned at `63deec3`. A fresh clone needs `git submodule update --init`. The user approved running it on October 18 ("I allow").

#### 1. Goals (unchanged; Section 1)
- **Contribution 1:** an online I-POMDP solver that is *verifiably* exact wherever exact references exist, and degrades gracefully where they do not. "First online I-POMDP planner" is not claimable (RELATED_WORK.md).
- **Contribution 2:** it beats the published baseline (IPF + RTS, as published and with the standard repair) and the modern one (I-NTMCP, as published and tuned).
- **Contribution 3:** interpretability and the value of depth. Depth pays exactly when the opponent's behaviour depends on its beliefs about you *and* the payoff depends on the opponent's action. Prediction gain alone is not enough (12.66).
- **The current aim:** make the suite publication-worthy for AAMAS (main track), and for AAAI or IJCAI if G3 lands. The user's standards hold: analysis fixed before each run, controls and large-n backups, big-tech code quality, and reports in the user's style.

#### 2. Next steps, Claude's recommended order
1. **G3, last piece: UAV episodes from the common prior, beyond exact reach, for every planner.** This is the most serious remaining reviewer gap: every exactly checked domain is small, and on Wumpus 3 × 3 at H2 every root is scored exactly.
   - **Why UAV episodes, not panel problems:** I-NTMCP's model class is a common prior with one opponent level. The UAV panel problems (arbitrary law chains per agent) are outside it.
   - **Cells:** the UAV at level 2 against the matrix's level-1 target (12.49's agents), 20 steps, at H4 and H5. At H4, exact references exist on the realised histories, which anchors the curve; H5 is beyond exact reach.
   - **The I-POMCP arm needs a point-mass prior on the target's level** (`level_weights={1: 1}`) to match I-NTMCP's single level. The matrix's agents hold uniform priors (`suite.hierarchy_agent`). Report the uniform-prior arm too.
   - **Arms:**
     - I-POMCP at root 1,000, 4,000 and 16,000;
     - IPF + RTS, published and smoothed, K = 2, 4, 8;
     - I-NTMCP tuned, M = 1,000 to 64,000.
     - Published I-NTMCP is optional: UAV's senses are noisy, so force-filling should be rare. Check `forced_shares` in the pilot.
   - **Endpoints:**
     - the UAV's 20-step reward and spotting time against median CPU per decision (anytime curves), paired by trial seed;
     - at H4, the share of realised decisions that are exactly optimal.
   - **Engineering:** an episode-planner wrapper around `INTMCPAgent` (`src/ipomdp/baselines/intmcp.py`). `experiments/episodes.run_episode` calls:
     - `get_action()`, which is `search()`;
     - `get_detailed_stats()`, which should return `{"action_values": values}`;
     - `update_root(action, observation)`, which exists.
     - `_prediction(planner)` reads `planner.belief` and returns None without it. I-NTMCP has no explicit belief, so return None, or add a prediction from its nested tree's softmax.
     - The wrapper must see only its own actions and observations. Then register it in `build_agents` and the study spec (`experiments/study.py`), and test it like `tests/test_intmcp.py`.
   - **Price first** (12.61's lesson): pilot at least 10 episodes per arm, including the hardest cell (I-NTMCP at M = 64,000 at H5, and I-POMCP at 16,000 at H5). Price from the mean plus the tail before writing `ENDPOINTS.md`. **Ask the user if it is over 100 CPU-h.**
   - Results go to RESULTS 12.67 and onward, PAPER.md Figure 2, the Claude Doc and the reviewer page.
2. **Write the paper draft from PAPER.md** once G3 is in. Every claim already has a RESULTS section.
3. **Optional, cheap:**
   - Test the screen's untested prediction (12.64): a third UAV level pays only against a level-2 target. That is a small episode run of UAV L3 against target L1 and L2, priced first.
   - Wumpus 3 × 3 at H3 (third level, longer lookahead): likely over 100 CPU-h, so ask.

#### 3. What happened October 18 (RESULTS 12.62–12.66)
- **G1, literature** (`docs/RELATED_WORK.md`):
  - I-NTMCP (IROS 2022) is the closest prior work. Hula, Montague & Dayan (2015) adapted POMCP to I-POMDPs without naming it, so "I-POMCP" is free. No paper checks an online I-POMDP planner against exact solutions.
- **G2, outline** (`docs/PAPER.md`).
- **12.62, the repaired IPF:**
  - `particle_filter="smoothed"` (ε = 0.01) passed the gate: at ε = 0, 420/420 rows were identical.
  - It removes 54–56 of 55–57 Wumpus failures but still loses: Wumpus 20–26/60 and UAV 49–55/80, against 60/60 and 80/80.
- **12.63, the ablation table:**
  - Root chance scoring off breaks 34 of 560 UAV decisions and fixes none (p = 1.2 × 10⁻¹⁰), and changes none of 1,786 Tiger decisions.
  - The other rows come from stored A/Bs; the level-prior row comes from 12.39 and 12.58.
- **12.64, the exact depth screen:**
  - Wumpus: a second level is worth 19.7 per decision against an intentional wumpus and costs 1.7 on random-wumpus histories.
  - UAV H2: worth 6.0 against a cost of 1.6.
  - The screen predicts that a third UAV level pays only against a level-2 target (6.1 against 5.8).
- **12.65, I-NTMCP against the exact I-POMDP** (Wumpus 3 × 3 L2 H2):
  - Tuned, at matched CPU (M = 64,000; median 97 CPU-s): 40/60, against 60/60, with 20 pairs to 0 (Holm 1.9 × 10⁻⁶).
  - Published: 19–32/60, force-filling up to 99% of its beliefs.
  - The adapter gate at H1 was 57/60.
- **12.66, prediction is not reward:**
  - The same prediction gain from a second level (about 55% of the gap) earns nothing in the published Tiger and up to +0.21 reward ranges with coordination payoffs.
  - Figure: `docs/figures/prediction_vs_reward.png`.

#### 4. Files touched October 18 (`git diff --stat b411729..HEAD`)
- **New:**
  - `src/ipomdp/baselines/intmcp.py` (the adapter: `POSGAdapter`, `INTMCPAgent`, variants `published` and `tuned`, `forced_shares`);
  - `tests/test_intmcp.py`;
  - `.gitmodules` and `third_party/i-ntmcp`;
  - `docs/RELATED_WORK.md`, `docs/PAPER.md`, `docs/figures/prediction_vs_reward.png`.
- **Changed:**
  - `src/ipomdp/belief/particle_filter.py`: `smoothing=`, `IPF_SMOOTHING`.
  - `src/ipomdp/planning/solver_bank.py`: the filter name `"smoothed"`.
  - `src/ipomdp/experiments/wumpus_qualification.py`: `--particle-filter`, `--planner intmcp`, `--intmcp-variant`, `_evaluate_intmcp`.
  - `oracle.py`, `panel.py`, `belief_qualification.py`: they list `sorted(FILTERS)`.
  - `tests/test_bounded_beliefs.py`.
  - `pyproject.toml`: pytest `pythonpath` for the submodule; ruff excludes `third_party`.
  - Docs: RESULTS 12.62–12.66, PLAN, this file.
- **Claude Doc report:** four new paragraphs (12.62 baseline repair, 12.63 components, 12.64 screen, 12.65 I-NTMCP).
- **The reviewer page was not changed today** (still version 26).

#### 5. Run folders October 18 (each holds its scripts, `ENDPOINTS.md` and, for the long ones, `KILL_CRITERION.txt`)
| Folder | What | RESULTS |
|---|---|---|
| `g3_ipf_fix_20261018` | smoothing gate (`old/`, `new/`, `compare.py`) and smoothed arms (`analyze.py`) | 12.62 |
| `g3_intmcp_20261018` | I-NTMCP grid, `analyze.py`, `analysis.txt`. `superseded_no_variant/` holds the rows from before the variant split (published defaults only); not reported | 12.65 |
| `g5_wumpus_screen_20261018` | `screen.py` (intentional histories), `control.py` (random-wumpus histories), `uav_screen.py` | 12.64 |
| `g5a_prediction_reward_20261018` | `assemble.py`, `rows.csv`, `figure.py` | 12.66 |

#### 6. Failed paths and lessons from October 18 (also in Section 6)
- **Running an external baseline with only its defaults, under a different sensor model.** I-NTMCP's published refill force-filled up to 99% of its beliefs on Wumpus. Before scoring a baseline on a new domain, measure its belief consistency (here `forced_shares`), and report a tuned variant beside the published one.
- **A gate that cannot separate an adapter bug from the method's own bias.** The first gate (L1 H2) failed for a legitimate reason: MCTS mean backups. The H1 gate, where values are plain particle means, is the clean adapter check. Design adapter gates where the method has no approximation of its own.
- **A hand-written consistency check gave a wrong count** (it reported most particles inconsistent when they were not). Instrument the library's own path, here a counter around the reinvigoration function and the particle's stored history, before trusting a diagnostic.
- **Pricing from the median problem.** The 4-step histories cost 30× the others at L2 (684 s against 13–23 s at M = 16,000). The planned grid would have been about 100 CPU-h. Price from the hardest stratum, and amend endpoints before the expensive cells, never after.
- **Capturing a PID with a pattern** (`ps | grep run.sh`) returned three PIDs. Record `$!` from a plain `nohup … &` line, or select by exact `comm` and `ppid`.
- **Scripted string edits after `ruff format`.** Ruff had rewrapped a call, so an exact-match replacement found nothing. The script asserted and wrote nothing, which is the safe failure. Re-read the file after formatting before scripting another edit.
- **WebFetch cannot read PDFs.** Install `pypdf` and `typing_extensions` into the scratchpad with `uv pip install --target <scratch>/pylib --python python3`, and extract with a script. The Read tool needs `pdftoppm`, which is not installed.
- **Claims written before checking.** "Depth pays after the wumpus is sensed" was false: 17 of 19 such histories had no stench. Check every mechanism sentence against the rows before committing.
- **Still binding:** kill by listed PID only, wait on `kill -0`, snapshot `src/` into long runs, one process per episode with `wrap.sh` and `watchdog.sh` on the shared machine, read `date` before writing times, never trust CPU recorded under load.

### 0.0b Session of October 18, in progress notes and the morning plan (history; superseded by 0.0a)

**Progress, October 18 (later the same day; this list supersedes the state line below):**
- **G1 done:** `docs/RELATED_WORK.md`.
  - I-NTMCP (Schwartz, Zhou & Kurniawati, IROS 2022; MIT code at github.com/RDLLab/i-ntmcp, commit `63deec3`) is the closest prior work: a nested MCTS over history particles, one fixed level per agent, a common prior, a generative model only, and no check against exact I-POMDPs.
  - The name "I-POMCP" is free.
  - Contribution 1 is reframed as *verified* exactness.
- **G2 done:** `docs/PAPER.md` (claims → RESULTS → figures; open slots `MISSING`).
- **G3, step 1 done:** RESULTS 12.62. `particle_filter="smoothed"` is the IPF with likelihood smoothing, ε = `IPF_SMOOTHING` = 0.01.
  - The gate passed: at ε = 0, 420/420 rows were identical to before the change and to the stored c16 and u5 rows. There are 350 tests.
  - The repair removes 54–56 of the 55–57 Wumpus failures, but the baseline still loses: Wumpus 20–26/60 and UAV 49–55/80, against 60/60 and 80/80 (Holm p ≤ 6 × 10⁻⁸).
  - Run folder: `results/g3_ipf_fix_20261018`.
- **G3, step 2 (I-NTMCP), unblocked by the user on October 18 ("I allow"):**
  - Submodule `third_party/i-ntmcp` at `63deec3`. Adapter `src/ipomdp/baselines/intmcp.py`. Driver `wumpus_qualification --planner intmcp --intmcp-variant published|tuned`. Tests: `tests/test_intmcp.py` (pytest adds the submodule to the path through pyproject). There are 357 tests.
  - **Levels:** our level L is their nesting L − 1, since their level 0 is a POMDP planner against a random opponent. **H** is set through their ε, as ε = γ^(H − 1.5).
  - **Finding:** the published variant (no reinvigoration function) force-fills up to 99% of a belief with particles that contradict the agent's own observations. The cause is deterministic senses, a 3,000-rejection cap and a large M. The tuned variant (kernel reinvigoration, a rejection budget of 50 per particle) has 0% forced. Both variants are run and reported, as in 12.62.
  - **Gate A′ passed (tuned, L1 H1, M = 64,000):** 57/60 optimal, max loss 1.5, median max |ΔQ| 4.7. The adapter is valid.
  - **Gate A (L1 H2)** climbs slowly with M: tuned 24 → 29 → (interim 37/53) at M = 1,000, 4,000 and 16,000. The cause is MCTS mean backups under the authors' c = r_max − r_min (about 2,000): depth-1 values average over explored actions.
  - **Done: RESULTS 12.65.** Wumpus 3 × 3 L2 H2: tuned I-NTMCP is 17, 32, 34 and 40 of 60 at M = 1,000, 4,000, 16,000 and 64,000. At M = 64,000 it uses a median 97 CPU-s in total, about I-POMCP's 118 s, and is 40/60 against 60/60, with 20 pairs to 0 (Holm 1.9 × 10⁻⁶). Published: 19, 32 and 29 of 60, force-filling up to 99%. 17.8 CPU-h in all; M = 256,000 was dropped before running (amendment 3). Nothing is running.
- **Next G3 piece, designed but not run: scale beyond exact reach.**
  - **Constraint found today:** I-NTMCP's model class is a common prior with one opponent level. The UAV panel problems, which are arbitrary law chains per agent, are outside it. Only histories and episodes from the common prior compare all planners.
  - **Design:** UAV L2 episodes from the common prior against the matrix's level-1 target (12.49's agents), at H4 (exact references exist, so I-POMCP's decisions are checkable on the realised histories) and H5 (beyond exact reach).
    - Arms: I-POMCP (root 1,000 to 16,000), IPF + RTS (published and smoothed, K = 2, 4, 8), and I-NTMCP (tuned, M = 1,000 to 64,000).
    - Endpoints: 20-step reward and spotting time against median CPU per decision (anytime curves), paired by trial seed.
  - **Engineering needed:** an episode driver that accepts an I-NTMCP agent. `INTMCPAgent` already has `search` and `update_root`; it needs the suite's planner interface (`get_action`, `update_root`), and the real target's actions must stay hidden from it.
  - **Price first:** pilot 10 episodes per arm on the hardest cell (I-NTMCP M = 64,000 at H5) before writing `ENDPOINTS.md`. Ask the user if it is over 100 CPU-h.
- **G4 done:** RESULTS 12.63, the ablation table. New: root chance scoring fixes 34 of 560 UAV decisions and breaks none (p = 1.2 × 10⁻¹⁰), and changes none of 1,786 Tiger decisions. Its last row, level-0 mass vs a single fixed level, comes from 12.39 and 12.58.
- **G5b, Wumpus part done:** RESULTS 12.64. A second level is worth 19.7 per decision against an intentional wumpus (0 at the prior, 51 after 4 steps). The matched control (histories from a random wumpus) costs 1.7 per decision, at 4 of 58 histories; 2 of 60 have zero evidence. The UAV screen (0 CPU, from the 12.41 gate rows) gives H2 1 → 2 worth 6.0 against 1.6, and 2 → 3 worth 6.1 against 5.8: a third level pays only against a level-2 target, an untested prediction.

**State at the start of October 18.** `main` was clean and pushed (`eefeb28` plus the HANDOFF commit), and nothing was running. There were 343 tests, and `ruff` was clean. Tiger, UAV and Wumpus 3 × 3 are complete (RESULTS 12.1–12.61). **The next free RESULTS section is now 12.67.** G5a done: RESULTS 12.66, `docs/figures/prediction_vs_reward.png`.

**First thing to tell the user:** back up `results/`. It has not been copied off the machine since September 30, and it holds every October dataset plus the reviewer-page generator.

#### 1. Goals (unchanged since September 30; Section 1)
- **Contribution 1:** an online I-POMDP solver that is exactly optimal wherever exact references exist, and degrades gracefully where they do not.
- **Contribution 2:** it beats the published baseline, IPF + RTS (Doshi & Gmytrasiewicz 2009).
- **Contribution 3:** interpretability and the value of depth. Depth pays exactly when the opponent's behaviour depends on its beliefs about you, and up to about one level above the opponent's.
- **The user's standards:** rigour, the analysis fixed before each run, controls and large-n backups for anything a reviewer could call noise, big-tech code quality, and reports in the user's style (chosen method only, interim n labelled, jargon defined).
- **The current aim, set October 17–18:** make the suite publication-worthy. The user asked whether it is. Claude's answer: a strong AAMAS or workshop paper as it stands, borderline for a main track (AAAI, IJCAI, NeurIPS). The gaps are below.

#### 2. Next steps: the "publication gaps" phase, Claude's order (write it as a new phase on top of PLAN.md first)

Each step gets a run folder, an `ENDPOINTS.md` fixed before any run, and a `KILL_CRITERION.txt`. Price every run first, and ask the user before anything over 100 CPU-h.

1. **G1: literature check** (0 CPU; do it first, since it can change the framing of contribution 1).
   - Use WebSearch (extended mode) and open the papers themselves, not snippets. Find modern online I-POMDP planners. Candidates from memory, unverified:
     - a nested-MCTS I-POMDP planner (Schwartz, Zhou & Kurniawati, about 2022);
     - IPOMDP-Lite (Hoang & Low);
     - the interactive particle filter and IPOMDP-Net line (Han & Gmytrasiewicz);
     - Panella & Gmytrasiewicz (finite-state models of others);
     - recent "I-POMCP" uses. Is the name taken?
   - Deliverable: `docs/RELATED_WORK.md`, one row per method (year, venue, belief representation, planner, levels, domains, exactness claims, code available) and the gap each leaves.
   - Decision gate: if a modern planner has public code and runs on Tiger or UAV, add it as a baseline in G3. Otherwise argue the comparison from its published numbers.
2. **G2: paper outline** (0 CPU). `docs/PAPER.md`: claims → evidence (RESULTS sections) → figure or table, with the missing pieces marked. Every later step must fill a slot in it.
3. **G3: a scale experiment beyond exact reach** (contributions 1 and 2; the most serious gap). A reviewer can say that on 3 × 3 boards at horizon 2 every root is scored exactly, so the MCTS does no work.
   - **Design:** anytime curves of reward (episodes) and decision quality against CPU per decision, I-POMCP vs IPF + RTS, CPU-matched across budgets.
   - **Where:** UAV L2 at H4–H5 (exact references exist up to H4, so H4 anchors the curve) and/or Wumpus 4 × 4 L1 (qualified at N = 4,096 and 10,000 simulations, 12.53). Pilot the cost of one hardest-case episode per arm first (the efficient-experiments rule).
   - **Make the baseline fair:** add a standard zero-likelihood fix to the IPF, so 12.59's failure cannot be called a strawman. Use likelihood smoothing (mix O with ε uniform) or rejuvenation, as a new option in `src/ipomdp/belief/particle_filter.py`. Gate: ε = 0 reproduces today's IPF rows bit for bit (a `u2_gate`-style before/after). Report the published IPF, the fixed IPF and I-POMCP side by side.
4. **G4: one ablation table** (mostly 0 CPU, from stored results).
   - The compressed-exact filter vs IPF (12.43–12.45, 12.59).
   - Chance scoring on vs off: free, because episode and qualification rows keep `search_values` beside the scored values (12.14).
   - The 1/n stopping rule (12.15); root budget (12.33); stratified vs systematic compression (12.57).
   - Certified bounds: cost and coverage (12.17).
   - Any piece with no paired number gets a small paired run.
5. **G5: a sharper contribution 3** (low CPU).
   - **(a) Prediction gain vs reward gain.** Pair them per cell across all three domains (Tiger 12.37, UAV 12.49, Wumpus 12.60–12.61) and show one scatter. A third level predicts better but earns the same (12.27, 12.32, 12.61): when does better prediction pay?
   - **(b) The exact screen as an a-priori predictor of when depth pays.** It exists for Tiger (`results/p2_payoff_screen_20261007/screen.py`, 12.22). Extend it to UAV and Wumpus 3 × 3 at H2, where exact solves are cheap, and check whether it predicts 12.49 and 12.60 before looking at the episodes.
   - **(c) One figure** for THEORY.md's proposition: the second-level gain against a mind vs against a random opponent, per domain.
6. **Optional:** Wumpus 3 × 3 at horizon 3, to test whether a third level pays with a longer lookahead. A decision costs 6–36 CPU-min (12.58), so price it first. It is likely over 100 CPU-h, so ask the user.

**The reviewer page and the Claude Doc** follow every new RESULTS section, as before (rebuild: 0.6 and the Wumpus note in item 3).

#### 3. Files touched October 17–18 (`git diff --stat cc5a30b..HEAD`)
- `docs/RESULTS.md`: 12.61 (Wumpus human L2 → L3) and a pointer from 12.60.
- `docs/THEORY.md`: a new section, "When a deeper model can pay":
  - Proposition 1: correct prediction is enough;
  - the corollary: no per-decision gain against a non-intentional opponent;
  - the remark: level-0 mass keeps nested updates defined.
- `docs/PLAN.md`: the Wumpus 3 × 3 phase marked complete. `docs/HANDOFF.md`: this section.
- **Outside the repository (git-ignored):**
  - `results/w4_report_20261007/wumpus_section.py` (new). It builds `wumpus.json` and parses the run folders' own analysis scripts, so the page shows exactly RESULTS' numbers. It also computes the mechanism (the probability given to the wumpus's actual move, and the belief over its level) for levels 2 and 3.
  - `build_report.py` and `template.html`: the Wumpus section (`wumpus_rts_table`, `wumpus_gains_table`, `wumpus_confirm_table`, `wumpus_mechanism`, `wumpus_level3`; JS `wumpus()` and `causeBars()`).
  - Backups: `*_20261017.bak`, plus `report_v24_20261017.bak` (the live page before this session).
  - **Rebuild:** `cd results/w4_report_20261007 && ../../.venv/bin/python wumpus_section.py && ../../.venv/bin/python build_report.py`, then republish `report.html` to the same URL (read the artifact first from a new session).
- **Reviewer page version 26:** https://claude.ai/artifact/PqpRuTLcuSq5afJdSSxoCB.
- **Claude Doc report:** a new section, "When a second level pays: three domains" (cross-domain table and mechanism), before "Checks behind". https://claude.ai/code/artifact/96fa514f-dc7c-477c-8f97-b31eb1b8fca7

#### 4. "Taste in research": directions Claude would pursue, best first (the user asked for these to be recorded)
- **Exactness as a design asset, extended to scale.** "Our planner *is* the exact I-POMDP wherever that is checkable (12.58: every root exactly scored), and degrades gracefully where it is not." G3 is the second half of that sentence, and the paper's spine.
- **Prediction is not reward.** Deeper levels predict better yet earn the same (Tiger 12.27 and 12.32, Wumpus 12.61). A value-of-information view would say when a better prediction changes an action at a belief the agent reaches. That is measurable exactly on small problems, and is a stronger contribution 3 than "depth pays only against a mind", which THEORY.md now states and a reviewer may call obvious.
- **The exact screen as a design tool:** predict, before any episode, which games reward depth (G5b). If it predicts 12.49 and 12.60, it is a reusable method, not just a check.
- **Deterministic-sensor domains break sampled filters.** IPF fails on 55–57 of 60 Wumpus problems (12.59) and on UAV (12.44–12.45), with zero-likelihood particles and nested consistency under a common prior. A short section; the fair-baseline fix in G3 tests how much of it is fixable.
- **Level-0 mass is necessary at level 3 and above with deterministic percepts.** The remark is in THEORY.md; it is worth a paragraph and a proof in the paper.
- **Not recommended:**
  - more Wumpus 3 × 3 episodes at horizon 2 (the question is answered);
  - the 4 × 4 Wumpus at L2 (12.55–12.56: over 8 GB per decision);
  - level 4 anywhere at H3 (over 100 CPU-h; the user said "not now", October 16);
  - micro-optimising exact enumeration (12.56: bit-identical speedups did not pay);
  - adopting either candidate branch (`resampling/optimal-fc`, `bounded-belief/compressors`) without a new gate.

#### 5. Failed paths from October 17–18 (also in Section 6), plus the binding older lessons
- **Interim effects again.** At n = 32, human L2 → L3 read +312 and +374 (Holm 0.02–0.03). At n = 100 both were not significant (+80, +120). Report and decide on the final n only.
- **Pricing a run from a 2-episode pilot.** The pilot said 13–14 CPU-min per level-3 episode; the run averaged 21–24 with a tail of about 4 CPU-h episodes, about 110 CPU-h in all (against about 70). Pilot at least 10 episodes, including the hardest cells, and price from the mean plus the tail.
- **Parsing analysis-script output with regexes.** One pattern (the confirmation table) silently matched nothing until checked. Assert the number of parsed rows every time.
- **Headless-Chrome libraries extracted into a session scratchpad** do not survive the session. Re-extract them with `apt-get download libnspr4 libnss3 libasound2` and `dpkg -x <deb> <dir>`, then run `LD_LIBRARY_PATH=<dir>/usr/lib/x86_64-linux-gnu ~/.cache/ms-playwright/chromium_headless_shell-1243/chrome-headless-shell-linux64/chrome-headless-shell --no-sandbox --screenshot=… --window-size=1000,3400 file://…`. To check one page section, inject CSS that hides the others.
- **Still binding from October 14–17:**
  - Kill by listed PID only; never match processes by pattern.
  - Wait on `kill -0 <pid>` inside the turn, never on a notification alone.
  - Snapshot `src/` into every long run's folder.
  - One process per episode, with `wrap.sh` and `watchdog.sh` on the shared machine.
  - Size workers from the largest RSS seen (Wumpus i2 episodes reach 2.6 GB).
  - Read `date` before writing times.
  - Never trust CPU recorded under load.

### 0.0c Next-session notes of October 17 (history)

**1. Done October 17, evening (this session):**
- **Step 4 finished** (300/300, 22:18) and was analysed with the fixed test: **RESULTS 12.61**. Human L2 → L3: −19 against a random wumpus, +80 and +120 against an intentional one, none significant (Holm 0.55–0.86). Level 3 predicts the wumpus better (0.54–0.55 of the probability on its actual move, against 0.48–0.50). It cost about 110 CPU-h for the human, against the pilot's estimate of about 70, because of a heavy tail of 4 CPU-h episodes. The interim numbers at n = 32 (+312, +374) were misleading.
- **Reviewer page version 25** (same URL): a Wumpus section built by `results/w4_report_20261007/wumpus_section.py` → `wumpus.json`. It contains the RTS decision table, heat maps, paired gains, death-cause bars, the confirmation table, the prediction and level-belief panel, and the level-3 paragraph. All inferential numbers are parsed from the run folders' own analysis scripts. Backups: `*_20261017.bak` and `report_v24_20261017.bak` (the live v24). **Rebuild after 12.61:** `python wumpus_section.py && python build_report.py`, then republish.
- **Claude Doc:** a new section, "When a second level pays: three domains" (a cross-domain table plus the mechanism), before "Checks behind".
- **THEORY.md:** "When a deeper model can pay": correct prediction is enough; no gain against a non-intentional opponent (per decision, at the planning horizon); and level-0 mass keeps nested updates defined.
- **Headless Chrome:** the extracted libraries had gone. Redo it with `apt-get download libnspr4 libnss3 libasound2`, `dpkg -x` into a scratch folder, and `LD_LIBRARY_PATH=<that>/usr/lib/x86_64-linux-gnu`; the browser is `~/.cache/ms-playwright/chromium_headless_shell-1243/chrome-headless-shell-linux64/chrome-headless-shell`. To screenshot one section, inject CSS that hides the others.

**2. Remind the user to back up `results/`.** It has not been backed up since September 30, and it holds every October dataset, including c12–c17.

**3. Goals** (unchanged since September 30; Section 1):
- **Contribution 1:** an online I-POMDP solver that is exactly optimal wherever exact references exist.
- **Contribution 2:** it beats the published baseline, IPF + RTS.
- **Contribution 3:** interpretability and the value of depth.

The evidence now spans three domains: Tiger (published and coordination variants), UAV reconnaissance, and Wumpus (4 × 4 at L1; 3 × 3 at L1–L3). The user's standards: rigour, the analysis fixed before each run, controls and large-n backups for anything a reviewer could call noise, and big-tech code quality.

**4. Next steps, Claude's recommended order:** close the publication gaps. The user asked on October 17 whether the work is publication-worthy. Claude's answer: a strong AAMAS or workshop paper as it stands, borderline for a main track. The gaps, most serious first:
- (a) **Literature check** (cheap; do it first): modern online I-POMDP planners. Check, among others, a nested-MCTS I-POMDP planner (Schwartz, Zhou & Kurniawati, about 2022, unverified), IPOMDP-Lite (Hoang & Low), and the Han & Gmytrasiewicz particle-filter line. Also check whether the name "I-POMCP" is already taken. This may change how contribution 1 is framed.
- (b) **A scale experiment beyond exact reach,** e.g. Wumpus 4 × 4 at L1–L2 or UAV at a larger horizon. Plot reward against CPU for I-POMCP and IPF + RTS, CPU-matched. Include IPF with a standard zero-likelihood fix (likelihood smoothing or rejuvenation), so the 12.59 failure cannot be called a strawman.
- (c) **One ablation table:** the compressed exact filter vs IPF, chance scoring on vs off, the stopping rule, and certified bounds (cost only), drawn from existing A/B results where possible.
- (d) **Contribution 3, sharper:** the exact screen as an a-priori predictor of when depth pays (extend 12.22 to UAV and Wumpus), and the link between prediction gain and reward gain.
- (e) **A paper outline,** so the remaining runs fill specific figures.
- Optional: Wumpus horizon 3 to test whether a third level pays at a longer horizon (6–36 CPU-min per decision; price it first).

**5. "Taste in research": directions Claude would pursue, best first** (the user asked for these to be recorded):
- **The mechanism of "depth pays only against a mind".** A clean theory statement: with a level-k prior that includes level 0, the value of modeling an opponent is zero when the opponent's policy does not depend on its beliefs. All three domains are instances, and the random-opponent controls are the exact backups (THEORY.md material, plus one figure).
- **Exactness as a design asset.** On small boards, horizon 2 with exact beliefs makes every I-POMCP root exactly scored (12.58), so episode results equal exact-I-POMDP behaviour, not search noise. A reviewer-proof story: "our planner is the exact I-POMDP where that is checkable, and degrades gracefully where it is not."
- **Deterministic-sensor domains break sampled filters.** IPF fails in 55–57 of 60 Wumpus problems (12.59) and on UAV (12.44–12.45). A short note on why: zero-likelihood particles, and nested consistency under a common prior. Exact or compressed-exact filtering is the remedy; stratified rather than systematic compression is the safe default (12.57).
- **Level-0 mass is necessary at level 3 and above with deterministic percepts:** without it, the exact I-POMDP is undefined on counterfactual branches (12.58). This is a theoretical point worth a paragraph and a small proof.
- **Not recommended:**
  - horizon 3+ on Wumpus (6–36 CPU-min per decision);
  - the 4 × 4 Wumpus at L2 (12.55–12.56);
  - more micro-optimisation of exact enumeration (12.56: bit-identical speedups did not pay);
  - level 4 on the coordination Tiger at H3 (over 100 CPU-h; the user said "not now", October 16).

**6. Files touched October 16–17** (`git diff --stat 562944f..HEAD`):
- `src/ipomdp/belief/resampling.py`: `stratified_draws`.
- `src/ipomdp/belief/compressed_filter.py`: compression uses stratified draws (12.57).
- `src/ipomdp/experiments/oracle.py`: `CERTIFICATION_EPSILON = 0.05`.
- `src/ipomdp/experiments/panel.py`: the `eps_correct_rate` and `eps_failure_high` summary columns (12.54).
- `src/ipomdp/experiments/wumpus_qualification.py`, now a general Wumpus qualification driver:
  - `--board W H P`; `--level 1|2|3` (exact I-POMDP references above L1);
  - `--modeled`; `--l0 <weight>|uniform` (level priors);
  - `--search-particles 0` (exact search);
  - `--script fixed|random|modeled` (histories from the exact wumpus and the wumpus's exact model of the human; `ModeledAgent`, `HistoryEnded`);
  - `--planner rts --particles N` (IPF + RTS baseline);
  - search and update failures recorded as rows.
- Tests: `tests/test_wumpus_qualification.py` (10 tests), `tests/test_bounded_beliefs.py` (the stratified bound, an aliasing regression test), `tests/test_panel.py` (ε columns). 343 tests in all.
- Docs: RESULTS 12.54–12.60, THEORY (certification standard), PLAN (the Wumpus 3 × 3 plan, on top), the Claude Doc report (Section 4 map).

**7. Run folders October 16–17** (each holds its scripts, `ENDPOINTS.md` and `KILL_CRITERION.txt`):

| Folder | What | RESULTS |
|---|---|---|
| `p1_pac_rescore_20261016` | ε re-scoring of every stored grid | 12.54 |
| `c6_wumpus_l2_pilot_20261016` | 4 × 4 L2 cost pilot | 12.55 |
| `c7`–`c10_*_20261016` | nested compression, interning, speedup gates (not adopted) | 12.56 |
| `c11_wumpus3_exact_20261016` | exact 3 × 3 L1–L3 costs | 12.56 |
| `c12_wumpus3_l2_20261016` | first 3 × 3 L2 grid (systematic compressor), shared references | 12.57 |
| `c13_compression_ab_20261016` | systematic vs stratified A/B; G1x exact-search L2 H2 | 12.57–12.58 |
| `c14_wumpus3_qualification_20261016` | L2 H3 (interim), L3 H2 | 12.58 |
| `c15_wumpus3_episodes_20261017` | gate, matrix, confirmation, level 3 (running) | 12.58, 12.60 |
| `c16_wumpus3_rts_20261017` | IPF + RTS vs I-POMCP decisions | 12.59 |
| `c17_wumpus3_level3_gate_20261017` | L3 episode-configuration gate | 12.58 |

**8. Run lessons from October 16–17** (also in Section 6):
- `wumpus_qualification` uses a spawn pool. Its workers are children of the python driver, so check `ps -o comm -p <pid>` says python before recording a PID. Killing only a shell wrapper orphans the workers.
- **Never match processes by pattern,** not even with a `while read` over `ps | grep`: the loop matched its own shell and killed it (October 17). Kill listed PIDs only.
- `pool.map` writes results in order: a slow job hides finished ones behind it, so count running workers, not rows.
- Wumpus episodes vary in memory: some i2 episodes reach 2.6 GB, against 0.75 GB in the pilot. Size workers from the largest RSS seen, not the pilot's.
- Read `date` before writing times into notes; two guessed times had to be corrected.

**Run lessons from October 14–15, still binding:**
- On the shared machine, use a run folder's `wrap.sh`: it waits for 3 GB free and always exits 0, so a killed job never stops xargs.
- `watchdog.sh` walks the process tree from the xargs PID, so it only ever kills this run's jobs.
- GNU xargs lowers its parallelism on SIGUSR2; raising it with SIGUSR1 did not take effect here.
- Two runners over one job list duplicate work where they meet.
- Never trust CPU recorded under heavy load or memory pressure. Rerun one deterministic job before quoting a cost (12.42's correction), and compare CPU only within paired runs (12.56).
- The machine can suspend: the wall clock then jumps, and the jobs pause without harm.
- Never wait with `pgrep -f` in a loop, since it matches its own shell. Wait with `kill -0 <pid>`.
- Do not rewrite a guard's PID with a digit-pattern `sed`: it also rewrote the 3 GB threshold. Replace the old PID literally.
- `R=… && nohup cmd &` puts the assignment in a background subshell, so later lines see an empty `R`. Set variables on their own line first (repeated on October 16; still true).

*(The October 15 list, items 1–6, is done: 12.51–12.54, with 5b deferred by the user.)*

### 0.0 State, rules and how to work

- **Repository.**
  - `main` is clean at the last HANDOFF commit. 357 tests (`.venv/bin/python -m pytest -q`; the submodule must be checked out for `tests/test_intmcp.py`, which otherwise skips; about 3 minutes on a quiet machine, over an hour at load 20, because the qualification tests spawn one process per belief), and `ruff check` and `ruff format --check` are clean. The four timing tests in `test_process_supervisor.py` fail while the machine is heavily loaded (load 16–20) and pass on a quiet machine (6/6 at load 0.5, October 14): a load artifact, not a defect.
  - **Running:** nothing (October 18, 20:53; the I-NTMCP grid of 12.65 is complete).
  - **Standing rule (user, October 11): after every implementation, commit and push (`git push origin main`) without asking.** That covers code, docs, plan and write-ups. Never force-push. `main` was pushed to `origin` (github.com/huytruong1810/ipomcp) on October 11. Candidate branches stay local unless asked.
  - Two branches hold candidates that were **not adopted**; do not merge either without a new gate:
    - `resampling/optimal-fc` (12.28);
    - `bounded-belief/compressors` (12.31): compressor strategies as bank filters, plus drivers that accept every filter name. It is a superset of the first, and its worktree is `../ipomcp-q2`.
- **The user's mandate** (Section 1):
  - rigorous, gated progress on the three contributions;
  - redesign rather than scale up when returns diminish;
  - A/B gates with paired statistics, the analysis fixed before the run;
  - big-tech code quality, with no legacy shims.

  The user wants autonomy: take the recommended option, state it in one line, and keep going. Ask only for decisions that are genuinely theirs. Examples: a run of over 100 CPU-h on the shared machine (they approved the October 9 H3 study when asked), and deletions. Pushing is not one of them any more (standing rule above).
- **The user's report style:** chosen method only, interim n labeled as interim, and jargon defined.
- **Before running anything:**
  - **Read Section 6 (do not repeat).**
  - **Snapshot every long run.** Run `mkdir -p <run>/src && cp -r src/ipomdp <run>/src/`, then check that `PYTHONPATH=<run>/src .venv/bin/python -c 'import ipomdp; print(ipomdp.__file__)'` prints the snapshot path. A wrong layout silently imports the live code.
  - **One process per episode.** Use `xargs -P <k> -L 1 env PYTHONPATH=<run>/src .venv/bin/python <run>/job.py < jobs.txt`, with a job script that skips finished files. The template is `results/q4_h3_matched_20261009/h3_job.py`. Never use a `ProcessPoolExecutor` under a memory watchdog: one kill breaks the pool and ends the run (October 9 lost a night this way).
  - **Memory.**
    - Since 12.36, level-3 H3 episodes peak at about 0.7 GB each over 10 steps (they plateaued at about 2 GB before). Level-4 episodes took 1–2 GB before 12.36 and have not been re-measured.
    - The machine has 23 GB, 24 cores, and is shared: `~/projects/jepa_ipomdp` runs jobs of up to about 12 GB. Never touch its processes.
    - Check `free -g`, then size the workers. Before 12.36, 6 level-3 H3 workers left about 10 GB free; re-measure on the first run.
    - **Watchdog:** `results/m1_level_matrix_20261011/watchdog.sh <runner pid> <log>` kills the youngest *child of that PID* when `MemAvailable` falls below 1.5 GB, and `memlog.sh <runner pid> <log>` logs memory every 5 minutes. Give it the **xargs** PID, not the `sh -c` wrapper's: the wrapper's only child is xargs, so the watchdog would kill the whole run. UAV jobs use 0.1–0.3 GB each.
    - **Job caps:** wrap each job as `timeout 3h …`. To lift the cap from a slow but healthy job, `kill -9` only its `timeout` wrapper; the job keeps running and still writes its file. The panel and qualification drivers have a supervisor timeout (`timeout_seconds`, default 2 h): size jobs below it.
    - **Scripts that call a supervised driver** (`run_panel`, `run_qualification`, `run_suite`) need `if __name__ == "__main__":`. Workers are spawned and re-import the script, so an unguarded script reruns itself in every worker (October 14 lost two gate runs this way).
  - **Processes.** Kill by PID only (`ps -eo pid,ppid,args`). Never use `pkill -f`. Never wait on `pgrep -f`: it matches its own command line. Wait on a PID or an output file instead.
  - **CPU.** Compare CPU only within paired, same-run measurements.
  - **Kill criteria.** Write a kill criterion before every run, in `KILL_CRITERION.txt`, and check it.
- **Results.**
  - Put each run in a new dated folder under `results/` (git-ignored), with its scripts inside.
  - Numbers go into `docs/RESULTS.md`. **The next free section is 12.67.**
  - When results land, update this file, PLAN.md's status, the Claude Doc report (Section 4) and, if relevant, the reviewer page (0.6).
  - `results/` has not been backed up since September 30. It holds every October dataset and the reviewer-page generator; remind the user to copy it off the machine.
- **Current task (new session):** 0.0a, item 2 (G3's UAV episodes). Suite tools: `python -m ipomdp.experiments.suite`; each run folder holds its launcher, `analyze.py` and `KILL_CRITERION.txt`.
- **Lesson (October 12–13): do not end a turn waiting on background-task notifications while unattended work remains.** About 21 hours were lost when every run had finished but no turn resumed. Wait inside a turn on a PID or output file (a background `until …; do sleep 60; done` you then block on), or keep working.

### 0.1 Where the science stands (October 14)

*Since October 14:*
- **UAV:** 12.41–12.50.
- **Certification standard:** ε = 0.05 (12.54).
- **Wumpus:** 4 × 4 L1 qualified at N = 4,096 with 10,000 simulations under stratified compression (12.53, 12.57); 3 × 3 qualified at L2 and L3 (12.58); I-POMCP against IPF + RTS (12.59); value of depth (12.60).

The lists below are as of October 14.

1. **Online solver (I-POMCP with bounded nested beliefs).**
   - **Optimal wherever exact references exist:** L1 H1–H5, 220/220 at H5 (12.1); L2 H3–H4, 380/380 (12.3–12.4); L3 H3, 162/162 (12.7); L4 H2, 486/486 (12.23).
   - **UAV reconnaissance (12.41–12.42), at qualified budgets:** L1 H2–H5 40/40 each (root 1,000 up to H3, 4,000 at H4, 64,000 at H5); L2 H2–H3 80/80 (root 1,000, target modeled at 400); L2 H4 80/80 with an exact target at root 16,000, and 32/32 with the target modeled at 6,400 with stopping; L3 H2 640/640. Certified bounds contained the exact Q in 1,016/1,016 decisions checked.
   - **The L3 H4 residual** (4/80 near-tie beliefs wrong at gaps ≤ 0.03; 12.14) is 80/80 ε-correct under the ε = 0.05 standard adopted October 16 (12.54). UAV has a similar near-tie (gap 0.027) for the modeled target at 6,400 with stopping, harmless to i's decisions (12.42).
   - **Certified bounds:** every real decision carries per-action intervals that contain the exact Q of the searched model (12.17).
   - **Beliefs:** the compressed filter (exact updates, systematic compression to N = 64; 12.31 found no better compressor at acceptable cost). **Root budget (12.33–12.35):** a decision's CPU is almost all nested solves, paid within its first 125 root simulations; more root simulations cost 0.2–0.35 ms each. Root 4,000 cuts L2 H4's optimal-action Q error 3.9× (0.121 → 0.031) at 1.06× CPU; no root stopping rule helps. Episodes keep 1,000 (a matched model of j makes 4,000 cost about 1.4× at L3).
2. **Benchmarks.** I-POMCP is 380/380 at L2 H3 and H4. IPF + RTS reaches 95–99.5% there and is 3–5× slower at H4 (12.20). **UAV L2 (12.43):** I-POMCP 80/80 at H2 and H3; IPF + RTS 53–68% at K = 2–8 (Holm p < 10⁻⁷), and 2–4× slower at H3; at H4 (12.46) 80/80 against 46–50%, at 1.6–11× less CPU. Its error is observation-sampling search error (the compressed-belief ablation matches it), and it converges at K = 128.
3. **Interpretability and the value of depth.**
   - **Modeling the opponent pays** on the coordination Tiger: level 1 → 2 gains 27–30 over 20 steps (Holm p < 0.001, 12.27). In the published Tiger it gains nothing, since payoffs depend only on one's own door.
   - **A third level predicts slightly better but earns the same:** at H2, 12.27; at H3 with a model matched to the opponent's configuration, +0.8 [−6.7, 8.6], n = 50 (12.32). The exact screen agrees: a third level is worth little, and only on rare beliefs (12.22).
   - **Episodes also show:** sunbursts, level grids (12.21, 12.27) and prediction-miss diagnostics (12.30).
   - **The Tiger suite (12.37–12.40), at H2 with N = 64 unless noted:**
     - **Value of depth on the coordination Tiger:** level 0 → 1 is worth about 600 over 20 steps; 1 → 2 is worth 15–30 against an intentional opponent (Holm p < 0.05); 2 → 3 gives +5 to +7, not significant; 3 → 4 gives nothing (level 4 acts as level 3 in 34–35 of 40 episodes). The published Tiger (control): nothing above level 0. Against a random opponent, no level above 1 adds anything.
     - **I-POMCP against IPF + RTS in episodes (12.38):** +28 to +38 for I-POMCP on the coordination Tiger at H3 (Holm p = 0.003), including against RTS K = 4, which costs more CPU; +24 to +26 at H4, not significant at n = 50; no difference on the published Tiger.
     - **Prior sensitivity (12.39):** at level 3 against a level-1 j, a point prior one level too high costs 17.5 and "random" costs 30.0; at level 4 against a level-2 j, wrong priors cost nothing, because levels 2–4 act alike. A uniform prior costs nothing significant at either level.
   - **Wumpus L1** is qualified on 3 distinct beliefs only (12.19). **UAV:** see item 1; no UAV episodes yet.
   - **Equilibria:** a design note only (`docs/EQUILIBRIA.md`).
- **Lessons that bind future designs:**
  - **A "correct model" of an opponent must match its whole configuration,** budgets included (12.30).
  - **Every bounded-belief run reports** `analysis.metrics.prediction_misses`.
- **Reviewer-facing material:**
  - **Reviewer page, version 23** (October 15: version 22 added the reviewer checks C1–C3, 12.47–12.48; version 23 the UAV section, 12.41–12.49): https://claude.ai/artifact/PqpRuTLcuSq5afJdSSxoCB (private to the user). How to rebuild it is in 0.6.
  - **Claude Doc report:** https://claude.ai/code/artifact/96fa514f-dc7c-477c-8f97-b31eb1b8fca7 (Section 4).

### 0.1a Reviewer concerns on the Tiger visuals (user, October 14): top priority after the H4 grid

**Status (October 15): C1–C3 done.** C1: RESULTS 12.47 (j's two hypotheses; exact fixed point of levels 2–5 at H2; a level-3 j identifies levels; exact replay). C2: RESULTS 12.48 (exact ceiling V*_3 = 2.72; no plotted interval above it; 15,600 fresh trials give 2.59 ± 0.22 at step 3; the plotted trials reproduce 100/100). C3: split y-axis plus a dotted ceiling. **Reviewer page version 22** published October 15 (same URL; generator backups `*_20261015.bak`, live v21 saved as `report_v21_20261015.bak`), with a new section "Checks behind two patterns in these plots". The C1 screen finished what fits: level 6 on 72 of 81 sampled chains (all identical to level 5); level 4 at H3 needs over 4 GB per exact solve and was not run; the Claude Doc report covers C1/C2 and UAV since October 15.

The user relayed a reviewer's skepticism about two patterns on the reviewer page (version 21), plus one plot request. The reviewer will not accept "the simulation happens to produce it" or "statistical noise": every explanation needs a **backup benchmark that cannot be argued with** (an exact computation, a positive control, or a much larger n), and a theory statement where one exists. Do this before U6.

**C1. "j's belief that i is level 1" (Opponent modeling) is the same curve for i at levels 1, 2 and 3.** The reviewer asks: why does a level-2 j converge to "i is level 1" whatever i's level? Should a higher-level i not act very differently? Would i at level 1000 give the same curve? And if j can never model an equal- or higher-level i (a finitely nested I-POMDP cannot), should j not conclude that i acts randomly (level 0)? Should exact value iteration check this?
- **Facts already in hand (October 14).**
  - The plotted curves (`results/w4_report_20261007/report_data.json`, `level_posterior_j`) are identical through step 3 (0.694, 0.807, 0.81) and close afterwards (0.99 at step 20 for all three).
  - The three conditions share trial seeds (`analyze.py`: L3 `w4_episodes_20261006`, L2/L1 `w4_controls_20261006`), and i takes the same early actions at every level.
  - **Positive control:** in the level grid (`w_suite_report_20261012/published_grid.json`), a level-2 j facing an i that *is* level 0 puts **0.986 [0.963, 0.999]** on level 0 by step 20 (n = 100). j detects randomness when it is there.
- **The explanation to establish.**
  - j's hypothesis space is {level 0 = uniform over 3 actions, level 1}. A Bayesian with a misspecified model concentrates on the hypothesis closest to the truth in KL divergence (Berk 1966; Kleijn & van der Vaart 2012), not on the most random one.
  - Any i of level ≥ 1 mostly listens until confident, then opens. Level 1 predicts that with high probability; level 0 gives each action 1/3. So each observed listen multiplies the odds for level 1 by up to about 3.
  - In the published Tiger an agent is paid for its own door only. So i's best response barely depends on how it models j: levels 1–3 act alike, which RESULTS 12.22 and 12.40 already show (level 4 acts as level 3 in 34–35 of 40 episodes). Identical actions give identical j posteriors.
- **Backup benchmarks (each exact or a control; analysis fixed before running):**
  1. **Exact replay of j's posterior** on the logged trials, with uncompressed exact filtering (`q1_miss_diagnosis_20261009/replay.py` is the template). The plotted (N = 64) curve must match the exact one within a stated tolerance.
  2. **Likelihood-ratio decomposition per step:** log P(i's action | level 1) − log P(i's action | level 0) as j computes it. It shows which actions move j, and that the movement comes from listens, not from deep strategy.
  3. **Exact policy agreement across levels:** with the exact I-POMDP, compute i's policy at levels 1–5 (H2; H3 where feasible, since level 4 at H3 took over 10 minutes per solve) over a belief grid plus the reachable beliefs. Report the share of beliefs where level k+1's action differs from level k's. If levels ≥ 2 coincide on every reachable belief, j's posterior is the same for every higher level. That is the level-1000 answer: by induction, plus the uniform-prior mixture argument. State the argument and its premise exactly.
  4. **The coordination Tiger as the contrast:** there depth changes behaviour (12.37). Plot j's posterior per i level there; if the curves separate, that shows the overlap in the published game comes from its payoffs, not from the code.
  5. **More controls:** i at level 0 (done above), and a "scripted" i that always listens (expected: j → level 1), or always opens (expected: j → level 0).
- **Write-up:** a RESULTS section plus a page panel with the exact replay overlaid, the level-0 control curve, and one paragraph of theory (misspecified Bayes).

**C2. "Performance" curves above the optimal 2.72 at step 3.** With 85% growl accuracy, the optimal expected cumulative reward of 3 steps from a uniform belief is 2.72 (listen twice and open only if both growls agree: 0.7225 × 8 + 0.0225 × (−102) + 0.255 × (−3); undiscounted, own-door payoffs). The page shows 4.48 (L2 and L3) and 3.49 (L1) at step 3.
- **Facts already in hand (October 14), from the trial files:**
  - Step-3 outcomes for L2 and L3 were **77 correct opens (+8), 1 wrong (−102), 22 not opened (−3)**, against the expected 72.25 / 2.25 / 25.5; L1 was 68 / 1 / 31.
  - Every interval contains 2.72 (L3 [1.73, 6.24]). The SD of a single trial's step-3 reward is about 16.6, so SE ≈ 1.66 at n = 100, and 4.48 is about 1.1 SE above 2.72 (p ≈ 0.3).
  - The conditions share trial seeds, so one lucky draw of growls appears in every curve at once. That is why the ceiling looks broken "so frequently".
  - L1 opens less by step 3 because it models j as random: j's random opens reset the tiger, so L1 is less confident after 2 agreeing growls.
- **Backup benchmarks:**
  1. **The exact ceiling:** compute the optimal expected cumulative reward V*_t from a uniform belief for t = 1–20, by exact finite-horizon value iteration for i alone. Check whether j's opens (which reset the tiger) can raise i's optimum; if they can, use a valid bound instead (for example, the optimum when i also controls j). Overlay the ceiling on the Performance plots.
  2. **Exact expected curve of the agents actually played:** for the cheap conditions (L1, L2 at H2/H3), evaluate the pair of policies exactly by enumerating the joint observation tree with memoization over (belief of i, belief of j, state), to step 4–6. Every empirical mean must sit within its interval of that exact curve.
  3. **Large n with independent seeds:** rerun the cheap conditions with n = 2,000–10,000 trials on fresh seeds (a published L2 H2 episode takes seconds), and show convergence to the exact curve. This answers "is n too low?" with data.
  4. **Show the luck directly:** a seed-level table of correct, wrong and no opens against their exact probabilities, with a multinomial test.
- **Any real breach is a bug:** if any exact-ceiling check fails at large n, stop and debug the reward accounting (resets, the step-0 row, double counting) before anything else.

**C3. Plot request, "What each agent earns, every pairing" (level grid):** a level-0 agent loses 550–600, so the shared symmetric y-scale flattens every positive curve. Use an asymmetric y-axis: keep the negative range as is and give the positive range its own, larger scale (a piecewise-linear axis split at 0, with a visible break marker and tick labels on both sides, so nobody misreads it). The generator is in `results/w4_report_20261007` (`template.html`, `build_report.py`); back up before editing (`*_<date>.bak`); republish as version 22 with C1 and C2 panels.

### 0.2 Next steps (input to the next plan)

1. **Tiger suite write-up (W): done October 13.** Reviewer page version 21 republished (generator in `results/w4_report_20261007`, backups `*_20261013.bak`; data from `results/w_suite_report_20261012/grid_data.py` and `results/r_priors_20261011/analyze.py`). Claude Doc report: level matrix, planner comparison in episodes and prior sensitivity added to the lead and the benchmark section; the stale horizon-3 sentence (−9.6) replaced by 12.32's +0.8.

2. **UAV reconnaissance (PLAN.md, top; started October 12).**
   - **Parameters.** The 2009 paper states none. The user could not be asked (away), so values come from Zeng & Doshi (2012), JAIR 43, Section 7 and Appendix B.3 (arXiv:1401.4600): listening reports the target's row correctly with 0.8; +50 for spotting, −5 per step. The rest are declared. Source table: `DOMAINS.md`, "UAV reconnaissance". **Ask the user whether they have the original problem files** (Zeng & Doshi say they are "available upon request"); if so, replace the declared values before the gates.
   - **Declared design choices to keep in mind:** 81 states, not the paper's 36 (the side-column collapse cannot express spotting); the UAV starts bottom-left, known to both; the target's own sector is in its observation, so its first decision is made without it (the package's prior is common and never conditions on the true state).
   - **U1 done** (`3819675`): `domains/uav.py` replaced (no shim); tests for kernels, spotting and mirror symmetry; RESULTS Section 2 marked as the retired pursuit game.
   - **U2 done** (`9321093`): `experiments/problems.py`. A problem is a chain of physical beliefs, one per level, each a function of the enclosing state (`chain_model`, `tiger_chain`, `uav_chain`). `oracle.evaluate_case(chain=..., problem=..., history_actions=...)`; level 1 off Tiger is scored by `BestResponseReference` (exact enumeration); `sample_history` takes a domain's actions and rejects terminal histories. **Gate:** 5 Tiger problems (L1; L2 vs exact; L3 declared; L2 history 3 with N = 64; RTS sampled on the coordination game) give identical rows before and after (`results/u2_gate_20261012`).
   - **U3 done** (`results/u3_uav_reference_20261012/probe.py <level> <horizon> [budget]`, one problem: i uniform over the target's 8 possible sectors; at level 2 the target uniform over the UAV's sectors). Exact reference plus planner, 1,000 simulations:

     | Level, horizon | Total CPU | Planner loss | optimal_q_error |
     |---|---|---|---|
     | L1 H2 | 0.1 s | 0 | 0 |
     | L1 H3 | 0.4 s | 0 | 0 |
     | L1 H4 | 2.2 s | 0 | 1.16 |
     | L1 H5 | 4.7 s | **6.29** | 2.40 |
     | L2 H2 | 1.3 s | 0 | 0 |
     | L2 H3 | 38 s (planner 19 s) | 0 | 0 |
     | L2 H4 | 87 s (planner 45 s), 0.2 GB | 0 | 0.48 |
     | L3 H2 | 1.4 s | 0 | 0 |

     Exact references are cheap enough for gate grids at L1 H2–H5, L2 H2–H4 and L3 H2 (logs: `probe_2_4.log`, `probe_3_2.log`).
   - **U4 engineering done** (October 13): `panel` takes `problems(level)`, a list of named `experiments.problems.Problem`s (the `evaluate_case` arguments plus extra row columns), with `tiger_problems` (the old point-belief grid, the default) and `uav_problems` (every combination, one per level, of 4 named laws over the other agent's sector from `uav_laws`: uniform; top-row, near-start and far-corner, each 0.75 on the named sectors plus 0.25 uniform; 4 problems at L1, 16 at L2, 64 at L3); levels 1–4; CLI `--domain uav`. **Gate:** 64 Tiger rows (6 cells × 2 seeds, published and coordination games, levels 2–3, exact, particle and RTS cells) identical before and after (`results/u4_panel_gate_20261013`).
   - **U4 runs done (RESULTS 12.41, `results/u4_uav_gates_20261013`, about 31 CPU-h).** Qualified budgets: root 1,000 at H2–H3 (L1, L2, L3); root 4,000 at L1 H4; 64,000 at L1 H5; 16,000 at L2 H4 with exact opponent models. The modeled target needs 400 at L2 H3 (100: 63/80). At L2 H4, 100 and 400 both lose up to 38 on near-start problems, which is opponent-model error.
   - **U4b engineering done (October 14):** `modeled_qualification` takes `problems(level)` and `physics` like `panel` (CLI `--domain uav`); rows carry the problem name plus its labels. Gate: 608 Tiger qualification rows (L2 H2–H3, L3 H2) identical before and after (`results/u4b_qualification_gate_20261014`). Under load ≥ 16, all four timing tests in `test_process_supervisor.py` fail on HEAD and on this change alike (327 others pass).
   - **U4b/U4c done (RESULTS 12.42):** with the target at 400, i is wrong in 16/32 problems at root 16,000. At the beliefs i queries, the target is wrong in 948 of 2,040 queries at 400, 368 at 1,600, and 4 at 6,400 with stopping (one near-tie, gap 0.027). Confirmation: 32/32 decisions optimal at root 16,000 with the target at 6,400 with stopping (seeds 0–1 after the kill-criterion cut; 27 CPU-h). **Corrected October 15 (12.46):** the recorded 50 CPU-min per decision was inflated by the machine's state during the run. A deterministic rerun gave the identical decision in 718 s, and 12.46 measured 366 s on average, so the qualified decision costs about 6 CPU-min. The "exact target is 10× cheaper" claim is withdrawn.
   - **Cost lessons:** an L2 H4 UAV problem costs about 270 CPU-s for the planner with exact models (at any root budget) plus about 230 s for its exact reference. Panel jobs of 16 such problems hit the 2-hour supervisor timeout, so run H4 as one process per (problem, rung, seed), with the reference solved once per problem (`results/u4_uav_gates_20261013/l2h4/job.py`, `score.py`).
   - **U5 step 1 done (RESULTS 12.43, `results/u5_uav_rts_20261014`, about 4 CPU-h):** see 0.1, item 2. Also: `panel.summarize` crashed on RTS-only grids (fixed, `db054bf`, with a test); a UAV `planner_comparison` episode runs end to end (spotting gives +50/−50 and ends the episode; later rows are padded with no actions and reward 0), so step 3 needs no code change. **H4 not run:** an IPF + RTS decision at H4 costs 4.2 CPU-min at K = 2, 17 at K = 4 and over 30 at K = 8, and I-POMCP's qualified H4 decision about 50, so 80 problems would exceed 100 CPU-h. **Ask the user** before running it.
   - **U5 steps 2–3 done (October 14):** episodes (12.44) and the particle sweep at L2 after a 2-step history (12.45; level 1 cannot take a history, see there). Both found IPF failures that are results: the sampled filter runs out of particles because the target observes its own sector exactly. I-POMCP is 77/80 with exact beliefs at H3 after a history (near-ties ≤ 0.46): the history-0 qualification does not fully carry over. **H4 done (12.46): I-POMCP 80/80, IPF + RTS 46–50%, and I-POMCP 1.6–11× cheaper.** Next: the reviewer concerns C1–C3 (0.1a, in progress), then U6, the UAV level matrix (levels 0–2). The episode configuration of 12.44 runs, but a 3-step horizon cannot see the far corner the target flees to, so consider depth 4 (a qualified decision costs about 6 CPU-min, 12.46) and pilot cost first.
   - The original U5 plan, in order:
     1. *(Done, 12.43.)* **Decisions against the exact I-POMDP** (as Tiger 12.20): `panel --domain uav` cells `("rts", 2, H, K, "K", 64, "sampled", 0)` for K = 2, 4 against I-POMCP at the qualified configuration, at H2 and H3, 16 problems × seeds 0–4. Checked October 14: an RTS UAV L2 H2 decision runs through `evaluate_case` (2 s). At H3 run as one process per job if RTS cost grows like `(|A|K)^(H−1)` (UAV has |A| = 5); profile one H3 problem first (the efficient-experiments rule). H4 RTS only if H3 is cheap.
     2. **The paper's reward-vs-particles shape** (Doshi & Gmytrasiewicz 2009, Fig. 16) at L1 H2–H3: IPF particle counts × K, scored against the exact best response.
     3. **Episodes** with `experiments.suite.planner_comparison(domain="uav", …)`. A UAV episode runs end to end (checked October 14 at depth 2 with 100 simulations, 8 steps). Endpoints must handle the padded rows after spotting. Done (12.44).
     4. **U6–U7** after: the UAV level matrix (levels 0–2), then the write-up in RESULTS, the reviewer page and the Claude Doc.
   - **Rollout heuristic:** not needed so far. Every horizon qualified with the uniform rollout, at the cost of large root budgets at H5 (64,000); revisit only if U5/U6 need H ≥ 5.

3. **Other breadth items, after UAV:**
   - Wumpus L1 beyond 3 beliefs (Section 0.10, item 3);
   - level 4 on the coordination Tiger (now cheaper, 12.36);
   - a solver-level change to the exact nested update (about a third of the CPU, 12.36; needs 12.15's replay gate).
4. **L3 H4 residual:** resolved under the ε = 0.05 standard (12.54, October 16).
5. **Housekeeping** (the user's call): legacy `results/` (about 12 GB), stale branches (Section 1), the `../ipomcp-q2` worktree (`git worktree remove ../ipomcp-q2`), and backing up `results/`.

### 0.3e What happened October 14 (PLAN.md, "UAV reconnaissance", U5)

| Work | Result | RESULTS |
|---|---|---|
| Supervisor timing tests on a quiet machine | 6/6 pass (their failures were load-related) | — |
| `panel.summarize` on RTS-only grids (`db054bf`) | Crashed on missing `search_*` columns; fixed, with a test | — |
| U5a: decisions against the exact I-POMDP, H2–H3 (`u5_uav_rts_20261014`: `u5.py`, `analyze.py`, `profile.py`, `diag/diag.py`) | I-POMCP 80/80 at both horizons; IPF + RTS 53–68%; RTS converges at K = 128 | 12.43 |
| UAV episode check | `planner_comparison("uav")` runs; spotting ends the episode | — |
| `uav_problems` history actions (`8c079fc`) | UAV mid-episode problems sampled Tiger's history (KeyError 'L'); now i listens in place; history-0 rows 160/160 identical | — |
| U5 step 3: episodes (`u5_uav_rts_20261014/episodes`) | Same reward; I-POMCP 8–18× cheaper per decision, better prediction; 1 IPF episode aborted | 12.44 |
| U5 step 2: particles (`u5_uav_rts_20261014/particles`: `job.py`, `score.py`, `analyze.py`, `refcheck.py`, `gate.py`) | I-POMCP below IPF + RTS at every N; IPF fails at N = 4 | 12.45 |
| U5a H4 (`u5_uav_rts_20261014/h4`) | I-POMCP 80/80, IPF + RTS 46–50%; I-POMCP 1.6–11× cheaper; 12.42's CPU corrected (about 6 min per decision, not 50) | 12.46 |
| C1: j's level posterior (`c1_level_posterior_20261014`: `odds.py`, `agreement.py`, `contrast.py`, `replay_j.py`, `screen_levels.py`, `screen_analyze.py`) | j's two hypotheses; levels 2–5 (and 6, on 81/81 chains) act identically at H2; a level-3 j identifies levels; exact replay agrees | 12.47 |
| C2: rewards above 2.72 (`c2_tiger_ceiling_20261014`: `ceiling.py`, `check_plots.py`, `large_n.py`, `gate.py`, `analyze_large_n.py`) | Exact ceiling; no plotted interval above it; 15,600 fresh trials 2.59 ± 0.22; plotted trials reproduce 100/100 | 12.48 |
| C3 and page versions 22–23 (`w4_report_20261007`, backups `*_20261015.bak`) | Split y-axis, ceiling line, a reviewer-checks section, a UAV section; Claude Doc updated | — |
| U6: UAV level matrix (`u6_uav_level_matrix_20261015`) | A second level pays for both UAV and target | 12.49 |
| Mid-episode qualification (`q_uav_midepisode_20261015`: `diag_tie.py`) | Root 4,000 + exact target 80/80; one tie-convention mismatch | 12.50 |
| Shared-machine lesson | Another project's process took 18 GB; our watchdog killed 46+ young jobs, and xargs exits when a child dies by signal. Fix: `h4/wrap.sh` (waits for 3 GB, exits 0) and `h4/watchdog.sh` (walks the tree from the xargs PID). Change xargs parallelism live with SIGUSR1/SIGUSR2 | — |

### 0.3d What happened October 13–14 (PLAN.md, "UAV reconnaissance", W and U3–U4)

| Work | Result | RESULTS |
|---|---|---|
| W: reviewer page v21, Claude Doc report | Tiger suite 12.37–12.40 published; stale H3 sentence fixed | — |
| U3 finish | L2 H4 exact reference 87 s, L3 H2 1.4 s | — |
| U4 engineering (`14c7044`) | `panel` takes `problems(level)` (`Problem`, `tiger_problems`, `uav_problems`, `uav_laws`), levels 1–4, `--domain uav`; Tiger rows 64/64 identical | — |
| U4 gates (`u4_uav_gates_20261013`) | Qualified budgets per level and horizon | 12.41 |
| U4b engineering (`ecb8ed0`) | `modeled_qualification` takes problems and physics; Tiger rows 608/608 identical | — |
| U4b qualification, U4c confirmation | Target at 6,400 with stopping at L2 H4; 32/32 decisions | 12.42 |

Run folders: `results/u4_panel_gate_20261013`, `u4_uav_gates_20261013` (`u4.py` chunks `l1_h*`, `l2_h2`, `l2_h3`, `l3_h2`, `l1_h5_64k`; `l2h4/` one-process jobs with `job.py`, `score.py`; `analyze.py`; the killed `l2_h4` panel chunk's partial folder is still there, unused), `u4b_qualification_gate_20261014`, `u4b_uav_qualification_20261014` (`u4b.py`, `q/`), `u4c_uav_confirm_20261014` (`job.py`, `score.py`, references copied from U4). About 75 CPU-h in all, about 10 of them lost in the stopped L2 H4 panel chunk.

Files: `src/ipomdp/experiments/problems.py`, `experiments/panel.py`, `experiments/modeled_qualification.py`; tests `test_panel.py`, `test_l4_oracle.py`, `test_modeled_qualification.py`; docs `RESULTS.md` 12.41–12.42, `ARCHITECTURE.md`, `PLAN.md`, this file.

### 0.3c What happened October 11–13 (PLAN.md, "Tiger experiment suite" and the start of "UAV reconnaissance")

| Work | Result | RESULTS |
|---|---|---|
| E: packaged suites (`df8803d`) | `experiments/suite.py` (`SuiteSpec`, `run_suite`: round-robin over cells on shared trial ids, one supervised process per episode, resumable; only `trials` may grow) and builders `level_matrix`, `planner_comparison`, `prior_sensitivity`; `analysis/endpoints.py`; `holm` moved to `analysis.statistics`. Gate E: 18/18 of 12.27's episodes reproduced exactly, so its 900 episodes were imported | 12.37 |
| M1: level matrix L0–3, both games, n = 100 | Above | 12.37 |
| M2: level-4 row (4, 0..4), coordination, n = 40 (user-approved, 69 CPU-h) | 3 → 4 gains nothing | 12.40 |
| P: I-POMCP vs IPF + RTS in episodes | Above | 12.38 |
| R: prior sensitivity, levels 3 and 4 | Above | 12.39 |
| UAV U1–U3 | 0.2, item 2 | — |

Run folders (each with its launcher, `analyze.py`, `KILL_CRITERION.txt`, snapshot `src/`): `results/e_suite_gate_20261011`, `m1_level_matrix_20261011` (`coordination/`, `published/`), `m2_level4_20261011` (`pilot/`, `row/`), `p_planners_20261011` (`h3-coordination/`, `h3-published/`, `h4-coordination/`), `r_priors_20261011` (`l3/`, `l4/`), `w_suite_report_20261012`, `u2_gate_20261012`, `u3_uav_reference_20261012`. Watchdog and memory log that only look at one runner's children: `m1_level_matrix_20261011/watchdog.sh <runner pid> <log>` and `memlog.sh`. Costs since 12.36: an L3 × L3 H2 episode about 1.9 CPU-min; level 4 against levels 0–4: 1–45 CPU-min, peak 1.3 GB.

Files: `src/ipomdp/experiments/suite.py`, `experiments/problems.py`, `analysis/endpoints.py`, `analysis/statistics.py`, `experiments/oracle.py`, `domains/uav.py`; tests `test_suite.py`, `test_domains.py`, `test_best_response.py`, and the oracle tests migrated to `problems`; docs `PLAN.md`, `RESULTS.md` 12.37–12.40 (and a note on Section 2), `DOMAINS.md`, `ARCHITECTURE.md`, this file.

### 0.3b What happened October 11, evening (PLAN.md, "cheaper nested solves" phase)

| Work | Result | RESULTS |
|---|---|---|
| S0 profile | L3 episode, 4 steps: canonical keys re-encoded the physics for every atom (about 32%), the rollout-memory update (14%), recursive belief comparisons (7%) | 12.36 |
| S1–S2 (`661526c`) | Per-frame canonical fragments (byte-identical keys); memoized rollout memory | 12.36 |
| S3 (`7183339`) | Compressed-filter cache 4,096 → 65,536 (it was recomputing 22% of updates, as duplicate objects); one-pass order and digest | 12.36 |
| Net (bit-identical) | 10-step L3 episodes 0.50–0.52× CPU, peak memory 1.9 → 0.65 GB; L2 0.57× | 12.36 |

Tools: `results/s0_nested_profile_20261011/profile_episode.py` (cProfile of any condition); `results/s1_gates_20261011/` and `results/s3_gates_20261011/` (paired old/new bit-identity gates: `jobs.sh`, `episode_job.py <version> <condition> <trial> <steps>`, `compare.py`); `results/s3_cache_sizes_20261011/cache_probe.py`; `results/s4_net_effect_20261011/`. Files: `belief/canonical.py`, `belief/compressed_filter.py`, `domains/tiger.py`, `tests/test_canonical.py` (new; 315 tests).

### 0.3a What happened October 11 (PLAN.md, "root adaptive budgets" phase)

| Work | Result | RESULTS |
|---|---|---|
| R0–R1 root replay | 542 gate problems, one root tree each, scored at 125–4,000 simulations against the exact I-POMDP. The first 125 simulations carry the whole fixed cost (solving every reachable modeled policy); later ones cost 0.2–0.35 ms. **No stopping rule qualifies** (at most 3% to save). **Root 4,000: 380/380 and 162/162, L2 H4 optimal-action Q error 0.121 → 0.031 at 1.06× CPU** | 12.33 |
| R2' episodes | 12.32's configuration: one decision at 4,000 costs 1.015× at L3; with a matched model of j (every root 4,000), L3 episodes cost 1.32–1.42× (n = 3). Episodes keep 1,000 | 12.34 |
| R3 error split at 4,000 | L2 H4 scored error 0.026; deeper choices plus rollouts about 0.009. No deeper-node redesign | 12.35 |

Tools: `results/r0_root_replay_20261011/` (`r0_job.py` per-problem checkpoint dataset, `replay.py` rule replay with the pre-fixed adoption test, `count_solves.py`); `results/r2_episode_root_cost_20261011/` (`probe.py` measures a root budget at real episode decisions without changing them, checked against 12.32's values; `matched_cost.py`); `results/r3_attribution_4000_20261011/`. The Claude Doc report has an "Update, October 11" paragraph in the root-vs-modeled-budget section. No `src/` change this phase.

### 0.3 What happened October 9–11 (PLAN.md, "bounded-belief" phase)

| Work | Result | RESULTS |
|---|---|---|
| Miss diagnosis | Exact replays of all 19 zero-probability predictions. 17 are compression losses: a recent, light branch dropped at the last one or two compressions (14 below 1/N), while systematic compression drops about 25% of the mass per update, mostly harmlessly. **2 (H3) were a nested-budget mismatch** (i modeled j's model of i with 100 simulations; the real j used 400). Those 2 trials carry −469 of 12.29's −481 summed deficit | 12.30 |
| Compressor gate | Optimal resampling, coverage-ε (expected dropped mass ≤ ε) and larger systematic N: every arm costs 1.7–3.8× the control (coverage 0.03 more than 10×), or fails belief accuracy (coverage 0.1). **Killed; systematic N = 64 kept.** The miss metric is now on `main` | 12.31 |
| H3 with a matched model of j | Stack `1000/400` (i models j exactly as j is configured). **L3 − L2 = +0.8 [−6.7, 8.6], p = 0.84, n = 50.** 12.29's deficit was the mismatch. Prediction loss 0.41 against 0.45 (not significant) | 12.32 |

12.29's own text carries a correction note pointing to 12.30. The reviewer page (version 20) and the Claude Doc report ("Bounded-belief caveat" paragraph) carry 12.30–12.32.

### 0.4 Reusable tools and data from October 9–11

- `results/q1_miss_diagnosis_20261009/`:
  - `replay.py <run> <trial csvs>` replays an episode's real belief updates exactly (support sizes and predictions checked against the log) and traces where a branch was lost, with compression profiles per update. Runs: `h3`, `cp32`, `cp64`; add a spec for others.
  - `budget_check.py <trial csv> <step> <stack>` replays with another modeled stack and prints i's prediction.
- `results/q3_compressor_gate_20261009/`:
  - `gate.py` runs paired arms of i's compressor on the 12.26 probe condition; `analyze.py` gives miss share, McNemar test, paired CPU ratio, and a reproduction check against 12.27.
  - `g1/`: `belief_qualification` outputs for every arm.
- `results/q4_h3_matched_20261009/`:
  - the study scripts: `h3_matched.py`, `h3_job.py` (the per-episode runner template) and `analyze.py`;
  - `KILL_CRITERION.txt`, `memory.log`;
  - `watchdog.sh` and `memlog.sh`.
- `analysis.metrics.predictions(frame)` and `prediction_misses(frame)`: realized-action probability, log-loss and miss flag per decision, plus a per-trial summary.

### 0.5 Files touched October 9–11

- `main`:
  - `src/ipomdp/analysis/metrics.py` (`predictions`, `prediction_misses`, `MISS_PROBABILITY`);
  - `tests/test_analysis.py`;
  - docs: `PLAN.md` (the bounded-belief phase), `RESULTS.md` 12.30–12.32 and the 12.29 correction note, this file.
- Branch `bounded-belief/compressors` only:
  - `belief/resampling.py` (`SystematicCompressor`, `OptimalCompressor`, `CoverageCompressor`, `expected_dropped_mass`, `COMPRESSORS`);
  - `belief/compressed_filter.py` (a `compressor` argument);
  - `planning/solver_bank.py` (`FILTERS` per compressor);
  - `experiments/{belief_qualification,oracle,panel}.py` (filter choices from `FILTERS`);
  - tests: `test_compressors.py`, `test_bounded_beliefs.py`.
- Outside the repository:
  - `results/w4_report_20261007/build_report.py` and `template.html`: the H3 paragraph now reads `q4_h3_matched_20261009`. The previous versions are saved as `*_20261011.bak`.

### 0.6 What happened October 7–9 (PLAN.md, "October 7 evening" phase)

| Work | Result | RESULTS |
|---|---|---|
| Joint payoffs | `JointTigerModel` (`domains/tiger.py`, study domain `tiger-joint`): published physics, payoffs `together_safe` and `solo_penalty`. `oracle` and `panel` take a `physics` and levels up to 4 | — |
| Exact screen | Value of one more level, exactly, over point-belief chains. Published Tiger: 2→3 worth 0 at H2, 0.009 at H3. **`together_safe = 40`** (the coordination Tiger): 2.10 at H3, 3→4 0.81 at H2 | 12.22 |
| Gates | Coordination Tiger vs the exact I-POMDP: 1,244/1,244 optimal, including the **first level-4 gate** (486/486 at H2) | 12.23 |
| Error attribution | Exact recursive split of search error: at H3, 98% chance error (removed by scoring); at L2 H4, 22% deeper choices, 9% rollouts | 12.24 |
| Level grid, N = 32 | Superseded (compression misses) | 12.25 |
| Compression misses | N = 32 drops the realized branch in 3–20% of trials above level 1; caveat on 12.21 and 12.25 | 12.26 |
| Level grid, N = 64 | Level 1→2 pays (27–30, Holm p < 0.001); 2→3 not significant at H2; level 3 predicts best | 12.27 |
| Optimal resampling A/B | 6–160× lower belief error per atom, 1.3–5× CPU at equal N; no win at matched CPU; on branch `resampling/optimal-fc` | 12.28 |
| H3 depth study | n = 50 (cut by kill criterion): level 3 −9.6 [−26.8, 4.2] vs level 2. **Corrected:** the deficit was a nested-budget mismatch in i's model of j (12.30); with a matched model, +0.8 (12.32) | 12.29 |

**Reviewer page** (version 20, October 11: H3 paragraph from `q4_h3_matched_20261009`): https://claude.ai/artifact/PqpRuTLcuSq5afJdSSxoCB, with a "coordination Tiger" section (screen table, N = 64 grid heatmaps, one-level paired forest plot with the published control, H3 paragraph). Rebuild: `cd results/w4_report_20261007 && ../../.venv/bin/python analyze.py && ../../.venv/bin/python build_report.py`, then republish `report.html` with the Artifact tool, passing that `url` (read it first from a new session). It reads `p2_payoff_screen_20261007`, `p4c_joint_level_grid_n64_20261007` (grid_data.json, endpoint.csv, endpoint_control.csv) and `q4_h3_matched_20261009/h3_summary.json` (since October 11). All page text is generated; edit `template.html` and `build_report.py` only. `*_20261007.bak` files there are the pre-phase versions.

**Claude Doc report** (Section 4): the "Coordination Tiger" and "Bounded-belief caveat" paragraphs after "Level grid"; the error-attribution row of "Not yet measured" is now measured.

### 0.7 Reusable tools and data from October 7–9 (each folder holds its scripts)

- `results/p2_payoff_screen_20261007/screen.py`, `analyze.py`: the exact value of one more level for any payoff variant (add a variant to `VARIANTS`).
- `results/p3_joint_gates_20261007/gates.py`: one job per (gate, seed, belief of i) panel launcher with JSONL rows; reuse for any physics.
- `results/p5_error_attribution_20261007/attribution.py`: exact recursive split (`split`) of root search error into chance error per depth, local action loss, frontier and exact tail; zero residual.
- `results/p4c_joint_level_grid_n64_20261007/`: `grid.py` (any level grid; args: out, N, trials, workers, levels, required level), `endpoint.py <grid> <out.csv> [levels]` (the pre-registered one-level paired endpoint, Holm), `analyze_grid.py` (per-cell table + joint safe openings).
- `results/p4b_correct_prior_20261007/`: `correct_prior.py`, `probe_n.py` (N from `PARTICLES`), `compare_n.py`, `analyze.py [trials dir] [grid dir]`: the zero-probability (compression miss) diagnosis.
- `results/p8_resampling_gate_20261008/`: the compressor A/B (G1 `belief_qualification` outputs, `g2_probe.py`, `g2_compare.py`, `g3.py`; arm chosen by `PYTHONPATH`, old arm = `p4b_correct_prior_20261007/src`).
- `results/p7_h3_depth_20261008/`: `h3_depth.py` (pre-registered H3 study), `analyze.py`, `KILL_CRITERION.txt`, `trials_quarantine/` (5 episodes of the discarded first launch; unknown code).
- Zero-probability check: now `analysis.metrics.prediction_misses` (October 9).

### 0.8 Files touched October 7–9

- `main`: `domains/tiger.py` (`JointTigerModel`), `experiments/oracle.py` (`physics`, levels 1–4, `nested_beliefs`, `_nested_chain`), `experiments/panel.py` (`physics`, levels 2–4, `--together-safe`), `experiments/modeled_qualification.py` (call site), `experiments/study.py` (`tiger-joint`); tests `test_l4_oracle.py` (new), `test_domains.py`, `test_domain_contracts.py`, `test_l3_oracle.py`, `test_panel.py`; docs `DOMAINS.md`, `ARCHITECTURE.md`, `RESULTS.md` 12.22–12.29, `PLAN.md`, this file.
- Branch `resampling/optimal-fc` only: `belief/resampling.py` (`optimal_resampling`), `belief/compressed_filter.py`, `tests/test_optimal_resampling.py`.

### 0.9 What happened October 5–7 (PLAN.md W1–W8)

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

### 0.10 Goals as of October 7 (superseded by 0.2; items 1, 2 and 5 done)

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

### 0.11 Files touched October 5–7

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
- **Commit and push to `origin main` after every implementation, without asking** (standing rule, October 11). Never force-push.
- Back up before deleting evidence. Kill processes by PID only.
- **Housekeeping.** The worktree and the legacy caches were removed on October 2. Two optional items remain; the permission classifier blocks Claude from deleting them, so the user decides:
  - **Legacy raw outputs, about 12 GB in `results/`:** `oracle/`, `archive/`, `deep_prior/`, `payoff_matrix/`, `review-20260923/`, `tiger/`, and the `*-202609*` folders (`bellman-backup`, `exact-tail`, `exploration`, `history-rewards`, `l2-contract`, `l2-finite`). They are evidence for the retired code only. Material up to September 30 is backed up on `C:\`. The recursive-chance panels of September 30–October 1 and Claude's audit (`results/oracle/l2_recursive_chance_development_claude_audit_20261002/`) are not backed up yet. Every other folder in `results/` belongs to the current package.
  - **Stale branches:**
    - Already in `main`'s history, so safe to delete with `git branch -d`: `claude/restructure`, `claude/l3-l2-tiger-experiment`, `claude/snapshot-and-hardening`, `codex/final-20261002` (also tagged), `refactor/bigtech-hardening` and `review/rigor-20260916`.
    - **Not** in `main` and not tagged (checked October 2): `archive/review-snapshot-notes-20260923`, `archive/tiger-baseline-20260923` and `backup/antigravity-before-supervision-20260919`. Deleting them loses their commits, so keep them or tag them first.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Report results with chosen methods only, label interim n, and define jargon (the user's report style).

## 2. Current state

- **Tests.** 311 pass in about 80 s on `main`; `ruff check` and `ruff format --check` are clean.
- **Running jobs.** None.
- **Archives.**
  - Tag `archive/pre-restructure-20260930`: the pre-restructure code.
  - Tag `archive/codex-final-20261002` (branch `codex/final-20261002`): Codex's and Antigravity's final working tree, preserved unchanged. It holds the legacy recursive-chance candidate (`src/solvers/recursive_chance.py`), its protocols (`docs/RECURSIVE_CHANCE_*.md`), the frozen contracts (`benchmarks/`) and the launchers. `main`'s merge has it as a parent, but `main`'s tree is the restructure.
- **Results.** Stored under `results/` (git-ignored). A backup of everything up to September 30 is in `C:\Users\Andyj\ipomcp-results-backup-20260930`. Nothing from October 1–11 is backed up yet (every current dataset and the reviewer-page generator); copy `results/` off the machine.

### What the package is now

```
src/ipomdp/
  model/        kernel-only domain contract (sampling derived from the kernels)
  domains/      tiger (published spec; JointTigerModel = the coordination Tiger), uav,
                wumpus (exact 94,500-state prior)
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
                oracle (one problem vs the exact reference; levels 1-4, any Tiger physics),
                panel (grids of cells vs the exact I-POMDP; levels 2-4),
                modeled_qualification (a modeled level scored at the beliefs it is queried at),
                l1_qualification, belief_qualification
  analysis/     hierarchy (sunburst nodes with content and error), metrics (incl. prediction misses),
                figures, statistics
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

### Validated configurations (Tiger; root 1,000 simulations unless noted; root 4,000 for Q-error reporting at gates, 12.33)

| Problem | Modeled opponent | Beliefs | Evidence |
|---|---|---|---|
| L1, H1–H5 | uniform L0 | exact | 220/220 at H5 (12.1) |
| L2, H3, k ≤ 4 | exact, or I-POMCP 400 (zero-wrong certified, 12.5) | exact or compressed 64 | 373–380/380 (12.3) |
| L2, H4 | I-POMCP 1,600 flat, or `6400s` (stopping, cap 6,400; 0.85× flat 1,600's CPU) | exact | 380/380 (12.4, 12.11) |
| L3, H3, k = 0 | I-POMCP 400 flat (i's decisions = exact); certified zero-wrong stack `6400s/100` | exact | 162/162 (12.7, 12.11) |
| L3, H4, k = 0 | **none zero-wrong.** `6400s/exact`: residual set 4/80 wrong after 12.14, all at 0.03-gap beliefs (loss ≤ 0.03) | exact | 12.10, 12.11, 12.14 |
| L3 vs L2 episodes, H3 | `6400s/100` for i, `400` for j, mixed level priors (gated 120/120) | compressed 64 (level TV ≤ 0.061 to step 5) | 12.16, 12.18 |
| Levels 1–4 × 1–4, H2 | decisions exact for the searched model at any budget (two steps left) | compressed 32 (drops realized branches above level 1: use 64) | 12.21, 12.26 |
| Coordination Tiger (`together_safe = 40`): L2 H3, L3 H3 k = 0, L2–L4 H2 | as for the published Tiger (`400`, `6400s/100`; `100` at H2): 1,244/1,244 | exact; compressed 32 at H2 | 12.23 |
| Coordination Tiger episodes, levels 1–3, H2 and H3 | as above, root 100 (H2) or 1,000 (H3); for a "correct model" of the opponent use the opponent's own configuration (H3: `1000/400`) | compressed 64 (systematic); misses in about 6% of level-3 H3 trials, cheap in reward | 12.27, 12.30–12.32 |
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
| 12.22 | **Exact screen:** a third level is worth 0.009 per decision in the published Tiger at H3, 2.10 with `together_safe = 40` (the coordination Tiger) |
| 12.23 | Coordination Tiger gates 1,244/1,244, including the first level-4 gate (486/486, H2) |
| 12.24 | **Search error at H3 is 98% chance error** (scoring removes it); at L2 H4, 22% deeper choices and 9% rollouts remain |
| 12.25–12.26 | **N = 32 compression drops the opponent's realized branch** (3–20% of trials above level 1); 12.25 superseded |
| 12.27 | Coordination grid at N = 64 (n = 100): **level 1→2 gains 27–30** (Holm p < 0.001); 2→3 not significant; level 3 predicts best |
| 12.28 | Optimal resampling: far more accurate per atom, no win at matched CPU; not adopted (branch `resampling/optimal-fc`) |
| 12.29 | H3 depth study (n = 50): level 3 −9.6 [−26.8, 4.2] vs level 2. **Corrected by 12.30:** its model of j used a smaller nested budget than the real j |
| 12.30 | **Why misses happen:** 17 of 19 are compression losses of a recent, light branch; 2 were the budget mismatch, carrying almost all of 12.29's deficit |
| 12.31 | **Compressor gate killed:** optimal resampling and coverage-ε cost 1.7–3.8× (or > 10×) or fail belief accuracy; systematic N = 64 kept |
| 12.32 | **H3 with a matched model of j (n = 50):** L3 − L2 +0.8 [−6.7, 8.6], p = 0.84; a third level neither gains nor loses |
| 12.33 | **Root CPU is a fixed cost** (nested solves within 125 simulations); no root stopping rule qualifies; root 4,000 cuts L2 H4 Q error 3.9× at 1.06× CPU |
| 12.34 | In episodes: one decision at 4,000 costs 1.015× (L3); with a matched model of j, 1.32–1.42× per episode; episodes keep 1,000 |
| 12.35 | Error split at root 4,000: deeper choices and rollouts about 0.009 of 0.026; no deeper-node redesign |
| 12.36 | **Nested solves about 2× faster, about ⅓ the memory, bit-identical:** per-frame canonical encodings, memoized rollout memory, filter cache 65,536 |

**Implication the user has been told about.** Contribution 1's bounded belief is now the compressed filter, not the published IPF; the IPF remains the baseline's belief. If the user wants the published IPF inside the proposed method too, that changes the story; ask before assuming.

## 4. Where Tiger stands, and next steps

### Assessment and next steps

Superseded by Sections 0.1–0.2 (October 11). Section 2's validated configurations and Section 3 hold the evidence.

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
- **Updated October 7–11:** "Coordination Tiger" and "Bounded-belief caveat" paragraphs after "Level grid" (the caveat now carries 12.30–12.32); the error-attribution row of "Not yet measured" is measured.
- **Updated October 11, evening:** an "Update, October 11" paragraph after the cost list (12.36: nested solves about 2× faster, bit-identical).
- **Updated October 11:** an "Update, October 11" paragraph closing the root-vs-modeled-budget section (12.33–12.35: root simulations are nearly free; root 4,000 for gates; episodes keep 1,000).
- **Updated October 16:** an "Update, October 16: certification standard" paragraph after the "Update, October 6" (residual) paragraph (12.54).
- **Updated October 16, afternoon:** a "Wumpus level 2" paragraph after "Wumpus, October 15" (12.55).
- **Updated October 17:** "Wumpus on a 3 × 3 board" (12.58) and "A compression bug, found and fixed" (12.57) paragraphs after "Wumpus level 2".
- **Updated October 17, morning:** "Wumpus decisions against the published baseline" (12.59) and "The value of a second level in an adversarial game" (12.60) after the compression paragraph.
- **Updated October 15, evening:** an "Update, October 15 (evening)" paragraph after the cost update (12.52), and "Tie-breaking" (12.51) and "Wumpus" (12.53) paragraphs after the UAV section's data line.
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
| An episode run without tracking cumulative CPU against its kill line (G3, October 18–19) | It used 173 CPU-h against a 120 CPU-h kill line and a 100 CPU-h ask line, and nobody noticed until the end | Sum the recorded CPU of finished trials and references at every progress check |
| Pricing exact references from the prior alone (G3) | References cost about ×3 per history step: 48 CPU-h, against about 25 priced | Price per history length from measured points; cap scoring by history |
| SIGUSR2 to throttle a running xargs (G3) | Parallelism did not drop | Kill the xargs (its running jobs finish), then relaunch the rest |
| Killing a job's `timeout` wrapper, or a watchdog that kills xargs's youngest child (N4(b), October 20) | GNU xargs stops the whole run when a child dies of a signal ("terminated by signal 15"); the old `watchdog.sh` kills exactly that child | Kill the Python grandchild (`timeout` then exits with 143 and xargs carries on): `watchdog2.sh` in the run folders. Relaunch `run.sh`; jobs skip finished outputs |
| Replaying many trials through the exact agent in one process without clearing caches (N4(a) pilot, N4(b)) | Level-2 replays grow to 3–4 GB within about 6–25 trials | `ExactReplay.clear_caches()` when RSS passes 1.5 GB (memory only) |
| Repository default estimator (sampled rewards, no exact tail) | Passed every test while 61–88% optimal | The estimator in Section 2 |
| Sampled IPF as the planner's belief | Quantizes small nested beliefs to 1/N steps and flips greedy modeled opponents (Q 3.3 → 6.3 at N = 64). Error accumulates with history. Slower than exact beliefs where exact is tractable | Compressed filter |
| Multinomial resampling | Noisier than systematic at every N (paired over 12 histories) | Systematic resampling |
| Judging IPF accuracy from 3 histories | They were easy ones; the error was understated 2–5× | At least 12 histories, reporting max and share of steps above 0.2 |
| Carrying one root particle down the tree with bounded beliefs | The particle can sit on an atom compression dropped; deterministic sensors then give zero evidence (Wumpus) | Draw each step's state from b_h |
| Compressed or sampled *real* beliefs on Wumpus | The 94,500-state prior compressed to N atoms loses the true world, and the next deterministic observation has zero evidence | Exact real updates (cheap: at most 2.9 s per step) + `search_particles` |
| Compressing *nested* beliefs in real updates (Wumpus L2, October 16) | 0/3 jobs: the real update enumerates every outer world, some dropped from the compressed nested belief, whose deterministic percept then has zero evidence (12.56) | Exact nested beliefs; a 3 × 3 board with exact references |
| Interning, cached hashes and kernel fast paths for Wumpus (October 16) | Bit-identical, but 0.91× CPU and 1.27× RSS, paired: the cost is enumeration volume (12.56) | Not adopted; patch in `results/c9_speedup_gate_20261016/new.diff` |
| Reading downloaded PDFs with WebFetch (October 18) | It returns raw PDF streams and cannot summarise them | Install `pypdf` and `typing_extensions` into the session scratchpad with `uv pip install --target <scratch>/pylib --python python3`, then extract the text in a script |
| Cloning and running an external planner (I-NTMCP) in auto mode (October 18) | Refused by the permission classifier ("Code from External") | Ask the user first; HANDOFF 0.0a, "G3, step 2" |
| I-NTMCP with its published defaults on a deterministic-sensor domain (October 18) | Its rejection-sampling refill stops after 3,000 rejections and force-fills the rest: up to 99% of a belief contradicted the agent's own observations (12.65) | Report it, plus the tuned variant (kernel reinvigoration, a rejection budget of 50 per particle) |
| An adapter gate at L1 H2 for an MCTS baseline (October 18) | It failed for the method's own reason (mean backups under c ≈ 2,000), so it could not reveal an adapter bug | A gate at H1, where values are particle means of the immediate reward |
| Pricing the I-NTMCP grid from one problem (October 18) | 4-step histories cost 30× the others; the full grid projected to about 100 CPU-h | Price from the hardest stratum; amendment 3 dropped M = 256,000 before it ran |
| Capturing a runner PID with `ps | grep` (October 18) | Matched three processes | Record `$!` or select by exact comm and ppid |
| Systematic resampling in canonical order for compression (until October 16) | Aliased on structured uniform beliefs: on the 3 × 3 Wumpus prior it dropped every world with a pit next to the start, at every N from 16 to 1,024 (12.57) | Stratified draws |
| The sampled IPF (published IPF + RTS) on Wumpus | Fails in 55–57 of 60 L2 problems: deterministic percepts leave no particle that explains an observation (12.59) | Exact or compressed-exact filtering |
| Random-script human histories at Wumpus L3 | The L2 wumpus models a near-deterministic L1 human, so scripted actions give it zero-evidence observations (12.58) | `--script modeled` |
| Point-mass level priors at Wumpus L3 | The exact I-POMDP is undefined on counterfactual branches: zero-probability actions plus deterministic percepts (12.58) | A level-0 weight (`--l0 0.1` or `uniform`) |
| Wumpus 3 × 3 at horizon 3 at scale | 334–2,176 CPU-s per decision; the grid was stopped by its kill criterion at 13/13 (12.58) | Horizon 2, exact (every root scored exactly) |
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
| `cp -r src/ipomdp <run>/src/` when `<run>/src` does not exist yet (October 7–8) | cp copies the package's contents *as* `<run>/src`, so `PYTHONPATH=<run>/src` finds no `ipomdp` and silently imports the live editable install. Two runs used live code; one (the H3 study) was hit by a mid-run source change and was stopped | `mkdir -p <run>/src && cp -r src/ipomdp <run>/src/`, then check `python -c 'import ipomdp; print(ipomdp.__file__)'` under that PYTHONPATH before launching |
| Bounded beliefs at N = 32 for level ≥ 3 episodes (October 7) | Dropped the opponent's realized history in up to 20% of trials; the agent then predicted the real action with probability 0, which reversed level-3 conclusions (12.25 → 12.27) | N = 64, and check the zero-probability rate of every bounded-belief run |
| Qualifying bounded beliefs by level-posterior TV alone (12.16) | Blind to a dropped light branch (mass about 0.05 moves TV little but removes what happened) | Also measure the probability given to realized opponent actions |
| Comparing compressors at equal N (12.28) | The systematic compressor keeps far fewer than N distinct atoms, so equal N is not equal cost | Compare at matched CPU or matched distinct atoms |
| A "correct model" of the opponent with a different nested budget (12.29: i modeled j's model of i with 100 simulations; the real j used 400) | i gave j's real action probability 0 at step 3 in 2 of 50 trials, costing −211 and −258; the whole H3 deficit | Match the opponent's whole configuration (budgets too), and check on logged steps that the model reproduces its actions (12.30) |
| Optimal resampling or coverage-ε compressors for episodes (12.31) | 1.7–3.8× (coverage 0.03: > 10×) the CPU of systematic N = 64; light kept atoms are distinct j-models, each solved; coverage 0.1 fails G1 (level TV up to 0.99) | Systematic N = 64; report `prediction_misses` |
| 16 workers of level-4 episodes on the shared machine (October 7) | Workers grew to 1–1.5 GB each; available memory fell to 4 GB | 8 workers for level ≥ 3 episodes, plus a memory watchdog |
| A stopping rule at the root (12.33): the modeled test of 12.11/12.15, or certified separation | Saves at most 3%: a decision's CPU is paid in its first 125 simulations (nested solves); stopping early only raises Q error. Certified separation certifies the decision, not the value (stops at 125–250 with Q error 0.28 at L3 H3) | A larger flat root budget (4,000 for gates) |
| Costing a flat arm from one checkpoint's incremental growth CPU (R1 pilot, October 11) | Flat 1,000 looked 50× cheaper than the rules | Sum growth over all checkpoints up to n, plus one scoring |
| A filter cache smaller than the caches keyed on its results (4,096 vs 16,384–65,536; until October 11) | Evicted posteriors were recomputed as new, equal objects: 22% of an L3 episode's updates were redone, downstream lookups compared whole beliefs recursively, and the duplicates kept memory at about 2 GB | Filter cache 65,536 (12.36) |
| `xargs -P` with the shared watchdog when another project starves memory (October 14) | The watchdog killed 46 young jobs; xargs stops reading its job list when a child dies by a signal, orphaning the rest | Run-folder `wrap.sh` (waits for 3 GB free, exits 0) and `watchdog.sh` (walks the tree from the xargs PID); a final pass reruns the missing files |
| Trusting CPU recorded during a loaded, memory-starved run (U4c, October 14) | 12.42 recorded about 50 CPU-min per qualified L2 H4 decision; a deterministic rerun gave the identical decision in 718 s (7.5× less), and 12.46 measured 366 s | Rerun one deterministic job on a quiet machine before quoting a cost; compare CPU only within one run |
| A memory watchdog over a `ProcessPoolExecutor` run (October 9, matched H3 study, 10 workers) | One kill raised `BrokenProcessPool` and the whole run exited; nothing finished overnight | One process per episode (`xargs -P`, a job script that skips done files), so a kill loses one episode |
| Reading an interim effect as a finding, again (Wumpus L2 → L3, October 17) | +312 and +374 at n = 32 (Holm 0.02–0.03) became +80 and +120, not significant, at n = 100 (12.61) | Decide on the final n only; label interim numbers |
| Pricing a run from a 2-episode pilot (Wumpus L3 episodes) | 13–14 CPU-min per episode predicted; 21–24 measured, with a tail of about 4 CPU-h; about 110 CPU-h against about 70 | Pilot at least 10 episodes, including the hardest cells; price from the mean plus the tail |
| Regex-parsing analysis-script output without checks (`wumpus_section.py`) | The confirmation pattern matched nothing, silently | Assert the number of parsed rows |

## 7. Merge record

On October 2, 2026, after the user confirmed that Codex and Antigravity had finished:
1. Codex's working tree was committed unchanged (`69097e6`, branch `codex/final-20261002`, tag `archive/codex-final-20261002`).
2. `claude/restructure` merged it with `-s ours` (`f40c525`), keeping the restructured tree and recording the ancestry.
3. `main` was fast-forwarded to `f40c525`. Nothing was pushed.

## 8. How to run

```bash
cd /home/andyj1810/projects/ipomcp
PY=.venv/bin/python
$PY -m pytest -q                                   # about 3 min quiet; far slower under load (Section 0.0)
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
# UAV: add --domain uav to panel or modeled_qualification (problems: experiments.problems.uav_problems)
$PY -m ipomdp.experiments.panel --domain uav --levels 1 --horizons 4 --budgets 4000 --modeled exact --output results/<new>
$PY -m ipomdp.experiments.modeled_qualification --domain uav --horizons 4 --budget 16000 --modeled 400 \
    --ladder 400 1600 6400s --seeds 0 1 --output results/<new>
# Studies
$PY -m ipomdp.experiments.runner studies/tiger_l3_vs_l2.toml --output results/<new>
```

- For custom grids, call `panel.run_panel(output, cells, seeds, workers, problems=partial(tiger_problems, beliefs=..., opponent_beliefs=..., nested_beliefs=...))` (or `partial(uav_problems, UAVModel())` with `physics=UAVModel()`) from a guarded launcher script, and copy the launcher into the output directory. Cells are `(planner, level, horizon, budget, modeled, particles, filter, history)`, with `modeled` a stack label such as `exact`, `1600` or `6400s/400`. The wide L2 grid (19 × 5 beliefs) and the L3 grid (9 × 3 × 3) are in the launchers in `results/l2_wide_n_20261002` and `results/l3_panel_20261003`.
- Wait on background jobs by PID or output file, never with `pgrep -f` (Section 6).
- Size runs from measured CPU. One machine has 24 cores and about 19 GB free; 12 workers are safe for Tiger L2 panels.
