# Paper outline (G2, October 18, 2026)

**Why this file:** the publication-gaps phase (PLAN.md, G2). Every claim maps to evidence (RESULTS sections) and to one figure or table. Every later run must fill a slot marked `MISSING`. Related work is in `docs/RELATED_WORK.md`.

**Working title:** *Exact where checkable: online planning for interactive POMDPs with verified nested beliefs.*

**Target:** AAMAS main track. With G3 done (scale beyond exact reach plus a modern baseline), consider AAAI or IJCAI.

**Terms used below.**
- *Exact I-POMDP reference*: the finite-horizon I-POMDP solved by full enumeration.
- *Optimal rate*: the fraction of decisions whose action is optimal under that reference.
- *ε-correct*: the chosen action's exact value is within ε = 0.05 of the best (12.54).
- *Qualified budget*: a budget at which every decision on a fixed grid of beliefs is optimal (or ε-correct).
- *CPU-matched*: the arms compared get the same CPU per decision.

## 1. Claims

### C1. I-POMCP is the exact I-POMDP wherever that is checkable, and degrades gracefully where it is not
| Evidence | RESULTS |
|---|---|
| Tiger: L1 H1–H5 220/220; L2 H3–H4 380/380; L3 H3 162/162; L4 H2 486/486 | 12.1, 12.3–12.4, 12.7, 12.23 |
| UAV: L1 H2–H5, L2 H2–H4, L3 H2 all optimal at qualified budgets | 12.41–12.42 |
| Wumpus 4 × 4 L1 qualified (52 beliefs, N = 4,096); 3 × 3 L2 H2 60/60, L3 H2 55/55 (every root exactly scored) | 12.53, 12.58 |
| Certified per-action bounds contain the exact Q in 1,016/1,016 decisions checked | 12.17, 12.41 |
| Held out: the qualified Tiger configurations on 960 random beliefs and seeds no qualification used, 960/960 optimal (L2 H3 and L3 H3, both games); UAV's held-out evidence is 12.67's exactly scored episode decisions | 12.69, 12.67 |
| The L3 H4 residual (4/80 near-ties, gaps ≤ 0.03) is 80/80 ε-correct | 12.14, 12.54 |
| **In whole episodes against the Bayes-optimal agent** (the definitive Tiger grid, levels 1–4, both games). Each agent is replayed along its own history through the exact agent over the exact window (10 / 8 / 4 / 3 steps at levels 1–4). Level 1 agrees on all 96,000 decisions; levels 2–4 on 97.7–100% per cell at the grid's N = 64. Every miss diagnosed (148 of 148) is belief compression, and at N = 256 (levels 3–4) or 1,024 (level 2) none remains: 0 of 2,400 trial windows, against 38 histories at N = 64 (p = 7 × 10⁻¹²) | 12.74, 12.75 |
| **Beyond exact reach** (UAV episodes from the common prior, H4–H5): I-POMCP is exactly optimal on every scored H4 decision (histories 0–4) from root 4,000 or own budget 1,000, and near the best reward from 1.4 s per decision. L2 H5 with the uniform prior needs root 64,000 (diagnosed: root-search error; fixed in replays, not yet in episodes). Only the first 5 (H4) or 2 (H5) decisions per episode are exactly scored | 12.67 |

**Figure 1** (done, October 19): `docs/figures/exactness.png` (script `results/g2_figure1_20261019/figure.py`, counts copied from RESULTS): the optimal rate of every checked cell with Clopper–Pearson 95% intervals, I-POMCP against each baseline at its best K.

**Figure 2** (`docs/figures/anytime_uav.png`, 12.67): anytime curves, i's 20-step reward against median CPU per decision, UAV H4 and H5, for every planner. Exact decision quality at H4 is in 12.67's Endpoint 3 table, which could become a second row of panels. Wumpus 4 × 4 was not run.

### C2. It beats the published baseline (IPF + RTS) and the modern one (I-NTMCP)
| Evidence | RESULTS |
|---|---|
| Tiger L2 H3–H4: I-POMCP 380/380 against IPF + RTS 95.3–99.5%, RTS 3–5× slower at H4 | 12.20 |
| Coordination Tiger L3 H2–H3: I-POMCP 162/162 against IPF + RTS 91–98% (published and smoothed, K = 2–8); never right where I-POMCP is wrong, but significant only at K = 2 (Holm p ≤ 0.005); I-POMCP 5–30× cheaper at H3. Tiger is the easy domain for the baseline at both levels | 12.68 |
| Tiger episodes against IPF + RTS | 12.38 |
| UAV L2 H2–H4: I-POMCP 80/80 against IPF + RTS 46–68%, and cheaper | 12.43, 12.46 |
| UAV episodes at H3: equal reward at 8–18× less CPU; decision quality against N particles (the 2009 paper's Fig. 16 shape) | 12.44, 12.45 |
| Wumpus 3 × 3 L2 H2: I-POMCP 60/60 against IPF + RTS 2–47%; the IPF fails on 55–57 of 60 problems (zero-likelihood particles) | 12.59 |
| IPF with the standard zero-likelihood repair (likelihood smoothing, ε = 0.01) still loses: Wumpus 20–26/60, UAV 49–55/80 against 60/60 and 80/80 (Holm p ≤ 6 × 10⁻⁸); once repaired, what remains is RTS's search | 12.62 |
| I-NTMCP (Schwartz et al., IROS 2022; authors' code through an adapter), against the exact I-POMDP on Wumpus 3 × 3 L2 H2: 40/60 at matched CPU (tuned, M = 64,000; median 97 CPU-s) against 60/60; 20 pairs favour I-POMCP and none favour I-NTMCP (Holm 1.9 × 10⁻⁶). The published variant force-fills up to 99% of its beliefs | 12.65 |
| CPU-matched episodes beyond exact reach (UAV from the common prior, H4–H5), every planner: at no more CPU, I-POMCP beats tuned I-NTMCP by +42.3 (M = 4,000, H4) and +3.5 to +4.7 (M = 16,000); both earn 35 in every episode at M = 64,000 (about 55 s per decision): a tie, stated as such. IPF + RTS K = 2 earns 30.0 at 1,130–1,340 s per decision. At H4, I-NTMCP is exactly optimal on 64–82% of decisions at M ≤ 16,000 | 12.67 |

**Table 1:** decisions against the exact I-POMDP, three planners (I-POMCP, IPF + RTS published and fixed, I-NTMCP) × three domains. Wumpus complete: published IPF 12.59, smoothed IPF 12.62, I-NTMCP 12.65. UAV has both IPF variants (12.43, 12.46, 12.62); I-NTMCP is out of scope for UAV panel problems, whose beliefs are not from a common prior.

### C3. Interpretability and the value of depth: a deeper model pays exactly when the opponent's behaviour depends on its beliefs about you
| Evidence | RESULTS |
|---|---|
| THEORY.md, "When a deeper model can pay": correct prediction is enough; no gain against a non-intentional opponent; level-0 mass keeps nested updates defined | THEORY.md |
| Tiger: depth pays only with joint payoffs (the coordination Tiger: level 1→2 gains 27–30, Holm p < 0.001; 2→3 not significant) | 12.22, 12.27, 12.37 |
| **The definitive Tiger grid** (i L0–4 × j L0–4, both games, n up to 15,000 per cell, Rao-Blackwellized rewards). The coordination Tiger's 1→2 gain by step 10 is +7.6 to +9.2 against levels 1–3 (Holm p < 10⁻⁴) and +7.0 against level 4 (n = 100, Holm p = 0.06); 2→3 and 3→4 are not significant. The published Tiger shows no gain. The exact Bayes-optimal pairs (zero variance, over their window) give the same profile: monotone to level 3, nothing at level 4. On the published Tiger at step 3 they earn 2.148 (levels 1–2) and 2.663 (levels 3–4), against the single-agent 2.72 | 12.72, 12.73 |
| Tiger: a third level with a matched model of j neither gains nor loses (+0.8 [−6.7, 8.6]) | 12.32 |
| UAV: a second level pays for both agents | 12.49 |
| Wumpus 3 × 3: a second level pays the human against an intentional wumpus (+169 [84, 259], p = 0.0004) and not against a random one (+20, n.s.) | 12.60 |
| A third level predicts better (0.54–0.55 against 0.48–0.50) but earns no more | 12.18, 12.61 |
| Interpretability: belief-hierarchy panels; why j's belief that i is level 1 follows the same curve for every level of i | 12.47, reviewer page |
| Prediction is not reward: the same prediction gain (about 55% of the gap) earns nothing in the published Tiger and up to +0.21 reward ranges with coordination payoffs; it pays only where the payoff depends on the opponent | 12.66 |
| The exact screen predicts where depth pays: on Wumpus 3 × 3, a second level is worth 19.7 per decision against an intentional wumpus, 0 at the prior and 51 after 4 steps, and costs 1.7 per decision on random-wumpus histories, matching 12.60 and its control. On UAV H2, it is worth 6.0 against 1.6, matching 12.49, and it predicts that a third level pays only against a level-2 target (6.1 against 5.8) | 12.22, 12.64 |

**Figure 3** (done, October 19): `docs/figures/depth_gain.png` (script `results/g5c_figure3_20261019/figure.py`), i's paired level 1 → 2 gain against a random (level-0) and an intentional (level-1) opponent, one panel per game: coordination Tiger −2.6 / +27.2 (12.37), UAV −7.2 / +5.8 (12.49), Wumpus 3 × 3 −70 / +179 (12.60 main trials; the fresh-trial confirmation is +169 against an intentional wumpus and +20 against a random one), published Tiger (control) −0.2 / −4.4 (12.37). n = 100 pairs each, 95% trial bootstrap.

**Figure 5** (done, October 20): `docs/figures/bayes_optimal.png` (script `results/n8_convergence_20261019/figure.py`): the share of replayed decisions that are exact maximizers by agent level, at N = 64 and at the larger N, both games (12.74–12.75).

**Figure 6** (done, October 20): `docs/figures/grid_curves.png` (script `results/n5_level_grid_20261019/figures.py`): i's expected cumulative reward by step for the diagonal pairs L1–L4, I-POMCP (Rao-Blackwellized mean, 95% band) with the exact pair's curve over its window and the 2.72 single-agent ceiling (12.72–12.73).

**Figure 7** (done, October 20): `docs/figures/grid_heatmap.png`: i's reward by step 10 over the 5 × 5 grid, per game (12.73).

**Figure 4** (done, 12.66): `docs/figures/prediction_vs_reward.png`, prediction gain against reward gain, one point per level step and opponent.

## 2. Ablation table (G4): RESULTS 12.63

Done, from paired stored results plus one new recomputation. Turning root chance scoring off breaks 34 of 560 UAV decisions and fixes none (p = 1.2 × 10⁻¹⁰); it changes none of 1,786 Tiger decisions. The table is in RESULTS 12.63.

| Component | Paired evidence | Status |
|---|---|---|
| Root chance scoring | 12.63 (new), 11 | done |
| Exact scoring with two steps left | 12.14 | done |
| Compressed exact filter vs sampled IPF | 12.59, Section 8 | done |
| Stratified vs systematic compression | 12.57 | done |
| Consistent 1/n stopping rule | 12.11, 12.15 | done |
| Root budget 1,000 vs 4,000 | 12.33 | done |
| Certified bounds | 12.17, 12.41 | done |
| Level priors with level-0 mass vs a single fixed level | 12.39 (uniform −6.0 n.s.; wrong point prior −17.5 / −30.0), 12.58 | done |

## 3. Paper sections and status

1. **Introduction.** Online I-POMDP planning. The gap: no online planner has been checked against exact I-POMDP solutions, and sampled nested filters break under deterministic sensors. Contributions C1–C3.
2. **Related work** (from RELATED_WORK.md):
   - Offline exact and approximate solvers: behavioural equivalence, I-PBVI, I-BPI, I-DIDs.
   - The IPF + RTS (Doshi & Gmytrasiewicz 2009) and I-POMDP Lite (Hoang & Low 2013).
   - POMCP adapted to I-POMDPs (Hula et al. 2015) and I-NTMCP (Schwartz et al. 2022), the closest work. It is a nested MCTS over history particles with one fixed opponent level; it needs only a generative model and has no exactness check.
   - Type-based POTMMCP (Schwartz et al. 2023).
3. **Background:** I-POMDPs, level-k priors with level-0 mass (THEORY.md).
4. **Method:**
   - I-POMCP over private histories with Bellman backups and horizon-bounded UCB;
   - nested solver bank, compressed-exact beliefs (stratified);
   - exact chance scoring; certified bounds; the consistent stopping rule.
   - Stated limitation: needs explicit kernels.
5. **Exactness** (C1, Figure 1).
6. **Against baselines** (C2, Table 1, Figure 2).
7. **The value of depth** (C3, Figures 3–4; theory statement and proof of the level-0 remark).
8. **Ablations** (Table 2).
9. **Limitations** (state each plainly; reviewer readiness, October 19):
   - two agents;
   - explicit kernels;
   - exact references limited to small horizons. In the UAV episodes (12.67) only the first 5 decisions (H4) or 2 decisions (H5) of each 20-step episode are exactly scored; later decisions are judged by reward alone;
   - at H5 with the uniform prior, I-POMCP below root 64,000 misses near-ties (37–46/60 exact, gap 0.67). The diagnosis is root-search error, and root 64,000 fixes it (19/19) at the same CPU, but only in single-decision replays, not yet in episodes (12.67);
   - on UAV, tuned I-NTMCP ties I-POMCP at M = 64,000 (both earn 35 in every episode, about 55 s per decision). I-POMCP's UAV advantage is at lower CPU; its decision-quality advantage at matched CPU is on Wumpus (12.65);
   - Wumpus L2 at 4 × 4 out of reach (12.55–12.56).
10. **Future work:** equilibria. `docs/EQUILIBRIA.md` defines a measure, and the episodes record the predictions it needs, but no result reports it (user's decision, October 19: future work, no run).

## 4. Open slots, in run order
1. **G3:** ~~the IPF zero-likelihood fix~~ (done, 12.62); the I-NTMCP adapter and its gate (blocked: running external code needs the user's permission); anytime curves (UAV L2 H4–H5, Wumpus 4 × 4 L1). Price first; ask above 100 CPU-h.
2. **G4:** assemble Table 2 from stored rows; run small paired runs only for empty rows.
3. **G5:** (a) the scatter; (b) the screen on UAV and Wumpus 3 × 3, with predictions recorded before comparing with 12.49 and 12.60; (c) Figure 3.
