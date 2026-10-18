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
| The L3 H4 residual (4/80 near-ties, gaps ≤ 0.03) is 80/80 ε-correct | 12.14, 12.54 |
| **Beyond exact reach: anytime decision quality and reward against CPU** | `MISSING (G3)` |

**Figure 1:** optimal rate by domain × level × horizon, with Clopper–Pearson intervals; a single panel of bars (from stored rows).

**Figure 2** `MISSING (G3)`: anytime curves, reward and decision quality against CPU per decision. Exact references anchor the curve at the largest checkable horizon (UAV L2 H4), and the curve extends beyond it (UAV L2 H5; Wumpus 4 × 4).

### C2. It beats the published baseline (IPF + RTS) and the modern one (I-NTMCP)
| Evidence | RESULTS |
|---|---|
| Tiger L2 H3–H4: I-POMCP 380/380 against IPF + RTS 95.3–99.5%, RTS 3–5× slower at H4 | 12.20 |
| Tiger episodes against IPF + RTS | 12.38 |
| UAV L2 H2–H4: I-POMCP 80/80 against IPF + RTS 46–68%, and cheaper | 12.43, 12.46 |
| UAV episodes at H3: equal reward at 8–18× less CPU; decision quality against N particles (the 2009 paper's Fig. 16 shape) | 12.44, 12.45 |
| Wumpus 3 × 3 L2 H2: I-POMCP 60/60 against IPF + RTS 2–47%; the IPF fails on 55–57 of 60 problems (zero-likelihood particles) | 12.59 |
| IPF with the standard zero-likelihood repair (likelihood smoothing, ε = 0.01) still loses: Wumpus 20–26/60, UAV 49–55/80 against 60/60 and 80/80 (Holm p ≤ 6 × 10⁻⁸); once repaired, what remains is RTS's search | 12.62 |
| **I-NTMCP** (Schwartz et al., IROS 2022; authors' code through an adapter): decisions scored against the exact I-POMDP, and CPU-matched episodes | `MISSING (G3)` |

**Table 1:** decisions against the exact I-POMDP, three planners (I-POMCP, IPF + RTS published and fixed, I-NTMCP) × three domains. Stored except I-NTMCP (`MISSING (G3)`): published IPF 12.43, 12.46, 12.59; smoothed IPF 12.62.

### C3. Interpretability and the value of depth: a deeper model pays exactly when the opponent's behaviour depends on its beliefs about you
| Evidence | RESULTS |
|---|---|
| THEORY.md, "When a deeper model can pay": correct prediction is enough; no gain against a non-intentional opponent; level-0 mass keeps nested updates defined | THEORY.md |
| Tiger: depth pays only with joint payoffs (the coordination Tiger: level 1→2 gains 27–30, Holm p < 0.001; 2→3 not significant) | 12.22, 12.27, 12.37 |
| Tiger: a third level with a matched model of j neither gains nor loses (+0.8 [−6.7, 8.6]) | 12.32 |
| UAV: a second level pays for both agents | 12.49 |
| Wumpus 3 × 3: a second level pays the human against an intentional wumpus (+169 [84, 259], p = 0.0004) and not against a random one (+20, n.s.) | 12.60 |
| A third level predicts better (0.54–0.55 against 0.48–0.50) but earns no more | 12.18, 12.61 |
| Interpretability: belief-hierarchy panels; why j's belief that i is level 1 follows the same curve for every level of i | 12.47, reviewer page |
| **Prediction gain against reward gain, paired per cell across domains** | `MISSING (G5a)` |
| The exact screen predicts where depth pays: on Wumpus 3 × 3, a second level is worth 19.7 per decision against an intentional wumpus, 0 at the prior and 51 after 4 steps, and costs 1.7 per decision on random-wumpus histories, matching 12.60 and its control On UAV H2, it is worth 6.0 against 1.6, matching 12.49, and it predicts that a third level pays only against a level-2 target (6.1 against 5.8) | 12.22, 12.64 |

**Figure 3** `MISSING (G5c)`: the second-level gain against an intentional opponent vs a random opponent, per domain (the proposition's figure).

**Figure 4** `MISSING (G5a)`: a scatter of prediction gain against reward gain, one point per cell.

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
| Level priors with level-0 mass vs a single fixed level | none | `MISSING (G4)`: a small paired run |

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
9. **Limitations:**
   - two agents;
   - explicit kernels;
   - exact references limited to small horizons;
   - Wumpus L2 at 4 × 4 out of reach (12.55–12.56).

## 4. Open slots, in run order
1. **G3:** ~~the IPF zero-likelihood fix~~ (done, 12.62); the I-NTMCP adapter and its gate (blocked: running external code needs the user's permission); anytime curves (UAV L2 H4–H5, Wumpus 4 × 4 L1). Price first; ask above 100 CPU-h.
2. **G4:** assemble Table 2 from stored rows; run small paired runs only for empty rows.
3. **G5:** (a) the scatter; (b) the screen on UAV and Wumpus 3 × 3, with predictions recorded before comparing with 12.49 and 12.60; (c) Figure 3.
