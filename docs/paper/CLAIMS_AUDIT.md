# Claims audit of the IJCAI-27 paper (PLAN.md S0–S1, October 22, 2026)

**Scope:** every number, table cell, figure and comparative claim in `docs/paper/latex/main.tex` at commit `65f248f`, plus the supplement's numbers where the main text leans on them.

**Method:** each number was traced to its RESULTS section and, where RESULTS rounds or pools, to the run folder's own output (`gains.csv`, `analysis.txt`). The cost was 0 CPU.

**Messages** (PLAN.md S-phase):
- **M1:** the compute needed to reach Bayes-optimal decisions.
- **M2:** when depth pays.
- **M3:** what agents believe about each other.
- **M4:** the cost of over-thinking.
- **F:** framing or method, which needs no message.

**Status codes:**
- **OK:** the number is traced and backed.
- **FIX:** the text is wrong or inconsistent and needs an edit, with no run (S7, or sooner if trivial).
- **BACK:** a backup is missing; the item becomes S2–S6 work.
- **CUT:** the item serves no message; move it to the supplement or drop it.

## A. Findings that need action (most serious first)

1. **FIX: "the variant as released gives no better results" (I-NTMCP, Section 6.3) is false on UAV.**
   - In 12.67, the released variant earns more than the tuned one at every M ≤ 16,000 at H4: 17.5 / 20.8 / 32.2 against −2.7 / −7.5 / 30.2.
   - It is also exactly optimal more often: 73–88% against 64–82%.
   - The supplement prints these numbers, so a reviewer will see the contradiction.
   - Section 7.5's headline ("I-NTMCP earns −7.5 at 4.2 s and 30.2 at 13 s") and the matched-CPU differences (+42.3, +4.7, +3.5) all use the tuned variant.
   - **Done (RESULTS 12.85, 0 CPU):** against the stronger variant at each M, I-POMCP leads by:
     - +14.0 [4.3, 27.5] at M = 4,000 (H4; Holm p = 0.12, n.s.);
     - +2.7 [1.7, 3.7] at M = 16,000 (H4; Holm p = 10⁻⁴);
     - +2.5 [1.2, 4.0] at M = 16,000 (H5; Holm p = 0.010);
     - 0 at M = 64,000.
   - S7 replaces +42.3 / +4.7 / +3.5 with these. S4 raises n above 30 for this comparison (cheap: no exact references are needed).
   - On Wumpus the tuned variant is as strong or stronger (12.65), so Table 1 stands.
2. **FIX: Table 1's n counts are inconsistent.**
   - "UAV 3, 2: 640" is two modeled arms (exact and 100 simulations) of the same 320 problems (12.41), so the cell should read 320.
   - The "3,107 decisions" total then becomes 2,787.
   - Every n counts (belief, seed) pairs. The distinct beliefs are fewer:
     - tiger L1 H5: 11 beliefs × 20 seeds;
     - tiger L2: 95 × 4;
     - L3: 81 × 2;
     - L4: 243 × 2;
     - UAV L1: 8 laws × 5 seeds;
     - UAV L2: 16 × 5;
     - UAV L3: 64 × 5;
     - Wumpus L2: 60 problems over 31 distinct beliefs;
     - Wumpus L3: 55 over 23.
   - A reviewer will ask about the effective n. **Action:** add a "distinct beliefs" column, or state the counts in the caption, and give one Clopper–Pearson bound per cell.
3. **FIX: Section 7.6 mixes two samples in one sentence.**
   - "+169 [84, 259]" comes from the fresh trials 100–199 (12.60, confirmation).
   - "killed in 41–44% against 57–60%" comes from trials 0–99.
   - The fresh trials' own mechanism numbers are 48% against 64–67%. Use those.
4. **FIX (minor): rounding and wording.**
   - Section 7.2's level-posterior TV is "0.005–0.011". The source (`results/p1_level_identification_20261021/analysis.txt`) gives a per-cell mean of 0.0043–0.0112, so write 0.004–0.011.
   - Section 7.4's "0.08 [0.07, 0.08]" is 0.077 [0.074, 0.080]; print three decimals.
5. **BACK (M2): the UAV per-decision screen in Section 7.6 ("6.0 and 1.6 … 16 belief chains") is at gate chains, not at beliefs the episodes reach.**
   - 12.64 says so itself.
   - The Wumpus screen has a matched random-opponent control on reachable histories; the UAV screen does not.
   - → S5: screen at beliefs reachable from the common prior.
6. **BACK (M2): the third-level claim ("A third level does not pay in our domains") rests on underpowered episode tests.**
   - Wumpus 2 → 3: +80 [−80, 239] and +120 [−50, 290], n = 100.
   - Coordination 2 → 3 against j2: +1.38 [0.25, 2.50]; unadjusted p = 0.016, Holm 0.16.
   - Absence of significance is not absence of effect, and the exact 2 → 3 gain on the coordination game is +0.2 by step 3.
   - → S5: exact values at reachable beliefs, then state "≤ x" with a zero-variance number rather than "does not pay".
7. **BACK (M2/M4): H-CH as written in PLAN.md is already contradicted by stored data.**
   - In UAV, a level-0 → level-1 step *loses* against an opponent two levels up:
     - the target, against a level-2 UAV: −3.5, Holm p = 0.032;
     - the UAV, against a level-2 target: −16.4, Holm p = 0.055.
   - The mechanism is predictability. A level-1 agent best-responds deterministically to a random opponent, and a level-2 opponent models exactly that agent.
   - **Refined hypothesis H-CH′:** the step k → k + 1 pays against level m = k + 1 (the step makes i's model contain the truth) and m > k + 1 in cooperative payoffs. In adversarial payoffs it can lose against m ≥ k + 2, because the opponent's model of level k + 1 is exact.
   - Sharp untested prediction: **in UAV, a level-2 UAV gains less, or loses, against a level-3 target than a level-1 UAV does.**
   - Test it exactly at H2 (UAV L3 H2 references exist, 12.41) before any episodes (S5).
8. **BACK (M1): there is no compute-to-benchmark curve.**
   - Table 1 gives one I-POMCP budget (the qualified one) against a K or M range, so "how much compute I-POMCP needs" is unanswered.
   - Stored ladders exist only for:
     - UAV L1 H4–H5 and L2 H3–H4 (12.41);
     - tiger L3 H3, modeled 100/400/exact (12.7);
     - UAV H4 episodes (12.67).
   - → S1 figure 1 and S4.
9. **BACK (S2): there is no fidelity check of either baseline against its authors' own numbers.**
   - IPF + RTS has indirect soundness evidence:
     - RTS converges to exact at K = 128 on UAV (12.43);
     - K = 32 gives 162/162 on the coordination tiger at L3 H2 (12.68).
   - It has no reproduction of Doshi & Gmytrasiewicz (2009).
   - The I-NTMCP adapter passed an L1 H1 gate (57/60, 12.65), but the authors' own runner was never compared with our adapter on one of their domains.
   - **I-NTMCP: done (RESULTS 12.86).** The authors' runner reproduces their Runner-Chaser 4 × 4 table bit for bit. Our adapter matches it (65.57 against 65.96, ±6.26, n = 1,000). The adapter's CPU overhead is 1.45×; S7 must state it.
   - **IPF + RTS:** Fig. 17(a) reproduction is running (`results/s2_ipf_fidelity_20261022`).
10. **BACK (S3): open "looks off" items, per HANDOFF 0.0000000000:**
    - (a) the tie convention: the 70 UAV disagreements; Section 8 already concedes it;
    - (b) UAV H5 near-ties need root 64,000 (Section 8 concedes it);
    - (c) the level-4 pairs above the exact pair at step 3 (z ≈ 2, n = 100, one sample counted several times);
    - (d) the exact screen against the exact episode curves at 3 → 4.
11. **CUT candidates:**
    - Section 7.1's sentence about the original tiger game at L3 H4 ("76 of 80 … lose at most 0.03"). It serves no message; it is a known near-tie residual (12.14, 12.54), so move it to the supplement.
    - The original-game rows of Table 1 serve M1 only if M1's figure uses them. Keep them for now and decide in S7.

## B. Claim-by-claim table

Each row gives: ID | claim (paper section) | message | source | n | precision | strongest reviewer attack | backup | status.

### Abstract, introduction and method (F)
| ID | Claim | Msg | Source | Status |
|---|---|---|---|---|
| F1 | "better decisions than IPF and nested MCTS at equal or lower computation" (abstract) | M1 | 12.20, 12.43, 12.46, 12.59, 12.62, 12.65, 12.67, 12.68 | **FIX:** true per cell, but tiger H3 K = 2 is cheaper than I-POMCP (1.0 s against 1.5 s, 12.20) and misses 1.6%. Say "at equal or lower computation in UAV and Wumpus, and at comparable computation in tiger" |
| F2 | "remains well defined under deterministic observations where sampled filters fail" | F | 12.59 (55–57/60 failures), 12.62 | OK |
| F3 | "one more level pays only where the other agent's actions depend on its beliefs" | M2 | Prop. 3; 12.73, 12.49, 12.60 | **FIX** after item A7: "only where" is the necessary condition (Prop. 3). Add the predictability caveat |
| F4 | Escalation at N = 64, N′ = 1,024 / 256, g = 3 (Section 4.5) | F | 12.80 | OK; the threshold was fixed before the held-out run |

### Section 7.1, decisions (Table 1)
| ID | Claim | Msg | Source | n (distinct) | Attack | Backup | Status |
|---|---|---|---|---|---|---|---|
| D1 | Tiger orig. L1 H5 220/220 | M1 | 12.1 | 220 (11 × 20) | 11 beliefs only | 12.69 held out (L2/L3 only) | OK, report the distinct count |
| D2 | Orig. L2 H3/H4 380/380; IPF + RTS 374–378, 362–368 | M1 | 12.20 | 380 (95 × 4) | K = 2 is cheaper at H3 | the CPU column in M1's figure | OK |
| D3 | Orig. L3 H3 162/162 | M1 | 12.7 | 162 (81 × 2) | — | 12.69 | OK |
| D4 | Coord. L3 H2/H3 162/162; IPF + RTS 148–158, 150–159 | M1 | 12.23, 12.68 | 162 (81 × 2) | significant only at K = 2; K = 32 also reaches 162/162 | 12.68 diagnosis | OK, but M1 must show K = 32 reaching the benchmark, and its CPU |
| D5 | Coord. L4 H2 486/486 | M1 | 12.23 | 486 (243 × 2) | point beliefs only | the n7 replay at L4 (1,793/1,800) | OK |
| D6 | UAV L1 H2–5 40 each | M1 | 12.41 | 40 (8 × 5) | H5 needs root 64,000 | the ladder in 12.41 | OK; the ladder is M1 data |
| D7 | UAV L2 H2–4 80; IPF + RTS 46–55, 42–48, 37–40 | M1 | 12.41, 12.43, 12.46, 12.62 | 80 (16 × 5) | H4 I-POMCP uses root 16,000 (366 s) | cheaper than IPF + RTS (601–3,980 s) | OK |
| D8 | UAV L3 H2 640 | M1 | 12.41 | **320** (64 × 5) | double count | — | **FIX** (A2) |
| D9 | Wumpus L2 H2 60/60; IPF + RTS 1–26; I-NTMCP 40 at 0.8× CPU | M1 | 12.58, 12.59, 12.62, 12.65 | 60 (31) | 31 distinct beliefs | McNemar 20/0 | OK |
| D10 | Wumpus L3 H2 55/55 | M1 | 12.58 | 55 (23) | 23 distinct beliefs | — | OK |
| D11 | Total 3,107 | M1 | sum | — | inflated by D8 | — | **FIX:** 2,787 |
| D12 | 960/960 held out | M1 | 12.69 | 960 | tiger only | UAV: 12.67 exact-scored episodes | OK; **BACK:** Wumpus held out (S6) |
| D13 | Orig. L3 H4 76/80, loss ≤ 0.03 | — | 12.14, 12.54 | 80 | serves nothing | — | **CUT** to the supplement |
| D14 | UAV target 100 simulations → 63/80 | M1 | 12.41 | 80 | — | — | OK; becomes a point on M1's curve |
| D15 | Certified intervals 1,016/1,016 | F | 12.17, 12.41, 12.63 | 1,016 | containment depends on the modeled solve | stated in Section 4.3 | OK |
| D16 | Exact scoring 0.06 → 0.0006; 34 of 560, p = 1.2 × 10⁻¹⁰ | F | 12.24, 12.63 | 560 | — | — | OK |

### Section 7.2, episodes against the exact agent
| ID | Claim | Msg | Source | n | Status |
|---|---|---|---|---|---|
| E1 | L1 48,000/48,000; L2–L4 8,779/8,800, 8,361/8,400, 1,793/1,800 | M1, M3 | 12.74 (coordination counts) | episodes up to 1,000 / 200 / 400 / 100 per cell | OK |
| E2 | Posterior TV 0.005–0.011 | M3 | p1 `analysis.txt` | — | **FIX:** 0.004–0.011 |
| E3 | 67 misses, median gap 0.09; 61 diagnosed as compression | F | 12.84 | 67 | OK |
| E4 | Larger N: 0 against 21 histories, p < 10⁻⁶ | F | 12.84 (p = 9.5 × 10⁻⁷) | 1,200 windows | OK |
| E5 | Escalation 32 of 34; 337/10,400; 0.41× CPU (n = 120) | F | 12.84 | 1,600 episodes | OK; the CPU ratio is on a 120-episode subsample (stated) |
| E6 | UAV 10,582/10,652; 70 are tie convention | F | 12.81 | 100 per cell | OK; S3(a) can make it 0 |

### Section 7.3, level grid (Figure 2)
| ID | Claim | Msg | Source | n | Status |
|---|---|---|---|---|---|
| G1 | Exact pairs by step 3: 14.83, 18.01, 18.38; level 4 adds nothing | M2 | 12.72 | exact | **BACK:** reconcile with the 12.22 screen (3 → 4 positive at point chains), S3(d)/S5 |
| G2 | Exact inside the I-POMCP 95% interval at 97/100 agent-steps; 42/42 for level 0 | M1 | 12.84 | — | OK. The level-4 step-3 excursion (z ≈ 2) is S3(c) |
| G3 | 1 → 2: +7.8 [7.4, 8.3], +7.6 [7.1, 8.1], +9.2 [7.3, 11.1], +7.0 [2.3, 11.8]; −0.9 [−1.1, −0.7] | M2, M4 | n5 `gains.csv` | 15,000 / 700 / 100 | OK (checked to the CSV) |
| G4 | No gain 2 → 3 or 3 → 4 significant (Holm p ≥ 0.16) | M2 | `gains.csv` | 700 / 100 | **BACK** (A6): 2 → 3 against j2 is +1.38 [0.25, 2.50] unadjusted. Report the exact 2 → 3 value instead |
| G5 | Exact 2 → 3 ≈ 0.2 by step 3, "below what 700 resolve" | M2 | 12.72 | exact | OK; this is the right zero-variance framing |

### Section 7.4, identification (Figure 1)
| ID | Claim | Msg | Source | n | Status |
|---|---|---|---|---|---|
| I1 | Level-2 observer: 0.92–0.95 on level 1 for planning others | M3 | p1 `analysis.txt` (0.917–0.954) | 15,000 / 700 / 100 | OK |
| I2 | Against random: 0.08 [0.07, 0.08] | M3 | 0.077 [0.074, 0.080] | 15,000 | **FIX** (rounding) |
| I3 | Level-3 observer: 0.90 on L0; mode = truth for L1 (0.52), L2 (0.53); L3/L4 → L2 (0.54, 0.59) | M3 | p1 `analysis.txt` | 700 / 100 | OK |
| I4 | Exact posterior coincides; paired difference ≤ 0.003 at step 8 | M3 | p1 `analysis.txt` | 100–1,000 | OK |
| I5 | KL closest-hypothesis argument | M3 | Berk (1966) | — | OK; a mechanism sentence exists |

### Section 7.5, baselines
| ID | Claim | Msg | Source | Status |
|---|---|---|---|---|
| B1 | IPF + RTS misses 0.5–4.7% (orig.), 2–9% (coord. L3), 31–54% (UAV); never right where I-POMCP is wrong | M1 | 12.20, 12.68, 12.43, 12.46 | OK |
| B2 | "I-POMCP is 5–30 times cheaper at horizon 3" (tiger) | M1 | 12.68 (coordination L3 H3) | **FIX:** this holds at L3 only. At L2 H3 IPF + RTS K = 2 is cheaper (12.20). Scope the sentence |
| B3 | Wumpus plain IPF fails on 55–57/60; smoothed 20–26; RTS on exact beliefs 20–28 | M1 | 12.59, 12.62 | OK |
| B4 | I-NTMCP 40/60 at 0.8× CPU; 20/0; Holm p = 1.9 × 10⁻⁶ | M1 | 12.65 | OK (tuned ≥ released on Wumpus) |
| B5 | UAV anytime: 34.8 at 2.9 s against −7.5 at 4.2 s and 30.2 at 13 s; +42.3, +4.7, +3.5 | M1 | 12.67 | **FIX** (A1): compare against the stronger variant per M |
| B6 | 120/120 at H4 vs 64–82% (I-NTMCP) vs 62–70% (IPF + RTS, over 1,100 s) | M1 | 12.67 | **FIX** (A1): the released variant scores 73–88%. IPF + RTS's n is 47 and 46 decisions (10 episodes); state that |

### Section 7.6, value of depth
| ID | Claim | Msg | Source | Status |
|---|---|---|---|---|
| V1 | 1 → 2 against level 1: +7.8, +5.8 [3.9, 8.0], +169 [84, 259] | M2 | 12.73, 12.49, 12.60 | OK |
| V2 | Against level 0: −0.9, −7.2 [−12.7, −2.6], +20 [−70, 110] | M4 | same | OK |
| V3 | UAV 5.0 against 6.2 steps; "against a random target, the same anticipation costs 7.2" | M4 | 12.49 | OK; **BACK:** no mechanism trace yet (S1 figure 4) |
| V4 | Human killed 41–44% against 57–60% | M2 | 12.60 trials 0–99 | **FIX** (A3): 48% against 64–67% on the fresh trials |
| V5 | Exact per decision: Wumpus 19.7 / 1.7 (60 / 58 histories); UAV 6.0 / 1.6 (16 chains) | M2, M4 | 12.64 | **BACK** (A5): the UAV numbers are at gate chains |
| V6 | Level-3 human predicts 0.54–0.55 against 0.48–0.50; gains +80, +120 n.s. | M2 | 12.61 | **BACK** (A6) |
| V7 | "Depth pays up to about one level above the opponent's, as the cognitive hierarchy model suggests" | M2 | 12.61 | **BACK** (A7): replace with H-CH′ once tested |

### Section 8, discussion
| ID | Claim | Status |
|---|---|---|
| L1 | H5 needs root 64,000 with a uniform prior | **BACK:** S3(b), near-tie root escalation |
| L2 | "A modeled agent plays its own choice among tied actions" | **BACK:** S3(a) would remove this limitation |

## C. Full level-gain table from stored data (input to S5 and H-CH′)

Paired one-level gain of the stepping agent against an opponent at level m (Holm-adjusted p in parentheses where reported). Episode lengths: coordination 10 steps, UAV 20, Wumpus 20.

| Domain, agent | Step | m = 0 | m = 1 | m = 2 | m = 3 | m = 4 |
|---|---|---|---|---|---|---|
| Coordination, i | 1 → 2 | −0.89 (< 10⁻⁴) | +7.84 | +7.57 | +9.23 | +7.02 (0.06) |
| | 2 → 3 | −0.84 | −0.75 | +1.38 (0.16; unadj. 0.016) | +1.39 | +0.97 |
| | 3 → 4 | +1.57 | −0.36 | +0.46 | +0.52 | +1.62 |
| UAV, UAV | 0 → 1 | +33.1 | +91.1 | **−16.4 (0.055)** | — | — |
| | 1 → 2 | **−7.2 (0.032)** | +5.8 | +26.0 | — | — |
| UAV, target | 0 → 1 | +18.5 | −2.0 (0.09) | **−3.5 (0.032)** | — | — |
| | 1 → 2 | −5.0 (0.09) | +35.0 | +26.0 | — | — |
| Wumpus, human | 0 → 1 | +473 | +404 | +346 | — | — |
| | 1 → 2 | −70 | +179 (0.087) | +179 (0.015) | — | — |
| | 2 → 3 | −19 | +80 (n.s.) | +120 (n.s.) | — | — |
| Wumpus, wumpus | 0 → 1 | +435 | +435 | +234 | — | — |
| | 1 → 2 | −20 | +31 | +30 (n.s.) | — | — |

**Reading:**
- m = k + 1 is where the step makes the model contain the truth. There the gain is positive and significant in every domain except the Wumpus's own 1 → 2 (+31, n.s.).
- m < k + 1: the gain is ≤ 0 or n.s. everywhere. This is Prop. 3.
- m ≥ k + 2: the gain is positive in the cooperative game (coordination 1 → 2 against m = 3–4) and **negative in UAV pursuit–evasion at k = 0**, the predictability effect.

## D. S1: the figure plan (at most 4 figures in the paper)

Each figure has one motivating sentence, the exact claim it shows and its variance class. Precision target: a CI half-width at most 10% of the effect or of the exact value it is compared with. For a 100% cell the target is n ≥ 459, so that the Clopper–Pearson lower bound is ≥ 0.99.

### Figure 1 (M1): compute to the Bayes-optimal benchmark
- **Motivation:** the question a practitioner asks is how much CPU per decision each planner needs before its decisions are Bayes-optimal.
- **Claim:** I-POMCP reaches 100% optimal at a lower median CPU than either baseline in UAV and Wumpus, and at a comparable one in tiger. The baselines' curves plateau below 100% (UAV RTS 37–55%, Wumpus I-NTMCP ≤ 40/60), or reach it only at large K (tiger K = 32).
- **Form:** one panel per cell, with the share optimal (Wilson 95%) against median CPU per decision (log x). One line per planner over its budget ladder (I-POMCP root/modeled; IPF + RTS K; I-NTMCP M), and the mean first-action loss as a secondary y-axis or an inset.
- **Cells:**
  - coordination tiger L3 H3;
  - UAV L2 H3;
  - UAV L2 H4;
  - Wumpus L2 H2.
  - Tiger L2 H3 is optional, if space allows.
- **Stored points:**
  - I-POMCP:
    - UAV L2 H3 at modeled 100 / 400 (12.41);
    - UAV L2 H4 at root 1,000 / 4,000 / 16,000 with an exact target, and modeled 100 / 400 (12.41);
    - coordination L3 H3 at the qualified point only;
    - Wumpus L2 H2 at modeled 400 / 1,600 (root 1,000).
  - IPF + RTS: K = 2 / 4 / 8 everywhere, and K = 32 at coordination L3 H2 only.
  - I-NTMCP: Wumpus only (M = 1,000 to 64,000).
- **Missing points (S4):**
  - an I-POMCP root ladder (100 / 250 / 400 / 1,000) at coordination L3 H3, UAV L2 H3 and Wumpus L2 H2;
  - IPF + RTS K = 16 / 32 at UAV L2 H3 and coordination L3 H3, to show whether it reaches the benchmark at all;
  - I-NTMCP on UAV L2 H3 decisions from common-prior histories (its model class);
  - n raised to 459 problems at each cell's I-POMCP qualified point. New exact references are needed for UAV and Wumpus.
- **Price (to be refined on the hardest cell under load before S4 starts):**
  - I-POMCP ladders: about 10 CPU-h.
  - IPF + RTS K = 16 / 32 at UAV H3: about 100–200 s per decision × 80 × 2, about 9 CPU-h.
  - I-NTMCP UAV H3: about 5 CPU-h.
  - Raising n to 459 is dominated by exact references: Wumpus about 90 s each (≈ 10 CPU-h for 400 more), UAV H3 minutes each.
  - Total about 40–60 CPU-h, under the 100 CPU-h ask line.

### Figure 2 (M2, M4): the level match-up matrix
- **Motivation:** which match-ups make an extra level pay, with zero variance where possible.
- **Claim:** H-CH′ (Section C).
- **Form:** one heat map per domain (stepping agent's k → k + 1 on the rows, opponent's level m on the columns). Each cell shows the exact per-decision value at beliefs reachable from the common prior, with a marker where the planner's paired episode gain has the same sign and is significant.
- **Missing (S5):**
  - a reachable-belief exact screen for UAV (both agents; k = 0 → 1, 1 → 2 and 2 → 3 against m = 0–3 at H2);
  - coordination 2 → 3 and 3 → 4 at reachable beliefs (this reconciles G1 with 12.22);
  - Wumpus 2 → 3 at H2 (where L3 references exist).
  - Monte Carlo is used only to confirm a sign, with paired seeds.
- **The headline test:** UAV, a level-2 UAV against a level-3 target, exact at H2. H-CH′ predicts a smaller gain or a loss.

### Figure 3 (M3): identification
- **Motivation and claim:** as now (Figure 1 of the draft).
- **Variance:** n = 15,000 / 700 / 100, with the exact overlay.
- **Status:** exists; keep it.

### Figure 4 (M4): mechanism panel, over-thinking and predictability
- **Motivation:** show *why* depth costs reward, not only that it does.
- **Content:**
  - (a) a UAV level-2 against a random target: the empirical first-five-step flight paths against where a level-1 target would be, with the exact per-decision cost;
  - (b) a level-1 target caught by a level-2 UAV: its path and capture step, against a random target.
- **Data:** the stored 12.49 episodes (n = 100 per cell; 0 CPU), plus the exact screen values from S5.

### Tables
- Table 1 stays, with the n fixes (A2) and a distinct-beliefs count.
- The baselines' fidelity results (S2) go into one sentence plus the supplement.

## E. What S2–S6 must deliver, from this audit
| S-step | Item from this audit |
|---|---|
| S0 text fixes (now, 0 CPU) | A1 recomputation from stored rows; A2–A4 numbers recorded here for S7 |
| S2 | A9: reproduce IPF + RTS (Doshi & Gmytrasiewicz 2009, multiagent tiger) and I-NTMCP (the authors' Runner-Chaser 3 × 3 through their runner and through our adapter) |
| S3 | A10 (a)–(d) |
| S4 | A8, Figure 1's missing points |
| S5 | A5–A7, Figure 2 |
| S6 | D12 (Wumpus held out); Table 1's dashes |
| S7 | every FIX row; D13 cut; Figure 4 |
