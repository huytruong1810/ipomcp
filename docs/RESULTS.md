# Results

All results come from the current code (`main`; Sections 1–8 from October 1, 2026) and the published Tiger specification unless stated otherwise. **n** is the number of evaluated decisions or steps behind each number.

Terms:
- **"Optimal":** every action the planner can choose is an exact maximizer.
- **"Loss":** first-action loss (see [THEORY.md](THEORY.md)).
- **"TV":** total variation distance.

## 1. Planner vs exact value iteration (L1 Tiger)

**Setup.** Agent i against uniform-random L0 j. The reference is alpha-vector value iteration. Each cell covers 11 beliefs × 3 seeds (n = 33). Driver: `experiments.l1_qualification`.

| Simulations | H1 | H2 | H3 | H4 | H5 |
|---|---|---|---|---|---|
| 100: optimal | 100% | 100% | 100% | 100% | 97% |
| 1,000: optimal | 100% | 100% | 100% | 100% | 100% |
| 5,000: optimal | 100% | 100% | 100% | 100% | 100% |
| 100: mean max Q error | 0 | 0.27 | 0.48 | 1.06 | 2.94 |
| 1,000: mean max Q error | 0 | 0.08 | 0.13 | 0.54 | 2.14 |
| 5,000: mean max Q error | 0 | 0.05 | 0.06 | 0.43 | 0.88 |

CPU per root decision at 1,000 simulations ranges from 0.02 s at H2 to 0.10 s at H5.

**With IPF beliefs inside the search** (1,000 simulations; 9 beliefs × 3 seeds, n = 27 per cell):
- 100% optimal at H3 and H4 for 64, 256 and 1,024 particles.
- Mean max Q error at H3: 0.45, 0.18 and 0.17 at 64, 256 and 1,024 particles.
- Mean max Q error at H4: 0.71, 0.64 and 0.62.

## 2. Planner vs exact best response on a terminating domain (UAV, L1)

> **Note (October 12, 2026):** this section used the earlier UAV *pursuit* game (compass sensors, ±1 rewards). `domains/uav.py` is now the reconnaissance domain of Doshi & Gmytrasiewicz (2009) (DOMAINS.md); these numbers describe the retired model.

**Setup.** Pursuer on a 2 × 3 grid against a uniform-random evader. Beliefs are random over two target cells. n = 12 per cell.

| Horizon | Simulations | Optimal | Mean loss | Mean max Q error |
|---|---|---|---|---|
| 2 | 300 | 12/12 | 0 | 0.020 |
| 2 | 3,000 | 12/12 | 0 | 0.006 |
| 3 | 300 | 11/12 | 0.0003 | 0.046 |
| 3 | 3,000 | 11/12 | 0.0003 | 0.015 |

The single H3 miss is a near-tie: its loss is 0.004.

## 3. Sampled interactive particle filter vs exact update (L2 Tiger)

**Setup.** i is L2 with level prior {L1 .5, L0 .5}. Modeled opponents are
I-POMCP with 50 simulations at depth 2. Each history is 8 steps (listen, with
one opening at step 4) generated against a uniform-random j; the IPF and the
exact update condition on the same history. The IPF resamples systematically.
Driver: `experiments.belief_qualification --particle-filters sampled`; data in
`results/ipf_resampling_probe_20261002`. The compressed filter, measured the
same way, is in Section 8.

**All 12 histories** (n = 96 steps per cell):

| Particles N | Level-posterior TV | Physical-belief error | Nested-path TV (depth 2) | Max level TV | Steps with level TV > 0.2 |
|---|---|---|---|---|---|
| 16 | 0.206 | 0.101 | 0.103 | 0.91 | 39% |
| 64 | 0.062 | 0.041 | 0.031 | 0.68 | 6% |
| 256 | 0.040 | 0.020 | 0.020 | 0.65 | 4% |

**The 3 hardest histories at N = 256** (mean level TV 0.11, 0.12 and 0.09)
were re-run at larger N (n = 24 steps per cell):

| Particles N | Level-posterior TV | Max level TV | Physical-belief error | Nested-path TV |
|---|---|---|---|---|
| 1,024 | 0.020 | 0.067 | 0.010 | 0.010 |
| 4,096 | 0.014 | 0.060 | 0.005 | 0.007 |

**Interpretation:**
- Mean errors shrink with N on every measure. Resampling systematically
  instead of multinomially lowered the mean level TV at every N (paired over
  the same 12 histories: by 0.021, 0.052 and 0.014 at N = 16, 64 and 256;
  better on 6, 9 and 8 histories).
- **The worst case does not shrink like the mean.** After the opening, level
  TV of 0.04–0.06 persists at N = 4,096 on two of the hard histories. The
  modeled opponents act greedily, so a nested particle belief slightly
  different from the exact one can flip a modeled action. That is a discrete
  change in the likelihood of j's behaviour, and more particles make it rarer
  but cannot make it continuous.
- An earlier 3-history run (multinomial resampling) reported a level TV of
  0.019 at N = 1,024 and at most 0.004 at N = 4,096. Those histories were easy
  ones; the 12-history sample above replaces it.
- Raising the modeled budget from 50 to 400 did not change errors (earlier
  run), so the floor is not search noise in the modeled policies.
- The exact belief reaches 18,152 atoms by step 8.

**Gate (sampled IPF).** Use N ≥ 1,024 where per-step level posteriors are
reported, and report the mean with the maximum. The compressed filter (Section
8) meets the same accuracy at N = 64.

## 4. Cost of an L3 decision

**Setup.** Tiger, i at L3. 2,000 root simulations, 25 modeled simulations, depth 3, after two observations. One decision each (n = 1), profiled.

| Beliefs | CPU (s) | Nested policy solves | Share of CPU in nested solves |
|---|---|---|---|
| Exact | 888 | 1,779 | 89% |
| IPF, 64 particles | 7.7 | 228 | 80% |

**Cross-step reuse.** Keeping the caches across real steps reduced CPU over six L3 decisions (IPF, 64 particles) from 26.6 s to 19.8 s. Decisions and the final belief were identical, which is what purity of the cached functions implies.

## 5. Root search vs opponent modeling (L2 Tiger, vs the exact I-POMDP)

**Setup.** i is L2 and models j as an I-POMCP L1 with `modeled` simulations
(depth equal to the horizon). The reference is the exact I-POMDP: j is an exact
L1 agent, so the loss includes both i's search error and its opponent-model
error. Each cell covers 5 beliefs of i × 3 beliefs of j × 2 seeds (n = 30).
CPU is per root decision, including the nested solves. Driver:
`experiments.panel`.

Optimal rate (mean max Q error in brackets):

| Horizon | Root | Modeled 5 | Modeled 25 | Modeled 100 | Modeled 400 | Modeled 1,000 | Modeled 4,000 |
|---|---|---|---|---|---|---|---|
| 3 | 250 | 90% (0.77) | 90% (0.81) | 100% (0.33) | 100% (0.33) | | |
| 3 | 1,000 | 87% (0.70) | 90% (0.68) | 100% (0.15) | 100% (0.15) | | |
| 3 | 4,000 | 87% (0.62) | 87% (0.62) | 100% (0.06) | 100% (0.06) | | |
| 4 | 250 | 80% (2.38) | 80% (2.22) | 93% (1.56) | 93% (1.56) | 93% (1.55) | 100% (1.08) |
| 4 | 1,000 | 80% (2.08) | 80% (2.07) | 93% (1.21) | 93% (1.21) | 93% (1.20) | 100% (0.76) |
| 4 | 4,000 | 80% (2.12) | 80% (1.98) | 93% (1.03) | 93% (1.03) | | |

Mean CPU per decision: H3 0.07–1.2 s; H4 0.26–3.5 s up to modeled 400, 4.2–4.8 s
at modeled 1,000 and 18–22 s at modeled 4,000.

**Interpretation:**
- **The modeled budget decides optimality; the root budget only reduces Q
  error.** In every row the optimal rate is fixed by the modeled column. A
  16-fold larger root budget never removes a miss caused by a wrong opponent
  model.
- **The H4 ceiling of 93% is opponent-model error.** The two misses (max loss
  1.85) come from one modeled j-L1 that opens where the exact L1 listens
  (Q gap 0.66 at 400 simulations). It persists at 1,000 modeled simulations
  and disappears at 4,000, for both root budgets.
- **Allocation rule.** Give the modeled budget priority: enough that each
  modeled agent's decisions match its exact solution at the beliefs it is
  modeled at (here 100 simulations at depth 3 and 4,000 at depth 4). Section 1
  is not sufficient evidence for this, because it scores L1 decisions at
  i's beliefs, not at the beliefs j is modeled at. Spend what remains on root
  simulations, which lower Q error but do not change the decision once the
  opponent model is right.

## 6. I-POMCP vs IPF + RTS (L2 Tiger, vs the exact I-POMDP; sampled IPF)

**Setup.** i is L2; j is modeled as an L1 agent solved by the *same kind* of
planner (I-POMCP with `modeled` simulations, or RTS with K samples per node, as
in IPF + RTS where every level is solved by RTS), at depth equal to the horizon.
The reference is the exact I-POMDP, solved on exact beliefs in its own bank.
Each cell covers 5 beliefs of i × 3 private beliefs of j × 2 seeds (n = 30).
CPU is per root decision, including nested solves. Driver:
`experiments.panel`; data in `results/l2_baseline_20261002`.

**Both planners with IPF beliefs (N = 64)**: the full systems being compared.

| Horizon | Planner | Budget (modeled) | Optimal | Mean max Q error | CPU per decision |
|---|---|---|---|---|---|
| 3 | I-POMCP | 250 (100) | 100% | 0.64 | 2.9 s |
| 3 | I-POMCP | 1,000 (100) | 100% | 0.63 | 3.2 s |
| 3 | I-POMCP | 4,000 (400) | 100% | 0.59 | 7.5 s |
| 3 | IPF + RTS | K = 2 (2) | 93% | 2.19 | 1.0 s |
| 3 | IPF + RTS | K = 4 (4) | 97% | 1.55 | 2.1 s |
| 3 | IPF + RTS | K = 8 (8) | 97% | 1.27 | 4.1 s |
| 4 | I-POMCP | 250 (100) | 97% | 1.60 | 21 s |
| 4 | I-POMCP | 1,000 (100) | 90% | 1.54 | 22 s |
| 4 | I-POMCP | 4,000 (400) | 93% | 1.24 | 60 s |
| 4 | IPF + RTS | K = 2 (2) | 77% | 3.09 | 11 s |
| 4 | IPF + RTS | K = 4 (4) | 80% | 2.66 | 43 s |
| 4 | IPF + RTS | K = 8 (8) | 83% | 1.97 | 96 s |

The panel covers every I-POMCP cell (root 250, 1,000 and 4,000 × modeled 100
and 400). At H3 all are 100% optimal; at H4 they are 90–97% optimal.

**Control: the same cells on exact beliefs.** These separate belief error from search error.
- **I-POMCP:** H3 is 100% optimal at 0.4–1.9 s, and its Q error falls from 0.33 to 0.06 with the root budget. H4 is 93% optimal in every cell, at 1.6–6.5 s.
- **RTS:** H3 is 87%, 97% and 100% optimal at K = 2, 4 and 8, taking 0.04–0.41 s. H4 is 77%, 90% and 83%, taking 0.5–13 s.

**Interpretation:**
- **Accuracy at matched CPU.** I-POMCP is more accurate at every horizon.
  - At H4, I-POMCP with 250 root and 100 modeled simulations is 97% optimal in 21 s. IPF + RTS reaches 80–83% at 43–96 s.
  - RTS's tree grows as (|A| K)^(H−1) at every nesting level. Its CPU rose 11–24× from H3 to H4, while I-POMCP's rose 6–8×.
- **On exact beliefs, the I-POMCP H4 ceiling (93%) is the opponent-model error of Section 5.**
  - With 4,000 modeled simulations it is 100% optimal at root 250 and at root 1,000 (n = 30 each; 38 s per decision; `results/l2_baseline_h4_modeled4000_20261002`).
- **With the IPF (N = 64), I-POMCP stays at 93% at H4 even with 4,000 modeled simulations (206 s per decision).**
  - Those misses are belief error, not search error: different problems from the exact-belief misses, and independent of both budgets.
  - At N = 64 the planner values opening at 6.2–6.5 against an exact 3.2–3.3.
  - Re-run at root 1,000 and modeled 100, the two missed problems become optimal at N = 256 and N = 1,024 respectively (`results/l2_ipf_miss_20261002`).
  - At L1, N = 64 was enough (Section 1); at L2 H4 it is not.
- **IPF costs more than exact beliefs on these problems.**
  - Each problem starts from a two-atom belief, so exact posteriors stay small within four steps, while the IPF always carries up to 64 particles per level.
  - The IPF earns its cost where exact supports grow: by step 8 of an episode, the exact L2 belief holds 18,152 atoms (Section 3).
  - So this panel compares the planners' accuracy with the IPF in place. It does not show that the IPF is faster here.
- **With the IPF, I-POMCP's Q error has a floor of about 0.6 at H3, whatever the root budget.**
  - The likely source is that each history's expected rewards are integrated over a 64-particle posterior, so particle error enters every backup. This has not been tested separately.
  - The floor never changed a decision at H3.

## 7. Exact vs sampled-IPF beliefs in mid-episode decisions (L2 Tiger, H3)

**Setup.**
- i decides after conditioning on the first k steps of a fixed history, generated by `oracle.sample_history` (i listens and opens once at step 4; j acts at random).
- j is the exact L1 agent, so only i's belief representation and search differ.
- Planner: I-POMCP with 1,000 root simulations.
- The reference is the exact I-POMDP at the exact posterior.
- Each cell covers 5 beliefs of i × 3 beliefs of j × 2 seeds (n = 30).
- Planning CPU is per decision; update CPU covers all k belief updates; support is the number of distinct interactive hypotheses i holds.
- Driver: `experiments.panel`; data in `results/l2_mid_episode_20261002`.

| History k | Beliefs | Optimal | Mean max Q error | Planning CPU | Update CPU | Support |
|---|---|---|---|---|---|---|
| 0 | exact | 100% | 0.15 | 0.3 s | 0 | 2 |
| 0 | IPF 64 | 100% | 0.63 | 0.8 s | 0 | 2 |
| 0 | IPF 256 | 100% | 0.15 | 1.7 s | 0 | 2 |
| 0 | IPF 1,024 | 100% | 0.13 | 4.8 s | 0 | 2 |
| 2 | exact | 97% | 0.14 | 2.1 s | 0.02 s | 23 |
| 2 | IPF 64 | 97% | 7.73 | 3.6 s | 0.06 s | 16 |
| 2 | IPF 256 | 97% | 1.89 | 8.2 s | 0.11 s | 26 |
| 2 | IPF 1,024 | 100% | 1.16 | 23 s | 0.45 s | 36 |
| 4 | exact | 100% | 0.11 | 26 s | 0.31 s | 241 |
| 4 | IPF 64 | 100% | 3.99 | 7.9 s | 0.32 s | 32 |
| 4 | IPF 256 | 100% | 2.77 | 23 s | 1.07 s | 81 |
| 4 | IPF 1,024 | 100% | 1.32 | 140 s | 1.83 s | 181 |

RTS (K = 4) on the same problems was 97–100% optimal. With exact beliefs it took 0.1, 1.4 and 17 s at k = 0, 2 and 4. With IPF 256 it took 0.9, 6.2 and 19 s, and its Q error was 1.3–2.9.

**Beyond k = 4** (one problem: i at 0.5, j at 0.5, seed 0):
- At k = 6, exact-belief I-POMCP did not finish one decision in 30 minutes.
- The exact I-POMDP reference exceeded its 200,000-branch enumeration budget.
- The IPF planners still run there, but they cannot be scored.

**Interpretation:**
- **Where the exact reference is tractable, exact beliefs dominate.** Up to k = 4, exact-belief I-POMCP is faster than the IPF at N = 1,024 and about as fast as at N = 256 (26 s against 23 s at k = 4). It is 8–25 times more accurate in Q than either. Only the IPF at N = 64 is clearly cheaper (8 s against 26 s at k = 4), at about 36 times the Q error, and it has one miss with a loss of 4.8 at k = 2.
- **Particle error accumulates with history.** At k = 0 the IPF matches exact beliefs (the prior has two atoms). By k = 2–4 its Q error is 1.2–7.7, because each in-tree update samples the opponent's private observations and actions afresh.
- **The compressed filter (Section 8) changes this conclusion.** The remaining points apply to the sampled IPF.
- **The IPF pays off only where exactness becomes intractable:** k ≥ 6 at L2 H3 here, or the 94,500-state Wumpus prior.
  - There, its accuracy can be checked only indirectly: by belief error against exact filtering along single histories (Section 3, up to 18,152 atoms), or against references that drop the nested enumeration.
  - Contribution 2 must therefore report IPF error where it can be measured and cost where it cannot. It must not claim measured accuracy in the regime that motivates the IPF.

## 8. The compressed filter: exact updates with supports bounded to N

**Why.** The diagnosis came from exact Bellman values over sampled-IPF beliefs, with no search involved.
- At N = 64, they reproduce the H4 misses of Section 6: Q(open) is 6.31 against an exact 3.32. At N = 256 they match the exact values to two decimals.
- The sampled IPF holds every belief, however small, as N equal-weight samples. A nested two-state belief therefore moves in steps of 1/N, and greedy modeled opponents flip their predicted action across thresholds.
- The compressed filter (`belief.compressed_filter`; see THEORY.md) enumerates each update exactly. It samples (systematically) only to bring a posterior with more than N atoms back to N.
- Prototype and diagnosis scripts are in `results/compressed_prototype_20261002`.

**Belief accuracy** (same 12 histories and modeled opponents as Section 3; n = 96 steps per cell; `results/belief_qualification_compressed_20261002`):

| N | Filter | Level-posterior TV | Physical-belief error | Nested-path TV | Max level TV | Steps with level TV > 0.2 |
|---|---|---|---|---|---|---|
| 16 | compressed | 0.049 | 0.013 | 0.025 | 0.43 | 4% |
| 16 | sampled | 0.206 | 0.101 | 0.103 | 0.91 | 39% |
| 64 | compressed | 0.008 | 0.001 | 0.004 | 0.09 | 0% |
| 64 | sampled | 0.062 | 0.041 | 0.031 | 0.68 | 6% |
| 256 | compressed | 0.002 | 0.0005 | 0.001 | 0.02 | 0% |
| 256 | sampled | 0.040 | 0.020 | 0.020 | 0.65 | 4% |

**Mid-episode decisions** (Section 7's problems: I-POMCP with 1,000 simulations, exact j-L1, H3; n = 30 per cell; `results/l2_compressed_20261002`):

| History k | Beliefs | Optimal | Mean max Q error | Planning CPU | Support |
|---|---|---|---|---|---|
| 2 | exact | 97% | 0.14 | 2.1 s | 23 |
| 2 | compressed 16 | 93% | 2.14 | 0.7 s | 9 |
| 2 | compressed 64 | 100% | 0.25 | 1.5 s | 23 |
| 2 | sampled IPF 64 | 97% | 7.73 | 3.6 s | 16 |
| 4 | exact | 100% | 0.11 | 26 s | 241 |
| 4 | compressed 16 | 100% | 0.68 | 0.8 s | 9 |
| 4 | compressed 64 | 100% | 0.18 | 1.9 s | 29 |
| 4 | compressed 256 | 100% | 0.12 | 4.6 s | 77 |
| 4 | sampled IPF 64 | 100% | 3.99 | 7.9 s | 32 |

RTS (K = 4) with compressed beliefs at N = 64:
- It was 97%, 97% and 100% optimal at k = 0, 2 and 4, with Q error 1.23, 1.00 and 0.48.
- It took 0.1, 0.8 and 1.3 s, against 0.1, 1.4 and 17 s on exact beliefs.

**From the prior at H4** (Section 6's problems, N = 64):

| Planner | Budget (modeled) | Compressed | Exact | Sampled IPF |
|---|---|---|---|---|
| I-POMCP | 250 (100) | 90%, 1.5 s | 93%, 1.6 s | 97%, 21 s |
| I-POMCP | 1,000 (100) | 93%, 2.1 s | 93%, 2.1 s | 90%, 22 s |
| I-POMCP | 250 (4,000) | 97%, 27 s | 100%, 38 s | 93%, 206 s |
| RTS | K = 2 | 77%, 0.4 s | 77%, 0.5 s | 77%, 11 s |
| RTS | K = 4 | 90%, 2.8 s | 90%, 3.1 s | 80%, 43 s |
| RTS | K = 8 | 83%, 9.9 s | 83%, 13 s | 83%, 96 s |

**Interpretation:**
- **At equal N, the compressed filter is far more accurate than the sampled IPF.**
  - Its belief errors are 4–40 times smaller. Its worst step at N = 64 (0.09) is close to the sampled filter's floor at N = 1,024–4,096 (0.06–0.07 on the hard histories).
  - In decisions, its Q error at N = 64 is close to exact (0.18 against 0.11 at k = 4).
- **It is also cheaper.**
  - Mid-episode at k = 4, N = 64 is 14 times faster than exact beliefs and 4 times faster than the sampled IPF.
  - From two-atom priors it costs the same as exact beliefs, because nothing exceeds N. The sampled IPF costs 5–24 times more there, because it always carries N samples.
- **So bounded beliefs now pay off inside the region where the exact I-POMDP is tractable,** and can be scored there. This replaces the Section 7 conclusion for the planner.
- **One problem differs from exact at H4 with 4,000 modeled simulations** (97% against 100%). Its loss, 1.53, is the exact-belief miss at low modeled budgets. The likely cause is that compressed nested beliefs change the modeled opponent's search seeds, so this is opponent-model noise; it has not been tested separately.
- **RTS with compressed beliefs keeps its exact-belief accuracy at exact-belief cost.** Its misses are therefore its own search error, and the sampled IPF added cost without changing them.

**Beyond the exact planner: k = 6** (same problems; n = 30 per cell; `results/l2_compressed_k6_20261002`):
- Raising the exact reference's enumeration guard to 50 million branches (`oracle.REFERENCE_BRANCHES`) made it complete at k = 6, in 146 s for one problem.
- Exact-belief I-POMCP did not finish one decision there in 30 minutes (n = 1).

| Planner | Beliefs | Optimal | Mean max Q error | Planning CPU | Support |
|---|---|---|---|---|---|
| I-POMCP 1,000 | compressed 64 | 100% | 1.02 | 2.9 s | 38 |
| I-POMCP 1,000 | compressed 256 | 100% | 0.22 | 7.6 s | 96 |
| RTS K = 4 | compressed 64 | 97% | 1.66 | 2.7 s | 38 |

- **Not yet measured:** L3.

## 9. Estimator revision: each step's state is drawn from b_h (October 2, 2026)

**What changed.** Sections 1–8 were run with root sampling: one particle drawn at the root and carried down the tree. Since commit `f793712`, every tree and rollout step draws its interactive state from the agent's history posterior b_h.
- With exact b_h the two have the same distribution.
- With bounded b_h only the new form is consistent: a carried particle can sit on an atom that compression removed from b_h. With deterministic sensors its observation then has zero evidence. This happened on Wumpus (Section 10).

**L1 gate** (Section 1's design, n = 33 per cell; `results/l1_qualification_belief_sampling_20261002`):
- 100% optimal at every budget and horizon, except H5 at 100 simulations: 94% (previously 97%).
- Mean max Q error at 1,000 simulations: 0.09, 0.11, 0.57 and 1.67 at H2–H5 (previously 0.08, 0.13, 0.54 and 2.14).

**L2 re-runs** (n = 30 per cell; `results/l2_revalidate_belief_sampling_20261002`):

| Problem | Beliefs | Optimal now (before) | Mean max Q error now (before) |
|---|---|---|---|
| H3, k = 0 | exact | 100% (100%) | 0.13 (0.15) |
| H3, k = 2 | exact | 100% (97%) | 0.12 (0.14) |
| H3, k = 4 | exact | 100% (100%) | 0.10 (0.11) |
| H3, k = 2 | compressed 64 | 97% (100%) | 0.26 (0.25) |
| H3, k = 4 | compressed 64 | 100% (100%) | 0.17 (0.18) |
| H3, k = 6 | compressed 64 | 97% (100%) | 1.03 (1.02) |
| H3, k = 2 | compressed 256 | 100% (97%) | 0.12 (0.15) |
| H3, k = 4 | compressed 256 | 100% (100%) | 0.11 (0.12) |
| H3, k = 6 | compressed 256 | 97% (100%) | 0.20 (0.22) |
| H4, root 250, modeled 100 | exact | 80% (93%) | 2.55 (1.56) |
| H4, root 250, modeled 100 | compressed 64 | 80% (90%) | 2.54 (1.51) |
| H4, root 1,000, modeled 100 | exact | 80% (93%) | 2.45 (1.21) |
| H4, root 250, modeled 4,000 | exact | 100% (100%) | 1.04 (1.08) |
| H4, root 250, modeled 4,000 | compressed 64 | 100% (97%) | 0.98 (0.97) |

**Interpretation:**
- **Within the sampling variation of n = 30, the revision reproduces every result except one.** The exception is the H4 rate at 100 modeled simulations, 93% → 80%.
- **That rate is a property of a few modeled opponents.** A modeled j-L1 at 100 simulations is one random search per distinct belief. The revision changed the random streams, so different modeled policies came out, and one wrong opponent policy is shared by several problems.
- **This strengthens Section 5's rule.** At 100 modeled simulations, H4 decisions depend on luck in the opponent models. At 4,000 they are 100% optimal for exact and compressed beliefs alike.
- **Compressed 64 now equals exact beliefs on every H4 cell** (same optimal rates and losses), consistent with nothing exceeding 64 atoms on those problems.

## 10. Wumpus pilot: exact real beliefs, compressed search (L1, H3)

**Setup (pilot).**
- The human (L1) against a uniform-random wumpus, on the full 4 × 4 prior (94,500 states).
- The human's real beliefs are exact; each search starts from the belief compressed to 64 or 256 atoms (`SolverBank(search_particles=N)`).
- The planner is I-POMCP with 1,000 simulations at depth 3, deciding after 0, 2 or 4 real steps of a scripted history.
- The reference is the exact best response at H3 on the exact belief.
- n = 18 decisions over 3 seeds. Data in `results/wumpus_split_pilot_20261002`.

**Measured:**
- **Exact filtering is cheap.** Along one history the support shrank from 94,500 to 46 atoms in 8 steps, at most 2.9 s per update (`results/wumpus_exact_filtering_20261002`).
- **The exact reference is tractable at H3:** 520–600 s at the prior, 92–142 s after 2–4 steps.
- **Planning took 1.2–34 s per decision.**
- **Losses were 0 in 5 of 18 decisions and 1.8–7.7 in the rest; max Q errors were 2.6–89** (rewards range over ±1,000).

**Status: not qualified.** The split removes the failure mode but is not accurate yet. The likely causes, not yet separated, are:
- compressing a 25,920–94,500-atom belief to 64–256 atoms;
- 1,000 simulations over Wumpus's 5 actions and 32 observations.

The next rung is a qualification grid over N and simulations against the exact reference.

## 11. Post-search chance scoring (October 2, 2026)

**What changed** (commit `192ea90`, `planning.chance_scoring`). After search, the planner rescores its tree bottom-up, weighting each observation by its exact probability P(o, alive | b_h, a) under the agent's belief at that history, instead of by how often search sampled it (THEORY.md). Two cases keep their approximate values:
- an action with an observation search never sampled keeps its **search value** (a "fallback");
- a branch reached once keeps its rollout estimate.

The rescored values decide, for real decisions and for modeled opponents alike.

**Design.** Search itself (exploration, backups, random streams) is unchanged, and every decision records both values:
- **"search":** the empirical-backup value, i.e. what the planner decided on before this change;
- **"scored":** the rescored value.

The "same-tree" columns compare the two on identical trees. The search columns reproduce Sections 1 and 9 exactly.

**L1 gate** (Section 1's design, n = 33 per cell; `results/l1_qualification_chance_scoring_20261002`). Mean max Q error, search → scored:

| Simulations | H2 | H3 | H4 | H5 |
|---|---|---|---|---|
| 100 | 0.29 → 0.02 | 0.45 → 0.38 | 1.16 → 1.17 | 2.43 → 3.52 |
| 1,000 | 0.09 → 0 | 0.11 → 0.004 | 0.57 → 0.61 | 1.67 → 1.57 |
| 5,000 | 0.05 → 0 | 0.05 → 0 | 0.34 → 0.38 | 1.46 → 1.50 |

("0" means below 1e-13.)
- **Decisions:** 100% optimal everywhere except H5 at 100 simulations, 94% → 91%. On the same trees, 0 decisions were corrected and 1 regressed, out of 297 at H3–H5.
- **Per action (H4, 5,000 simulations):** the mean Q error of listening, the most-sampled action, halves (0.078 → 0.038). The rarely sampled door openings get slightly worse (0.25 → 0.27, 0.17 → 0.19). Their children have only one or two visits, and exact weights give those noisy estimates their full probability.

**L2 panel** (Section 9's 15 cells, n = 30 per cell, against the exact I-POMDP; `results/l2_chance_scoring_20261002`). All values are search → scored:

| Problem | Beliefs | Optimal | Mean max Q error | CPU (s) |
|---|---|---|---|---|
| H3, k = 0 | exact | 100% → 100% | 0.13 → 0.05 | 0.33 → 0.25 |
| H3, k = 2 | exact | 100% → 100% | 0.12 → 0.08 | 2.1 → 1.9 |
| H3, k = 4 | exact | 100% → 100% | 0.10 → 0.04 | 23 → 22 |
| H3, k = 2 | compressed 64 | 97% → 97% | 0.26 → 0.24 | 1.5 → 1.4 |
| H3, k = 4 | compressed 64 | 100% → 100% | 0.17 → 0.14 | 1.6 → 1.6 |
| H3, k = 6 | compressed 64 | 97% → 93% | 1.03 → 1.01 | 2.7 → 4.2 |
| H3, k = 2 | compressed 256 | 100% → 100% | 0.12 → 0.08 | 2.3 → 2.2 |
| H3, k = 4 | compressed 256 | 100% → 100% | 0.11 → 0.05 | 4.4 → 4.1 |
| H3, k = 6 | compressed 256 | 97% → 100% | 0.20 → 0.16 | 5.9 → 8.0 |
| H4, root 250, modeled 100 | exact | 80% → 80% | 2.55 → 2.54 | 0.8 → 0.8 |
| H4, root 250, modeled 100 | compressed 64 | 80% → 80% | 2.55 → 2.58 | 0.9 → 0.9 |
| H4, root 1,000, modeled 100 | exact | 80% → 80% | 2.45 → 2.37 | 1.1 → 1.0 |
| H4, root 1,000, modeled 100 | compressed 64 | 80% → 80% | 2.45 → 2.39 | 1.2 → 1.2 |
| H4, root 250, modeled 4,000 | exact | 100% → 100% | 1.04 → 1.05 | 35 → 33 |
| H4, root 250, modeled 4,000 | compressed 64 | 100% → 100% | 0.98 → 0.98 | 34 → 33 |

CPU columns are the previous run (Section 9) and this run. The k = 6 runs shared the machine with unrelated jobs.

- **Same tree:** 1 decision corrected and 1 regressed, out of 450. Both are the same near-tie problem (k = 6, seed 1, p = 0.05: exact Q of listen 3.86, open-right 4.19). In both, open-right kept its search value (fallback) while listen was rescored, so the root compared a scored value against a search value.
- **Modeled opponents:** scoring them did not change any H4 result. Every search column equals Section 9 to the last digit, so no modeled policy's greedy action changed.
- **Exactness flag:** the scored root was never certified exact in these panels. Once-visited deep histories leave actions untried, so the flag is conservative.

**Rejected variant.** Instead of falling back, weight the sampled branches by their exact probabilities, renormalized to the continuation mass. On the L1 gate it gave the same decisions and more upward bias (listen at H4, 1,000 simulations: bias +0.04 → +0.13). Script: `rejected_renormalized_variant.py` in the L1 results folder.

**Verdict.**
- **Decisions are neutral within this n.** Across the two gates, 1 was corrected and 2 regressed out of 747 paired decisions. The legacy evidence (4 corrected, 0 regressed) was on a different estimator and does not transfer.
- **Q accuracy improves where branches are well sampled:** H2–H3 Q error falls 30–100%, and complete subtrees are exact.
- **It does not help at H4–H5**, where error is dominated by under-sampled actions and opponent models.

Kept as the default because it improves the accuracy of reported Q values at no decision cost. The comparison stays one commit away, since every row records both values.

## 12. Wider n (October 2, 2026)

Sections 1–11 rest on 30–33 decisions per cell. This section re-measures the questions those cells left open, with more problems.

### 12.1 L1 H5 Q error: no regression from per-step belief sampling

**Question.** At 5,000 simulations, L1 H5 mean max Q error was 0.88 in Section 1, 1.46 after per-step belief sampling (Section 9's code) and 1.50 with chance scoring (Section 11). Did `f793712` make long-horizon values worse?

**Design.** Section 1's problem (11 beliefs) at H5, 1,000 and 5,000 simulations, seeds 0–19: n = 220 per cell. Three arms, paired by (seed, belief):
- **before:** `f793712^` (one root particle carried down the tree);
- **after:** `f793712` (each step's state drawn from b_h);
- **HEAD:** `6f6c49d` (after + chance scoring).

The before and after arms ran from temporary worktrees of those commits. `results/gap2_belief_sampling_paired_20261002`.

**Results** (mean max Q error; differences are paired, with 95% t-intervals):

| Simulations | before | after | HEAD (scored) | after − before | HEAD − after |
|---|---|---|---|---|---|
| 1,000 | 1.71 | 1.47 | 1.52 | −0.24 [−0.53, +0.06] | +0.05 [−0.02, +0.12] |
| 5,000 | 1.05 | 1.03 | 1.14 | −0.02 [−0.22, +0.18] | +0.11 [+0.05, +0.17] |

Every decision in every arm is optimal (220/220; 95% Clopper–Pearson lower bound 98.3%).

- **No regression from `f793712`.** On seeds 0–2, the arms reproduce the earlier figures exactly (0.88, 1.46/1.67 and 1.50). Per-seed means at 5,000 simulations range from 0.56 to 2.15, so three-seed averages cannot resolve differences of this size.
- **Chance scoring slightly raises L1 H5 Q error at 5,000 simulations** (+0.11; Wilcoxon p = 0.0005). HEAD's search values equal the after arm's to the last digit, so this is scoring alone, on identical trees (`scoring_on_identical_trees.py`). Every one of the 440 roots has at least one action that falls back to its search value. This corrects Section 11's "flat at H4–H5": at H5 scoring is mildly harmful to Q values, though not to decisions.
- **The long-horizon plateau is real, and it is heavy-tailed.** Mean max Q error at H5 falls only from 1.47 to 1.03 between 1,000 and 5,000 simulations, and single decisions reach errors of 6–12. Section 12.2 locates it.

### 12.2 The long-horizon Q plateau is mostly the metric

**Question.** Mean max Q error barely falls with budget at long horizons (L1 H5: 1.47 → 1.03 from 1,000 to 5,000 simulations, Section 12.1; L2 H4 about 1.0 in 100%-optimal cells). Where does the error enter?

**Design.** L1 H5, seeds 0–4 × Section 1's 11 beliefs (n = 55 problems per budget). Each root action's search Q error is split exactly into immediate, chance and continuation error (`analysis.error_attribution.describe_node`). `results/q_plateau_diagnosis_20261002/root_attribution.py`.

**Findings.**
- **The error is continuation error.** Immediate error is 0, since E[R | b_h, a] is exact, and chance error is at most 0.007 in mean. Every root action sampled all 6 observation branches, and no child was a rollout frontier.
- **It sits on clearly wrong actions.** The eight largest errors (2.2–5.7) are all on the door the tiger is likely behind (exact Q −79 to −99). These eight had 121–205 of 5,000 visits (door openings average 800–1,200), as UCB should give them, so the value of the 4-step game after the reset is estimated from few samples. "max_a |Q̂ − Q*|" takes its maximum over these actions.
- **The decision-relevant value converges.** Q error of the optimal action(s) falls with budget, while the all-action maximum barely moves:

| L1 H5 (n = 55) | 1,000 simulations | 5,000 simulations |
|---|---|---|
| Max Q error over all actions | 1.57 | 1.35 |
| Q error of the optimal action | 0.40 | 0.12 |

- **L2 shows the same pattern** (Section 11's rows, n = 30 per cell). Optimal-action Q error vs the all-action maximum:
  - H4, modeled 4,000: 0.44 vs 1.05;
  - H3, k = 6, compressed 64: 0.30 vs 1.01;
  - H4, modeled 100: 0.85–1.09 vs 2.4–2.6, where opponent-model error also enters.
- **Chance scoring is neutral on the optimal action** (0.120 → 0.120 at 5,000 simulations). Its +0.11 in Section 12.1 is on the poorly sampled wrong actions, where every root falls back on at least one action.

**Consequence for contribution 2.** Error bounds should be reported on the decision-relevant value: optimal-action Q error, or the value error |max Q̂ − V*|. The all-action maximum is still recorded, but it measures how well search has valued actions it rightly neglects.

### 12.3 L2 decisions over 380 problems per cell

**Design.** Six cells of Sections 9 and 11, each over 19 beliefs of i (0.05, 0.10, …, 0.95) × 5 private beliefs of j (0.085, 0.3, 0.5, 0.7, 0.915) × seeds 0–3, so n = 380 per cell (was 30). With k > 0, each seed also draws a different history (states, j's actions, i's observations; i's actions follow `oracle.HISTORY_ACTIONS`). Reference: the exact I-POMDP. Intervals are exact 95% Clopper–Pearson. `results/l2_wide_n_20261002` (launcher inside).

Exact beliefs at H3, k = 4 were left out: one problem took 279 s of planner CPU (Section 8's mean was 22 s), and compressed 64 is the planner default.

| Problem | Beliefs | Optimal | 95% interval | Mean max Q error | Optimal-action Q error | CPU (s) |
|---|---|---|---|---|---|---|
| H3, k = 0 | exact | 380/380 | 99.0–100% | 0.05 | 0.05 | 0.3 |
| H3, k = 2 | exact | 379/380 | 98.5–100% | 0.09 | 0.06 | 2.7 |
| H3, k = 2 | compressed 64 | 373/380 | 96.2–99.3% | 0.21 | 0.14 | 1.6 |
| H3, k = 4 | compressed 64 | 380/380 | 99.0–100% | 0.13 | 0.13 | 1.8 |
| H4, root 250, modeled 4,000 | exact | 380/380 | 99.0–100% | 1.00 | 0.48 | 36 |
| H4, root 1,000, modeled 100 | exact | 370/380 | 95.2–98.7% | 1.45 | 0.46 | 2.2 |

**Findings.**
- **H3 decisions are near-optimal at this n.** The 8 misses (H3, k = 2) are near ties: each lost exactly the gap to the second-best action, 0.11–0.40.
- **The H4 misses at modeled 100 are confined to the four "corner" problems,** where both i (0.05 or 0.95) and j (0.085 or 0.915) are confident:
  - corner problems: 6/16 optimal (95% interval 15–65%);
  - every other problem: 364/364 (99.0–100%).
  The old grid's 80% (Sections 9 and 11) came from its problem mix: corners were 4 of its 15 problems, but are 4 of 95 here. A headline rate over a belief grid depends on the grid, so this cell is reported split. At modeled 4,000 the corners are 16/16. This is Section 5's opponent-model error, localized: j-L1 at a confident belief is mis-solved by 100 simulations.
- **Chance scoring** corrected 5 decisions and regressed 1 out of 2,280, all near ties.
- **Optimal-action Q error is 2–3× below the all-action maximum at H4** (Section 12.2).

### 12.4 Modeled budget, qualified at the beliefs j is modeled at

**Question.** Section 12.3's H4 misses at modeled 100 sit on the corner problems. Are they wrong j-L1 decisions, and what modeled budget removes them?

**Probe** (`results/modeled_budget_probe_20261002/probe.py`). On the 16 corner problems (H4, root 1,000, modeled 100), every j-L1 belief that i's search asked a modeled policy for was logged: 480 beliefs, 12–60 per problem. Each was re-solved by j-L1 I-POMCP at several budgets and scored against the exact L1 agent (depth 4). Re-solving at 100 reproduced the in-search policy in 480/480 cases.

| Modeled simulations | Wrong j-L1 decisions (of 480) |
|---|---|
| 100 | 10 |
| 400 | 6 |
| 1,600 | 0 |
| 4,000 | 0 |

All 10 wrong decisions at 100 fall in the problems where i's decision was wrong. In the problems i got right, 0 of 360 were wrong. They occur where j is confident but not certain (P(tiger left) in 0.05–0.15 or 0.85–0.95), where listening and opening are close for j.

**End to end** (wide grid, n = 380 per cell, root 1,000; `results/l2_h4_modeled_sweep_20261002`, with modeled 100 from 12.3):

| Modeled simulations | Optimal | 95% interval | Corners | Optimal-action Q error | CPU (s) |
|---|---|---|---|---|---|
| 100 | 370/380 | 95.2–98.7% | 6/16 | 0.46 | 2.2 |
| 400 | 374/380 | 96.6–99.4% | 10/16 | 0.35 | 4.9 |
| 1,600 | 380/380 | 99.0–100% | 16/16 | 0.19 | 13.7 |

**Conclusion.** Qualifying j-L1 at the beliefs it is modeled at predicts the budget that i's decisions need: 1,600 here, not the 4,000 used so far. H4 with root 1,000 and modeled 1,600 is 100% optimal at 2.6× less CPU than root 250 and modeled 4,000 (13.7 s vs 36 s), with lower optimal-action Q error (0.19 vs 0.48).

### 12.5 Modeled-agent qualification over the whole grid (October 3, 2026)

**Design.** `experiments.modeled_qualification` (the 12.4 probe as a driver). For each L2 problem (19 beliefs of i × 5 of j × seeds 0–1, n = 190 problems per horizon), i searches with root 1,000 and modeled 100. Every j-L1 belief that i's search queries is recorded and re-solved over a budget ladder, then scored against the exact L1 agent. On every query, the driver checks that the rung equal to 100 reproduces the policy i used. `results/modeled_qualification_20261003` and `results/modeled_qualification_h4_6400_20261003` (6,400 rung).

| Horizon | j-L1 simulations | Wrong / queries | Distinct beliefs wrong / queried | Wrong in problems i got wrong | Max j loss |
|---|---|---|---|---|---|
| 3 | 100 | 19 / 3,325 | 1 / 87 | 0 | 0.46 |
| 3 | 400 | 0 / 3,325 | 0 / 87 | 0 | 0 |
| 4 | 100 | 152 / 8,284 | 6 / 235 | 6 | 1.01 |
| 4 | 400 | 95 / 8,284 | 4 / 235 | 4 | 0.66 |
| 4 | 1,600 | 38 / 8,284 | 2 / 235 | 0 | 0.32 |
| 4 | 6,400 | 0 / 8,284 | 0 / 235 | 0 | 0 |

**Findings.**
- **Mistakes cluster on a few beliefs.** At H4 and 100 simulations, 152 wrong queries come from 6 distinct beliefs.
  - The harmful one is j's **prior** (P(tiger left) = 0.085 or 0.915). j opens a door where the exact choice is to listen (Q −0.67 vs −0.01). The prior is queried in every problem with that j.
  - The two left at 1,600 (P(tiger left) = 0.077 or 0.924) are near ties: j listens where opening is better by 0.20–0.32. i is unaffected.
- **j's own loss does not separate harmful from harmless mistakes.** Harmful ones lose 0.66; harmless ones lose 0.2–1.0. No loss tolerance selects the budget i needs.
- **Zero wrong modeled decisions is the qualifying rule.** It is sufficient for i to be unaffected, and it uses only the modeled level's exact reference. That makes it usable where the higher level's exact solution is out of reach (L3 and beyond). It certifies 400 at H3 and 6,400 at H4 (0/8,284; correct rate ≥ 99.96%).
- **It is conservative.** At H4, i's decisions are already 380/380 at 1,600 (12.4), with 2 harmless near-tie beliefs left. Where the higher level's exact reference exists, it can lower the budget, as in 12.4. Where it does not, the certified budget is the honest choice.

### 12.6 Why near-tie modeled decisions need thousands of simulations

**Question.** At H4, wrong j-L1 queries fall 152 → 95 → 38 → 0 from 100 to 6,400 simulations (12.5). That is a 64× budget for a few near-tie beliefs. Is this a bug, a poor allocation, or estimator variance?

**Design.** j-L1 (depth 4) at P(tiger left) = 0.0767: exact Q is open-left 0.246, listen 0.043, open-right −92.9. Budgets 100–6,400 × seeds 0–19 (n = 20 per budget). Root visits and Q estimates are logged per action. `results/near_tie_diagnosis_20261003/near_tie.py`.

| Simulations | Correct | Visits listen / open-left / open-right | Search Q(open-left): mean ± sd | Scored Q(open-left): mean ± sd |
|---|---|---|---|---|
| 100 | 11/20 | 42 / 43 / 15 | 0.35 ± 1.00 | 0.30 ± 1.09 |
| 400 | 12/20 | 181 / 183 / 36 | 0.30 ± 0.50 | 0.28 ± 0.43 |
| 1,600 | 18/20 | 761 / 769 / 70 | 0.26 ± 0.22 | 0.26 ± 0.19 |
| 6,400 | 20/20 | 3,120 / 3,169 / 112 | 0.24 ± 0.07 | 0.27 ± 0.03 |

**Findings.**
- **No bug.** Estimates centre on the exact values and their spread shrinks as 1/√n.
- **No allocation problem.** UCB splits visits evenly between the two contenders and spends little on the clearly wrong door. Simple-regret root selection, such as Sequential Halving (Karnin et al. 2013, as used at the root of Gumbel MuZero, Danihelka et al. 2022), would allocate the same way here, so it was not tried.
- **The floor is return variance.** About 6.5 per simulation means a 0.2 gap needs thousands of visits per action. Only variance reduction moves this floor. Chance scoring already cuts the spread 2.5× at 6,400 (0.07 → 0.03).
- **Consequence.** A flat modeled budget pays the near-tie price at every queried belief, even though most beliefs are clear. Adaptive per-belief budgets would cut that cost: stop when the decision is statistically separated, spend more on near ties. Such a rule is not designed or tested yet.

### 12.7 First level-3 decisions against the exact L3 I-POMDP (October 3, 2026)

**Design.** `experiments.panel` at level 3: i-L3 models j-L2, who models i-L1, who faces uniform L0 j. Point beliefs: 9 of i (0.1–0.9) × 3 of j × 3 of the modeled i-L1 (0.085, 0.5, 0.915) × seeds 0–1, n = 162 per cell. Root 1,000; modeled 0 means every lower level is exact. Before the run, the L3 reference was checked against `build_exact_agent`'s independently assembled hierarchy: equal to 0.0 at H3. `results/l3_panel_20261003`.

| Horizon | Modeled simulations | Optimal | 95% interval | Optimal-action Q error | Max Q error | CPU (s) |
|---|---|---|---|---|---|---|
| 2 | exact | 162/162 | 97.7–100% | 0 | 0 | 0.16 |
| 3 | exact | 162/162 | 97.7–100% | 0.06 | 0.42 | 17 |
| 3 | 100 | 162/162 | 97.7–100% | 0.17 | 0.48 | 14 |
| 3 | 400 | 162/162 | 97.7–100% | 0.06 | 0.42 | 26 |

- **Every L3 decision checked is optimal**, with exact or finite modeled opponents.
- **At 400 simulations, every modeled j-L2 policy equals the exact one.** i's trees and values are identical to the exact-opponent cell to the last digit. At 100, 144 of 162 problems are identical, and Q error rises (0.17 vs 0.06) without changing a decision.
- **Mid-episode exact checks reach their limit at L3.** All four H3, k = 2 jobs exceeded the 4 GB memory limit. One such problem alone ran over 30 minutes and reached a 5 GB peak. At H2, k = 2 takes 6–8 s, but i's support there is 32 atoms, so compressed 64 equals exact and nothing is tested. H2 at k = 4 with exact beliefs did not finish in 26 minutes.
- **Tractable frontier for exact L3 decision checks (Tiger): H3 at k = 0 and H2 at k ≤ 2.** Compressed-belief error at L3 must be measured as belief error against exact filtering (`belief_qualification`), not as decisions.

### 12.8 Sequential replicate searches for modeled policies (October 3, 2026)

*Superseded by 12.11: this rule is biased for j-L2 at L3 H4 (12.10) and was replaced by a growing-tree rule. The `6400s` stacks of 12.8–12.10 are this old rule.*

**Design change** (commit `c0cba21`, `planning.replicates`, opt-in through `PlannerConfig.modeled_stopping`). A modeled policy is solved by independent replicate searches of 200 simulations, a form of root parallelization (Chaslot et al. 2008). After at least 3 replicates, the solve stops once the two best mean scored values are 3 standard errors apart, which is fixed-confidence best-arm identification (cf. LUCB, Kalyanakrishnan et al. 2012). The cap is `modeled_simulations`. Search seeds exclude the rule, so single-search streams and all earlier results are unchanged. Motivation: 12.6 shows the near-tie floor is return variance, so spending adaptively is the lever.

**Offline A/B** (235 distinct j-L1 beliefs queried at L2 H4 × seeds 0–4, n = 1,175; `results/sequential_replicates_ab_20261003`):

| Modeled solve | Correct | Mean simulations |
|---|---|---|
| single search, 400 | 1,171/1,175 | 400 |
| single search, 1,600 | 1,175/1,175 | 1,600 |
| single search, 6,400 | 1,175/1,175 | 6,400 |
| replicates with stopping, cap 1,600 | 1,172/1,175 | 627 |
| replicates with stopping, cap 6,400 | 1,175/1,175 | 699 |

At an equal 1,600 cap, replicates are worse on near ties (17/20 vs 20/20), because small trees lose depth. The gain comes from adaptivity: about 4,070 simulations on the 4 near-tie beliefs, about 600 elsewhere.

**Gate on i's decisions** (L2 H4, root 1,000, the 380-problem grid of 12.3; `results/l2_h4_stopping_gate_20261003`):
- **Accuracy.** Stopping with cap 1,600 or 6,400 is 380/380 optimal (99.0–100%), the same as flat 1,600. Optimal-action Q error is 0.16 vs 0.19.
- **Cost, paired.** For 40 problems, both arms ran back to back in one process, in alternating order. i's decision CPU with stopping (cap 6,400) is 0.47× flat 1,600 (median 0.47; Wilcoxon p = 2×10⁻¹²): 9.8 s vs 20.6 s under that machine load. Per modeled solve it is 0.16 s vs 0.37 s, with 3.5 replicates on average and no per-replicate overhead.
- **The unpaired panel CPU is not comparable.** The gate panel showed 13.1–13.8 s against 13.7 s for arm A, but arm A ran on a quieter machine (another project's 8 jobs were running during the gate). CPU comparisons across runs at different load are invalid; compare only paired, same-process measurements.

**L3 gate** (L3 H3, root 1,000, stopping with cap 1,600 for both modeled levels, the 162 problems of 12.7; `results/l3_stopping_gate_20261003`):
- **Accuracy.** 162/162 optimal. Values are identical to the exact-opponent cell, so every modeled policy matched the exact one.
- **Cost, paired** (16 problems, alternating order). Stopping costs 1.30× flat 400 (median 1.23; Wilcoxon p = 3×10⁻⁵): 37.6 s vs 29.5 s. Flat 400 already reproduces the exact policies at L3 H3 (12.7), while the rule's floor is 3 replicates × 200 = 600 simulations. That floor applies at each nested level.

**Verdict.**
- **Use stopping where the qualified flat budget is well above its 600-simulation floor.** At L2 H4 it halves CPU at equal accuracy, and caps near-tie error at the level flat 6,400 reaches (12.5).
- **Use a flat budget where a small one already qualifies.** At L3 H3, flat 400 matches exact and the rule costs 1.3×.
- **It stays opt-in, chosen per modeled level.** It is not a default.

### 12.9 Per-level modeled solves, and the first level-3 modeled qualification (October 3–4, 2026)

**Design change** (commits `2700e7a`, and the oracle's `reference="none"`). `experiments.modeled_levels` names each modeled level's solve, top-down, with one label per level: `400` (one search of 400), `6400s` (sequential stopping, cap 6,400; 12.8), `exact`. At level 3, `6400s/100` solves j-L2 with stopping and the i-L1 that j models with a flat 100. `oracle`, `panel` and `modeled_qualification` build the hierarchy from that label. `modeled_qualification` now also runs at level 3: it scores j-L2 at the beliefs i-L3's search queries, against the exact L2 agent, and each ladder rung is a whole stack. Its qualifying rung is the cheapest one, by mean CPU per query, with zero wrong decisions. Only the modeled level's exact reference is needed, so `score_decision=False` drops i's level-3 reference where it is intractable.

**L3 H3 qualification** (12.7 grid: 9 beliefs of i × 3 of j × 3 of the modeled i-L1 × seeds 0–1, n = 162 problems; root 1,000; in-search stack `400`; 3,294 queried j-L2 models, 183 distinct). `results/l3_modeled_qualification_h3_20261003` and `results/l3_modeled_qualification_h3_ladder2_20261004`.

| j-L2 / i-L1 | Wrong / queries | Distinct beliefs wrong | 95% interval, correct | CPU per query (s) |
|---|---|---|---|---|
| 100 / 100 | 63 / 3,294 | 6 | — | 0.65 |
| 100 / 400 | 63 / 3,294 | 6 | — | 1.09 |
| 400 / 100 | 36 / 3,294 | 4 | 98.5–99.2% | 1.01 |
| 400 / 400 | 36 / 3,294 | 4 | 98.5–99.2% | 1.45 |
| 1,600 / 100 | 9 / 3,294 | 1 | 99.5–99.9% | 1.71 |
| **6,400 stopping / 100** | **0 / 3,294** | **0** | **99.9–100%** | **1.28** |

CPU is comparable only within a run: rows 1–5 ran in one run, the last two in the second.

- **i's decisions were optimal in every problem (162/162).** All the wrong j-L2 queries are harmless to i. This agrees with 12.7, which scored i's decisions, not j's.
- **i-L1's budget inside j-L2 does not matter at H3.** At 100 and at 400, every j-L2 policy is identical (3,294 of 3,294 queries), so `/100` is enough and costs about 30% less.
- **The j-L2 mistakes are near ties.** The 4 beliefs still wrong at 400 have P(tiger left) ≈ 0.03 or 0.97, with exact gaps of 0.08–0.15 between listening and opening the safe door. Re-solved over 10 seeds (`results/l3_near_tie_probe_20261003`), flat 400 is wrong 12/40, while flat 1,600 and flat 6,400 are 0/40, with i-L1 exact or at 400 alike. That is search noise, not a bug.
- **Certified L3 H3 stack: `6400s/100`.** It has zero wrong queries, and it costs 0.75× flat 1,600, which still has one wrong belief.

**Stopping floor A/B** (`results/stopping_floor_ab_20261004`). With 3 replicates, the stopping test's standard error has 2 degrees of freedom, so a 3-SE separation occurs by chance with two-sided probability ≈ 0.095. Is the floor too low?

| Minimum replicates | L2 H4 (n = 1,175): wrong, mean simulations | L3 H3 near ties (n = 40): wrong, mean simulations |
|---|---|---|
| 3 (current) | 0, 695 | 1, 4,570 |
| 4 | 0, 894 | 1, 4,955 |
| 5 | 0, 1,091 | 1, 5,245 |

The one wrong case (gap 0.083) reached the 6,400 cap in every arm, so it is a cap limit, not an early stop. Raising the floor costs 29–57% more simulations and corrects nothing, so **the floor stays at 3**. Lowering it to 2 replicates is unsound: the SE has 1 degree of freedom and a 3-SE separation occurs by chance about 20% of the time. It was not run.

### 12.10 Level-3 horizon-4 qualification: the stopping rule is biased for j-L2 (October 4–5, 2026)

**Design.** `modeled_qualification` at L3 H4 without i's exact L3 reference (`score_decision=False`), which is intractable here. Grid: 3 beliefs of i (0.1, 0.5, 0.9) × 3 of j × 3 of the modeled i-L1 × seed 0, n = 27 problems; root 1,000; in-search stack `400/100`. The 9 problems with nested belief 0.5 are in `results/l3_modeled_qualification_h4_20261004`. The 18 with a confident nested belief (0.085 or 0.915) are in `results/l3_modeled_qualification_h4_confident_20261004`, run with the two-phase driver below.

| Problems | j-L2 / i-L1 | Wrong / queries | Distinct beliefs wrong / queried | Max j loss |
|---|---|---|---|---|
| nested 0.5 (9) | 400 / 100 | 0 / 633 | 0 / 211 | 0 |
| nested 0.5 (9) | 6,400 stopping / 100 | 0 / 633 | 0 / 211 | 0 |
| nested 0.085, 0.915 (18) | 400 / 100 | 90 / 1,998 | 30 / 666 | 0.98 |
| nested 0.085, 0.915 (18) | 6,400 stopping / 100 | 24 / 1,998 | 8 / 666 | 0.30 |

Neither stack qualifies at L3 H4. Every wrong query is at a confident j belief (P(tiger left) 0.03–0.07 or 0.93–0.97), where listening and opening the safe door are close (gaps of 0.03–0.30). Confident nested beliefs make i's problem much larger: 103 queries of up to 86 atoms, with i's search taking 2,224 s and the exact L2 reference 140 s per belief, against 67 queries, 50 atoms, 600 s and 50 s at nested 0.5.

**Residual probe** (the 8 beliefs still wrong at `6400s/100`, re-solved over seeds 0–9, n = 80 per row; `results/l3_h4_residual_probe_20261005`):

| j-L2 / i-L1 | Wrong / 80 | Wrong on the three 0.19-gap beliefs / 30 | Mean CPU (s) |
|---|---|---|---|
| stopping, cap 6,400 / 100 | 54 | 18 | 112 |
| stopping, cap 6,400 / exact | 57 | 20 | 93 |
| stopping, cap 25,600 / exact | 51 | 16 | 149 |
| one search, 6,400 / exact | 14 | 0 | 157 |

- **The i-L1 budget is not the cause.** An exact i-L1 gives the same error rate as 100 simulations (57 vs 54 of 80).
- **The stopping rule is biased here, not merely noisy.** On the 0.19-gap beliefs one search of 6,400 is 0/30 wrong, while replicate searches stay 16–20/30 wrong even with a 4× higher cap. Each replicate is a 200-simulation tree, too shallow at depth 4 for j-L2. Averaging many of them converges to the shallow trees' value, and more replicates only make that biased answer more confident. 12.8 noted that small trees lose depth. At L2 H4 (j-L1) and L3 H3 (j-L2) the bias was too small to change a decision, but at L3 H4 it is not.
- **One search of 6,400 still misses the near ties** (gaps 0.03–0.04, 14/80). These are close to exact ties.

**Consequences.**
- The 12.8 rule (replicates of 200, cap 6,400) is not a consistent estimator, and must not be used beyond the cells where it was gated (L2 H4, L3 H3).
- L3 H4 has no certified modeled stack yet. Its j-L2 needs either a consistent adaptive rule or flat budgets above 6,400. Whether any of these j mistakes changes i's decision cannot be checked exactly at L3 H4.

**Engineering for this run** (commits `f2754fa`, `bf0608e` and later):
- `FiniteBelief` unpickling now restores masses bit for bit. Renormalizing them had changed 504 of 549 canonical digests, and so search seeds, which flipped 5 near-tie policies. Earlier results never sent beliefs between processes, so they are unaffected.
- `modeled_qualification` runs in two phases: a search per problem, then a job per distinct (horizon, seed, belief). This is exactly equivalent to the old driver (1,098 of 1,098 rows on 27 L3 H3 problems) at a third of the scoring work.
- The exact reference is built fresh per belief. A shared cache had grown to 9.6 GB.

### 12.11 A consistent stopping rule: one growing tree, calibrated by pilot trees (October 5, 2026)

**Question.** 12.10 showed that averaging 200-simulation replicates converges to a shallow tree's biased value. Can an adaptive modeled solve be consistent, as accurate as flat 6,400 at L3 H4, and cheaper?

**Diagnosis first** (`results/single_tree_bias_20261005`; the 8 residual beliefs of 12.10 × seeds 0–9, n = 80 per checkpoint). One j-L2 tree over an exact i-L1 was grown to 6,400 and chance-scored at 200, 400, …, 6,400:
- On the three 0.19-gap beliefs, wrong decisions fall 13/30 → 13/30 → 8/30 → 9/30 → 5/30 → 0/30.
- The spread of the scored gap across seeds falls from about 1.0 at 200 to about 0.06 at 6,400, roughly n^-0.8. Averaging k replicates reduces spread only as k^-0.5, so depth beats replication on both bias and variance.
- Replaying k = 2–4 independent growing trees on these beliefs: at an equal total budget they are worse than one tree (gap 0.19: 3–4% wrong with three trees of 3,200, against 0/30 with one tree of 6,400).

**Design change** (commit `f5287f5`, `planning.stopping`, which replaces `planning.replicates`):
- **One main tree.** It is seeded exactly as the flat search of the cap and grown by doubling from a pilot size `b`. The decision is always the main tree's, so at the cap the rule *is* the flat search, and it is consistent.
- **A calibrated test.** A tree with Bellman backups has no honest standard error, so `m − 1` independent pilot trees of `b` simulations measure one. With the main tree's checkpoint at `b`, they give σ, the spread over `m` trees of the scored difference between the main tree's two leading actions. At checkpoint n the main tree stops when its gap exceeds t(1 − α/2K, m − 1) · σ · sqrt(b/n). K is the number of checkpoints (Bonferroni). sqrt(b/n) assumes variance falls at least as 1/n (measured: about n^-1.6), so the test is conservative.
- **Defaults:** b = 200, m = 4, α = 0.01. Labels keep their form: `6400s` (default) or `6400s[200,4,0.01]`.

**Offline replay at L3 H4** (`results/l3_h4_tree_trajectories_20261005`). All 666 distinct j-L2 beliefs queried in the 18 confident-nested problems of 12.10 come from the 6 problems with i's belief 0.1, since i's own belief does not change which j beliefs it queries. Each belief has one main tree grown to 6,400 and 3 pilot trees of 200 and 400. Each problem's queries were solved in query order in one bank, as inside i's search, so CPU is the in-situ marginal cost. Rules replayed from these checkpoints give exactly the decisions they would make online with these seeds. n = 666; gap bands are exact gaps between the two best actions.

| Rule | Wrong | gap < 0.05 (8) | 0.05–0.3 (34) | 0.3–1 (68) | ≥ 1 (556) | Mean simulations | In-situ CPU per query (s) |
|---|---|---|---|---|---|---|---|
| flat 400 | 30 | 4 | 12 | 13 | 1 | 400 | 33 |
| flat 1,600 | 15 | 3 | 7 | 5 | 0 | 1,600 | 76 |
| flat 6,400 | 6 | 5 | 1 | 0 | 0 | 6,400 | 178 |
| **stopping, b 200, m 4, α 0.01** | **6** | 5 | 1 | 0 | 0 | 2,584 | **92** |
| stopping, b 200, m 4, α 0.05 | 6 | 5 | 1 | 0 | 0 | 2,211 | 83 |
| stopping, b 400, m 4, α 0.01 | 6 | 5 | 1 | 0 | 0 | 3,162 | 122 |
| stopping, b 200, m 3, α 0.01 | 6 | 5 | 1 | 0 | 0 | 3,189 | 109 |

- **Same decisions as flat 6,400 at 0.52× its CPU.** Every remaining miss runs to the cap, where the rule is flat 6,400. Five are near-exact ties (gaps 0.03–0.04); one has a gap of 0.14.
- **Where the main tree stops:** 334 beliefs at 200, 63 at 400, 47 at 800, 29 at 1,600, 28 at 3,200, and 165 at the cap.
- **CPU is front-loaded.** In situ, a query's first 200 simulations cost about 19 s, while its new nested i-L1 beliefs are solved. Flat 6,400 costs 178 s per query on average and up to 340 s, so it is about 5 CPU-hours per i decision at L3 H4 (about 100 queries). The earlier estimate of "hours" stands; a probe of the first 24 queries (16–20 s each, `results/shared_bank_cost_20261005`) had understated the later ones.

**Online checks of the implementation** (`results/stopping_redesign_gate_20261005`, `results/stopping_redesign_gates_20261005`):

| Set | n | New rule | Reference |
|---|---|---|---|
| L3 H4 residual beliefs × seeds 0–9 (12.10) | 80 | **14 wrong; 0/30 on the 0.19-gap beliefs** | old rule 57 (20/30); flat 6,400 14 (0/30), the same per belief |
| L2 H4 j-L1 beliefs × seeds 0–4 (12.8) | 1,175 | 0 wrong | flat 1,600 0 wrong |
| L3 H3 j-L2 near ties × seeds 0–9 (12.9) | 40 | 0 wrong | flat 1,600 0; old rule 1 |
| L3 H3 qualification, rung `6400s/100` (12.9 grid) | 3,294 queries (183 beliefs) | **0 wrong** (99.89–100%) | old rule 0 wrong |
| i's decisions, L2 H4, root 1,000, modeled `6400s` | 380 | **380/380** (99.0–100%); optimal-action Q error 0.155 | flat 1,600 380/380, 0.19 |
| i's decisions, L3 H3, root 1,000, stack `6400s/100` | 162 | **162/162** (97.7–100%); optimal-action Q error 0.06 | exact opponents 0.06 |

Paired CPU, measured in one process with alternating order and fresh banks:
- **L2 H4 j-L1** (n = 1,175): 0.37 s per solve against 0.44 s for flat 1,600, with 1,127 simulations on average. With α = 0.05 it is 0.28 s.
- **L3 H3 j-L2** (n = 183 beliefs, cold banks): 3.67 s against 3.77 s for flat 1,600/100 (median ratio 0.93, Wilcoxon p = 1×10⁻⁸) and 3.28 s for flat 400/100 (1.08×). In this cold setting flat 1,600/100 has 1 wrong belief and flat 400/100 has 3.

**Verdict.**
- **The rule replaces 12.8's.** It is consistent and as accurate as flat 6,400 wherever it was checked, and it costs 0.5× flat 6,400 in situ at L3 H4 and about 0.9× flat 1,600 at L2 H4 and L3 H3. It is still opt-in, chosen per level.
- **Certified stacks:** `6400s` at L2 H4 and `6400s/100` at L3 H3 (re-gated above).
- **L3 H4 still has no zero-wrong stack.** Its remaining misses are flat 6,400's misses: near-exact ties (gaps 0.03–0.04) and one 0.14 gap. Fixing them takes a larger cap or variance reduction (HANDOFF step 3), not a different stopping rule.

**Also measured, kept out of the code:**
- Memoizing the hashes of frames, models and atoms gives bit-identical results but costs 8% more CPU. Most of these objects are hashed only once or twice.
- A larger exact-reference cache changes nothing: every one of a solve's 173k i-L1 reference nodes is a distinct belief.
- Persistent nested trees (R6) cannot be reused exactly. Every modeled query is a fresh receding-horizon problem of depth H, while a subtree of an earlier j tree has fewer steps left.

### 12.12 Simulating minds, 2.2× faster in situ with outputs bit-identical (October 5, 2026)

**Profile first** (`results/w1_profile_20261005`, cProfile of one confident-nested L3 H4 problem's j-L2 queries in one bank, stack `1600/exact`). In situ, the exact i-L1 best response is 19% of CPU, not the 60% measured earlier on one cold solve. j-L2's own belief updates take 64%. On the later queries (21–40), one function took 33% on its own: `MentalModel` validation, which re-scanned the whole support on every one of 67M constructions.

**Changes** (commits `e17ab33`, `dd56728`, `5454665`):
- `AgentFrame` and `MentalModel` cache their hashes. The pickled state stays the declared fields, and hashes are recomputed on load.
- `FiniteBelief` summarizes its opponents once, so `MentalModel` validation is O(1).
- `InteractiveFilter.advance` memoizes successor models. Atoms built on one successor share it and compare by identity.
- `FiniteBelief` validation is one pass, and equality has an identity shortcut. Frozenset equality walks the support even for the same object.
- The enumerating filter groups rows by atom as it builds them (`FiniteBelief.from_groups`). This is bit-identical because `fsum` is correctly rounded.

Kept out: grouping inside the exact reference (no measurable gain), and memoizing Tiger's kernels (a lookup costs as much as the kernel).

**Gate** (`gate_w1.py`): every chance-scored tree's values, compared with `repr`. The old and new code ran concurrently under the same load.

| Set | n | Identical | CPU, new / old | Wilcoxon p |
|---|---|---|---|---|
| L3 H4 in situ, first 40 queries of problem 0, `1600/exact` | 40 queries | 40/40 | **0.45** (median 0.46; last 20 queries 0.43) | 2×10⁻¹² |
| L3 H3 cold, the 183 queried j-L2 beliefs, `6400s/100` | 183 (26,673 trees) | 183/183 | 0.84 (median 0.84) | 4×10⁻²⁷ |

The speedup grows with belief size, because the removed costs scaled with the support. An i-L3 H4 decision at `6400s/exact` should fall from about 2.5 CPU-hours to about 1.1. That figure is extrapolated from the per-query ratio, not measured on a whole decision.

### 12.13 Are near-tie mistakes of the modeled opponent harmful to i? (October 5, 2026)

**Design** (`results/w2_near_tie_harm_20261005/harm.py`). In i's exact I-POMDP, the directly modeled j level is perturbed adversarially: at every queried belief whose exact gap lies in (tie, ε), j plays its second-best action. i's exact decision under this hierarchy is then scored against i's true exact Q. Problems: L2 H4, the 95 distinct k = 0 problems of the wide grid; L3 H3, the 81 distinct problems of 12.7. ε ∈ {0.05, 0.15, 0.3}. A perturbed run is skipped, exactly, when no queried gap is below ε, since nothing can flip. 0.8 CPU-h.

| Problems | ε | Problems with a flip | Flipped j beliefs | i's decision changed | i's loss |
|---|---|---|---|---|---|
| L2 H4 (95) | 0.05, 0.15 | 0 | 0 | — | — |
| L2 H4 (95) | 0.3 | 38 | 38 | 0 | 0 |
| L3 H3 (81) | 0.05 | 0 | 0 | — | — |
| L3 H3 (81) | 0.15, 0.3 | 36 | 72 | 0 | 0 |

- **No flip of a j decision with gap up to 0.3 changed i's decision**, in 74 problems with 110 flipped beliefs. This holds even with all of a problem's near ties flipped at once.
- **Limit:** these exact hierarchies have no j gaps below 0.05. The 0.03–0.04 ties of L3 H4 (12.11) lie below anything tested here, so calling them harmless would extrapolate.
- **For the certification standard** (HANDOFF 0.1): the evidence supports a PAC-style standard with ε up to 0.3 at L2 H4 and L3 H3. Adopting it is the user's decision. 12.14 attacks the near ties directly.

### 12.14 The near-tie floor was the chance-scoring fallback: exact scoring with two steps left (October 5–6, 2026)

**Question** (PLAN W3). 12.6 attributed the near-tie floor of modeled solves to return variance and suggested variance reduction. Which part of a scored value's spread is it?

**Diagnosis on fixed trees** (`results/w3_variance_20261005`, `a0_frontiers.py`, `a0b_fallbacks.py`). j-L1 at L2 H4 (depth 4) at the 8 queried beliefs with exact gaps 0.2–1.0, seeds 0–29 (n = 60 per gap band). One tree per (belief, seed) at 400, 1,600 and 6,400 simulations. Each tree is scored several ways, so the comparison is exactly paired.

| Gap | Simulations | Spread of the scored gap: planner | frontiers exact | missing branches exact at two steps left | all missing branches exact (oracle) | Wrong: planner / two steps left |
|---|---|---|---|---|---|---|
| 0.205 | 400 | 0.61 | 0.55 | 0.55 | 0.36 | 18 / 13 of 60 |
| 0.205 | 1,600 | 0.25 | 0.25 | **0.03** | 0.02 | 15 / **0** of 60 |
| 0.205 | 6,400 | 0.053 | 0.053 | **0.000** | 0.000 | 0 / 0 of 60 |
| 0.320 | 1,600 | 0.25 | 0.26 | 0.02 | 0.02 | 3 / 0 of 60 |
| 0.656 | 1,600 | 0.26 | 0.26 | 0.02 | 0.02 | 0 / 0 of 60 |
| 1.008 | 1,600 | 0.23 | 0.23 | 0.02 | 0.01 | 0 / 0 of 60 |

- **Rollout noise is not the floor.** Valuing every frontier exactly leaves the spread unchanged from 1,600 simulations on.
- **The floor is whole-action fallbacks.** A tree keeps about 500 fallback (history, action) pairs at 1,600 simulations, nearly all with two steps left. An action falls back when one of its 6 observations was never sampled, and it then keeps its noisy search Q.
- With two steps left, every child is the last decision, whose exact value max_a E[R | b, a] needs only the child's posterior. Valuing those branches exactly cuts the spread 8.5× at 1,600, and the optimal-action Q error falls from 0.15 to 0.00. It is exact backward induction, so it adds no bias. This is unlike renormalizing over the sampled branches (Section 6).

**Design change** (`planning.chance_scoring`): an action with two steps left is scored over all its positive branches, sampled or not. Fallback remains only with three or more steps left. Search, UCB and random streams are unchanged.

**Gates** (new code against the code of 12.12):

| Set | n | Before | After |
|---|---|---|---|
| **L3 H4 residual beliefs × seeds 0–9, `6400s/exact`** (12.10) | 80 | 14 wrong (11/30 at gap 0.03, 3/10 at 0.039) | **4 wrong** (4/30 at gap 0.03, 0/10 at 0.039); 12 fixed, 2 broken, exact McNemar p = 0.013; max loss 0.030 |
| L3 H4 residual, flat `1600/exact` | 80 | — | 24 wrong |
| L2 H4 j-L1 beliefs × seeds 0–4 (12.8), every arm | 1,175 | 0 wrong | 0 wrong |
| L3 H3 j-L2 near ties × seeds 0–9 (12.9), every arm | 40 | 0 wrong | 0 wrong |
| L3 H3 183 queried j-L2 beliefs, `6400s/100`, cold | 183 | 0 wrong | 0 wrong, every policy unchanged |
| i's decisions, L2 H4, root 1,000, modeled `6400s` (`g3_decisions`) | 380 | 380/380; optimal-action Q error 0.155 | **380/380** (99.0–100%); optimal-action Q error **0.117** |
| i's decisions, L3 H3, root 1,000, stack `6400s/100` | 162 | 162/162; optimal-action Q error 0.06 | **162/162** (97.7–100%); optimal-action Q error **0.0006**; 83% of i's scored roots exact |

**Cost.** Scoring now needs posteriors for branches search never sampled. At L3 those posteriors bring new nested i-L1 beliefs to solve. Paired on the 183 L3 H3 beliefs (cold, same load): 1.18× CPU (median 1.20, Wilcoxon p = 5×10⁻²⁸). The residual set ran at 114 s per solve, unpaired, against 146 s before 12.12's speedups.

**Verdict.**
- **Kept as the planner's scoring.** It roughly halves L3 H4's residual misses (14 → 4 of 80, p = 0.013) and leaves no regression on any certified set. Q errors at i's root fall, at L3 H3 by 100×.
- **What remains at L3 H4:** misses only at the 0.03-gap beliefs, with loss ≤ 0.030. Every exact check (12.13) found such mistakes harmless to i, but those checks never contained a gap that small.
- **Follow-up: the stopping rule is now conservative.** Its pilots (200 simulations) calibrate σ and scale it as sqrt(200/n). Scored spreads now collapse much faster than 1/n, so near-tie solves still run to the cap (L2 H4 near ties: 6,760 simulations on average). A rule that also uses the scored values' exactness, or certified intervals (PLAN W6), could stop them far earlier.

### 12.15 The stopping rule scales with the measured rate, 1/n (October 6, 2026)

**Question.** After 12.14, near-tie modeled solves still ran to the cap. The rule scales the pilots' spread at 200 simulations to the main tree's size n as sqrt(200/n), the Monte Carlo rate. Is that still the right rate?

**Dataset** (`results/stopping_v2_20261006/dataset.py`). 2,016 solves, each in a fresh bank:
- L3 H4: all 110 queried j-L2 beliefs with exact gaps below 1, 110 clear ones, and the 8 residual beliefs × seeds 0–9;
- L3 H3: the 183 j-L2 beliefs × seeds 0–2;
- L2 H4: the 235 j-L1 beliefs × seeds 0–4.

Each solve has a main tree scored at 200–6,400 simulations, plus pilot trees. A rule replayed from these checkpoints (`replay.py`) decides exactly as it would online with these seeds.

**Measured rate.** Across 10 seeds at each of the 8 residual beliefs, the spread (sd) of the main tree's scored gap fell as n^−1.0 between 200 and 3,200 simulations (median; range 0.76–1.15).

**Replay** (wrong decisions / mean simulations; the threshold scales as (200/n)^β):

| Rule | L3 H4 (n = 292) | L2 H4 (n = 1,175) | L3 H3 (n = 549) |
|---|---|---|---|
| flat 6,400 | 4 / 6,400 | 0 / 6,400 | 0 / 6,400 |
| β = 0.5 (12.11's rule) | 4 / 5,081 | 0 / 1,424 | 0 / 1,283 |
| **β = 1.0 (measured)** | **4 / 4,332** | **0 / 1,103** | **0 / 1,107** |
| β = 1.25 | 4 / 3,622 | 0 / 1,033 | 0 / 1,000 |
| β = 1.5 | 7 / 2,971 | 0 / 980 | 0 / 937 |
| β = 2.0 | 22 / 2,000 | 0 / 934 | 0 / 879 |
| flat 1,600 | 37 / 1,600 | 0 / 1,600 | 0 / 1,600 |

The 4 L3 H4 misses are flat 6,400's own, all at exact gaps below 0.05. β = 1.25 is the steepest rate with no added mistakes, but choosing it would tune on the test data. The measured rate, β = 1.0, is adopted (commit `d7ba56c`).

**Online.**
- In situ, the first 40 queries of L3 H4 problem 0 (`6400s/exact`, the old and new rules run concurrently): 40/40 policies equal; CPU 0.88× in total, median 0.95 per query, Wilcoxon p = 0.09. In situ, CPU is dominated by solving new nested beliefs, so 15% fewer simulations save little CPU.
- The 12.11 near-tie sets, online (`gate/near_ties.py`): `6400s` is 0 wrong on the L2 H4 j-L1 set (n = 1,175) and on the L3 H3 near ties (n = 40), as in the replay. Mean simulations at L2 H4 fell from 1,199 (12.14) to 882, a 26% cut (23% in the replay). This harness counts simulations by tree `id()`, which freed pilot trees can reuse, so its absolute counts run about 220 below the replay's for both rules.

**Verdict:** same decisions, 14–23% fewer simulations, and a modest in-situ CPU saving that is not significant.

### 12.16 Gates for the episode study (October 6, 2026)

The study (`studies/tiger_l3_vs_l2.toml`) now runs at horizon 3 with root 1,000 and compressed 64-atom beliefs. Modeled levels are named per level (`AgentSpec.modeled`): `6400s/100` for i and `400` for j. Its mixed-level priors had never been gated.

- **Mixed priors vs the exact I-POMDP** (`results/w4_mixed_prior_gate_20261006`, seeds 0–19):
  - **Problems:** i-L3 at its prior; j-L2 at its prior and after 2 steps (4 histories).
  - **Result:** 120/120 optimal, with optimal-action Q error 0–0.09.
  - **Limit:** every exact gap here is at least 1.6, so this checks the hierarchy's construction, not near-tie accuracy.
- **Compressed beliefs at level 3** (`results/w4_belief_l3_20261006`, `belief_qualification --steps 5`): 12 histories under the study's priors, against exact beliefs whose supports reach 4,328 atoms by step 5.

  | N | Level TV, mean | Level TV, max | Steps above 0.2 | Nested TV, mean | Physical error, mean |
  |---|---|---|---|---|---|
  | 64 | 0.012 | 0.061 | 0 of 60 | 0.010 | 0.002 |
  | 256 | 0.003 | 0.017 | 0 of 60 | 0.003 | 0.0001 |

  At step 8, the exact L3 belief outgrew memory (stopped at 3.3 GB), so exact checks end at step 5.
- **Pilot** (`results/w4_episode_pilot_20261006`, 2 trials × 20 steps):
  - **CPU:** i's planning took 3,063 s and 480 s per trial, j's about 90 s. i's first steps cost 30–240 s each, while later steps reuse cached nested policies.
  - **Memory:** peak RSS 1.8 GB.
  - **Implication:** the full run (100 trials, 6 workers) needs about 50 CPU-h, within 3× of the plan's estimate.
  - **Bug found:** episode rows had empty `action_values`, because rows were written after the belief update cleared them. Fixed in commit `14ac0e2`, with a regression test.
- **Also recorded per step** (W6, W8): each decision's certified bounds, and each agent's predicted distribution of the opponent's next action (`docs/EQUILIBRIA.md`).

### 12.17 Certified bounds on every real decision (October 6, 2026)

`planning.chance_scoring(..., bounds=True)` returns, for each root action, an interval that contains the exact Q of the searched model, whatever the budget. Exact pieces enter exactly. Rollout frontiers, unsampled branches and untried actions are bounded by the declared reward bounds (THEORY's chance-scoring section; commit `95d11d5`).

- **Checks:** containment and tightness in 15 cases (L1 Tiger, depths 2–4, 3–800 simulations, 3 beliefs).
- **Where recorded:** `oracle` rows (`bound_width`, `bounds_contain_reference`) and episode rows (`bounds_<agent>`).
- **Scored values are unchanged,** and the cost is negligible, because successors are already cached.
- **What the bounds certify:** search error relative to the searched model. When the reference includes opponent-model or belief error, containment is not guaranteed and is reported as measured.

### 12.19 Wumpus L1 qualification: the simulation budget, not compression, was the pilot's problem (October 7, 2026)

**Design** (`experiments.wumpus_qualification`, `results/w7_wumpus_20261006`):
- **Agents:** the human (L1) against a uniform-random wumpus.
- **Beliefs:** exact real beliefs; each search starts from the belief compressed to N atoms.
- **Problems:** a scripted history cut after 0, 2 or 4 steps. The three seeds produced the same observations, so there are 3 distinct beliefs (94,500, 25,920 and 25,920 atoms), each decided with 3 planner seeds: n = 9 decisions per cell.
- **Reference:** the exact H3 best response at the exact belief, computed once per problem (1,034 s at the prior, 178–195 s after 2–4 steps).
- **Exact gaps:** 7.74 at the prior, 1.82 and 1.83 later. All rows use 12.14's scoring.

| N (search atoms) | Simulations | Optimal / 9 | Max loss | Optimal-action Q error | CPU per decision (s) |
|---|---|---|---|---|---|
| 64 | 1,000 | 5 | 7.74 | 9.44 | 10 |
| 64 | 3,000 / 10,000 | 9 / 9 | 0 | 9.43 | 10–11 |
| 256 | 1,000 | 3 | 7.74 | 3.54 | 17 |
| 256 | 3,000 / 10,000 | 7 / 7 | 7.74 | 1.98 | 17–18 |
| 1,024 | 1,000 | 6 | 1.83 | 4.13 | 41 |
| **1,024** | **3,000** | **9** | **0** | **0.63** | **41** |
| 1,024 | 10,000 | 9 | 0 | 0.63 | 42 |
| 4,096 | 1,000 | 4 | 1.83 | 4.34 | 144 |
| 4,096 | 3,000 / 10,000 | 9 / 9 | 0 | 0.16 | 146–147 |

- **The simulation budget caused the Section 10 pilot's losses.** At 1,000 simulations every N misses, and the misses fall at all three history lengths.
- **Compression sets the Q error,** and at N = 256 the decision too. N = 256 is wrong on 2 of 9 decisions at every budget, losing the full 7.74 gap at the prior. N = 64 decides right but misvalues by 9.4.
- At 3,000 simulations or more, every scored root at H3 is exact for its searched belief (certified bound width 0, 12.17). The remaining Q error is therefore all compression error.
- **Qualified Wumpus L1 H3 configuration: N = 1,024, 3,000 simulations** (9/9, optimal-action Q error 0.63, 41 s). N = 4,096 cuts the Q error to 0.16 at 3.5× the CPU.
- **Limit:** 3 distinct beliefs. A wider set of histories is needed before scaling.

### 12.18 Episode study: level 3 vs level 2 in Tiger, with level-2 and level-1 controls (October 6–7, 2026)

**Design** (`studies/tiger_l3_vs_l2.toml`; controls in `results/w4_controls_20261006/l{1,2}/study.toml`):
- **Agents:** j is level 2 in every condition. i is level 3 (prior over j: L1 0.8, L2 0.1, L0 0.1), level 2 (L1 0.89, L0 0.11) or level 1 (j uniform L0).
- **Planning:** horizon 3, root 1,000 simulations, compressed 64-atom beliefs; modeled stacks `6400s/100` for i-L3, `400` for i-L2 and for j.
- **Trials:** 100 trials of 20 steps per condition. The same trial seeds are used throughout, so the environment and j's streams are paired across conditions.
- **Gates:** 12.16.
- **Intervals:** trial-bootstrap 95%. Paired differences of j's reward also get Bonferroni intervals over the three pairs (98.3%).
- **Data:** `results/w4_episodes_20261006`, `results/w4_controls_20261006`; analysis in `results/w4_report_20261007`.
- **Reviewer page:** https://claude.ai/artifact/PqpRuTLcuSq5afJdSSxoCB

| Condition | Trials | i's reward | j's reward | i's prediction log-loss | i's hit rate | Doors opened per trial | Safe-door rate | Exact decisions | i's CPU per step (s) |
|---|---|---|---|---|---|---|---|---|---|
| Level 3 i | 100 | 17.1 [9.7, 23.9] | 10.8 [2.3, 18.5] | 0.462 [0.418, 0.515] | 0.83 [0.81, 0.84] | 4.4 | 98% | 98% | 90.7 |
| Level 2 i | 100 | 15.6 [7.4, 22.8] | 13.6 [5.1, 20.8] | 0.452 [0.416, 0.499] | 0.82 [0.80, 0.84] | 4.3 | 97% | 97% | 4.9 |
| Level 1 i | 100 | 18.9 [10.2, 26.4] | 8.5 [-0.4, 16.5] | 1.099 [1.099, 1.099] | 0.80 [0.79, 0.81] | 4.7 | 97% | 98% | 0.2 |

| Paired difference (same trials) | n | i's reward | j's reward | i's prediction log-loss |
|---|---|---|---|---|
| L3 − L2 | 100 | 1.4 [-2.4, 5.3] | -2.8 [-6.2, -0.1] | 0.010 [-0.045, 0.069] |

- **Modeling j is what predicts j.** i's log-loss on j's next action is 0.46 at level 3 and 0.45 at level 2, against 1.10 (uniform) at level 1 (−0.64 paired, [−0.68, −0.58]). **Level 3 adds nothing over level 2 here:** paired difference +0.010 [−0.045, 0.069].
- **i's level is weakly identified.** Level 3 i's belief that j is level 2, the truth, moves only from 0.10 to 0.18 by step 20; level 1 keeps 0.82. In Tiger, level-1 and level-2 agents both listen until confident and then open, so j's actions carry little evidence about its level. j rules out a random i by step 5 in every condition (P(i is L1) 0.89–0.90 at step 5, 0.99 at step 20).
- **No robust reward effect, for either agent.**
  - **i's reward:** level 3 − level 2 is +1.4 [−2.4, 5.3], level 3 − level 1 is −1.9 [−6.5, 3.3]. i's reward depends only on its own door choice.
  - **j's reward:** two of three paired differences narrowly exclude zero at 95%: level 3 − level 2 −2.8 [−6.2, −0.1], level 2 − level 1 +5.1 [0.8, 9.6]. Neither survives Bonferroni: 98.3% intervals [−7.2, 0.2] and [−0.3, 10.8].
  - **The interim effect did not hold.** At n = 77 an apparent effect of i's level on j's reward looked significant. A tested mechanism (i's openings resetting the tiger and erasing j's information) was not supported: a level-1 i causes fewer such resets (3.8 per trial against 4.0–4.1), and per-trial reset differences barely track j's reward differences (correlation 0.12).
- **Decisions were almost always exact for the searched model:** 97–98% had certified bound width 0 (12.17).
- **Cost:** level 3 i spends 91 s of planning CPU per step, against 4.9 s at level 2 and 0.2 s at level 1, about 50 CPU-h per 100 trials.

### 12.20 I-POMCP vs IPF + RTS at scale, on the current estimator (October 7, 2026)

**Design** (`results/w5_rts_baseline_20261006`, `launch.py full`):
- **Problems:** L2 Tiger, the wide grid of 12.3 (19 beliefs of i × 5 of j × seeds 0–3, n = 380 per cell), scored against the exact I-POMDP.
- **I-POMCP:** root 1,000 simulations, compressed beliefs N = 64, the gated modeled solve (H3 flat 400, H4 `6400s`).
- **IPF + RTS:** K observation samples per node at every level, with the published sampled IPF (N = 64).
- **Ablation:** RTS on compressed beliefs.
- **CPU:** per root decision, nested solves included. All cells ran in one run under the same load, so CPU is comparable, though not paired in-process.

| Planner | H | K (RTS) | Optimal (95% interval) | Mean loss | Max loss | Optimal-action Q error | CPU (s) |
|---|---|---|---|---|---|---|---|
| **I-POMCP, compressed 64** | 3 | — | **380/380 (99.0–100%)** | 0 | 0 | **0.000** | 1.5 |
| IPF + RTS | 3 | 2 | 98.4% (96.6–99.4) | 0.051 | 6.58 | 1.83 | 1.0 |
| IPF + RTS | 3 | 4 | 98.9% (97.3–99.7) | 0.029 | 3.08 | 1.37 | 2.2 |
| IPF + RTS | 3 | 8 | 99.5% (98.1–99.9) | 0.013 | 2.87 | 0.94 | 4.0 |
| RTS, compressed 64 | 3 | 4 | 98.4% (96.6–99.4) | 0.043 | 3.08 | 1.30 | 0.15 |
| **I-POMCP, compressed 64** | 4 | — | **380/380 (99.0–100%)** | 0 | 0 | **0.127** | 16.7 |
| IPF + RTS | 4 | 2 | 95.3% (92.6–97.2) | 0.102 | 6.14 | 1.49 | 12.1 |
| IPF + RTS | 4 | 4 | 95.3% (92.6–97.2) | 0.090 | 6.04 | 0.93 | 49.7 |
| IPF + RTS | 4 | 8 | 96.8% (94.5–98.4) | 0.053 | 1.86 | 0.62 | 91.8 |
| RTS, compressed 64 | 4 | 4 | 95.5% (92.9–97.4) | 0.109 | 6.14 | 0.87 | 1.2 |

- **I-POMCP is exact-optimal on every problem at both horizons.** IPF + RTS misses 0.5–1.6% of decisions at H3 and 3.2–4.7% at H4, at every K.
- **At H4, I-POMCP is more accurate and cheaper:** 16.7 s, against 49.7–91.8 s for IPF + RTS at K = 4–8. Its optimal-action Q error is 0.13, against 0.62–1.49.
- **At H3, IPF + RTS at K = 2 is slightly cheaper** (1.0 s against 1.5 s), but not exact.
- **The ablation separates belief from search error.** RTS on compressed beliefs makes about the same decisions as on the IPF, at 1/14–1/40 of the CPU. Its misses are search error (K samples per node), not belief error.
- **This replaces Section 6's 15-problem comparison,** whose I-POMCP cells predated the last estimator revisions.

### 12.21 Level grid: i at levels 1–4 against j at levels 1–4 (October 7, 2026)

> **Caveat (12.26):** N = 32 drops low-probability branches of the opponent's history (for example after a false creak), which gives the opponent's actual action zero predicted probability. Prediction log-loss and level identification here are affected; see 12.26.

**Design** (`results/w9_level_grid_20261007/grid.py`; analysis `analyze_grid.py`):
- **Priors:** every agent holds a uniform prior over the opponent's lower levels (0 … L−1), and so does every nested model.
- **Planning:** both agents plan with I-POMCP at horizon 2. With two steps left, chance scoring (12.14) makes each decision the exact best response to the agent's modeled opponents for its searched belief, at any budget (100 simulations).
- **Beliefs:** compressed to N = 32. At N = 64, a level-4 decision grew to 65 s per step by step 8, against 1–2 s at N = 16 (probe logs in the same folder).
- **Trials:** 20 steps. All 16 cells share trial seeds. Trials were scheduled round-robin and the run was stopped at 13:29, leaving n = 39–42 per cell.
- **Intervals:** trial bootstrap, 95%.

| i | j | n | i's reward | j's reward | Joint welfare | i's log-loss of j | j's log-loss of i | i's belief in j's level | j's belief in i's level | CPU per step (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 42 | 13.0 [-2.5, 25.3] | 12.5 [-0.6, 23.5] | 25.5 [5.0, 43.0] | 1.10 [1.10, 1.10] | 1.10 [1.10, 1.10] | — | — | 0.0 |
| 1 | 2 | 42 | 13.0 [-3.0, 26.4] | 14.6 [4.1, 23.2] | 27.6 [7.9, 44.1] | 1.10 [1.10, 1.10] | 0.55 [0.47, 0.66] | — | 0.99 [0.98, 1.00] | 0.2 |
| 1 | 3 | 42 | 17.2 [2.5, 29.8] | 15.9 [4.9, 24.8] | 33.1 [14.7, 49.3] | 1.10 [1.10, 1.10] | 0.59 [0.45, 0.78] | — | 0.50 [0.40, 0.60] | 3.0 |
| 1 | 4 | 42 | 19.0 [4.4, 31.1] | 18.2 [8.5, 26.4] | 37.3 [20.8, 51.7] | 1.10 [1.10, 1.10] | 0.72 [0.51, 0.95] | — | 0.42 [0.31, 0.53] | 25.1 |
| 2 | 1 | 42 | 13.3 [1.7, 23.5] | 19.5 [9.1, 28.5] | 32.8 [17.4, 46.7] | 0.61 [0.50, 0.74] | 1.10 [1.10, 1.10] | 0.97 [0.93, 1.00] | — | 0.2 |
| 2 | 2 | 42 | 14.8 [2.5, 25.3] | 20.3 [12.5, 26.9] | 35.2 [21.0, 48.5] | 0.58 [0.49, 0.69] | 0.55 [0.47, 0.66] | — | — | 0.4 |
| 2 | 3 | 42 | 16.9 [2.5, 28.7] | 22.7 [14.3, 29.2] | 39.6 [23.9, 53.5] | 0.54 [0.46, 0.64] | 0.50 [0.42, 0.60] | — | 0.49 [0.39, 0.58] | 3.4 |
| 2 | 4 | 41 | 18.9 [3.9, 30.7] | 19.7 [10.3, 27.8] | 38.6 [22.2, 52.6] | 0.57 [0.47, 0.69] | 0.63 [0.45, 0.83] | — | 0.43 [0.33, 0.52] | 26.7 |
| 3 | 1 | 42 | 10.4 [-4.3, 22.4] | 17.7 [6.2, 27.9] | 28.1 [10.0, 45.1] | 0.50 [0.45, 0.57] | 1.10 [1.10, 1.10] | 0.65 [0.54, 0.75] | — | 2.5 |
| 3 | 2 | 42 | 9.6 [-6.6, 23.2] | 18.2 [8.8, 26.1] | 27.8 [9.5, 44.1] | 0.48 [0.43, 0.56] | 0.56 [0.45, 0.68] | 0.53 [0.42, 0.64] | — | 3.3 |
| 3 | 3 | 42 | 13.8 [-0.9, 26.4] | 21.9 [12.0, 29.8] | 35.7 [17.6, 51.9] | 0.46 [0.40, 0.54] | 0.54 [0.41, 0.71] | — | — | 6.6 |
| 3 | 4 | 40 | 19.3 [5.8, 29.5] | 21.8 [11.6, 30.1] | 41.1 [26.0, 54.0] | 0.51 [0.43, 0.62] | 0.55 [0.40, 0.74] | — | 0.25 [0.18, 0.34] | 32.5 |
| 4 | 1 | 40 | 11.3 [-2.4, 22.9] | 23.4 [13.6, 31.4] | 34.8 [17.5, 50.2] | 0.56 [0.45, 0.68] | 1.10 [1.10, 1.10] | 0.47 [0.37, 0.59] | — | 26.3 |
| 4 | 2 | 40 | 12.2 [-1.9, 24.3] | 19.9 [10.5, 27.6] | 32.0 [13.6, 47.5] | 0.49 [0.41, 0.59] | 0.51 [0.45, 0.60] | 0.34 [0.25, 0.43] | — | 30.4 |
| 4 | 3 | 41 | 11.1 [-5.2, 25.1] | 24.0 [13.5, 32.0] | 35.1 [16.9, 51.8] | 0.47 [0.39, 0.57] | 0.54 [0.40, 0.73] | 0.32 [0.24, 0.40] | — | 35.4 |
| 4 | 4 | 39 | 16.1 [-0.3, 28.8] | 20.3 [10.7, 28.2] | 36.4 [18.9, 51.9] | 0.51 [0.40, 0.63] | 0.50 [0.39, 0.63] | — | — | 61.8 |

- **Modeling the opponent is what predicts it.**
  - Every agent at level ≥ 2 predicts its opponent's next action with log-loss 0.46–0.72, whatever the opponent's level.
  - A level-1 agent sits at 1.10 (a uniform guess) by construction.
  - Prediction does not keep improving with level: level 4 is no better than level 2.
- **Identifying the opponent's level gets harder with level.** At step 20, against an opponent whose level is in its prior:
  - level 2 identifies level 1 almost surely (0.97–0.99);
  - level 3 reaches 0.65 for level 1 and 0.53 for level 2, against a prior of 1/3;
  - level 4 reaches 0.47, 0.34 and 0.32 for levels 1–3, against a prior of 1/4.

  Agents at levels 1–3 act alike in Tiger (listen until confident, then open), so their actions carry little evidence about their level. This is consistent with 12.18.
- **No consistent effect of an agent's own level on its reward.**
  - Per-cell reward intervals are about ±20.
  - Paired by seeds, the agent at level k minus the same agent at level 1 against the same opponent: 2 of 24 comparisons exclude zero, both small and negative, for j facing a level-4 i (−3.6 and −3.5). About 1 would by chance.
  - Comparing i with j inside a cell is confounded by each agent's growl luck, which the shared seeds repeat across cells. For example, i's tiger hits at n = 19 were 5 distinct events counted 67 times.
- **Cost:** CPU per step, both agents, ranges from 0.0 s (level 1 vs level 1) to 62 s (level 4 vs level 4). A level-4 agent costs 25–35 s per step against lower levels.

### 12.22 Which payoff makes one more level of reasoning pay? An exact screen (October 7, 2026, evening)

**Why.** In the published Tiger an agent's reward depends only on its own door, so a deeper model of the opponent cannot pay in reward (12.18, 12.21). `JointTigerModel` (DOMAINS.md) keeps the published states, transitions and observations and changes only the payoff for opening a door: `together_safe` is each agent's reward for opening the safe door when the other opens the same door in the same step (published: 10), and `solo_penalty` is subtracted from an agent that opens while the other listens (published: 0).

**Design** (`results/p2_payoff_screen_20261007/screen.py`, `analyze.py`):
- Exact I-POMDP Q of i at point-belief chains: i's tiger belief in {0.05, 0.2, …, 0.95} (7 values), and each nested model's in {0.085, 0.5, 0.915}, down to a uniform level 0.
- Levels 1–4 at H2, 1–3 at H3, 1–2 at H4. Every lower level is exact at the same depth.
- **Value of one more level:** at each chain, the exact loss, under the level-(k+1) model, of the action the level-k agent chooses at the same beliefs (ties played uniformly). A positive value means i gains by modeling one more level, if the deeper model is right.
- Cost: 2,793 exact solves, about 1 CPU-h.

**Result** (mean loss over chains; share of chains with a positive loss in parentheses):

| Payoffs | H | 1→2 | 2→3 | 3→4 |
|---|---|---|---|---|
| published (together 10) | 2 | 0.64 (29%) | 0 | 0 |
| published | 3 | 0.78 (29%) | 0.009 (10%) | — |
| together 20 | 3 | 0.75 (10%) | 0.53 (13%) | — |
| **together 40** | 2 | 1.45 (19%) | 0.63 (10%) | 0.81 (6%) |
| **together 40** | 3 | 1.59 (19%) | **2.10 (16%)** | — |
| race (together 0) | 3 | 0.50 (19%) | 0.30 (6%) | — |
| race (together 5) | 3 | 0.53 (19%) | 0 | — |
| solo penalty 10 | 3 | 2.44 (29%) | 0 | — |
| solo penalty 30 | 3 | 0 | 0 | — |

- **The published game confirms 12.18 and 12.21 exactly:** level 3 changes no decision at H2, and is worth 0.009 on average at H3.
- **A coordination bonus makes every level count.** With `together_safe = 40`, a third level is worth 2.10 at H3 and a fourth 0.81 at H2. Penalizing solo openings separates level 1 from level 2 only.
- **Chosen:** `together_safe = 40`, the largest minimum over 1→2 and 2→3 (the rule fixed in PLAN.md). It is the "coordination Tiger" from here on.

### 12.23 Gates on the coordination Tiger against the exact I-POMDP (October 7, 2026, evening)

**Design** (`results/p3_joint_gates_20261007/gates.py`): the configurations the episode studies use, on the problem grids of 12.3 and 12.7. Losses are first-action losses against the exact I-POMDP (every lower level exact); intervals are Clopper–Pearson 95%.

| Gate | Configuration | Optimal | Interval |
|---|---|---|---|
| L2 H3 | root 1,000, modeled `400`, exact beliefs; 19 × 5 beliefs × 4 seeds | 380/380 | [0.990, 1] |
| L3 H3, k = 0 | root 1,000, modeled `6400s/100`; 9 × 3 × 3 beliefs × 2 seeds | 162/162 | [0.977, 1] |
| L2 H2 | root 100, modeled `100`, compressed 32 (12.21's configuration) | 54/54 | [0.934, 1] |
| L3 H2 | the same | 162/162 | [0.977, 1] |
| **L4 H2** | the same; 9 × 3 × 3 × 3 beliefs × 2 seeds | **486/486** | [0.992, 1] |

- The certified configurations carry over to the new payoffs unchanged; no modeled re-qualification was needed. Mean optimal-action Q error is 0.002 at L2 H3 and 0 elsewhere.
- **First level-4 gate against the exact I-POMDP.** The drivers now run levels up to 4 (`oracle`, `panel`; the level-4 reference is cross-checked against `build_exact_agent` in `tests/test_l4_oracle.py`).
- Planner CPU per decision: 0.9 s at L2 H3, 8.4 s at L3 H3, 0.8 s at L4 H2.

### 12.24 Where search error comes from, at L2 and L3 (October 7, 2026, evening)

**Design** (`results/p5_error_attribution_20261007/attribution.py`, `analyze.py`): published Tiger, root 1,000, the certified modeled stacks (L2 H3 `400`, L2 H4 `6400s`, L3 H3 `6400s/100`), one seed, the grids of 12.3 (95 problems) and 12.7 (81). Each root action's search value (empirical Bellman backups) is split exactly, against the exact best response to i's own modeled hierarchy (search error only), into:
- **chance error at depth d:** sampled minus exact observation frequencies, weighted by exact values;
- **local action loss:** the exact loss of the greedy choice at a deeper node;
- **frontier:** the error of a rollout below the tree;
- the immediate-reward error, which is zero by construction (the planner uses the exact E[R | b_h, a]).

The parts sum to the search error with a residual below 1e-14 on every action.

| Problems | Search error, optimal action (mean) | Chance error | Local action loss | Frontier | Error after chance scoring |
|---|---|---|---|---|---|
| L2 H3 | 0.141 | 98% (depth 1: 45%, depth 2: 53%) | 2% | 0 | 0 |
| L3 H3 | 0.146 | 98% (39%, 58%) | 2% | 0 | 0.0006 |
| L2 H4 | 0.164 | 69% (12%, 26%, 32%) | 22% | 9% | 0.132 |

(Shares are of the summed absolute parts.)
- **At H3, search error is almost all sampling noise in observation frequencies,** which chance scoring replaces by exact probabilities. This is why scoring brought i's L3 H3 Q error from 0.06 to 0.0006 (12.14).
- **At H4, a fifth of the error is in choices deeper in the tree,** plus rollouts below it. Scoring is exact only where two steps are left, so 0.13 remains on the optimal action. This is where variance reduction or adaptive root budgets would act (HANDOFF 0.2 item 4).

### 12.25 Level grid on the coordination Tiger: levels 1–3 (October 7, 2026)

> **Caveat (12.26):** at N = 32, beliefs dropped the opponent's realized history in 3–20% of trials above level 1. The grid is being rerun at N = 64 with the same seeds; read the level-3 rows below as provisional. **Superseded by 12.27** (N = 64).

**Design** (`results/p4_joint_level_grid_20261007/grid.py`): 12.21's grid unchanged except for the payoffs (`together_safe = 40`, 12.22). Uniform level priors at every level, H2 (each decision exact for the searched model), compressed 32, 20 steps, the same trial seeds in every cell. Levels 1–3: **n = 100 per cell, final.** The level-4 cells are running to n = 40 and will be reported separately.

**Primary endpoint** (fixed in PLAN.md before the run; `endpoint.py`): the same agent one level higher, against the same opponent and seeds, difference in 20-step reward; paired t-test, Holm over the family of 12 one-level steps among levels 1–3; 95% trial-bootstrap intervals. The published-Tiger grid (12.21, n = 42) is the control, restricted to the same 12 steps.

| Step | Opponent level | i: difference [95%] | Holm p | j: difference [95%] | Holm p |
|---|---|---|---|---|---|
| 1 → 2 | 1 | **+25.5 [13.6, 37.9]** | 0.001 | **+28.7 [16.1, 41.4]** | < 0.001 |
| 1 → 2 | 2 | +13.6 [3.7, 23.8] | 0.10 | +5.1 [−5.0, 15.0] | 1 |
| 1 → 2 | 3 | +13.4 [2.6, 23.8] | 0.13 | +12.9 [2.7, 23.7] | 0.13 |
| 2 → 3 | 1 | −13.3 [−25.4, −2.4] | 0.14 | −2.7 [−11.4, 5.9] | 1 |
| 2 → 3 | 2 | −6.4 [−15.9, 2.4] | 0.90 | +1.3 [−6.2, 9.0] | 1 |
| 2 → 3 | 3 | +1.0 [−7.0, 9.2] | 1 | +8.1 [2.1, 14.7] | 0.13 |

Control (published Tiger, same 12 steps): 0 of 12 survive (smallest Holm p = 0.14).

**Per cell** (mean 20-step reward; joint safe openings = steps in which both open the same safe door):

| i | j | i's reward | j's reward | Welfare [95%] | Joint safe openings | i's log-loss of j | i's belief in j's level |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 63.9 | 53.9 | 117.8 [95.6, 140.6] | 1.7 | 1.10 | — |
| 1 | 2 | 92.8 | 82.6 | 175.4 [150.5, 200.1] | 2.6 | 1.10 | — |
| 1 | 3 | 92.4 | 80.0 | 172.3 [145.1, 198.8] | 2.6 | 1.10 | — |
| 2 | 1 | 89.4 | 93.6 | 183.1 [158.6, 207.2] | 2.7 | 0.58 | 0.99 |
| 2 | 2 | 106.4 | 98.7 | 205.1 [181.4, 228.3] | 3.1 | 0.54 | — |
| 2 | 3 | 105.8 | 100.0 | 205.7 [179.4, 231.4] | 3.2 | 0.61 | — |
| 3 | 1 | 76.1 | 82.5 | 158.7 [132.2, 185.8] | 2.5 | 0.63 | 0.62 |
| 3 | 2 | 99.9 | 95.4 | 195.4 [169.3, 220.9] | 3.0 | 0.55 | 0.65 |
| 3 | 3 | 106.8 | 103.5 | 210.3 [184.6, 235.5] | 3.3 | 0.55 | — |

- **With joint payoffs, modeling the opponent pays in reward.** Against a level-1 opponent, a level-2 agent earns 25–29 more over 20 steps than a level-1 agent (both agents, Holm p ≤ 0.001), where the published game shows nothing. Welfare rises from 118 (both level 1) to 205–210 (both at level 2 or 3), and joint safe openings from 1.7 to 3.1–3.3 per episode.
- **A third level does not pay here.** No 2 → 3 step survives the correction, and against a level-1 opponent the direction is negative (i: −13.3, nominal p = 0.02). The exact screen (12.22) valued level 3 at point beliefs where its model of the opponent is right. In these episodes a level-3 agent holds a uniform prior over levels 0–2, so it must first identify the opponent. By step 20 it puts only 0.62–0.65 on the true level, against 0.99 for a level-2 agent facing level 1, whose prior has only two candidates.
- **Implication:** what pays is a correct model, not depth. The test of depth alone is a level-3 agent with a correct (or concentrated) prior. That is the next run, alongside the level-4 cells.

### 12.26 Belief compression at N = 32 drops the opponent's real history (October 7, 2026)

**How it was found.** In a check of 12.25 (`results/p4b_correct_prior_20261007`), a level-3 i given a *correct* prior (all mass on j at level 2, with j's true nested prior) predicted j worse than a level-2 i (log-loss 0.64 vs 0.54), which a correct model should not do. Tracing one trial: j heard a false creak (probability 0.05) and so did not register a reset; i's belief, compressed to 32 atoms, had dropped every j-model on that branch; at the next steps i gave j's actual action probability 0, and all later updates rested on wrong models.

**Probe** (same 10 trial seeds, condition above; `probe_n.py`, `compare_n.py`):

| Support bound N | Zero-probability predictions of j's action | Mean log-loss | Mean 20-step reward of i | i's CPU per episode |
|---|---|---|---|---|
| 32 | 8 (in 3 of 10 trials) | 1.38 | 125.4 | 146 s |
| 64 | 0 | 0.34 | 142.4 | 235 s |
| 128 | 0 | 0.33 | 139.4 | 1,007 s |

- **N = 64 removes every miss; N = 128 changes nothing more** at 4.3× the CPU. The reward changed in 1 of 10 trials (a tiger hit after the misses: −155 at N = 32, +15 at N = 64).
- **In the N = 32 coordination grid,** trials with at least one zero-probability prediction: 0% at level 1 vs level 1; 3–9% with level 2 only; 8–20% where level 3 is involved. Level 4 cells (n = 14–15): 0–7 of 15.
- **Why the gate missed it.** 12.16 qualified compressed beliefs by the total-variation distance of the level posterior (≤ 0.061 at N = 64). Dropping a branch of mass about 0.05 barely moves that distance but removes the realized history. The gate for a bounded belief should include the probability it assigns to what actually happens.
- **Consequences.** The prediction and identification numbers of 12.21 (published Tiger, N = 32) and 12.25 are affected, most at level 3. The levels 1–3 coordination grid and the correct-prior check are being rerun at N = 64 with the same seeds (`results/p4c_joint_level_grid_n64_20261007`). Level 4 at N = 64 costs about 7× more per step (12.21's probes) and is deferred.

### 12.27 The coordination grid and the correct-prior check at N = 64 (October 8, 2026)

**Design:** 12.25 and its correct-prior check rerun at N = 64 with the same trial seeds (`results/p4c_joint_level_grid_n64_20261007`, `results/p4b_correct_prior_20261007/trials_n64`), n = 100 per cell, levels 1–3, H2. Endpoint, family and control as in 12.25.

**Compression misses** (trials in which some prediction gave the opponent's actual action probability 0): 0–6 per 100 per cell, against 0–20 at N = 32. Lower, not zero.

**Primary endpoint** (one level higher, same opponent and seeds; Holm over 12):

| Step | Opponent level | i: difference [95%] | Holm p | j: difference [95%] | Holm p |
|---|---|---|---|---|---|
| 1 → 2 | 1 | **+27.2 [15.8, 38.4]** | < 0.001 | **+29.8 [16.5, 43.5]** | < 0.001 |
| 1 → 2 | 2 | +10.4 [1.1, 19.9] | 0.25 | +6.2 [−5.3, 17.2] | 1 |
| 1 → 2 | 3 | **+15.5 [5.5, 25.7]** | 0.031 | **+14.7 [5.8, 23.8]** | 0.021 |
| 2 → 3 | 1 | −3.1 [−9.5, 3.3] | 1 | −1.0 [−7.5, 5.5] | 1 |
| 2 → 3 | 2 | +6.8 [1.8, 12.1] | 0.10 | +5.0 [0.2, 9.5] | 0.25 |
| 2 → 3 | 3 | +6.2 [0.1, 12.3] | 0.25 | +1.7 [−3.5, 6.2] | 1 |

**Per cell:**

| i | j | i's reward | j's reward | Welfare | i's log-loss of j | j's log-loss of i | CPU per step (s) |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 63.9 | 53.9 | 117.8 | 1.10 | 1.10 | 0.0 |
| 1 | 2 | 93.2 | 83.7 | 176.9 | 1.10 | 0.53 | 0.3 |
| 1 | 3 | 91.1 | 82.7 | 173.8 | 1.10 | 0.54 | 7.7 |
| 2 | 1 | 91.1 | 94.2 | 185.3 | 0.54 | 1.10 | 0.3 |
| 2 | 2 | 103.6 | 100.4 | 204.0 | 0.50 | 0.46 | 0.5 |
| 2 | 3 | 106.6 | 105.4 | 211.9 | 0.49 | 0.39 | 11.4 |
| 3 | 1 | 88.0 | 89.4 | 177.3 | 0.56 | 1.10 | 7.6 |
| 3 | 2 | 110.4 | 104.1 | 214.5 | 0.43 | 0.50 | 11.1 |
| 3 | 3 | 112.8 | 105.8 | 218.6 | 0.41 | 0.40 | 22.3 |

**Correct prior** (paired with the N = 64 grid):

| i | Against | i's reward | Grid comparison | Difference [95%] | Holm p | i's log-loss |
|---|---|---|---|---|---|---|
| level 3, all prior on j at level 2 | j at level 2 | 105.3 | i at level 2 (103.6) | +1.7 [−5.5, 7.9] | 0.63 | 0.39 |
| | | | i at level 3, uniform prior (110.4) | −5.1 [−11.7, 0.3] | 0.31 | |
| level 2, all prior on j at level 1 | j at level 1 | 93.9 | i at level 1 (63.9) | **+30.0 [18.6, 41.5]** | < 0.001 | 0.48 |
| | | | i at level 2, uniform prior (91.1) | +2.8 [−2.1, 7.8] | 0.50 | |

- **Modeling the opponent pays, robustly.** One level of modeling (level 1 → 2) gains 27–30 against a level-1 opponent and 15 against a level-3 opponent, for both agents. The published game shows none of this (12.21, 12.25).
- **With fewer compression errors, a third level predicts better** (log-loss 0.41–0.43 against 0.49–0.50 for level 2; 0.39 with a correct prior). **Its reward gain is small at horizon 2:** +5 to +7 against opponents at levels 2–3 (nominal p 0.01–0.05), none surviving the correction at n = 100. The exact screen predicts this: a third level is worth 0.63 on 10% of beliefs at H2, against 2.10 on 16% at H3 (12.22).
- **The N = 32 results changed direction for level 3** (12.25: i 2 → 3 against level 1 was −13.3, now −3.1; against level 2, −6.4 now +6.8). Bounded-belief error was large enough to reverse level-3 conclusions; 12.25's level-3 rows are superseded.
- **Next, if the depth question is pursued:** a horizon-3 study of level 3 against level 2 on the coordination Tiger at N = 64, where the screen puts the value of a third level. 12.18's cost (about 30 CPU-min per episode for level 3 at H3) gives about 100 CPU-h for n = 100 in two conditions. PLAN's gate for it (an H2 effect against j at level 2) was not met (Holm p = 0.10), so it was not launched.

### 12.28 Optimal resampling for the compressed filter: an A/B, not adopted (October 8, 2026)

**Why.** 12.26 found that compression to N atoms dropped light branches of the opponent's history. The compressor draws N systematic samples and merges identical ones. A heavy atom then spends many of the N draws, so the belief keeps fewer than N distinct atoms (17 of 32 at k = 4 below). The candidate is Fearnhead & Clifford's (2003) optimal resampling: atoms heavier than a threshold 1/c keep their exact mass, and the light tail is resampled without bias into the remaining slots at mass 1/c each, so all N slots hold distinct atoms.

**Design** (`results/p8_resampling_gate_20261008`; old and new arms each on a frozen source, import paths checked; same seeds):
- **G1, belief accuracy:** `belief_qualification`, levels 2–3, N ∈ {16, 32, 64}, 12 histories × 5 steps, against exact beliefs.
- **G2, realized-branch coverage:** the 12.26 probe (level-3 i with a correct prior, coordination Tiger, H2, 10 seeds); zero-probability predictions of j's action, log-loss, i's CPU per episode.
- **G3, decisions:** published Tiger, L2 H3, root 1,000, modeled 400, compressed beliefs after k = 2 and 4 real steps, 190 problems per k, against the exact I-POMDP.

**G1** (mean level-posterior TV against exact; worst in parentheses):

| Level | N | Systematic (old) | Optimal (new) |
|---|---|---|---|
| 2 | 16 | 0.033 (0.17) | 0.0012 (0.059) |
| 2 | 32 | 0.0075 (0.031) | 0.0007 (0.031) |
| 3 | 16 | 0.088 (0.55) | 0.013 (0.079) |
| 3 | 32 | 0.051 (0.27) | 0.0032 (0.037) |
| 3 | 64 | 0.012 (0.061) | 0.0018 (0.044) |

**G2 and G3** (CPU is i's per episode, or per decision):

| Arm, N | G2 zero-probability predictions | G2 log-loss | G2 CPU | G3 optimal, k = 2 | G3 optimal-action Q error, k = 2 / 4 | G3 CPU, k = 2 / 4 |
|---|---|---|---|---|---|---|
| new, 16 | 3 | 0.69 | 82 s | 184/190 (max loss 0.61) | 0.15 / 0.25 | 5.3 / 6.1 s |
| old, 32 | 8 | 1.38 | 146 s | 184/190 (max loss 0.11) | 0.24 / 0.23 | 8.0 / 9.2 s |
| new, 32 | 2 | 0.60 | 414 s | 190/190 | 0.024 / 0.082 | 10.4 / 18.1 s |
| old, 64 | 0 | 0.34 | 235 s | — | — | — |
| new, 64 | 0 | 0.35 | 1,244 s | — | — | — |

(G3 at k = 4 was 190/190 in every arm.)

- **Per atom, optimal resampling is far more accurate:** belief error is 6–160× lower at equal N, Q error 3–10× lower, and at N = 32 the six k = 2 misses disappear.
- **Per CPU-second it does not dominate.** Holding N distinct atoms costs 1.3–5× more at equal N. At the low end it wins: new N = 16 is cheaper than old N = 32 and has fewer misses. At the operating point used for episodes it loses: old N = 64 has no misses at 235 s, against 414 s and 2 misses for new N = 32. Merged systematic draws act as an adaptive support, small when the belief is concentrated.
- **Decision:** not adopted, under the rule that a redesign must win at matched cost. The implementation and its tests are on branch `resampling/optimal-fc`. A fair next comparison would match the number of distinct atoms rather than N, or adapt N to the belief's spread. Episodes keep the systematic compressor at N = 64 (12.27, and the H3 study now running).

### 12.29 Does a third level pay at horizon 3? (October 9, 2026)

**Design** (`results/p7_h3_depth_20261008/h3_depth.py`, `analyze.py`; fixed before the run): coordination Tiger, both agents at H3, root 1,000 simulations, compressed beliefs N = 64 (systematic, 12.28), j at level 2 (uniform prior over i at levels 0–1, modeled `400`).
- **L3:** i at level 3 with all prior on j at level 2 and j's true nested prior: a correct model (modeled `6400s/100`).
- **L2:** i at level 2, uniform prior over levels 0–1 (modeled `400`).
- Same trial seeds, so the two conditions are paired.
- **Kill criterion:** the first 8 level-3 episodes took a median 138 CPU-min, above 3× 12.18's 32. So n was cut to 50 as fixed in advance (`KILL_CRITERION.txt`). Level-2 episodes: 0.8 CPU-min.

**Primary endpoint** (i's 20-step reward, L3 − L2, paired t-test, n = 50): **−9.6 [−26.8, 4.2], p = 0.24.** No gain from a third level.

| Measure | L3 | L2 | Difference [95%] | p |
|---|---|---|---|---|
| i's reward | 103.2 | 112.8 | −9.6 [−26.8, 4.2] | 0.24 |
| j's reward | 102.9 | 112.3 | −9.4 [−20.7, 0.2] | 0.09 |
| i's log-loss of j | 0.51 | 0.45 | +0.06 [−0.09, 0.23] | 0.50 |
| Trials with a zero-probability prediction | 10% | 0% | | |

**Exploratory (not fixed in advance):** the whole deficit sits in the 5 trials where level 3's belief dropped the realized branch. There the difference averages −101 (a tiger hit or a lost coordination). In the other 45 trials it averages +0.6.

- **What limits level 3 here is the bounded belief, not reasoning depth.** A level-3 belief holds j-models that carry their own beliefs about i, so 64 atoms still drop a realized branch in 1 trial in 10. When it holds, level 3 earns what level 2 earns.
- **The screen's 2.10 per decision does not appear in episodes.** That value assumed exact beliefs at point priors. In play, beliefs spread over many histories, and the decisions where a third level differs are rare.
- **Next:** a bounded belief that never drops the realized branch at level 3 at acceptable cost. The optimal-resampling candidate (12.28, branch `resampling/optimal-fc`) is far more accurate per atom; an adaptive N or a matched-distinct-atom comparison is the open design question. A larger H3 study is not worth its CPU (about 2.3 CPU-h per level-3 episode) until then.

**Correction (12.30):** two of the five "compression" trials (t016, t018) were not compression misses but a nested-budget mismatch in the L3 condition, and they carry almost all of the deficit. The bullets above are revised there.

### 12.30 Why bounded beliefs lose the realized branch, and a correction to 12.29 (October 9, 2026)

**Design** (`results/q1_miss_diagnosis_20261009/replay.py`, `budget_check.py`). For every trial with a zero-probability prediction (i gave j's actual action probability 0), i's real belief updates were replayed along the logged (a_i, o_i). The filter and the modeled policies are pure functions of their arguments, so the replay is exact: it matched the logged support size at every step and the logged prediction at every miss. For a miss at step t, the atoms dropped at each earlier compression were grouped by j-model, propagated exactly through the remaining logged updates, and tested for predicting j's actual action. The first such compression is where the realized branch was lost. Trials: 12.26's N = 32 probe (3 trials, 8 misses), 12.27's N = 64 probe (5 trials, 5 misses), and 12.29's five H3 trials (6 misses).

**Where and why** (17 of 19 misses):

| Run | Misses | Lost at | Branch mass × N: range (median) | At least 1/N | Atoms in the branch | Mass dropped by that compression |
|---|---|---|---|---|---|---|
| N = 32 probe | 8 | last or second-to-last compression | 0.04–1.6 (0.22) | 2 | 1–18 | 0.13–0.43 |
| N = 64 probe | 5 | last or second-to-last | 0.17–1.25 (0.39) | 1 | 1–32 | 0.17–0.41 |
| H3, N = 64 | 4 | last or second-to-last | 0.03–0.37 (0.23) | 0 | 2–26 | 0.09–0.39 |

- **A lost branch is young and light.** It is lost at the last or second-to-last compression before the miss, typically right after j hears something unlikely. In 14 of 17 cases its mass is below 1/N, so any compressor bounded by N atoms loses it with some probability. In 3 cases it weighs 1.2–1.6/N but is spread over 16–32 atoms, each below 1/N. Systematic draws can then drop all of them.
- **Systematic compression discards a quarter of the belief per update.** At N = 64 it keeps a median of 47 distinct atoms of a 246-atom posterior and drops 25% of the mass (27% at N = 32). Most dropped mass is harmless: other atoms predict the same opponent actions. That explains why misses are rare (about 5% of trials) despite the large dropped mass.
- **Optimal resampling at the same N would usually still lose them.** Its expected number of surviving branch atoms is a median of about 0.3 at N, and 1.3–2.4 at 2N. Removing most misses therefore needs roughly twice the distinct atoms, which is about twice the CPU at these sizes.
- **The coverage target needs more atoms than N.** To hold the expected dropped mass to ε, optimal resampling needs a median of 90, 129 and 169 atoms at ε = 0.1, 0.03 and 0.01, on posteriors produced by an N = 64 belief. The count grows further once more atoms are kept.

**Two H3 misses were not compression.** In trials 16 and 18, the miss is at step 3 and no compressed-away atom predicts j's action. With i's uncompressed posterior, j's action (OR, preferred by j's own search by a gap of 5.5 and 17.0) still has probability 0.

- **Cause:** the L3 condition modeled j with the stack `6400s/100`, i.e. j's model of i at level 1 solved with 100 simulations. The real j uses 400 (`modeled = 400`).
- **Check:** replaying with i's stack set to `6400s/400` gives OR probability 0.37 in both trials; the original stack reproduces the logged prediction exactly.
- **Why it matters:** with joint payoffs, j's choice depends on what it predicts i will do. So the "correct model" of 12.29 was correct in prior but not in j's computation.

**12.29's deficit, split by cause** (L3 − L2, i's 20-step reward):

| Trials | n | Mean difference |
|---|---|---|
| Nested-budget mismatch (t016, t018) | 2 | **−234.5** (−211, −258) |
| Compression miss (t004, t014, t017) | 3 | −12.7 (0, −79, +41) |
| No miss | 45 | +0.6 |

- **Revised reading of 12.29:**
  - At H3, almost the whole deficit (−469 of −481 summed over 50 trials) comes from the two trials where i's model of j computed differently from the real j.
  - Compression misses cost little reward there.
  - The bounded belief remains the cause of prediction misses: 17 of 19, and all of them at H2.
- **A correct model must match the opponent's computational frame,** that is, its nested budgets, not only its prior. Matched budgets still leave seed noise, since search trees differ, but they remove a systematic difference.
- **Consequences for the plan:**
  - Q3 (the compressor gate at H2, where the stacks match: 100 at every level for both agents) is unaffected.
  - An H3 re-run (Q4) should use a model of j that matches j's own configuration, whatever the compressor decision.

### 12.31 Compressor gate: no candidate keeps the realized branch at acceptable cost (October 9, 2026)

**Design** (`results/q3_compressor_gate_20261009`: `gate.py`, `analyze.py`, `pilot_table.txt`, `g1/`; source branch `bounded-belief/compressors`). Condition: the 12.26 probe (level-3 i with a correct prior against the grid's level-2 j, coordination Tiger, H2, root and modeled 100 simulations, 20 steps). Only i's compressor changes, and j stays systematic at N = 64. The control arm reproduced 12.27's trials exactly (7/7). Arms:
- systematic at N = 64 (control) and N = 96;
- optimal resampling at N = 48 (the control keeps about 41 distinct atoms, so this matches distinct atoms) and N = 64;
- coverage (optimal resampling to the fewest atoms with expected dropped mass ≤ ε, capped at 256) at ε = 0.1 and 0.03.

Kill criterion, fixed before the run: no pilot arm with fewer misses at ≤ 1.2× the control's CPU.

**Pilot cost** (i's CPU per episode, paired by seed; stopped at n = 6–7 per arm once the criterion was decided):

| Arm | n | CPU / control: median (range) | Mean support | Trials with a miss | i's log-loss |
|---|---|---|---|---|---|
| systematic 64 (control) | 7 | 1 (409 s) | 41 | 0 | 0.38 |
| systematic 96 | 7 | 2.08 (1.88–2.16) | 57 | 1 | 0.52 |
| optimal 48 | 7 | 2.03 (1.94–2.36) | 45 | 0 | 0.40 |
| optimal 64 | 6 | 3.77 (3.66–4.55) | 60 | 0 | 0.43 |
| coverage ε = 0.1 | 7 | 1.69 (1.07–1.85) | 138 | 0 | 0.38 |
| coverage ε = 0.03 | 0 | > 10 (two episodes at 67 CPU-min each, unfinished) | — | — | — |

**G1, belief accuracy against exact beliefs** (`belief_qualification`, 12 histories × 5 steps, cap 256): coverage ε = 0.03 passes (level TV mean 0.023, worst 0.063 at level 3; systematic N = 64: 0.012 and 0.061, 12.28). **Coverage ε = 0.1 fails:** level TV mean 0.32, worst 0.99, and 45% of level-3 steps above 0.2. Dropping up to 10% of the mass at every nesting level and every step compounds. Fixed-size arms at level 3 (mean level TV, worst in parentheses): systematic 64 0.012 (0.061), 96 0.0070 (0.040); optimal 48 0.0015 (0.032), 64 0.0018 (0.044). So optimal 48 is 8× more accurate than the control at matched distinct atoms, at 2× its CPU.

- **Decision: Q3 killed; episodes keep the systematic compressor at N = 64.** No candidate costs ≤ 1.2× the control. Optimal resampling costs about 2× per distinct atom, because the light atoms it keeps are distinct j-models, each needing its own nested solve. Coverage either fails G1 (ε = 0.1) or costs over 10× (ε = 0.03).
- **Consistent with 12.30:** keeping the young, light branches needs roughly twice the distinct atoms, and nested solves per distinct j-model dominate the cost.
- **Kept on `main`:** the miss metric (`analysis.metrics.predictions`, `prediction_misses`), so every bounded-belief run reports it.
- **Kept on branch `bounded-belief/compressors` only:** the compressor strategies.
- **Not tested, since 12.30 makes them low value:** reward effects (12.30 found compression misses cost little reward at H3) and the main run of 200 seeds.

### 12.32 A third level at horizon 3 with a matched model of j: no gain, no loss (October 9–11, 2026)

**Design** (`results/q4_h3_matched_20261009`: `h3_matched.py`, `h3_job.py`, `analyze.py`, `KILL_CRITERION.txt`; fixed before the run). The design is 12.29's, with one change: in the L3 condition, i models j exactly as j is configured. The stack is `1000/400`: j at level 2 as a 1,000-simulation depth-3 search, and its model of i at level 1 with 400 simulations. 12.29 used `6400s/100` (12.30). Only search and compression seeds differ from the real j. Other settings are unchanged: coordination Tiger, H3, root 1,000, compressed beliefs N = 64 (systematic, 12.31), trials 0–49 paired by seed, endpoint i's 20-step reward L3 − L2.
- **Precheck:** with this stack, j's actual action in 12.29's two mismatch trials gets probability 0.37 instead of 0.
- **Kill criterion:** the first 8 L3 episodes took a median 120 CPU-min (99–143), under the 414 threshold, so n stayed at 50.
- **Reproducibility:** the L2 episodes reproduce 12.29's exactly (50/50).

**Primary endpoint** (n = 50, paired t-test): **+0.8 [−6.7, 8.6], p = 0.84.**

| Measure | L3, matched | L2 | Difference [95%] | p |
|---|---|---|---|---|
| i's reward | 113.6 | 112.8 | +0.8 [−6.7, 8.6] | 0.84 |
| j's reward | 111.4 | 112.3 | −1.0 [−8.3, 5.6] | 0.79 |
| i's log-loss of j | 0.41 | 0.45 | −0.04 [−0.13, 0.06] | 0.38 |
| Trials with a prediction miss | 6% | 0% | | |
| i's CPU per episode (median) | 114 min | 0.6 min | | |

L3 matched vs 12.29's L3 on the same seeds: +10.4 [−2.0, 27.2], p = 0.19.

- **12.29's deficit was the model mismatch.** With a matched model, the two mismatch trials go from −211 and −258 to 0 and +41, and the mean difference goes from −9.6 to +0.8.
- **A third level still does not pay at H3.** The interval excludes a loss larger than 6.7 and a gain larger than 8.6 over 20 steps. The exact screen's 2.10 per decision (12.22) would add at most a few points per episode on the rare beliefs where level 3 decides differently, which is within that interval.
- **Prediction is slightly better but not significant,** consistent with 12.27 at H2 (0.41–0.43 against 0.49–0.50 there).
- **Compression misses remain rare and cheap.** Three L3 trials have a miss (two the same as in 12.29, plus trial 19). They average +3.7 against L2; the other 47 average +0.6.
- **Contribution 3, as of now:** modeling the opponent pays (level 1 → 2, +27 to +30, 12.27). A third level predicts somewhat better but earns the same reward at H2 (12.27) and H3 (here), in a game where the exact value of the third level is small and rare.

### 12.33 Root budgets: the root's cost is fixed, so stopping early saves nothing (October 11, 2026)

**Question.** Can a stopping rule at the root spend fewer simulations on easy real decisions, or buy lower Q error per CPU-second (PLAN, root adaptive budgets, R0–R1)?

**Design** (`results/r0_root_replay_20261011`: `r0_job.py`, `replay.py`, `count_solves.py`, `KILL_CRITERION.txt`; arms and the adoption rule fixed in PLAN.md before the run). The problems are the current gates, published Tiger:
- L2 H4, modeled `6400s`, 380 problems;
- L3 H3 k = 0, stack `6400s/100`, 162 problems.

For each problem, one root tree was seeded as a flat search of 4,000, scored with certified bounds at 125, 250, 500, 1,000, 2,000 and 4,000, and compared with the exact I-POMDP. Each checkpoint of n is a flat search of n with another seed, so all arms are paired. Growth and scoring CPU were timed separately. Three pilot trees of 125 were grown in a fresh bank; this overstates the cost of rule (a), which shares one bank online.

The arms:
- flat budgets;
- (a) the modeled stopping test (12.11, 12.15) applied at the root, with a cap of 4,000;
- (b) stop once certified intervals separate the leading action;
- (a or b).

**Check:** the 1,000 checkpoint reproduces 12.14: 380/380 with optimal-action Q error 0.121 (12.14: 0.117), and 162/162.

**Cost structure.** Almost all of a decision's CPU goes into the first 125 simulations. The median is 16.9 s at L2 H4 and 23.3 s at L3 H3. That is when the planner solves every modeled policy reachable from the tree (in two traced problems: 56 j-L1 policies at L2 H4, 215 policies at L3 H3). No further policy is solved up to 4,000. After that, a root simulation costs 0.35 ms (L2 H4) or 0.20 ms (L3 H3) at the median.

| Arm | L2 H4 optimal | CPU vs flat 1,000 | Q error | L3 H3 optimal | CPU vs flat 1,000 | Q error |
|---|---|---|---|---|---|---|
| flat 125 | 374/380 | 0.97 | 0.763 | 162/162 | 0.99 | 0.320 |
| flat 250 | 380/380 | 0.98 | 0.407 | 162/162 | 0.99 | 0.077 |
| flat 500 | 380/380 | 0.99 | 0.244 | 162/162 | 1.00 | 0.006 |
| flat 1,000 | 380/380 | 1.00 (16.0 s) | 0.121 | 162/162 | 1.00 (23.1 s) | 0.0001 |
| flat 2,000 | 380/380 | 1.02 | 0.064 | 162/162 | 1.01 | 0 |
| **flat 4,000** | **380/380** | **1.06** | **0.031** | **162/162** | **1.02** | **0** |
| (a) test | 380/380 | 1.97 | 0.617 | 162/162 | 2.00 | 0.279 |
| (b) certified | 380/380 | 1.02 | 0.210 | 162/162 | 0.99 | 0.275 |
| (a or b) | 380/380 | 1.97 | 0.617 | 162/162 | 2.00 | 0.298 |

Q error is the optimal-action Q error, as a mean. 380/380 is 99.0–100% and 162/162 is 97.7–100% (Clopper–Pearson).

- **No rule qualifies** under the pre-fixed criteria. The most any rule can save is the 3% between flat 125 and flat 1,000. Every rule stops early on most problems and so pays in Q error. Rule (a)'s pilots also re-pay the fixed cost; even if they were free, it would stop at a mean of 291 simulations.
- **Certified separation is sound but does not cut the error.** It stops once the decision is certified, which at L3 H3 is usually by 125–250 simulations, while the optimal action's own value is still off by 0.28. At L2 H4 the rollout frontiers keep every interval wide (median width 23.5 even at 4,000; no root exact), so it rarely fires.
- **The cheap lever is the flat root budget.** Going from 1,000 to 4,000 cuts L2 H4's optimal-action Q error by 3.9× (0.121 → 0.031; median 0.073 → 0.022; maximum 0.78 → 0.34; paired Wilcoxon p = 5 × 10⁻³⁹) at 1.06× the CPU. At L3 H3, 4,000 makes 161 of 162 scored roots exact.
- **Scope:** exact beliefs and the gate problems. Episodes use bounded beliefs (N = 64) and the coordination Tiger, where the number of reachable j-models may keep growing with the root budget. That is measured next before any default changes.

### 12.34 Root budgets in episodes: nearly free for one decision, about 1.4× with a matched model of j (October 11, 2026)

**Question.** Does 12.33's cost structure hold at real episode decisions (bounded beliefs, the coordination Tiger)? It could fail there, because the number of reachable j-models might keep growing with the root budget.

**Design** (`results/r2_episode_root_cost_20261011`: `probe.py`, `matched_cost.py`, `KILL_CRITERION.txt`). The configuration is 12.32's: `together_safe = 40`, N = 64, H3, root 1,000. The L3 condition uses stack `1000/400`; the L2 condition's i is the grid's agent.
- **Probe:** at each of i's real decisions, a measurement tree seeded as a flat 4,000 search is grown first. It records growth CPU and newly solved modeled policies at 125, 250, 500, 1,000, 2,000 and 4,000. The real 1,000-simulation decision then runs unchanged. 3 trials × 6 steps per condition. **Check:** all 36 decisions reproduce 12.32's root values exactly (to 1e-9).
- **Matched-model cost:** the L3 condition with every root at 4,000 (i, the real j, and i's model of j: stack `4000/400`) against 1,000. Same run, paired by trial, 3 trials × 4 steps.

**Results.**
- **One decision.** The fixed cost again sits in the first 125 simulations. In L3, 9,789 of 9,917 modeled policies were solved there, and the median growth to 125 was 101 s. Growing from 1,000 to 4,000 costs 1.015× the 1,000 tree (max 1.025×). The L2 decision costs 2.9 s at 1,000 and 1.21× at 4,000.
- **Episodes with a matched model of j.** CPU is **1.32–1.42×** (per trial: 360 → 511 s, 391 → 535 s, 444 → 588 s), and i's and j's actions are identical in all three trials. This cost does not front-load. Each distinct j-model that i solves runs j's own root, and that root is now 4,000.

**Verdict.**
- **Gates and Q-error reporting** (a real decision against a fixed opponent model): use root 4,000. The cost is 1.02–1.06× and the Q error falls 3.9× (12.33).
- **Episodes keep root 1,000:** decisions do not change, and 12.30 requires the model of j to match j's root, which costs about 1.4× at L3.
- **Interim:** n = 3 trials of 4–6 steps. These are cost measurements only; no accuracy claim rests on them.

### 12.35 Error split at root 4,000: nothing left for a deeper-node redesign (October 11, 2026)

**Design** (`results/r3_attribution_4000_20261011`). 12.24's exact split (`attribution.py`, `analyze.py`), unchanged except root 4,000: L2 H4, modeled `6400s`, the 95 problems, seed 0. The residual is below 2 × 10⁻¹⁵.

| Root | Search error, optimal action | Scored error, optimal action (mean / max) | Chance error | Local action loss | Frontier |
|---|---|---|---|---|---|
| 1,000 (12.24) | 0.164 | 0.132 / 0.78 | 69% | 22% | 9% |
| **4,000** | **0.068** | **0.026 / 0.093** | 87% | 9% | 4.5% |

(Shares are of the summed absolute parts of the search error.)
- **Deeper choices and rollouts now come to about 0.009** of the remaining error (13% of 0.068). These were the targets of R3's candidate redesign (variance-reduced backups, better rollouts). There is too little left for such a change to pay off, so none is planned.
- What remains is mostly chance error with three steps left, where scoring keeps the sampled weights because some branches are missing.

### 12.36 Nested solves twice as fast, with a third of the memory, bit-identical (October 11, 2026)

**Question.** Nested solves are almost all of a decision's CPU (12.33–12.34). Can they be made cheaper without changing any output (W1's standard, 12.12)?

**Profile** (`results/s0_nested_profile_20261011`): cProfile of 12.32's L3 condition, trial 0, 4 steps (462 s profiled, 445 s of it inside modeled solves). The waste was in three places:
- **Canonical sort keys and digests (about 32%).** Every compression and digest re-encoded the opponent's physics for every atom: 5.4M encodings for a handful of frames.
- **The rollout-memory update (14%).** 2M Tiger Bayes updates, over few distinct arguments.
- **Recursive belief comparisons (7%).** These came from a cache-size mismatch.

**Changes** (commits `661526c`, `7183339`):
- Each frame's part of an atom's canonical encoding is computed once per frame. Rows are assembled byte-identically to `json.dumps`. Tests check keys, digests and canonical order against the JSON definitions on Tiger, UAV and Wumpus beliefs.
- The rollout-memory update is memoized by value at module level, so physics gains no attribute.
- The compressed filter's caches went from 4,096 to 65,536 entries. At 4,096, an L3 episode recomputed 22% of its updates. Each recomputed posterior is a new object equal to ones cached downstream, so later lookups compared whole beliefs recursively, and the duplicates stayed in memory.
- Compression computes the canonical order and the digest in one pass.

**Gates** (fixed in PLAN before each change). Old and new snapshots ran paired in the same run.
- **Bit-identical outputs:** 40 R0 gate problems (L2 H4 and L3 H3; every checkpoint's values, bounds and exactness, plus references and pilots). 12.32-configuration episodes: 4 for each change, two of them 10 steps long. Every non-timing column matched, and decisions matched 12.32's stored episodes.
- 315 tests pass; ruff is clean.

**Net effect** (`results/s4_net_effect_20261011`): pre-phase code against final `main`, paired, 10-step episodes, all bit-identical.

| Episode (10 steps) | CPU before | CPU after | Ratio | Peak memory before | after |
|---|---|---|---|---|---|
| L3, trial 0 | 2,092 s | 1,079 s | 0.52 | 1,910 MB | 678 MB |
| L3, trial 1 | 1,917 s | 992 s | 0.52 | 1,748 MB | 635 MB |
| L3, trial 3 | 2,029 s | 1,015 s | 0.50 | 1,868 MB | 612 MB |
| L2, trial 0 | 66 s | 38 s | 0.57 | 168 MB | 281 MB |

Exact-belief gate problems (no compression) took 0.74× the CPU at the median (range 0.60–0.90) from the first change alone (`results/s1_gates_20261011`).

- **Level-3 episodes are about 2× faster and use about a third of the memory.** So about 3× more of them fit on the shared machine at once.
- **What remains is essential work.** After the changes, the largest item is the exact enumeration inside the belief update (about a third of the CPU). Belief equality is about 7%, and the Tiger kernels are about 11% (memoizing those did not help before, HANDOFF 6). No single bit-identical change left would save more than about 7%. A further gain needs a solver-level change, which would alter outputs and needs its own gate (12.15's replay standard).
- Earlier CPU figures (12.29, 12.32: 114 CPU-min per L3 episode) predate these changes.

### 12.37 The packaged suite, and the level matrix at levels 0–3 on both games (October 11–12, 2026)

**Packaging** (PLAN.md, "Tiger experiment suite", E; commit `df8803d`):
- `experiments.suite` runs named study cells on shared trial ids (so cells pair by trial), round-robin, with one supervised process per episode. Its builders are `level_matrix`, `planner_comparison` and `prior_sensitivity`. Endpoints are in `analysis.endpoints`.
- **Gate E** (`results/e_suite_gate_20261011`): the suite reran 12.27's cells (levels 1–3 × 1–3, trials 0–1, 18 episodes). Every row equalled the stored episodes (all columns except CPU, wall time and memory). So 12.27's 900 episodes were imported as this matrix's coordination cells at levels 1–3 instead of being rerun.

**Design** (`results/m1_level_matrix_20261011`; `m1.py`, `analyze.py`, `KILL_CRITERION.txt`, all fixed before the run):
- **Games:** the coordination Tiger (`together_safe = 40`) and the published Tiger.
- **Agents:** levels 0–3 for both agents. Level 0 acts uniformly at random. An agent at level L ≥ 1 holds a uniform prior over the opponent's levels 0 … L − 1, and so does every nested model.
- **Configuration:** H2, 100 simulations, compressed beliefs N = 64, 20 steps, n = 100 per cell, all cells on the same seeds.
- **Primary endpoint:** one more level, against the same opponent level and seeds (the agent at k + 1 minus the same agent at k), for both agents. Holm over the 24 comparisons per game. Intervals are 95% trial bootstrap.

**Primary endpoint, comparisons at level ≥ 1** (the 8 comparisons per game from level 0 to level 1 all gain 535–653, Holm p < 0.001):

| Step | Opponent | Coordination, i | Holm p | Coordination, j | Holm p | Published, i | Holm p | Published, j | Holm p |
|---|---|---|---|---|---|---|---|---|---|
| 1 → 2 | 0 | −2.6 [−9.8, 4.0] | 1 | −4.0 [−11.8, 3.8] | 1 | −0.2 [−0.6, 0.0] | 1 | +0.1 [0.0, 0.3] | 1 |
| 1 → 2 | 1 | **+27.2 [15.9, 38.4]** | < 0.001 | **+29.8 [16.5, 43.5]** | < 0.001 | −4.4 [−8.9, 0.1] | 0.83 | −0.6 [−7.4, 6.3] | 1 |
| 1 → 2 | 2 | +10.4 [1.5, 20.0] | 0.38 | +6.2 [−5.1, 17.1] | 1 | −2.0 [−6.4, 2.4] | 1 | +0.6 [−4.2, 5.7] | 1 |
| 1 → 2 | 3 | **+15.5 [5.5, 26.1]** | 0.044 | **+14.7 [5.9, 23.7]** | 0.029 | −3.0 [−7.0, 1.0] | 1 | +2.3 [−4.1, 8.7] | 1 |
| 2 → 3 | 0 | +2.3 [−3.0, 8.0] | 1 | −3.7 [−13.1, 4.9] | 1 | −1.4 [−4.5, 0.8] | 1 | −2.6 [−6.4, 0.3] | 1 |
| 2 → 3 | 1 | −3.1 [−9.7, 3.3] | 1 | −1.0 [−7.5, 5.6] | 1 | +5.2 [1.9, 9.4] | 0.15 | +2.5 [−4.3, 9.0] | 1 |
| 2 → 3 | 2 | +6.8 [1.9, 12.2] | 0.15 | +5.0 [0.3, 9.6] | 0.40 | +3.3 [−0.6, 7.7] | 1 | +1.7 [−3.9, 6.2] | 1 |
| 2 → 3 | 3 | +6.2 [0.2, 12.3] | 0.44 | +1.7 [−3.5, 6.2] | 1 | +4.3 [0.8, 8.8] | 0.62 | −1.0 [−7.5, 4.8] | 1 |

**Per cell** (20-step reward; log-loss is the agent's prediction of the opponent's next action, where a uniform guess is log 3 = 1.10; "misses" are trials in which a prediction gave the opponent's actual action probability 0):

| i | j | Coordination: i's, j's reward | Welfare | Published: i's, j's reward | Welfare | Misses (i, j), coord. / publ. |
|---|---|---|---|---|---|---|
| 0 | 0 | −500, −533 | −1,033 | −561, −593 | −1,155 | — |
| 0 | 1 | −556, 26 | −530 | −586, −1 | −586 | 0 / 0 |
| 0 | 2 | −548, 22 | −526 | −589, −1 | −589 | 0 / 1 |
| 0 | 3 | −539, 19 | −520 | −583, −3 | −586 | 2 / 1 |
| 1 | 0 | 35, −554 | −519 | 7, −582 | −575 | 0 / 0 |
| 2 | 0 | 33, −559 | −526 | 7, −582 | −576 | 0 / 0 |
| 3 | 0 | 35, −561 | −526 | 5, −584 | −578 | 3 / 3 |
| 1–3 | 1–3 | 12.27's table (coordination) | | 8–17 each | 22–29 | 0–6 per cell |

- **Reasoning at all is what pays most.** Against every opponent, a level-1 agent earns 535–653 more over 20 steps than a level-0 one, which opens doors at random.
- **Against a random opponent, more levels add nothing** on either game (all |difference| ≤ 4, Holm p = 1). The level posterior identifies a random opponent almost surely by step 20 (0.95–1.00), so modeling buys no information that changes the best response.
- **The coordination results at levels 1–3 are 12.27's.** Level 1 → 2 pays against opponents at levels 1 and 3, and a third level gains nothing that survives the correction.
- **The published Tiger is the control, and it holds.** At N = 64, no comparison above level 0 survives the correction (smallest Holm p 0.15). This replaces 12.21's N = 32 grid for levels 1–3.
- **Bounded-belief misses stay rare:** 0–6 trials per 100 per cell. Against a random opponent, the level-3 agent misses in 3 trials per game (its compressed belief had dropped the level-0 hypothesis).
- **Cost:** a level-3 against level-3 episode now takes about 1.9 CPU-min (12.27: 6.3, before 12.36). The CPU figures stored with the imported cells predate 12.36.

### 12.38 I-POMCP against IPF + RTS in episodes (October 11–12, 2026)

**Design** (`results/p_planners_20261011`; `p.py`, `analyze.py`, `KILL_CRITERION.txt`, fixed before the run):
- **Agents:** i is level 2 (prior uniform over j at levels 0–1). The real j is a level-1 I-POMCP agent with root R: R = 1,000 at H3, and a flat R = 6,400 at H4 (at least the gated `6400s`).
- **Arms for i:**
  - I-POMCP: root 1,000, compressed N = 64. It models j with j's own solve (R simulations), so the model matches j's configuration.
  - IPF + RTS: the sampled IPF with N = 64, at K = 2 and K = 4 sampled observations per node. As in Doshi & Gmytrasiewicz (2009), it also models j with RTS.
- **Suites:** H3 on both games, n = 100; H4 on the coordination Tiger, n = 50. 20 steps.
- **Primary endpoint:** i's 20-step reward, I-POMCP minus each RTS arm, paired by trial, Holm over the 6 comparisons.
- **Kill check:** the median H4 K = 4 episode took 15.8 CPU-min, under the 34 limit, so n stayed 50.

| Game, H | Arm | i's reward [95%] | I-POMCP − arm [95%] | Holm p | j's reward | i's log-loss | i's CPU per decision (s) | I-POMCP / arm CPU (median per trial) |
|---|---|---|---|---|---|---|---|---|
| Coordination, H3 | I-POMCP | 87.3 [72.4, 102.1] | — | — | 92.3 | 0.52 | 4.7 | — |
| | RTS K = 2 | 49.6 [30.4, 68.2] | **+37.8 [17.8, 58.9]** | 0.003 | 66.4 | 0.67 | 2.4 | 1.9 |
| | RTS K = 4 | 59.3 [44.5, 73.7] | **+28.0 [13.3, 43.6]** | 0.003 | 70.2 | 0.63 | 6.6 | 0.69 |
| Published, H3 | I-POMCP | 7.6 [−1.1, 15.8] | — | — | 13.7 | 0.50 | 4.7 | — |
| | RTS K = 2 | 11.7 [4.5, 17.6] | −4.1 [−10.3, 2.0] | 0.39 | 8.3 | 0.56 | 3.1 | 1.5 |
| | RTS K = 4 | 11.7 [5.4, 17.5] | −4.1 [−10.3, 1.7] | 0.39 | 8.4 | 0.57 | 8.9 | 0.52 |
| Coordination, H4 | I-POMCP | 87.0 [64.0, 107.2] | — | — | 92.5 | 0.53 | 67.3 | — |
| | RTS K = 2 | 62.6 [41.3, 82.9] | +24.4 [−0.6, 48.1] | 0.17 | 74.5 | 0.66 | 10.1 | 6.5 |
| | RTS K = 4 | 61.1 [36.4, 83.7] | +25.9 [2.6, 50.1] | 0.16 | 78.5 | 0.64 | 48.7 | 1.4 |

- **Where modeling pays (the coordination Tiger), I-POMCP earns more than IPF + RTS.** At H3 the gain is 28–38 over 20 steps (Holm p = 0.003), including against K = 4, which costs more CPU per decision than I-POMCP (I-POMCP uses 0.69× its CPU). j earns more too, by 22–26, so the gain is better coordination, not taking reward from j.
- **At H4 the difference is the same size** (+24 and +26) but does not survive the correction at n = 50. I-POMCP's flat 6,400-simulation model of j makes its decisions cost 1.4× RTS K = 4 and 6.5× K = 2.
- **On the published Tiger, where modeling cannot pay, the planners do not differ** (−4.1, Holm p = 0.39).
- **I-POMCP predicts j better** (log-loss 0.50–0.53, against 0.56–0.67 for RTS). This matches 12.20's ablation: RTS's error is search error, and RTS also models j with RTS.
- **This complements 12.20,** which compared decisions against the exact I-POMDP (I-POMCP 380/380; RTS 95–99.5%).

### 12.39 Prior sensitivity at levels 3 and 4 (October 12–13, 2026)

**Design** (`results/r_priors_20261011`; `r.py`, `analyze.py`, `KILL_CRITERION.txt`, fixed before the run):
- **Agents:** coordination Tiger, H2, N = 64, 100 simulations, 20 steps, n = 100. i is at level 3. The real j is the level matrix's agent at level 1.
- **Arms (i's prior over j's level):**
  - correct, {1: 1};
  - over-estimated, {2: 1};
  - under-estimated, {0: 1};
  - uniform, {0, 1, 2: 1/3}. This is 12.37's cell (3, 1), imported.
- Every model of j at a given level is the matrix's real agent at that level.
- **Primary endpoint:** i's 20-step reward, each arm minus correct, paired. Holm over the 6 comparisons (3 arms × levels 3 and 4). Level 4: i at level 4 against the matrix's j at level 2, n = 40 (12.40's configuration; its uniform arm is 12.40's cell (4, 2)).

| Prior | i's reward | j's reward | Arm − correct [95%] | Holm p (6) | i's log-loss | Miss trials | i's posterior on level 1 at step 5 / 10 / 20 |
|---|---|---|---|---|---|---|---|
| Correct | 93.9 | 94.9 | — | — | 0.47 | 3 | 1.00 / 1.00 / 1.00 |
| Over-estimated | 76.4 | 81.1 | **−17.5 [−28.8, −6.6]** | 0.011 | 0.88 | 31 | 0 / 0 / 0 |
| Under-estimated | 63.9 | 53.9 | **−30.0 [−41.2, −18.7]** | < 0.001 | 1.10 | 0 | 0 / 0 / 0 |
| Uniform | 88.0 | 89.4 | −6.0 [−13.1, 1.1] | 0.41 | 0.55 | 4 | 0.44 / 0.53 / 0.67 |

- **A wrong point prior costs reward, and in both directions.**
  - Over-estimating j's level costs 17.5.
  - Under-estimating it as random costs 30.0. This arm reproduces cell (1, 1) of 12.37 exactly: an agent sure its opponent is random best-responds as a level-1 agent.
  - A point prior cannot be revised, so the posterior stays at 0.
- **The over-estimated model is refuted by the data but cannot be dropped.** It gives j's actual action probability 0 in 31 of 100 trials. With every atom a level-2 model, nothing in the belief can absorb an action only a level-1 j takes. So most of these misses are presumably the model's, not compression's (the correct arm has 3); no replay has separated the two (12.30's method would). Either way, the miss metric also flags a misspecified opponent model.
- **A uniform prior costs little** (−6.0, not significant): i learns j's level (0.67 by step 20) while acting nearly as well as with the correct prior.
**Level 4** (i at level 4, real j at level 2, n = 40):

| Prior | i's reward | j's reward | Arm − correct [95%] | Holm p (6) | i's log-loss | Miss trials | i's posterior on level 2 at step 20 | CPU per step (s) |
|---|---|---|---|---|---|---|---|---|
| Correct, {2: 1} | 112.6 | 119.0 | — | — | 0.33 | 0 | 1.00 | 5.8 |
| Over-estimated, {3: 1} | 114.6 | 119.5 | +1.9 [−3.6, 9.6] | 1 | 0.41 | 3 | 0 | 177.4 |
| Under-estimated, {1: 1} | 113.9 | 119.9 | +1.2 [−4.1, 6.1] | 1 | 0.38 | 0 | 0 | 0.3 |
| Uniform, {0–3: 1/4} | 117.9 | 115.9 | +5.2 [−1.3, 14.0] | 0.57 | 0.40 | 1 | 0.35 | 65.4 |

- **At level 4 a wrong prior costs nothing.** Against a level-2 opponent, believing it is level 1 or level 3 earns the same as the truth. Levels 2–4 act alike in this game (12.40), so a model one level off still predicts the opponent well (log-loss 0.38–0.41 against 0.33). At level 3 the wrong models were level 0 (random) and level 2 against a real level-1 opponent, and level-1 play differs from both.
- **So what a misspecified prior costs depends on whether the wrong level acts differently, not on the size of the error.** The correct prior is also the cheapest to plan with (5.8 s per step, against 177 s when i must solve a level-3 opponent model).

### 12.40 Level 4 against levels 0–4 on the coordination Tiger (October 12, 2026)

**Design** (`results/m2_level4_20261011`; `m2.py`, `analyze.py`, `KILL_CRITERION.txt`, fixed before the run): 12.37's matrix extended by the level-4 row.
- **Agents:** i at level 4 against j at levels 0–4. The coordination Tiger is symmetric in the agents, so the column would repeat the row on other seeds. The scope was approved by the user on October 12.
- **Configuration:** H2, 100 simulations, N = 64, 20 steps, n = 40, on 12.37's seeds. Trials 0–3 of (4, 4) and (4, 2) are the cost pilot's episodes (same code and configuration).
- **Primary endpoint:** one more level (3 → 4):
  - i at level 4 minus i at level 3 against j at levels 0–3, paired with 12.37's cells (3, j) on trials 0–39;
  - j at level 4 minus j at level 3 against i at level 4.

  Holm over these 5 comparisons.
- **Cost:** 69 CPU-h. Median CPU-min per episode: 0.9 against level 0, 13 against level 1, 22–23 against levels 2–3, 45 for level 4 against level 4. The pilot's 61–93 had been measured under full load. Peak memory was 1.3 GB per episode; no episode failed.

| Step 3 → 4 | Opponent level | Difference [95%] | Holm p |
|---|---|---|---|
| i | 0 | −1.7 [−11.3, 6.4] | 1 |
| i | 1 | +3.8 [−1.6, 9.4] | 0.99 |
| i | 2 | +1.7 [−3.8, 8.1] | 1 |
| i | 3 | +1.7 [−3.8, 8.1] | 1 |
| j | 4 | −1.0 [−4.1, 2.1] | 1 |

| i | j | i's reward | j's reward | Welfare | i's log-loss | Miss trials (i) | i's posterior on j's level at step 20 | CPU per step (s) |
|---|---|---|---|---|---|---|---|---|
| 4 | 0 | 39.4 | −548.3 | −508.9 | 1.40 | 2 | 0.94 | 6.1 |
| 4 | 1 | 92.4 | 99.8 | 192.1 | 0.59 | 3 | 0.63 | 38.8 |
| 4 | 2 | 117.9 | 115.9 | 233.8 | 0.40 | 1 | 0.35 | 65.4 |
| 4 | 3 | 119.6 | 119.3 | 238.8 | 0.39 | 1 | 0.38 | 64.0 |
| 4 | 4 | 119.1 | 118.3 | 237.3 | 0.39 | 1 | — | 137.9 |

- **A fourth level changes almost nothing.** Level 4 takes the same actions as level 3 in 34–35 of 40 trials against opponents at levels 2 and 3. The few trials where they differ are the same ones with the same reward swings (+41, +99, −41, +11, −41), which is why those two rows match exactly. No difference exceeds 4.
- **Prediction no longer improves:** log-loss 0.39–0.40 at level 4 against 0.41–0.43 at level 3 (12.27). Identification of the opponent's level stays weak: 0.35–0.38 for levels 2–3 against a prior of 1/4.
- **Together with 12.37, the coordination Tiger's value of depth is settled at H2:**
  - level 0 → 1 is worth about 600;
  - level 1 → 2 is worth 15–30 against intentional opponents;
  - level 2 → 3 is worth +5 to +7, not significant;
  - level 3 → 4 is worth nothing measurable.

  The exact screen (12.22) predicted this ordering.

### 12.41 UAV reconnaissance against exact references: L1 H2–H5, L2 H2–H4, L3 H2 (October 13–14, 2026)

**Design** (`results/u4_uav_gates_20261013`; `u4.py`, `l2h4/job.py`, `l2h4/score.py`, `analyze.py`, `ENDPOINTS.md`, `KILL_CRITERION.txt`; endpoints fixed before the run, rungs added during it are listed in `ENDPOINTS.md`):
- **Domain:** UAV reconnaissance with the declared values of `DOMAINS.md` (3×3 grid, 81 states; the UAV i spots the target j by entering its sector, +50, −5 per step; γ = 0.95).
- **Problems** (`experiments.problems.uav_problems`): at each level, one named law per agent over the other agent's sector. The laws are uniform, and top-row, near-start and far-corner, each with 0.75 of the mass on the named sectors and 0.25 uniform. Each agent knows its own sector. This gives 4 problems at level 1, 16 at level 2 and 64 at level 3. History 0, so seeds add planner randomness only.
- **References:** level 1 is scored against the exact best response to a uniformly random target. Levels 2–3 are scored against the exact I-POMDP, in which every lower level is exact.
- **Planner:** I-POMCP with compressed beliefs, chance scoring and per-action certified bounds. "Root" is i's simulations per decision; "modeled" is how the target's level is solved (`exact`, or I-POMCP with that many simulations).
- **Pass rule:** 100% optimal at some rung; the smallest such rung is the qualified budget.
- **Cost:** about 31 CPU-h, of which 28 were planner CPU, plus about 10 CPU-h lost in a first L2 H4 attempt that was stopped (below). Peak memory was 0.3 GB per process; no job failed.

**Results** (n = decisions; optimal = zero first-action loss; 95% Clopper–Pearson intervals; optimal-action Q error = |Q̂ − Q*| on the exactly optimal actions; "bounds" = decisions whose certified intervals contain every exact Q, reported where the reference is the searched model):

| Level, horizon | Root | Modeled | n | Optimal [95%] | Max loss | Mean optimal-action Q error | Bounds |
|---|---|---|---|---|---|---|---|
| L1 H2 | 1,000 | — | 40 | 40/40 [0.912, 1] | 0 | 0 | 40/40 |
| L1 H3 | 1,000 | — | 40 | 40/40 [0.912, 1] | 0 | 0 | 40/40 |
| L1 H4 | 1,000 | — | 40 | 34/40 [0.702, 0.943] | 3.82 | 0.84 | 40/40 |
| L1 H4 | **4,000** | — | 40 | **40/40** [0.912, 1] | 0 | 0 | 40/40 |
| L1 H5 | 1,000 | — | 40 | 26/40 [0.483, 0.794] | 8.39 | 5.35 | 40/40 |
| L1 H5 | 4,000 | — | 40 | 30/40 [0.588, 0.873] | 3.16 | 1.04 | 40/40 |
| L1 H5 | 16,000 | — | 40 | 39/40 [0.868, 0.999] | 0.65 | 0.16 | 40/40 |
| L1 H5 | **64,000** | — | 40 | **40/40** [0.912, 1] | 0 | 0 | 40/40 |
| L2 H2 | 1,000 | exact / 100 / 400 | 80 each | **80/80** each [0.955, 1] | 0 | 0 | 80/80 (exact) |
| L2 H3 | 1,000 | exact | 80 | **80/80** [0.955, 1] | 0 | 0 | 80/80 |
| L2 H3 | 1,000 | 100 | 80 | 63/80 [0.682, 0.871] | 2.02 | 0.52 | — |
| L2 H3 | 1,000 | **400** | 80 | **80/80** [0.955, 1] | 0 | 0 | — |
| L2 H4 | 1,000 | exact | 80 | 54/80 [0.561, 0.776] | 2.85 | 2.30 | 16/16 |
| L2 H4 | 4,000 | exact | 80 | 78/80 [0.913, 0.997] | 0.44 | 0.22 | — |
| L2 H4 | **16,000** | exact | 80 | **80/80** [0.955, 1] | 0 | 0 | — |
| L2 H4 | 1,000 | 100 | 80 | 40/80 [0.386, 0.614] | 37.97 | 3.80 | — |
| L2 H4 | 1,000 | 400 | 80 | 34/80 [0.315, 0.541] | 37.97 | 3.82 | — |
| L3 H2 | 1,000 | exact / 100 | 320 each | **320/320** each [0.989, 1] | 0 | 0 | 320/320 (exact) |

- **I-POMCP is optimal on UAV wherever an exact reference exists, at a qualified budget.** The qualified budgets are: root 1,000 at H2–H3 for every level run; root 4,000 at L1 H4; root 64,000 at L1 H5; root 16,000 at L2 H4 with exact opponent models.
- **Long horizons need more root simulations than on Tiger** (Tiger L1 H5 is 220/220 at 1,000, 12.1). UAV has 5 actions and 3 observations per step against Tiger's 3 and 2, and a terminal spotting event, so the tree at H5 is about 15 times wider. The misses are overestimates of suboptimal moves, the max bias of search on noisy children: at L1 H5 with 4,000 simulations, E is valued 10.9 against an exact 8.3. They vanish as the root tree becomes exact; at every qualified rung the optimal-action Q error is 0.
- **Root budget at level 2 is nearly free** when the target is solved exactly: CPU per decision is 267–269 s at roots 1,000, 4,000 and 16,000 alike, all spent on the exact nested solve (12.33's finding on Tiger).
- **The modeled target's budget must be qualified separately at H4.** At H3, 400 simulations for the target suffice (80/80); 100 do not (63/80). At H4, both 100 and 400 lose up to 38 on near-start problems. There, the exact-model planner rightly picks E (exact Q 30.2), but a finite model of the target values E at 15.8, so i picks N. That is opponent-model error, not i's own search error. Its qualification (a modeled ladder at the beliefs the search queries, `modeled_qualification`, as for Tiger in 12.4) is the next step.
- **Certified bounds held in every decision checked:** 1,016 of 1,016 (520 at level 1, 176 exact-model rows at level 2 with the full reference, 320 at level 3).
- **How L2 H4 was run.** The first attempt ran it as panel jobs of 16 problems each, and those were heading past the 2-hour job timeout, so it was stopped. It reran as one process per (problem, rung, seed), with the exact reference solved once per problem. The reference does not depend on the seed or the rung; at L2 H3 it was identical in all 15 rows of each problem. Planner-only rows (`reference="none"`) were scored offline with `evaluate_case`'s formulas; a planner-only rerun of an L2 H3 row reproduced its values, loss and Q error exactly. Certified bounds are therefore recorded only for the 16 reference rows at L2 H4.

### 12.42 UAV L2 H4: the target's modeled budget, qualified where i's search queries it (October 14, 2026)

**Question.** In 12.41, L2 H4 decisions were optimal at root 16,000 only with the target solved exactly; with the target modeled at 100 or 400 simulations they lost up to 38. Which modeled budget makes i's decisions exact?

**U4b: qualification** (`results/u4b_uav_qualification_20261014`; `u4b.py`, `ENDPOINTS.md`, `KILL_CRITERION.txt`; `experiments.modeled_qualification` with UAV problems, as for Tiger in 12.4):
- **Phase 1:** i at level 2, H4, root 16,000, with the target modeled at 400. All 16 UAV problems × seeds 0–1 = 32 problems; each of i's decisions is scored against the exact I-POMDP.
- **Phase 2:** every target model that i's searches queried (2,040 queries at 378 distinct beliefs) is re-solved at each rung and scored against the exact level-1 target at depth 4. The rung equal to 400 reproduced i's policy at every query.
- **Engineering:** the driver now takes `experiments.problems` problem lists and a `physics` (`ecb8ed0`). Gate: 608 Tiger qualification rows identical before and after (`results/u4b_qualification_gate_20261014`).
- **Cost:** about 5 CPU-h for phase 1 and 2 CPU-h for phase 2.

| Rung for the target | Wrong of 2,040 queries | Distinct wrong beliefs (of 378) | Correct rate [95%] | Max loss | CPU per query |
|---|---|---|---|---|---|
| 400 | 948 | 200 | [0.513, 0.557] | 0.73 | 0.9 s |
| 1,600 | 368 | 80 | [0.802, 0.836] | 0.10 | 2.4 s |
| 6,400, sequential stopping | **4** | **1** | [0.995, 0.9995] | 0.027 | 12.6 s |

- **With the target at 400, i is wrong in 16 of 32 problems** (losses 0.30–11.96), confirming 12.41's diagnosis: at H4 the binding error is the target's model, not i's search.
- **The one belief 6,400 with stopping still gets wrong is a near-tie.** Its exact values are listen −5.004 against N −5.031, a gap of 0.027. 400 and 1,600 get it right; 6,400 with stopping stops early (2.5 s against a mean of 12.6 s) and picks N. It is queried in 4 problems, in all of which i's decision is right. This is the same kind of residual as Tiger's L3 H4 near-ties (gaps ≤ 0.03, 12.14). Under the driver's zero-wrong rule no rung qualifies, so 6,400 with stopping was taken to the confirmation below.

**U4c: confirmation in i's decisions** (`results/u4c_uav_confirm_20261014`; `job.py`, `score.py`, `ENDPOINTS.md`, `KILL_CRITERION.txt`): i at level 2, H4, root 16,000, the target at 6,400 with stopping. Decisions are scored against 12.41's exact references, which do not depend on the seed or the rung. The plan was 16 problems × seeds 0–4. The first decisions cost over 36 CPU-min each, which projected past the 50 CPU-h kill line, so by the fixed criterion the run was cut to seeds 0–1 (32 decisions).

| Configuration | n | Optimal [95%] | Max loss | Mean optimal-action Q error | CPU per decision (median, range) |
|---|---|---|---|---|---|
| Root 16,000, target 6,400 with stopping | 32 | **32/32** [0.891, 1] | 0 | 0 | 50.6 min (22–106) |

- **UAV L2 H4 is qualified at root 16,000 with the target at 6,400 with stopping.** That configuration is now fixed for UAV episodes at H4. At H2–H3, root 1,000 with the target at 400 suffices (12.41).
> **Correction (October 15, 12.46):** the CPU per decision in this section is inflated by the machine's state during the run. A deterministic rerun of one U4c decision (seed 0, uniform/uniform) on October 15 gave the identical Q values and action in 718 CPU-s, against 5,368 recorded here. In 12.46's run, the same configuration averaged 366 CPU-s (6.1 min) over 48 decisions. The comparison with an exact target below is withdrawn: CPU from different runs is not comparable (HANDOFF Section 6).

- **Cost:** 27.4 CPU-h. A qualified L2 H4 UAV decision costs about 50 CPU-min, against about 4.5 min with an exact target (12.41), because the exact level-1 target is cheap on UAV. For UAV episodes at level 2, the exact target model is therefore the cheaper correct choice wherever its depth fits.

### 12.43 UAV L2: I-POMCP vs IPF + RTS against the exact I-POMDP, H2–H3 (October 14, 2026)

**Design** (`results/u5_uav_rts_20261014`; `u5.py`, `analyze.py`, `ENDPOINTS.md`, `KILL_CRITERION.txt`; endpoints fixed before the run). This is 12.20's design on UAV:
- **Problems:** the 16 level-2 UAV problems of 12.41 × seeds 0–4 (n = 80 per cell), history 0, scored against the exact I-POMDP. Each problem's exact reference was identical in every cell and seed.
- **I-POMCP:** root 1,000, compressed beliefs N = 64, with the target modeled at its qualified budget (100 simulations at H2, 400 at H3; 12.41).
- **IPF + RTS:** reachability-tree sampling with K sampled observations per node at every level, on the published sampled particle filter (N = 64).
- **Ablation:** RTS with K = 4 on compressed beliefs, which separates belief error from search error.
- **Test:** an exact McNemar test (on the problems where exactly one of the two planners is optimal) of I-POMCP against each IPF + RTS cell, paired on (problem, seed), Holm-corrected over the three values of K.
- **CPU:** per decision, nested solves included. All cells ran in the same run under the same load (16 workers), so CPU is comparable, though not paired in-process. Total cost was about 4 CPU-h.

| Planner | H | K | Optimal [95%] | Mean loss | Max loss | Optimal-action Q error | CPU (s) |
|---|---|---|---|---|---|---|---|
| **I-POMCP** (target 100) | 2 | — | **80/80** [0.955, 1] | 0 | 0 | **0** | 0.60 |
| IPF + RTS | 2 | 2 | 46/80 [0.459, 0.685] | 2.89 | 43.73 | 5.55 | 0.65 |
| IPF + RTS | 2 | 4 | 51/80 [0.522, 0.742] | 1.66 | 14.98 | 4.04 | 0.90 |
| IPF + RTS | 2 | 8 | 54/80 [0.561, 0.776] | 1.47 | 14.36 | 4.28 | 0.98 |
| RTS, compressed 64 | 2 | 4 | 56/80 [0.587, 0.797] | 1.90 | 16.83 | 3.05 | 0.23 |
| **I-POMCP** (target 400) | 3 | — | **80/80** [0.955, 1] | 0 | 0 | **0** | 11.8 |
| IPF + RTS | 3 | 2 | 42/80 [0.410, 0.638] | 1.27 | 17.13 | 7.48 | 24.4 |
| IPF + RTS | 3 | 4 | 44/80 [0.435, 0.662] | 0.95 | 17.13 | 4.33 | 42.7 |
| IPF + RTS | 3 | 8 | 48/80 [0.484, 0.708] | 1.04 | 17.13 | 3.89 | 47.3 |
| RTS, compressed 64 | 3 | 4 | 50/80 [0.510, 0.731] | 0.44 | 4.49 | 5.19 | 3.5 |

- **I-POMCP is optimal on every UAV problem at both horizons; IPF + RTS is optimal on 53–68% of them.** On no problem was IPF + RTS optimal where I-POMCP was not. I-POMCP alone was optimal on 26–34 problems at H2 and 32–38 at H3 (Holm p < 10⁻⁷ for every K).
- **At H3, I-POMCP is also cheaper:** 11.8 s per decision, against 24–47 s for IPF + RTS.
- **The gap is much larger than on Tiger** (there, IPF + RTS reached 95–99.5%; 12.20). RTS's error is in sampling observations: UAV has 5 actions, 3 observations and a terminal spotting event whose value depends on predicting the target's move, so a few sampled branches misjudge that value. The misses are spread over all 16 problems, not concentrated on a few.
- **The ablation shows that the error is search error, not belief error:** RTS on compressed beliefs is about as accurate as on the sampled particle filter (McNemar p = 0.33 at both horizons), at 1/4 to 1/12 of the CPU.
- **RTS itself is sound on UAV.** In a check at H2 (`diag/`, 2 seeds), RTS converges to the exact decision as K grows: at level 1, 6/8 optimal at K = 2 and 8/8 at K = 128; at level 2 with an exact target, 27/32 at K = 2 and 32/32 at K = 128; at level 2 with RTS at every level, 32/32 at K = 128. The shortfall at K ≤ 8 is therefore sampling error, not a defect of the implementation.
- **H4 was not run.** One IPF + RTS decision at H4 costs 4.2 CPU-min at K = 2 and 17 at K = 4, and more than 30 at K = 8 (the profile was stopped at its 30-minute cap). I-POMCP's qualified H4 decision was then believed to cost about 50 CPU-min (12.42; corrected in 12.46 to about 6). An 80-problem H4 grid would therefore exceed 100 CPU-h, which needs the user's approval.

### 12.44 UAV episodes: I-POMCP vs IPF + RTS, horizon 3 (October 14, 2026)

**Design** (`results/u5_uav_rts_20261014/episodes`; `episodes.py`, `analyze.py`, `ENDPOINTS.md`, `KILL_CRITERION.txt`; endpoints fixed before the run). This is 12.38's design on UAV:
- **Suite:** `suite.planner_comparison("uav", depth=3)`. i is a level-2 UAV, planning either with I-POMCP (root 1,000, modeling the target exactly as the target is configured) or with IPF + RTS at K = 2 or 4 (sampled particle filter, N = 64). The real target j is a level-1 I-POMCP agent with 400 simulations (qualified at H3, 12.41). Episodes last up to 20 steps; spotting ends one.
- **Trials:** n = 100 per cell, on shared trial seeds.
- **Primary endpoint:** i's episode reward, I-POMCP minus each RTS arm, paired by trial, with a bootstrap 95% interval and Holm correction over the 2 comparisons.
- **Cost:** about 20 CPU-h (both agents' planning and updates).

| i plans with | n | i's reward [95%] | Steps to spot | Spotted | i's log-loss of j | Prediction misses | CPU per decision (s) |
|---|---|---|---|---|---|---|---|
| **I-POMCP** | 100 | 30.0 [30.0, 30.0] | 5.0 | 100% | **0.93** | 0 | **5.3** |
| IPF + RTS, K = 2 | 100 | 28.6 [27.1, 29.8] | 5.3 | 100% | 1.21 | 0 | 43.2 |
| IPF + RTS, K = 4 | 99 | 30.3 [29.1, 31.4] | 5.0 | 100% | 1.20 | 0 | 95.0 |

| Comparison | n | Mean difference [95%] | p | Holm p | Median CPU ratio per decision |
|---|---|---|---|---|---|
| I-POMCP − (IPF + RTS, K = 2) | 100 | +1.45 [0.25, 2.90] | 0.035 | 0.070 | 0.12 |
| I-POMCP − (IPF + RTS, K = 4) | 99 | −0.25 [−1.36, 0.91] | 0.66 | 0.66 | 0.055 |

- **The planners earn the same reward in UAV episodes at horizon 3** (neither difference is significant after correction). I-POMCP spends 8–18× less CPU per decision.
- **Why episodes barely separate the planners:** the level-1 target always flees to the far corner, (0, 2), and stays there. The UAV starts 4 moves away, beyond the 3-step horizon, so the first move is close to a tie for every planner at this depth. I-POMCP listens once and then makes the four moves (reward 30 in all 100 episodes). IPF + RTS sometimes skips the listen (35, in 28 and 39 episodes for K = 2 and 4) and sometimes wanders (25 or less, in 34 and 18; as low as 0). The decision-level gap of 12.43 shows up on problems that a fleeing target never presents.
- **I-POMCP predicts the target better:** log-loss 0.93, against 1.20 for both RTS arms (log 5 = 1.61 is a uniform guess).
- **IPF failed once:** IPF + RTS K = 4 aborted trial 52, deterministically (twice). Its sampled filter of the target had no particle consistent with the target's observation ("no particle explains action = S"); the target observes its own sector exactly, which the published IPF does not handle. The K = 4 comparison is paired on the 99 remaining trials. I-POMCP's compressed filter makes exact updates and cannot fail this way.

### 12.45 UAV: decision quality against the number of particles (the paper's Fig. 16 shape), L2 after a 2-step history (October 14, 2026)

**Design** (`results/u5_uav_rts_20261014/particles`; `job.py`, `score.py`, `analyze.py`, `ENDPOINTS.md` with amendment 1, `refcheck.py`, `gate.py`). Doshi & Gmytrasiewicz (2009, Fig. 16) plot IPF + RTS's reward against the number of particles. Here the same axis is scored against exact references:
- **Why level 2 with a history:** the level-1 exact reference is solved only at the prior, where i's UAV belief is uniform over 8 sectors and N barely matters. So i acts at level 2 after a 2-step history, listening twice in place (`UAV_HISTORY_ACTIONS`, added for this run, `8c079fc`; history-0 rows unchanged, 160/160). The exact belief then has about 20 atoms.
- **Problems:** the 16 level-2 problems × seeds 0–4 (each seed draws a history; all 5 histories are non-terminal), n = 80 per cell, at horizons 2 and 3.
- **References:** the exact I-POMDP at the exact posterior, solved once per (problem, seed). They were checked to be independent of the planner's particle setting (32/32).
- **Cells:** IPF + RTS at K = 2, 4, 8 with N = 4–64 sampled particles, and on exact beliefs; I-POMCP (root 1,000; target 100 at H2, 400 at H3) with compressed N = 4–64, and exact.
- **Failures (amendment 1, made before any H3 table was read):** the sampled filter can run out of particles inside RTS's look-ahead. Such a decision is recorded as failed and counted as not optimal; losses are means over the completed decisions.
- **Cost:** 21 CPU-h of planning, plus references.

Horizon 3 (mean loss = expected reward lost against the exact decision; CPU includes the 2 history updates):

| Planner | N | Failed | Optimal [95%] | Mean loss [95%] | Max loss | CPU (s) |
|---|---|---|---|---|---|---|
| I-POMCP | 4 | 0 | 29/80 [0.26, 0.48] | 4.22 [2.70, 6.14] | 54.8 | 2.6 |
| I-POMCP | 8 | 0 | 46/80 [0.46, 0.69] | 0.58 [0.32, 0.87] | 6.2 | 9.1 |
| I-POMCP | 16 | 0 | 51/80 [0.52, 0.74] | 0.42 [0.16, 0.77] | 9.7 | 15.2 |
| I-POMCP | 32 | 0 | 70/80 [0.78, 0.94] | 0.071 [0.006, 0.16] | 2.4 | 24.3 |
| **I-POMCP** | **64** | 0 | **74/80** [0.84, 0.97] | **0.012** [0.002, 0.026] | 0.46 | 34.5 |
| I-POMCP | exact | 0 | 77/80 [0.89, 0.99] | 0.007 [0.000, 0.019] | 0.46 | 55.0 |
| IPF + RTS K = 2 | 4 / 8 / 16 / 32 / 64 | 22 / 0 / 0 / 0 / 0 | 22 / 35 / 38 / 43 / 43 of 80 | 4.48 / 3.71 / 2.90 / 2.20 / 1.48 | 10.8–46.8 | 2–58 |
| IPF + RTS K = 2 | exact | 0 | 46/80 [0.46, 0.69] | 1.24 [0.73, 1.84] | 10.4 | 8.8 |
| IPF + RTS K = 4 | 4 / 8 / 16 / 32 / 64 | 41 / 2 / 0 / 0 / 0 | 18 / 38 / 40 / 51 / 47 of 80 | 4.41 / 3.30 / 2.71 / 1.19 / 0.86 | 9.3–42.4 | 4–135 |
| IPF + RTS K = 4 | exact | 0 | 49/80 [0.50, 0.72] | 1.39 [0.73, 2.19] | 20.7 | 10.7 |
| IPF + RTS K = 8 | 4 / 8 / 16 / 32 / 64 | 46 / 0 / 1 / 0 / 0 | 18 / 38 / 41 / 49 / 48 of 80 | 5.26 / 3.12 / 2.65 / 1.00 / 0.61 | 7.8–46.8 | 6–198 |
| IPF + RTS K = 8 | exact | 0 | 48/80 [0.48, 0.71] | 0.64 [0.27, 1.09] | 10.4 | 15.5 |

Horizon 2 (same layout; full table in `particles/analysis.txt`): I-POMCP is 46/80 at N = 4, 72/80 at N = 32, and **80/80 at N = 64 and exact**. IPF + RTS is 38–43/80 at N = 4 (2–5 failures) and 62–69/80 at N = 64; on exact beliefs it reaches 61–71/80.

- **The paper's shape holds for IPF + RTS:** loss falls as particles grow (at H3, K = 8: 5.3 at N = 4 to 0.61 at N = 64). It flattens at RTS's own limit with exact beliefs (loss 0.64–1.39 at H3, 46–49/80 optimal), which is set by K, not by N.
- **At every N, I-POMCP loses less than IPF + RTS at any K**, at H3 from N = 8 up. I-POMCP at N = 8 (loss 0.58, 9 s) matches IPF + RTS at K = 8, N = 64 (loss 0.61, 198 s) at 1/22 of the CPU. At N = 64 its loss is 0.012, 50× lower than the best IPF + RTS cell.
- **Small sampled filters break on UAV.** At N = 4 the published IPF ran out of particles in 22–46 of 80 H3 decisions (2–5 at H2). The target observes its own sector exactly, so a small sample of the target's possible situations often contains none consistent with what it saw. I-POMCP's compressed filter makes exact updates before compressing, and never failed.
- **I-POMCP's qualified budget does not fully carry over to mid-episode beliefs at H3:** with exact beliefs it is 77/80, not 80/80. The 3 misses are near-ties: gaps of 0.46 (E 36.82 against S 36.36) and 0.044 twice. Its budget (root 1,000, target 400) was qualified at the prior (12.41). At H2 it is 80/80 here too. Like 12.14's L3 H4 residual, this is a near-tie floor, recorded rather than tuned away.

### 12.46 UAV L2 H4: I-POMCP vs IPF + RTS against the exact I-POMDP (October 14–15, 2026)

**Design** (`results/u5_uav_rts_20261014/h4`; `job.py`, `score.py`, `wrap.sh`, `watchdog.sh`, `KILL_CRITERION.txt`; analysis `analyze.py`, chunk `h4`). This is 12.43's design at horizon 4, approved by the user (over 100 CPU-h):
- **Problems:** the 16 level-2 problems × seeds 0–4 (n = 80 per cell), history 0. Exact references come from U4 (one per problem).
- **I-POMCP:** the qualified H4 configuration (12.42): root 16,000, target modeled with 6,400 simulations and stopping, exact beliefs. Seeds 0–1 are U4c's rows. At H3, compressed N = 64 and exact beliefs gave identical estimates on all 80 rows, so these are the same decisions.
- **IPF + RTS:** K = 2, 4, 8, sampled particle filter, N = 64. **Ablation:** RTS with K = 4 on compressed beliefs.
- **Jobs:** one process per (problem, cell, seed), planner only, scored offline with `evaluate_case`'s formulas.
- **Failures:** a decision whose sampled filter ran out of particles is recorded as failed and counted as not optimal (12.45's amendment 1).
- **Cost:** 151 CPU-h. The run lost about 5 CPU-h of killed and duplicated jobs: another project's 18 GB job forced memory kills, and two runners overlapped at the end.

| Planner | n | Failed | Optimal [95%] | Mean loss | Max loss | Optimal-action Q error | CPU per decision (s), seeds 2–4 |
|---|---|---|---|---|---|---|---|
| **I-POMCP** (root 16,000, target 6,400 with stopping) | 80 | 0 | **80/80** [0.955, 1] | 0 | 0 | **0.05** | **366** |
| IPF + RTS, K = 2 | 80 | 0 | 40/80 [0.386, 0.614] | 1.68 | 11.96 | 7.32 | 601 |
| IPF + RTS, K = 4 | 80 | 0 | 37/80 [0.350, 0.578] | 2.00 | 11.96 | 5.30 | 2,143 |
| IPF + RTS, K = 8 | 80 | 1 | 40/80 [0.386, 0.614] | 1.40 | 11.96 | 4.07 | 3,980 |
| RTS, compressed 64, K = 4 | 80 | 0 | 43/80 [0.422, 0.650] | 1.62 | 11.96 | 5.49 | 53 |

- **I-POMCP is optimal on all 80 problems at H4; IPF + RTS on 46–50%.** On no problem was IPF + RTS optimal where I-POMCP was not. I-POMCP alone was optimal on 40, 43 and 40 problems against K = 2, 4, 8 (exact McNemar test, Holm p < 10⁻¹¹).
- **I-POMCP is also cheaper:** 366 CPU-s per decision, against 601–3,980 s for IPF + RTS (1.6–11×). All rows were measured in this run under the same machine load, but not paired in-process.
- **More samples do not close the gap:** IPF + RTS reaches the same 37–40/80 at K = 2, 4 and 8. Its misses are spread over 15 of the 16 problems.
- **The ablation again shows search error, not belief error:** RTS on compressed beliefs is as accurate as on the sampled particle filter (McNemar p = 0.38), at 1/40 of the CPU.
- **One IPF failure:** one K = 8 decision (seed 3, near-start/far-corner) ran out of particles in i's own filter. This is the failure mode of 12.44–12.45.
- **Across horizons (12.43, 12.46):** I-POMCP is 80/80 at H2, H3 and H4; IPF + RTS falls from 58–68% (H2) to 53–60% (H3) to 46–50% (H4).
- **Correction to 12.42:** the qualified H4 decision costs about 6 CPU-min, not 50 (see the note there).

### 12.47 Why j's belief that i is level 1 is the same curve for every level of i (reviewer concern C1, October 15, 2026)

**The concern** (HANDOFF 0.1a). On the reviewer page, a level-2 j's posterior that i is level 1 follows nearly the same curve whether i is level 1, 2 or 3, and converges to level 1. The reviewer asked:
- Would a level-1000 i give the same curve?
- Should a deeper i not act differently, and change how j models it?
- Since j cannot model an i of its own level or above, should j not conclude that i is random (level 0)?

Each answer below rests on an exact computation or a control, not on the plotted episodes alone.

**1. j updates on behaviour, and a level-2 j has only two hypotheses: "random" (level 0) or "level 1".**
- **Theory.** A Bayesian whose hypotheses exclude the truth concentrates on the hypothesis closest to the truth in Kullback–Leibler divergence (Berk 1966; Kleijn & van der Vaart 2012), not on the most random one. Level 0 gives each of i's 3 actions probability 1/3. Level 1 predicts what any planning i does: listen until the growls agree, then open the safe door.
- **Per-step decomposition** (`results/c1_level_posterior_20261014/odds.py`, the page's trials). j hears i's action only through a creak that is correct 90% of the time. Each step changes j's log-odds of level 1 against level 0 by:

  | i | Mean change per listen | Mean change per open | Share of steps listening | j's P(level 1) at step 20 |
  |---|---|---|---|---|
  | level 1, 2, 3 | +1.08 to +1.15 | +1.65 to +2.08 | 0.76–0.78 | 0.991–0.994 |
  | **level 0 (control)** | −0.49 | −1.47 to −1.91 | 0.35 | **0.014** |
- **Positive control** (level grid, n = 100): against an i that really is random, j's posterior on level 1 falls to 0.014 [0.001, 0.037] in the published game and 0.019 in the coordination game. j does conclude "random" when i is random. It does not when i plans, because a planning i is far closer to level 1 than to uniform random behaviour.

**2. A deeper i does not act differently in the published Tiger, exactly.** Exact I-POMDP Q at point-belief chains (12.22's screen, extended in `screen_levels.py`; today's code reproduces 12.22's rows). The table gives the share of chains where level k+1's optimal action set differs from level k's at the same beliefs:

| H | 1 → 2 | 2 → 3 | 3 → 4 | 4 → 5 | 5 → 6 |
|---|---|---|---|---|---|
| 2 | 29% (21 chains) | **0%** (63) | **0%** (189) | **0%** (567) | **0%** (81 sampled chains) |
| 3 | 29% (21) | 10% (63; value 0.009) | not run (one exact solve exceeds 12 GB) | — | — |

Level 6 was completed on all 81 sampled chains on October 15 (evening). One level-4 H3 exact solve, run alone, reached 12.2 GB after about 30 CPU-min, still growing about 0.4 GB a minute, and was stopped: it does not fit the shared 23 GB machine (`guard.log`).

- From level 2 up, the hierarchy is at a fixed point at H2. A level-(k+2) agent best-responds to a level-(k+1) opponent that acts exactly like a level-k one, so it acts exactly as level (k+1) does. By induction, a level-1000 i acts exactly as a level-2 i, and j's posterior follows the same curve.
- In the episodes (shared seeds), levels 2 and 3 take identical actions over all 20 steps in 52 of 100 trials, and levels 1 and 2 in 30. Divergences come late (median step 10–12).

**3. j distinguishes levels when its hypotheses include them.** A level-3 j holds a uniform prior over i at levels 0–2. Its posterior at step 20 (M1, n = 100, 95% bootstrap intervals):

| Game | i's level | P(level 0) | P(level 1) | P(level 2) |
|---|---|---|---|---|
| published | 0 | **0.985** | 0.004 | 0.011 |
| published | 1 | 0.005 | **0.609** [0.557, 0.660] | 0.386 |
| published | 2 | 0.002 | 0.450 | **0.548** [0.497, 0.599] |
| published | 3 | 0.002 | 0.461 | 0.537 |
| coordination | 1 | 0.012 | **0.652** [0.591, 0.712] | 0.336 |
| coordination | 2 | 0.003 | 0.344 | **0.653** [0.601, 0.703] |
| coordination | 3 | 0.000 | 0.322 | 0.678 |

- **The true level wins whenever it is inside j's hypotheses**, from a prior of 1/3. The win is clearer in the coordination game, where levels 1 and 2 behave more differently.
- **A level-3 i, outside j's hypotheses, is classified as level 2**, the closest model available, as item 1's theory predicts. j does not fall back to "random".

**4. The plotted posterior is accurate.** `replay_j.py` replays j's real belief updates on the page's logged trials (5 trials × 3 conditions). Configured as in the run, it reproduces the logged posterior to 1e-16. The exact filter (uncompressed; stopped by its 100,000-branch budget at step 8) and N = 256 agree within 0.015 at every step. The plotted N = 64 posterior can dip below them by up to 0.28 within a trial, when compression drops some level-1 mass, but by step 20 it is 0.996–0.998 against 1.000. Compression slightly *understates* j's confidence that i is level 1, so the exact posterior shows the pattern more strongly.

**Summary for the reviewer:** j's posterior answers "is i random, or does it plan like a level-1 agent?" For any planning i the answer is "it plans". In the published Tiger, levels ≥ 2 act identically by exact computation, so the curves coincide. A j that can represent level 2 separates the levels, and classifies a level-3 i as its closest representable model, level 2.

### 12.48 Reward curves above the "2.72 ceiling" at step 3: an exact ceiling and 15,600 fresh trials (reviewer concern C2, October 15, 2026)

**The concern** (HANDOFF 0.1a). With 85% growl accuracy, the optimal expected cumulative reward after 3 steps from a uniform belief is 2.72. The reviewer page's reward curves sit above it at step 3: 4.48 for i at levels 2 and 3, 3.49 at level 1. The reviewer asked whether n = 100 is too low, or whether there is a bug.

**1. The exact ceiling** (`results/c2_tiger_ceiling_20261014/ceiling.py`).
- **Claim:** in the published two-agent Tiger, V*_t (the single-agent optimum from a uniform belief, undiscounted) bounds i's mean cumulative reward at step t, whatever j does.
- **Why:** i's reward depends only on its own action and the tiger. j affects i only by resetting the tiger when it opens a door, and through creaks. With k steps left, i's optimal value is convex and symmetric in its belief, so it is lowest at 50/50. A reset puts i's belief at 50/50, so by induction over k it can never raise i's optimum, even for an i that is told when j opens.
- **Computed two ways:** a DP over the net growl count, and brute-force enumeration with the package's own `TigerModel` kernels (equal for t ≤ 5). Values: V*_1 = −1, V*_2 = −2, **V*_3 = 2.72**, V*_4 = 2.42, V*_5 = 3.61, V*_10 = 9.44, V*_20 = 20.39.

**2. Every plotted curve against the ceiling** (`check_plots.py`; 30 curves: the page's 3 matchups and the level grid, both agents).
- No 95% interval lies wholly above the ceiling at any step.
- Means exceed it only at steps 3–4, by +0.22 to +1.76.
- The same excesses repeat across cells because the cells share trial seeds, so each agent's growl luck is the same in every cell. On the page's matchups, at step 3 the 100 trials had 77 correct opens, 1 wrong and 22 without an open, against 72.25 / 2.25 / 25.5 expected under the optimal policy: a lucky draw, about 1.1 standard errors (p ≈ 0.3).

**3. Large n with fresh seeds** (`large_n.py`, `analyze_large_n.py`).
- **Design:** four configurations rerun with trial ids 0 to n − 1. **Gate: trials 0–99 reproduce the plotted episodes exactly in all four (100/100 each)**, so the plotted data come from the same code. Trials ≥ 100 are new, independent seeds.

| Configuration | Fresh trials | i at step 3 (mean ± 95%) | Ceiling | Steps with interval above the ceiling (i or j) | i at step 20 | Ceiling |
|---|---|---|---|---|---|---|
| grid, i L1 vs j L1 (H2) | 9,900 | 2.48 ± 0.29 | 2.72 | none | 12.41 ± 0.80 | 20.39 |
| grid, i L2 vs j L2 (H2) | 2,900 | 2.82 ± 0.47 | 2.72 | none | 14.11 ± 1.38 | 20.39 |
| grid, i L1 vs j L2 (H2) | 1,900 | 2.63 ± 0.59 | 2.72 | none | 14.68 ± 1.81 | 20.39 |
| page, i L1 vs j L2 (H3) | 900 | 2.95 ± 0.77 | 2.72 | none | 15.05 ± 2.59 | 20.39 |
| **pooled** | **15,600** | **2.59 ± 0.22** | 2.72 | — | — | — |

- With fresh seeds, the step-3 mean converges to just below the exact ceiling: 2.59 ± 0.22 pooled. The plotted excess was a sample of 100 shared across cells. The step-3 outcome mix with fresh seeds is 63–65% correct opens, 1.0–1.7% wrong and 33–36% no open.
- **No accounting bug:** the code that produced the plots reproduces them exactly and, with fresh seeds, stays below the exact optimum at every step for both agents.
- **Cost:** about 40 CPU-h.

**For the page:** every published-Tiger reward panel now draws the exact ceiling as a dotted line (version 22).

### 12.49 UAV level matrix: levels 0–2 for the UAV and the target (U6, October 15, 2026)

**Design** (`results/u6_uav_level_matrix_20261015`; `u6.py`, `analyze.py`, `KILL_CRITERION.txt`; endpoints fixed in PLAN.md, "U6 design", before the run):
- **Suite:** `suite.level_matrix("uav", levels=(0, 1, 2), depth=3, simulations=1000, particles=64, steps=20)`. i is the UAV (+50 for spotting, −5 per step) and j the target (−50 when spotted). Level 0 acts at random.
- **Priors and models:** each agent holds a uniform prior over the other's lower levels, and every model matches the real agent of that level. Depth 3 with root 1,000 is qualified at levels 1–2 (12.41).
- **Trials:** n = 100 per cell, on shared trial seeds. Paired one-level differences, Holm-corrected over the family of 12.
- **Cost:** about 15 CPU-h; no failures; prediction misses 0 in every cell.

| UAV i \ target j | Level 0 | Level 1 | Level 2 |
|---|---|---|---|
| **Level 0** | −15.4 (spotted 73%, 11.1 steps) | −66.9 (36%, 17.3) | −55.2 (46%, 16.1) |
| **Level 1** | 17.7 (96%, 7.0) | 24.2 (100%, 6.2) | −71.6 (30%, 17.6) |
| **Level 2** | 10.6 (93%, 8.1) | **30.0** (100%, **5.0**) | −45.6 (48%, 14.4) |

(Each cell: the UAV's mean 20-step reward, then the share of episodes in which it spots the target, then the mean number of steps played.)

| Agent | Step | Against | Paired difference [95%] | Holm p |
|---|---|---|---|---|
| UAV | 1 → 2 | target level 1 | **+5.8** [3.9, 8.0] | < 0.001 |
| UAV | 1 → 2 | target level 2 | **+26.0** [15.4, 37.1] | < 0.001 |
| UAV | 1 → 2 | target level 0 | **−7.2** [−12.7, −2.6] | 0.032 |
| UAV | 0 → 1 | target levels 0 / 1 / 2 | +33.1 / **+91.1** / −16.4 | < 0.001 / < 0.001 / 0.055 |
| Target | 1 → 2 | UAV level 1 | **+35.0** [30.5, 39.5] | < 0.001 |
| Target | 1 → 2 | UAV level 2 | **+26.0** [21.0, 31.0] | < 0.001 |
| Target | 1 → 2 | UAV level 0 | −5.0 [−10.0, −0.5] | 0.090 |
| Target | 0 → 1 | UAV levels 0 / 1 / 2 | +18.5 / −2.0 / −3.5 | < 0.001 / 0.090 / 0.032 |

- **On UAV, a second level of reasoning pays for both agents, against the opponents it models.**
  - A level-2 UAV models a level-1 target's flight to the far corner. It goes straight there, spotting in 5.0 steps in every episode, against 6.2 for a level-1 UAV (+5.8).
  - A level-2 target models the UAV's pursuit and evades. It escapes for all 20 steps in 52–70% of episodes against every UAV level, and gains 26–35 over a level-1 target.
  - This is the domain-level counterpart of the coordination Tiger (12.37). Here the opponent's model matters through the game itself, not through a designed payoff variant.
- **Depth costs something against an opponent it models wrongly:**
  - A level-2 UAV does worse than a level-1 UAV against a random target (−7.2, Holm p = 0.032): it spends moves anticipating a flight that never comes.
  - A level-1 target is caught slightly faster than a random one by a modeling UAV (−2.0 and −3.5): its flight to the corner is predictable.
- **Prediction:** a level-2 UAV predicts a level-1 target's moves with log-loss 0.90 (a uniform guess over 5 actions is 1.61), and a level-2 target predicts a level-1 UAV with 0.96. Against an opponent of its own level, the loss rises to 1.43 (UAV) and 0.99 (target).
- **Identification at the last step:** a level-2 UAV puts 0.64 on a level-1 target and 0.75 on a random one (from a prior of 0.5). A level-2 target puts 0.70 on a level-1 UAV and 0.63 on a random one.

### 12.50 UAV mid-episode qualification: two search misses and one tie-convention mismatch (October 15, 2026)

**Design** (`results/q_uav_midepisode_20261015`; `ENDPOINTS.md` fixed before the run, with two rungs added after the first ladder; `job.py`, `score.py`, `diag_tie.py`). 12.45 found I-POMCP (root 1,000, target 400, exact beliefs) optimal on only 77/80 decisions at level 2, H3, after a 2-step history, against 80/80 at the prior (12.41). This run asks which error source causes each miss.
- **Problems:** 12.45's (16 problems × seeds 0–4), with its exact references reused.
- **Rungs:** root × target budget.

| Root | Target model | Optimal [95%] | Max loss | Optimal-action Q error | CPU per decision (s) |
|---|---|---|---|---|---|
| 1,000 | I-POMCP 400 | 77/80 [0.894, 0.992] | 0.463 | 0.026 | 51 |
| 1,000 | exact | 78/80 [0.913, 0.997] | 0.044 | 0.008 | 21 |
| 4,000 | I-POMCP 400 / 1,600 / 6,400 with stopping | 79/80 each | 0.463 | 0.019 | 32 / 55 / 65 |
| 16,000 | I-POMCP 400 | 79/80 [0.932, 1] | 0.463 | 0.018 | 32 |
| **4,000** | **exact** | **80/80** [0.955, 1] | **0** | 0.001 | **21** |

- **Two misses (gaps 0.044, seed 3) are i's own search error.** Root 4,000 removes them with any target model.
- **One miss (uniform/top-row, seed 0, 0.463) stays at every modeled budget** with the same loss. Only an exact target removes it. It is a tie-convention mismatch, not an error (`diag_tie.py`):
  - i's search queried 170 distinct target models. **75 have an exact tie among the target's best actions**: on UAV a move into the grid's edge leaves the agent in place, and often has exactly the value of listening.
  - The exact reference has modeled agents play all tied actions uniformly. I-POMCP's modeled target played the whole tie set in 37 of the 75, and a subset of it in 38.
  - Every one of the 170 modeled decisions is optimal for the target (loss 0 at budgets 400, 1,600 and 6,400 with stopping). But tied actions give the target different observations, and so different futures, which changes i's best response. The reference and the modeled target are two equally rational opponents that break ties differently. No budget can remove the gap; only a modeled solve that plays ties as the reference does (the exact target) can.
- **Qualified mid-episode configuration on UAV (L2, H3):** root 4,000 with an exact target, 80/80. It is also the cheapest rung here (21 s against 51 s), as an exact level-1 target is cheap on UAV.
- **What this means for episodes:** the episode runs (12.44, 12.49) model the target as the real target is configured (I-POMCP, the same budget and seed derivation; 12.30's rule), so i predicts the real target's tie-breaking. In U6 no prediction ever gave the real action probability 0. Against the exact reference, the H3 decision has search error at root 1,000 (2 of 80 decisions, 0.044 each) and the tie convention above (1 of 80).
- **Not the cause of Tiger's L3 H4 residual (12.14):** that residual was measured with the stack `6400s/exact`, so the level below the searched j was exact and played ties as the reference does. Its 4 misses are j's own search at a true gap of 0.03, not a tie convention.

### 12.51 Modeled agents' tie-breaking made explicit (October 15, 2026)

**Decision** (HANDOFF 0.0a, item 4; option (a)): keep I-POMCP's modeled solves as they are, and record the tie convention instead of changing behaviour. The reasons:
- Episodes model the real opponent as it is configured (12.30), so they need the opponent's own tie-breaking, not the reference's uniform one.
- Option (b) (uniform play over actions whose certified intervals cannot be separated) would change episode behaviour, and an interval test cannot certify an exact tie anyway.

The definition is in THEORY.md, "Tie convention".

**Change** (`experiments.modeled_qualification`):
- Each row carries the exact `tie_set` (the actions within 10⁻⁹ of the best exact Q) and, per rung, `uniform_on_ties` (the policy is exactly uniform on that set, as the reference plays).
- The summary adds `tied` (queries with more than one best action) and `optimal_off_convention` (optimal decisions that break a tie differently from the reference).
- Qualification is unchanged: a rung still fails only on a wrong decision.

**Gate** (`results/t1_tie_columns_gate_20261015`; pass criterion fixed in `KILL_CRITERION.txt` before the run): the U4b grid (608 Tiger qualification rows, L2 H2–H3 and L3 H2), HEAD against the change, run from snapshots. Every stored column is identical (horizon, seed, query, rung, belief digest, policy, loss, decision loss, exact Q). That grid has no exact ties (`tied` = 0 everywhere).

**Positive control:** the new definition applied to 12.50's stored diagnosis (`q_uav_midepisode_20261015/diag_tie.json`) reproduces its counts exactly. Of the 170 distinct target models, 75 are tied, and 38 are optimal off convention at each of the 400, 1,600 and 6,400-with-stopping rungs.

### 12.52 Where a level-3 decision's CPU goes, and what level 4 at H3 costs (October 15, 2026)

**Question** (HANDOFF 0.0a, 5a): would a solver-level speedup of the exact nested update make level 4 at H3 affordable? `results/s5_exact_update_20261015`; today's code, quiet machine.

**Profile** (`profile_episode.py`, 12.36's L3 condition, trial 0, 4 steps; cProfile self time, which does not double-count the recursion):
- **The search itself** (tree descent, rollouts, sampling): 20%.
- **The exact update proper:** about 20%. That is its loop (7%), branch bookkeeping (5%) and the Tiger kernels (10%, shared with expected rewards).
- **Compression and digests:** 5%. **Belief equality:** 5%.
- Everything else is below 2% each.

12.36's "a third of CPU" was the update's cumulative time, which includes the nested updates it calls. Kernel memoization already failed (Section 6). So no change to the update alone can save more than about 20%.

**Where the nested solves come from** (`count_solves.py`, the same 4 steps without the profiler: 114 CPU-s, the same actions):

| Modeled level | Solves | CPU including nested (s) | Mean (s) | Distinct policies |
|---|---|---|---|---|
| j at level 2 | 457 | 113 | 0.25 | 3, all deterministic |
| i at level 1 (inside j's) | 1,298 | 36 | 0.03 | 3, all deterministic |

- **Level-1 models are exact thresholds on the physical marginal** (2 switches along it). A level-1 model facing a level-0 opponent is a POMDP over the physical state.
- **Level-2 models are not** (46 switches): their policy also depends on their model of i.
- Solving the level-1 models exactly (an exact bottom level, which stacks such as `1000/exact` already support) would save at most the level-1 share, about 30% at level 3. To keep i's model matched to j (12.30), the real j's own model of i would then also have to be exact. Episode studies reject exact modeled levels today (`study._modeled_configs`).

**Level 4 at H3** (`pilot_l4h3.py`; coordination Tiger; i at level 4 against j at level 2; root 1,000, every modeled level 400, N = 64, uniform level priors; 1 episode, 4 steps):

| Step | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| i's CPU (s) | 57 | 161 | 353 | 431 |

- About 9× level 3 over the same steps (1,006 s against 108 s), with a flat peak of 0.6 GB. Actions L, L, OL, L, as level 3's.
- A 20-step episode is about 2–2.5 CPU-h, so a paired comparison of levels 3 and 4 at n = 50 needs over 100 CPU-h.
- **Conclusion:** no available change to the update, nor exact level-1 models (at most about 1.4×), brings this below the user-approval threshold. 5b is the user's call at its full cost, and 5a stops here (no code change).

### 12.53 Wumpus L1 qualification over 52 distinct beliefs: N = 4,096 qualifies, N = 1,024 does not (October 15, 2026)

**Why** (HANDOFF 0.0a, 5c): 12.19 qualified N = 1,024 with 3,000 simulations on only 3 distinct beliefs, because its scripted history gave every seed the same observations.

**Design** (`results/c5_wumpus_beliefs_20261015`; `ENDPOINTS.md` fixed before the run, with 3 rungs added after the first ladder and recorded before they ran; `distinct.py`, `analyze.py`):
- **Problems:** `wumpus_qualification --script random`, which draws each history action uniformly per seed. Seeds 0–19, histories cut after 2, 4 and 6 steps: 60 problems.
- **Beliefs:** 52 distinct exact beliefs (by digest), supports 2,100–74,880 atoms, with breeze, bump, stench and scream observations. Duplicated beliefs are decided with different planner seeds.
- **Reference:** the exact H3 best response at the exact belief (as 12.19), median 260 s, 3.7 CPU-h in all. 11 problems have an exact tie at the best action, which first-action loss scores correctly. The median gap is 29.6.
- **Code** (`f003a17`): the fixed script is unchanged (histories identical, seeds 0–9, 0–5 steps). A Forward move at the 94,500-atom prior needs more than the default 100,000-branch guard of the exact filter, so `SolverBank` takes `max_branches` (the guard only raises, so no completed update changes).

| N (search atoms) | Simulations | Optimal / 60 [95%] | Wrong beliefs | Max loss | Optimal-action Q error (median, max) | Root certified exact | CPU per decision, median (s) |
|---|---|---|---|---|---|---|---|
| 64 | 3,000 | 43 [0.586, 0.825] | 16 | 43.5 | 16.3, 74.7 | 46/60 | 2.8 |
| 1,024 | 3,000 | 58 [0.885, 0.996] | 2 | 8.42 | 1.08, 24.2 | 30/60 | 14.5 |
| 1,024 | 10,000 | 59 [0.911, 1] | 1 | 1.27 | 1.33, 12.4 | 44/60 | 14.6 |
| **4,096** | **3,000** | **60 [0.940, 1]** | **0** | **0** | 0.50, 27.7 | 28/60 | **44.8** |
| 4,096 | 10,000 | 60 [0.940, 1] | 0 | 0 | 0.45, 3.75 | 44/60 | 52.4 |

- **12.19's qualification does not carry over.**
  - N = 64, which was 9/9 there, is wrong on 17 of 60 decisions, losing up to 43.5.
  - N = 1,024 misses 2 of 60:
    - seed 5 after 4 steps loses 1.27 with its root certified exact (bound width 0) at both budgets, so it is **compression error**;
    - seed 13 after 4 steps loses 8.42 with an uncertified root (width 15.9) at 3,000 simulations, and is optimal at 10,000: **search error**.
- **Qualified Wumpus L1 H3 configuration: N = 4,096, 3,000 simulations** (60/60 over 52 distinct beliefs, median 45 CPU-s per decision). 10,000 simulations cut the worst optimal-action Q error from 27.7 to 3.7 at 1.2× the CPU.
- CPU figures are from one run on a quiet machine. 12.19's 41 s per N = 1,024 decision predates 12.36's speedups.

### 12.54 Certification standard ε = 0.05: every stored grid re-scored; the L3 H4 residual is 80/80 ε-correct (October 16, 2026)

**Why** (HANDOFF 0.0a, item 6): 12.14 left 4 of 80 modeled-j decisions at L3 H4 wrong at near-ties (gap 0.03). On October 16 the user adopted the PAC-style standard proposed in 12.13, with ε = 0.05 (THEORY.md, "Certification standard").

**Design** (`results/p1_pac_rescore_20261016/rescore.py`, `rescore.csv`). Stored rows only; nothing was rerun. For each configuration cell of every certified grid (12.1, 12.3, 12.4, 12.7, 12.14, 12.23, 12.41, 12.42, 12.46, 12.53; 66 cells, 13,721 decisions), the script reports n, the exactly optimal count, the ε-correct count (first-action loss ≤ 0.05), the maximum loss and δ. δ is the upper end of the exact 95% Clopper–Pearson interval on the ε-failure rate, the package's two-sided convention. `panel.summarize` gains the columns `eps_correct_rate` and `eps_failure_high` (with a test). **Gate:** its existing columns reproduce 30 stored summary rows (U4 and U5 panels) exactly.

**Checks.** The exactly optimal counts reproduce each section's headline: L1 H5 220/220, L2 H4 380/380, L3 H3 162/162, L4 H2 486/486, UAV L3 H2 640/640, and UAV L2 H4 32/32 with the modeled target.

**Where ε changes a count.** Only in the L3 H4 residual set (the modeled j-L2 at the 8 residual beliefs × seeds 0–9):

| Rung | n | Exactly optimal | ε-correct | Max loss | δ |
|---|---|---|---|---|---|
| **6400s/exact (the planner's)** | 80 | 76 | **80** | 0.030 | **0.045** |
| 1600/exact | 80 | 56 | 74 | 0.295 | 0.156 |

- **The residual is resolved under the standard:** 80/80 ε-correct, with δ ≤ 4.5%. 12.13 found that flips of j decisions with gaps up to 0.3 never changed i's decision. That is the evidence behind ε, though those hierarchies held no gap as small as 0.03.
- **No other cell changes.** Every other grid's qualified configuration was already exactly optimal on every decision. Every cell that still fails ε is a sub-qualified rung (for example UAV L1 H5 below 64,000, Wumpus N = 1,024, or IPF + RTS), and its misses exceed 0.05, so the standard does not soften any comparison between planners.
- **Exactly optimal stays reported** alongside ε-correct in every new summary.

### 12.55 Wumpus L2 cost pilot: 6–28 CPU-min per decision at a sub-qualified budget, and over 8 GB after 4 steps (October 16, 2026)

**Why** (HANDOFF 0.0a, 5c): after Wumpus L1 qualified (12.53), the next Wumpus step is L2. First, price it.

**Design** (`results/c6_wumpus_l2_pilot_20261016`; `KILL_CRITERION.txt` was written before the first run: 2 CPU-h per job, an 8 GB RSS guard, cheapest first). The human is level 2 and models the wumpus at level 1, which models a uniform-random human. Real beliefs are exact. Searches use N = 1,024 atoms (bank-wide, so the wumpus's nested searches too), with root 1,000, H3, and histories drawn with `--script random`, seed 0. These budgets are **below** the qualified L1 configuration (N = 4,096, 3,000 simulations), so this measures cost only, not accuracy. There is one job per process, and the CPU is from a quiet machine, with at most 2 jobs at once.

| History steps | Modeled wumpus simulations | Human support (atoms) | Belief update CPU (s) | Decision CPU (s) | Peak RSS |
|---|---|---|---|---|---|
| 0 (prior) | 100 | 94,500 | 0 | 386 | 1.2 GB |
| 2 | 100 | 19,440 | 326 | 1,351 | 3.4 GB |
| 2 | 400 | 19,440 | 299 | 1,446 | 3.6 GB |
| 4 | 100 | — | — | **stopped by the guard at 8.1 GB after about 50 CPU-min** | > 8 GB |

- **The cost is in the nested belief updates, not the nested search.** Four times the modeled simulations cost only 1.07× (2 steps). The wumpus's exact beliefs stay large, because its only informative percepts are footsteps within 2 cells, breeze and glitter. Every nested update runs over them.
- **The cost grows with the history, not with the human's support.** The prior (94,500 atoms) is the cheapest case. Later beliefs reach more distinct nested wumpus beliefs, and memory grows with them.
- **Exact L2 reference: not feasible here.** It needs the exact level-1 best response of the wumpus at every nested belief the human's H3 tree reaches. One such solve costs about 260 s (12.53), and a single decision queries hundreds or more of them.
- **Conclusion: Wumpus L2 is not run on this machine.** A qualification grid (about 60 problems × rungs, at qualified budgets) plus its references would far exceed 100 CPU-h, and decisions after 4 steps do not fit in 8 GB. It would first need a memory-lean nested belief for the modeled wumpus. For example, compressed nested beliefs (the bank's `particles` for nested levels), qualified against the exact nested update on Tiger and UAV, where exact references exist. That is a design item, not a run.

### 12.56 Making Wumpus L2 lighter: nested compression fails, bit-identical speedups do not pay, and a 3 × 3 board has exact references (October 16, 2026)

**Why** (12.55): a 4 × 4 Wumpus L2 decision costs 6–28 CPU-min at budgets below the qualified ones, and over 8 GB after 4 steps.

**1. Compressing nested beliefs in real updates: 0 of 3 jobs survive** (`results/c7_wumpus_l2_nested_20261016`, not adopted). A filter kept the human's own update exact but held every nested wumpus belief at M atoms. Three jobs ran: M = 4,096 after 2 and 4 steps, and M = 1,024 after 4 steps. All three raised `UnsupportedObservation` in the real history update (for example "Nested j L1 cannot condition on Forward, (Breeze,)").
- **Why:** the real update enumerates every outer world, and a compressed nested belief has dropped some of them. In such a world the wumpus's deterministic percept has zero evidence under its own belief. Under a common prior, the exact nested belief always contains the outer world.
- In-tree updates already compress nested beliefs, but there only sampled worlds are reached, and no failure occurred in 4 pilot jobs.
- The code was removed (Section 6 of HANDOFF).

**2. Where an L2 decision's CPU goes** (cProfile of the exact-nested decision after 2 steps, root 200; profiled time 5,151 s): almost all of it is exact enumeration (`_update`, 4,867 s).

| Part | Profiled s | Calls |
|---|---|---|
| In-tree history successors (first advances of the root's exact nested models) | 3,932 | 51,156 |
| Real history updates (2 steps) | 1,316 | 2 |
| Transitions (state construction) | 1,558 | 151 M |
| Kernel validation (`checked_distribution`) | 1,453 | 227 M |
| Canonical encodings in compression | 1,007 | 2,832 compressions |
| Dataclass state hashes | 603 | 323 M |

**3. Bit-identical speedups: correct, but no gain** (`results/c8_wumpus_interning_20261016`, `c9_speedup_gate_20261016`, `c10_wumpus_speedup_20261016`; not adopted; patch kept as `c9_speedup_gate_20261016/new.diff`).
- **The changes:** weakly interned Wumpus states, cached state hashes, a point-mass fast path in kernel validation, cached pose moves, and cached state encodings.
- **Gates, all identical:** the U4 Tiger panel gate (6 cells × 2 seeds, every row), and 12.55's Wumpus rows at 0 and 2 steps (values, support, digest).
- **Paired cost** (old and new code concurrently, same decision at the prior): **0.91× the CPU speed (424 s → 466 s) and 1.27× the peak RSS (1.17 → 1.50 GB).** Most states are hashed only once, so the tables cost more than they save.
- **With interning alone,** the 4-step decision completed: 2,321 s of belief update plus 3,016 s of decision, peaking at 6.4 GB (12.55: over 8 GB, stopped). That is still about 1.5 CPU-h per problem at budgets below the qualified ones.
- **Conclusion:** the cost is the volume of exact enumeration, not per-branch overhead. On the 4 × 4 board, Wumpus L2 stays out of reach on this machine.

**4. A 3 × 3 board with 1 pit has exact references at L2 and L3** (`results/c11_wumpus3_exact_20261016/probe.py`; `WumpusModel(width=3, height=3, n_pits=1)`: 8 × 4 × 8 × 8 = 2,048 prior states). The exact I-POMDP at the prior (`reference.exact_agent`, every level exact):

| Level, horizon | CPU | Peak RSS |
|---|---|---|
| L1 H2 | 1.6 s | 32 MB |
| L1 H3 | 8.0 s | 62 MB |
| L2 H2 | 13.5 s | 70 MB |
| **L2 H3** | **226 s** | **0.5 GB** |
| L3 H2 | 28 s | 0.1 GB |

- **Next:** qualify I-POMCP at Wumpus L2 (and L3 H2) against these exact references on the 3 × 3 board, over distinct beliefs, as was done on Tiger and UAV. This follows the redesign-before-scale rule: a smaller instance of the same domain, with references that cannot be argued with.

### 12.57 Systematic compression aliased on structured uniform beliefs; stratified draws adopted (October 16, 2026)

**Found** (3 × 3 Wumpus L2, 12.56's follow-up). In 12 of 60 problems, the planner lost 102.5 at N = 256 and N = 1,024, whatever the root or modeled budget, with its root scored exactly. The problem was the prior, where Q(Forward) came out −56.1 against an exact −174.1. The modeled wumpus's policies agreed with the exact ones at every belief checked. The compressed search belief did not: it held **zero** mass on "pit at (1, 0)", whose exact mass is 1/8.
- **Cause:** systematic resampling (`belief.resampling`) takes evenly spaced points through the canonical order. On a uniform belief over 2,048 atoms, the points are every 2,048/N-th atom. A field that cycles in the canonical order with a commensurate period (here the pit cell) is then always drawn at the same phase.
- **Survey** of worst physical-marginal error (`results/c13_compression_ab_20261016`, scratch survey):
  - 3 × 3 prior: systematic 0.875 at N = 16–256 and 0.125 at N = 1,024, against stratified 0.25–0.018.
  - 4 × 4 prior: systematic is *better* (0.0036 against stratified 0.009 at N = 1,024), because canonical order stratifies when nothing resonates.
  - 4 × 4 seed 5 after 4 steps (12.53's compression miss): systematic 0.0225, stratified 0.0087.
- **Change** (`resampling.stratified_draws`; `CompressedInteractiveFilter.compress`): one independent point per 1/N stratum, still in canonical order. Per-atom error is below 2/N, and compression stays a pure function of the belief.
  - **Not changed:** the sampled IPF baseline (`ParticleInteractiveFilter`). It resamples its own N particles, so its support never far exceeds N and it cannot alias this way. Its published comparisons stand.

**A/B** (paired, `ENDPOINTS.md` fixed before the runs; exact references shared by both arms):

| Grid | Cell | Systematic wrong | Stratified wrong | Fixed / broken | Exact McNemar p |
|---|---|---|---|---|---|
| G1 Wumpus 3 × 3, L2 H2 (60) | N = 256 | 19 (max loss 949) | **6** | 18 / 5 | 0.011 |
| G1 | N = 1,024 | 12 | **1** | 12 / 1 | 0.0034 |
| G2 Wumpus 4 × 4, L1 H3 (60; 12.53) | N = 1,024, 3,000 sims | 2 | 2 | 2 / 2 | 1 |
| G2 | N = 4,096, 3,000 sims | 0 | 1 | 0 / 1 | 1 |
| G2 | N = 4,096, 10,000 sims | 0 | 0 | 0 / 0 | 1 |
| G3 Tiger L2 H3, N = 64, history 2 (380) | — | 5 | 6 | 2 / 3 | 1 |
| G3, history 4 (380) | — | 0 | 0 | 0 / 0 | 1 |

- **Adopted** by the pre-registered rule: no cell is worse by more than 2 decisions, none is significant against stratified, and it fixes the aliasing grid (31 of 34 discordant pairs on G1).
- **Consequence for 12.53:**
  - Under stratified draws, N = 4,096 with 3,000 simulations is 59/60. The miss has an uncertified root (width 21.7, loss 6.97, a search error), and the same problem is optimal at 10,000 simulations.
  - **The 4 × 4 L1 qualified configuration is now N = 4,096 with 10,000 simulations** (60/60), at a median of about 52 CPU-s.
  - 12.53's compression miss (seed 5 after 4 steps, at N = 1,024) is fixed by stratified draws.
- **Remaining 3 × 3 errors under stratified draws at N = 1,024** are in nested solves. The one miss (loss 949, seed 11 after 4 steps) has a root support of 35 atoms, scored exactly; its error is the modeled wumpus's compressed nested search. 12.58 removes compression on this board.

### 12.58 Wumpus 3 × 3 qualified at L2 H2 and L3 H2 against the exact I-POMDP (October 16, 2026)

**Design** (`results/c13_compression_ab_20261016/g1x_exact`, `results/c14_wumpus3_qualification_20261016`; `ENDPOINTS.md` fixed before the runs, amendments recorded before each rerun).
- **Setup:** board 3 × 3 with 1 pit (2,048 prior states). Search uses exact beliefs at every level (`--search-particles 0`), so no compressor is involved. The reference is the exact I-POMDP after the history.
- **Problems:** seeds 0–19 × history steps {0, 2, 4}.
- **Histories at L2:** human actions are random, and the wumpus acts as its exact model.
- **Histories at L3** (`--script modeled`): the human acts as the wumpus's exact model of it, among actions that do not end the episode.
- **The level-0 weight at L3:** both the planner and the reference give 0.1 to level 0 at each level ≥ 2 (`--l0 0.1`). The reasons are in the next list.

| Level, horizon | Problems (distinct beliefs) | Optimal [95%] | ε-correct | Median CPU per decision |
|---|---|---|---|---|
| L2 H2, root 1,000, modeled 400 | 60 (31) | **60/60** [0.940, 1] | 60/60 | 116 s |
| L2 H2, root 1,000, modeled 1,600 | 60 (31) | 60/60 [0.940, 1] | 60/60 | 120 s |
| L3 H2, root 1,000, modeled 400, l0 0.1 | 55 (23); 5 ended before a decision | **55/55** [0.935, 1] | 55/55 | 104 s |

- **Every root was scored exactly**, and no planner update failed.
- **Why L3 needs these choices** (each found in a failed run, and kept in `l3h2_*_failed/`):
  1. With scripted human actions, the level-2 wumpus models a near-deterministic level-1 human, and its own observations had zero evidence.
  2. Some modeled humans' only likely action ends the episode. Such problems are recorded as ended and skipped.
  3. With point-mass levels, the exact reference's own lookahead was undefined. A counterfactual human action that the wumpus model gives zero probability, with deterministic percepts, leaves its nested update with zero evidence. Any level-0 weight defines it.
- **L2 H3** (root 1,000, modeled 400; **interim, n = 13**): 13/13 optimal [0.753, 1] over the problems of seeds 0–4. Only 4 of 13 roots were scored exactly, since exact scoring covers the last two steps (12.14).
  - **Cost:** a decision costs 334–2,176 CPU-s (median 506), and is most expensive after histories with supports near 1,000 atoms. The 60 exact references cost 9.1 CPU-h (median 486 s).
  - **Stopped by its kill criterion** (a decision over 30 CPU-min; in-flight decisions reached 38–52). The full 60 would cost about 12 more CPU-h.
- **The episode agents' configuration** (`results/c15_wumpus3_episodes_20261017/qual`):
  - **Setup:** L2 H2 with a uniform prior over the opponent's lower levels (`--l0 0.5`, as `suite.hierarchy_agent` builds every agent), root 400, modeled 400, exact search. It was scored against fresh exact references (median 89 s).
  - **60/60 optimal** [0.940, 1] over 31 distinct beliefs, every root scored exactly, at a median of 112 CPU-s. This is the gate for the episode study (PLAN, October 17).
- **The level-3 episode agent's configuration** (`results/c17_wumpus3_level3_gate_20261017/gate`): L3 H2 with a uniform prior over all lower levels (`--l0 uniform`), root 400, modeled 400, exact search, `--script modeled`. It is **55/55 optimal** [0.935, 1] over 23 distinct beliefs; 5 problems ended before a decision. Every root was scored exactly, at a median of 145 CPU-s. This is the gate for 12.60's step 4.

### 12.59 Wumpus 3 × 3 L2 H2 decisions: I-POMCP 60/60, IPF + RTS 2–47% (October 17, 2026)

**Design** (`results/c16_wumpus3_rts_20261017`, `ENDPOINTS.md` fixed before the runs).
- **Problems:** the 60 problems of 12.58 (31 distinct beliefs), with their exact I-POMDP references.
- **I-POMCP:** 12.58's qualified cell (exact search, root 1,000, modeled 400).
- **IPF + RTS** (`--planner rts`, every level RTS): K = 2, 4 or 8 sampled observations per node, on two kinds of belief: the published sampled IPF at 64 particles, or exact beliefs, which isolates RTS's search.
- **Tests:** exact McNemar tests, paired by problem, with Holm correction over the 6 arms.

| Planner | Optimal [95%] | Failures | Max loss | Median CPU per decision | Pairs only I-POMCP / only RTS got right | Holm p |
|---|---|---|---|---|---|---|
| **I-POMCP** | **60/60** [0.940, 1] | 0 | 0 | 118 s | — | — |
| IPF 64 + RTS, K = 2 | 2/60 [0.004, 0.115] | 55 | 52 | 0.1 s | 58 / 0 | 3.5 × 10⁻¹⁷ |
| IPF 64 + RTS, K = 4 | 1/60 [0.000, 0.089] | 56 | 52 | 0.2 s | 59 / 0 | 2.1 × 10⁻¹⁷ |
| IPF 64 + RTS, K = 8 | 2/60 [0.004, 0.115] | 57 | 24 | 0.2 s | 58 / 0 | 3.5 × 10⁻¹⁷ |
| Exact beliefs + RTS, K = 2 | 20/60 [0.217, 0.467] | 0 | 949 | 113 s | 40 / 0 | 5.5 × 10⁻¹² |
| Exact beliefs + RTS, K = 4 | 23/60 [0.261, 0.518] | 1 | 173 | 104 s | 37 / 0 | 2.9 × 10⁻¹¹ |
| Exact beliefs + RTS, K = 8 | 28/60 [0.337, 0.600] | 0 | 173 | 90 s | 32 / 0 | 4.7 × 10⁻¹⁰ |

- **The sampled IPF fails on Wumpus**, as on UAV (12.44–12.45). In 55–57 of 60 problems a modeled wumpus's sampled belief cannot explain its own sampled percept (Breeze, Glitter or Footstep after Forward), because deterministic percepts leave no particle with positive likelihood. This is the published method as specified, not a tuning choice.
- **With exact beliefs, RTS's search is still wrong on 53–67% of problems,** at about the same CPU as I-POMCP. Its errors include a 949 loss, a walk into death. Raising K from 2 to 8 helps only slowly (20 → 28 of 60).
- **The one exact-belief failure** (K = 4, seed 9 after 4 steps) is the planner's own update. RTS's modeled wumpus gives zero probability to the move the history's exact wumpus made, so the human's observation has zero evidence: opponent-model error made visible.
- **No pair favours RTS** in any arm.

### 12.60 Wumpus 3 × 3 level matrix: a second level pays the human against an intentional wumpus only (October 17, 2026)

**Design** (`results/c15_wumpus3_episodes_20261017`, `ENDPOINTS.md` fixed before any run; PLAN, October 17).
- **Agents:** a human i and a wumpus j at levels 0–2. Each holds a uniform prior over the opponent's lower levels at every nesting, and every model of the opponent matches the real opponent's configuration.
- **Planning:** exact real and search beliefs, horizon 2, I-POMCP with 400 simulations at the root and in every modeled solve, at most 20 steps.
- **Trials:** 100 per cell, paired by trial; 900 episodes in about 45 CPU-h.
- **Gate:** this exact agent configuration is 60/60 optimal against the exact I-POMDP at L2 H2 (12.58), and every root in the gate was scored exactly.
- **Payoffs:** the human gets +1,000 for gold and −1,000 for dying; the wumpus gets +1,000 for a kill and −1,000 when shot.

| Cell | Human's return [95%] | Wumpus's return [95%] | Gold | Killed by wumpus | Pit | Wumpus shot |
|---|---|---|---|---|---|---|
| i0 j0 | −336 [−469, −203] | 107 [−22, 229] | 0.14 | 0.27 | 0.17 | 0.15 |
| i0 j1 | −793 [−892, −681] | 542 [401, 682] | 0.06 | 0.69 | 0.14 | 0.14 |
| i0 j2 | −784 [−883, −673] | 522 [380, 662] | 0.06 | 0.67 | 0.15 | 0.14 |
| i1 j0 | 136 [7, 266] | −185 [−314, −55] | 0.30 | 0.14 | 0.00 | 0.31 |
| i1 j1 | −389 [−539, −229] | 251 [70, 422] | 0.20 | 0.57 | 0.00 | 0.31 |
| i1 j2 | −438 [−588, −288] | 281 [101, 461] | 0.18 | 0.60 | 0.00 | 0.31 |
| i2 j0 | 67 [−65, 197] | −144 [−277, −14] | 0.28 | 0.18 | 0.01 | 0.31 |
| i2 j1 | −209 [−378, −40] | 90 [−80, 251] | 0.28 | 0.41 | 0.06 | 0.31 |
| i2 j2 | −259 [−418, −97] | 120 [−50, 291] | 0.25 | 0.44 | 0.05 | 0.31 |

**Paired gains of one level step** (same opponent, by trial; sign-flip permutation tests, Holm over the 12):

| Step | vs opponent L0 | vs opponent L1 | vs opponent L2 |
|---|---|---|---|
| Human L0 → L1 | +473 [306, 638], Holm < 10⁻⁴ | +404 [236, 574], Holm < 10⁻⁴ | +346 [186, 507], Holm < 10⁻⁴ |
| **Human L1 → L2** | −70 [−160, 20], Holm 0.47 | **+179 [49, 319], Holm 0.087** | **+179 [70, 299], Holm 0.015** |
| Wumpus L0 → L1 | +435 [284, 585], Holm < 10⁻⁴ | +435 [334, 537], Holm < 10⁻⁴ | +234 [124, 335], Holm 0.0003 |
| Wumpus L1 → L2 | −20 [−60, 20], Holm 0.47 | +31 [−10, 80], Holm 0.33 | +30 [−30, 100], Holm 0.47 |

- **The first level is worth about 350–470 to either agent.** A level-0 human dies in 82–83% of episodes against an intentional wumpus.
- **A second level pays the human only when the wumpus has a mind to model.**
  - Against a level-1 or level-2 wumpus, a level-2 human gains about 179. It is killed by the wumpus in 41–44% of episodes, against 57–60% at level 1, and reaches the gold more often (0.25–0.28 against 0.18–0.20), at a small pit cost (0.05–0.06 against 0).
  - Against a random wumpus, the second level gains nothing (−70, not significant). This is the control: there is no mind to predict.
  - This matches Tiger's coordination game (12.37) and UAV (12.49).
- **A second level does not pay the wumpus** (+30, not significant). The wumpus's lever is pursuit, which a level-1 model of a level-0 human already captures at horizon 2.
- **Every human of level ≥ 1 opens with Shoot** (100 of 100), killing the wumpus in 31% of episodes whatever its level.
- **Confirmed on fresh trials 100–199** (the six human L1/L2 cells, with new environment and planner streams; `confirm/`, `confirm_analyze.py`). The primary test was fixed before the run (`ENDPOINTS.md`, step 3) and is analysed on these trials alone.

  | Test | Human L1 → L2 gain [95%] | p |
  |---|---|---|
  | **Primary:** against an intentional wumpus (j1 + j2 pooled, 200 pairs) | **+169 [84, 259]** | **0.0004** |
  | Secondary: against wumpus L2 | +209 [88, 348] | Holm 0.004 |
  | Secondary: against wumpus L1 | +129 [9, 259] | Holm 0.067 |
  | Control: against wumpus L0 | +20 [−70, 110] | 0.76 |

  - **The mechanism replicates:** the L2 human is killed by the wumpus in 48% of episodes against 64–67% at L1, and reaches the gold in 19–20% against 12–15%.
  - **A third level:** see 12.61 (human L2 → L3; not significant against any wumpus level).

### 12.61 Wumpus 3 × 3: a third level for the human predicts better but does not earn significantly more (October 17, 2026)

**Design** (`results/c15_wumpus3_episodes_20261017/level3`, `ENDPOINTS.md` step 4, fixed before the run).
- **Cells:** a level-3 human against wumpus levels 0, 1 and 2, trials 0–99, on the same snapshot and trial streams as 12.60's matrix, so each episode pairs with the level-2 human's episode of the same trial.
- **Agent:** the episode configuration of 12.58's level-3 gate (55/55 optimal against the exact I-POMDP): a uniform prior over all lower levels at every nesting, horizon 2, root 400, modeled 400, exact beliefs and exact search.
- **Test:** human L2 → L3 against each wumpus level, sign-flip permutation, Holm over 3 (`level3_analyze.py`).
- **Run:** 300 of 300 episodes, every attempt complete, no failures. The human's planning costs a median of 11–17 CPU-min per episode (mean 21–24), with a heavy tail: a few long episodes took about 4 CPU-h. Peak memory 1.5 GB. About 110 CPU-h for the human, more than the pilot's 13–14 CPU-min per episode predicted.

| Wumpus level | Human L2 → L3 gain [95%] | p | Holm | Killed by the wumpus, L3 (L2) | Gold, L3 | Pit, L3 |
|---|---|---|---|---|---|---|
| 0 (random) | −19 [−138, 100] | 0.86 | 0.86 | 0.22 (0.18) | 0.30 | 0.01 |
| 1 | +80 [−80, 239] | 0.36 | 0.72 | 0.36 (0.41) | 0.33 | 0.08 |
| 2 | +120 [−50, 290] | 0.18 | 0.55 | 0.35 (0.44) | 0.32 | 0.09 |

- **A third level is not significantly better against any wumpus level.** The point estimates against an intentional wumpus are positive (+80, +120), with fewer deaths by the wumpus (0.35–0.36 against 0.41–0.44), but both intervals include zero. Against a random wumpus the gain is −19 (the control).
- **The level-3 human predicts the wumpus better.** Averaged over each episode, it gives the wumpus's actual move probability 0.55 [0.53, 0.58] against a level-1 wumpus and 0.54 [0.52, 0.57] against a level-2 one, against 0.50 and 0.48 for the level-2 human (12.60; `w4_report_20261007/wumpus_section.py`). Against a random wumpus both give 0.33, the best possible. The same pattern held in Tiger: a third level predicts better but earns the same (12.27, 12.32).
- **Interim numbers were misleading,** and only the final n is reported. At 32 trials per cell, the gains read +312 and +374, with Holm p of 0.02–0.03. This is why the analysis was fixed in advance and run on the full n only.
- **Across domains,** depth pays up to about one level above the opponent's (Tiger 12.37 and 12.40, UAV 12.49, Wumpus 12.60–12.61), as in the cognitive-hierarchy literature. Whether a longer horizon would let a third level pay in Wumpus is untested: horizon 3 costs 6–36 CPU-min per decision (12.58).

### 12.62 IPF with the standard zero-likelihood repair still loses to I-POMCP: Wumpus 20–26/60, UAV 49–55/80 (October 18, 2026)

**Why:** a reviewer could call 12.59's IPF failure a strawman, because the published filter has no repair for zero-likelihood particles. This adds the standard repair, likelihood smoothing, and re-runs the baseline (PLAN.md, publication gaps, G3).

**Change:** `ParticleInteractiveFilter(smoothing=ε)` weights each particle by (1 − ε) O_i + ε / |Ω_i| at every nesting level. The new filter name `"smoothed"` uses ε = `IPF_SMOOTHING` = 0.01, declared before the runs. `"sampled"` is still the published filter.

**Gate** (`results/g3_ipf_fix_20261018`, `ENDPOINTS.md` fixed before the runs). Identical cases were run under the code before the change (`75ea9e1`) and after it, with the published filter: Wumpus 3 × 3 L2 H2 (60 problems × K = 2, 4, 8) and UAV L2 H2 (80 problems × K = 2, 4, 8). **420/420 rows were identical**, apart from CPU, and also identical to the stored rows of 12.59 and 12.43. There are 350 tests.

**Design:**
- Smoothed IPF (64 particles) + RTS on the same problems.
- Paired exact McNemar tests against the published IPF at the same K, and against I-POMCP's qualified cells (Wumpus: 12.58, 60/60; UAV: 12.43, 80/80).
- Holm correction over the 6 tests against I-POMCP.

| Domain, planner | Optimal [95%] | Failures | Max loss | Median CPU per decision | vs published IPF (pairs only smoothed / only published right) | vs I-POMCP (pairs only I-POMCP / only smoothed right), Holm p |
|---|---|---|---|---|---|---|
| Wumpus, I-POMCP | **60/60** [0.940, 1] | 0 | 0 | 118 s | — | — |
| Wumpus, smoothed IPF + RTS, K = 2 | 22/60 [0.246, 0.501] | 3 | 949 | 0.8 s | 20 / 0 (published 2/60, 55 failures) | 38 / 0, 3.6 × 10⁻¹¹ |
| Wumpus, K = 4 | 20/60 [0.217, 0.467] | 1 | 949 | 1.0 s | 19 / 0 (published 1/60, 56 failures) | 40 / 0, 1.1 × 10⁻¹¹ |
| Wumpus, K = 8 | 26/60 [0.306, 0.568] | 1 | 949 | 1.1 s | 24 / 0 (published 2/60, 57 failures) | 34 / 0, 4.7 × 10⁻¹⁰ |
| UAV, I-POMCP | **80/80** [0.955, 1] | 0 | 0 | 0.6 s | — | — |
| UAV, smoothed IPF + RTS, K = 2 | 49/80 [0.497, 0.719] | 0 | 43.7 | 0.24 s | 3 / 0 (published 46/80), p = 0.25 | 31 / 0, 2.8 × 10⁻⁹ |
| UAV, K = 4 | 54/80 [0.561, 0.776] | 0 | 15.0 | 0.34 s | 3 / 0 (published 51/80), p = 0.25 | 26 / 0, 6 × 10⁻⁸ |
| UAV, K = 8 | 55/80 [0.574, 0.787] | 0 | 14.9 | 0.34 s | 3 / 2 (published 54/80), p = 1 | 25 / 0, 6 × 10⁻⁸ |

- **The repair works where it should.** On Wumpus it removes 54–56 of the 55–57 failures, and lifts the baseline from 1–2/60 to 20–26/60 (McNemar p ≤ 4 × 10⁻⁶). On UAV, where the sensors are already noisy and nothing failed, it changes almost nothing (p ≥ 0.25).
- **It does not close the gap.** I-POMCP is right on every problem the smoothed baseline gets right, and on 25–40 more per arm (Holm p ≤ 6 × 10⁻⁸ in all six). On Wumpus, the smoothed baseline's 20–26/60 matches RTS on *exact* beliefs (20–28/60, 12.59), as does its 949 maximum loss (a walk into death). Once the filter is repaired, what remains is RTS's search.
- **The remaining 5 failures** (K = 2: 3; K = 4 and K = 8: 1 each; all after 4 steps) are a nested wumpus model, all of whose particles predicted the episode would end when it had not. With ε > 0 every declared observation has positive weight, so the update's only remaining rejection is this terminal-consistency check, which likelihood smoothing does not address.
- **Reading, fixed in advance:** the smoothed baseline still loses significantly in both domains, so 12.59's gap is not a strawman artefact. The paper should report the published IPF, the smoothed IPF and I-POMCP side by side (PAPER.md, Table 1).

### 12.63 Ablation table: what each component of I-POMCP buys, from paired results (October 18, 2026)

**Why:** the publication-gaps phase (G4) needs one table of the planner's components. Each row compares the planner with one component removed or replaced, on the same problems and seeds. Most rows come from earlier paired A/Bs. One row is new: root chance scoring off, recomputed from stored rows that keep the search-only values beside the scored ones.

**Terms.**
- *Chance scoring:* after search, the root's values are recomputed with exact chance probabilities, and actions with two steps left are scored exactly over every branch (12.14).
- *Search-only:* the decision taken from the raw search values instead. Modeled opponents keep their scoring, so this row ablates the root's scoring only.

| Component removed or replaced | Problems (n) | With the component | Without it | Paired test | RESULTS |
|---|---|---|---|---|---|
| **Root chance scoring** → raw search values (new) | UAV qualified cells: L1 H4 at 4,000, L2 H2, L2 H3, L3 H2, L1 H5 at 64,000 (560) | 560/560 optimal | 526/560 (L2 H3: 63/80) | 34 fixed, 0 broken; sign test p = 1.2 × 10⁻¹⁰ | this section; 12.41 |
| Root chance scoring → raw search values | Tiger gates: published L2 H4 and L3 H3 (542); coordination L2 H3, L3 H3, L2–L4 H2 (1,244); 1,786 in all | 1,786/1,786 | 1,786/1,786 | no decision changes; it lowers Q error (L2 H3 k = 0: 0.13 → 0.05) | this section; 11; 12.23 |
| Exact scoring with two steps left → fallback to search Q | L3 H4 residual beliefs × 10 seeds (80) | 4 wrong | 14 wrong | 12 fixed, 2 broken; exact McNemar p = 0.013 | 12.14 |
| Compressed exact filter → the sampled IPF, same RTS search, Wumpus 3 × 3 L2 H2 | 60 problems × K = 2, 4, 8 | exact beliefs: 20–28/60 | IPF 64: 1–2/60 (55–57 failures) | — | 12.59 |
| Compressed filter → sampled IPF at equal N (belief error) | Tiger L2 histories | belief error 4–40× smaller at equal N | — | paired over histories | Section 8 |
| Stratified → systematic compression | Wumpus 3 × 3 L2 H2, N = 256 / 1,024 (60 each) | 6 / 1 wrong | 19 / 12 wrong | p = 0.011 / 0.0034 (no other grid differs) | 12.57 |
| Consistent stopping (1/n) → flat cap for modeled solves | L2 H4, L3 H3, L3 H4 near-tie sets (2,016 solves) | same decisions | same decisions | 14–23% fewer simulations; in-situ CPU 0.88×, n.s. | 12.11, 12.15 |
| Root budget 4,000 → 1,000 | Tiger L2 H4 (380) | 380/380; optimal-action Q error 0.031 | 380/380; 0.121 | Wilcoxon p = 5 × 10⁻³⁹ on Q error; CPU 1.06× | 12.33 |
| Certified bounds → none | every real decision | intervals contain the exact Q in 1,016/1,016 checked | — | cost negligible (successors cached) | 12.17, 12.41 |
| Uniform level prior (level 0 included) → a single fixed level | coordination Tiger, i at level 3 against the matrix's level-1 j, n = 100 trials | uniform: −6.0 [−13.1, 1.1] against the correct point prior (Holm 0.41) | wrong point prior: −17.5 (over-estimated, Holm 0.011) or −30.0 (under-estimated, Holm < 0.001) | paired by trial | 12.39 |
| Level-0 mass → point-mass levels, level 3 with deterministic percepts | Wumpus 3 × 3 L3 H2 | exact I-POMDP defined; 55/55 | the exact reference's nested update has zero evidence (undefined) | — | 12.58 |

- **What the table says:**
  - The components that change *decisions* are chance scoring (on UAV, not Tiger), exact two-step scoring, the compressed exact filter, and stratified compression.
  - The stopping rule and the root budget change only cost or Q error.
  - Certified bounds change nothing but add a guarantee.
- **The level prior** (12.39, imported): when the opponent's level is unknown, a uniform prior that includes level 0 costs little (−6.0, not significant) relative to knowing it. A single fixed level costs 17.5–30.0 when it is wrong, because a point prior can never be revised. With deterministic percepts, level-0 mass is also what keeps the level-3 update defined (12.58; THEORY.md).

### 12.64 The exact screen on Wumpus 3 × 3 and UAV: a second level is worth 19.7 (Wumpus) and 6.0 (UAV) per decision against an intentional opponent, and costs 1.7 and 1.6 if the opponent is one level lower (October 18, 2026)

**Why:** contribution 3, sharper (PAPER.md, G5b). Can an exact, episode-free screen say in advance whether depth pays? It exists for Tiger (12.22). This applies it to Wumpus 3 × 3 at H2, where 12.58 has the exact level-2 I-POMDP Q for 60 problems.

**Caveat:** 12.60's episodes were already known, so this checks the screen's consistency; it is not a blind prediction.

**Design** (`results/g5_wumpus_screen_20261018`, `ENDPOINTS.md` fixed before the run). At each problem's history h, the exact level-1 human's Q was also computed: its best response at its exact belief, which models the wumpus as uniformly random.
- **Value of a second level:** the exact level-2 loss of the level-1 action, V₂(h) − Q₂(h, a₁), with ties averaged. This is what modeling the wumpus as an agent buys if the wumpus really is the exact level-1 wumpus.
- **Cost of a second level:** V₁(h) − Q₁(h, a₂). This is what it costs if the wumpus is in fact random.
- About 2 CPU-min.

| Histories | Value of a second level: share positive, mean (max) | Cost if the wumpus is random: share positive, mean (max) |
|---|---|---|
| All 60 | 19/60, **19.7** (712) | 5/60, 9.1 (450) |
| At the prior (20) | 0/20, 0 | 0/20, 0 |
| After 2 steps (20) | 8/20, 7.8 | 1/20, 0.1 |
| After 4 steps (20) | 11/20, 51.2 | 4/20, 27.2 |

- **Direction matches 12.60:**
  - Against an intentional wumpus, the value is positive and grows as the history reveals the wumpus (0 → 7.8 → 51 per decision).
  - 12.60's episodes found +169 [84, 259] per episode.
- **The cost column is not the matched control.** On these intentional-wumpus histories, a second level would cost 9.1 per decision if the wumpus were random, all at 5 histories. The matched control below uses random-wumpus histories.
- **Limit on the comparison:** the histories here come from the exact *intentional* wumpus, while 12.60's control episodes run against a random one. The screen therefore bounds the cost at histories an intentional wumpus produces, not at the ones a random wumpus produces.
- **Matched control** (`control.py`; added after the main screen, with its endpoints fixed before its run). The same 60 problems, but on histories from a uniformly random wumpus, the situation of 12.60's control episodes.
  - 2 of 60 histories have zero evidence for the exact level-2 human, whose level-1 wumpus model gives the random move probability 0; they are reported, not scored.
  - On the other 58, the cost of a second level is **1.7 per decision** (positive at 4 of 58; mean 0 at the prior, 0.1 after 2 steps, 5.2 after 4).
  - The value it would have against an intentional wumpus is 9.2 per decision.
  - So the 9.1 cost above came from histories an intentional wumpus produces. Where the wumpus really is random, depth costs little, consistent with 12.60's control (+20, not significant).
- **UAV, the same screen for the UAV at no CPU** (`uav_screen.py`). It uses the exact Q stored with the 12.41 gate rows, at the gate's law chains (i's law first; the deeper chain adds the opponent's).

  | Horizon, step | Chains | Value of the extra level: share positive, mean (max) | Cost if the target is one level lower: share positive, mean |
  |---|---|---|---|
  | H2, 1 → 2 | 16 | 11/16, **6.0** (21.8) | 8/16, 1.6 |
  | H3, 1 → 2 | 16 | 7/16, 2.2 (23.1) | 9/16, 2.6 |
  | H2, 2 → 3 | 64 | 41/64, 6.1 (35.0) | 40/64, 5.8 |

  - **H2, 1 → 2** agrees with 12.49, where a second level pays the UAV (+5.8 against a level-1 target).
  - **H2, 2 → 3 is a prediction no episode has tested.** A third level would gain as much against a level-2 target as it loses against a level-1 one. So it should pay only when the target really is level 2.
  - These are per-decision values at the gate's chains, not the episode distribution. The target's own screen needs references from its side, which are not stored.
- **Reading:** the screen tells *where* depth pays: never at the prior, and at 19 of the 40 later histories. It is not tied to sensing the wumpus: 17 of those 19 histories contain no stench. This is a reusable, episode-free check. A matched control screen (histories from a random wumpus) is the obvious next step, at about the same cost.

### 12.96 S5: exact pair curves on Wumpus 3 × 3. A second level pays both agents against an intentional opponent and costs a little against a random one; 12.60's "a second level does not pay the wumpus" was a power failure, not an absence of effect (October 23, 2026)

**Why:** M2/M4 on Wumpus rested on episodes with n = 100–200 (12.60–12.61). This applies 12.72's zero-variance exact pair curves.

**Design** (`results/s5_wumpus_exact_curves_20261023`; `ENDPOINTS.md` with a probe rule fixed before the run; `gain_table.py` → `gain_table.txt`):
- Exact agents at levels 0–2 for the human and the wumpus, H2, γ 0.95, uniform nesting, ties uniform (12.60's agents, exact).
- Up to 20 steps; a 3 GB cap.
- **Windows:** 5–8 steps (20 for random against random).
- **Probe:** i2_j1 reached 5 steps in about 3 CPU-min, past the rule's threshold of 4.

**Exact one-level gains** (cumulative reward by step; the opponent's level in columns):

| Agent, step | By step | m = 0 | m = 1 | m = 2 |
|---|---|---|---|---|
| Human 1 → 2 | 3 / 4 / 5 | 0 / −3.8 / **−3.4** | 0 / +3.9 / **+11.7** | 0 / +3.9 / **+17.3** |
| Wumpus 1 → 2 | 3 / 4 / 5 | 0 / +0.6 / **−1.4** | 0 / +13.7 / **+24.6** | 0 / +13.7 / **+18.9** |
| Human 0 → 1 | 5 | +262.1 | +218.8 | +190.4 |
| Wumpus 0 → 1 | 5 | +130.1 | +150.8 | +132.7 |

**Findings (zero variance, inside the window):**
- **The second level is worth nothing for the first 3 steps** for either agent. At H2, levels 1 and 2 act alike until the history reveals the opponent, as 12.64's screen found at the prior.
- **Then it pays against an intentional opponent and costs a little against a random one,** for both agents. This is the coordination game's and UAV's pattern (12.88, 12.93), and Proposition 3's corollary in episodes.
- **12.60's "a second level does not pay the wumpus" is withdrawn.** The exact gain is +19 to +25 by step 5 against an intentional human. 12.60's +30 [−30, 100] (n = 100, 20 steps) was underpowered, not null. The wumpus's depth is worth about as much as the human's at this horizon. The paper (S7 draft) does not use the old claim.
- **The human's second level against an intentional wumpus** grows from +3.9 to +11.7–17.3 between steps 4 and 5, consistent with 12.60's +169 [84, 259] over 20 fresh-trial steps.
- **Cost:** 2.6 CPU-h.

### 12.95 S4, cell 2: Wumpus 3 × 3 L2 H2 at reachable histories. I-POMCP is optimal on all 219 distinct beliefs (CP ≥ 0.983) at 75–89 s; the best baseline reaches 77% (I-NTMCP tuned, M = 64,000); the sampled IPF fails on about 90%. This is also S6's held-out Wumpus check (October 23, 2026)

**Why:** message M1 on the domain with deterministic percepts, and S6's held-out check: 12.58's configuration was chosen on 60 problems, and these 219 beliefs are new.

**Design** (`results/s4_compute_benchmark_20261022`, ENDPOINTS "Cell 2" and amendment 7; `analyze.py wumpus` → `wumpus_analysis.txt`; `figure_m1.py`):
- 360 problems from the common prior (history lengths 2–4; i uniform among non-ending actions; the wumpus as its exact level-1 model). 25 ended during the history; **219 distinct beliefs**.
- Exact references with point-mass nesting (about 123 CPU-s each).
- Arms:
  - I-POMCP 400/400 (12.60's episode configuration) on all 219;
  - all other arms on a fixed subset of 100.
  - Amendment 7 trimmed I-POMCP 1,000/400, and published I-NTMCP at M = 64,000, for cost.
- **Cost:** about 50 CPU-h, references included. **S4 in total is about 160 CPU-h** (both cells), above the 100 CPU-h line, and flagged. The user granted autonomy while away.
- **Incomplete:** 1 I-NTMCP tuned M = 64,000 job hit the 2-hour timeout (n = 99 for that arm).

| Planner, budget | Optimal [95% CP] | Mean / max loss | CPU per decision (median) | Failures |
|---|---|---|---|---|
| **I-POMCP 400/400, exact beliefs (all 219)** | **219 / 219 [0.983, 1]** | 0 / 0 | 89 s | 0 |
| I-POMCP 400/400 (subset) | 100 / 100 [0.964, 1] | 0 / 0 | 75 s | 0 |
| I-POMCP 100/100 (subset) | 100 / 100 [0.964, 1] | 0 / 0 | 73 s | 0 |
| I-NTMCP tuned M = 1,000 / 4,000 / 16,000 / 64,000 | 55 / 64 / 70 / 77 (of 99) | 19–55 / up to 639 | 0.4 / 1.4 / 5.8 / 23 s | 0 |
| I-NTMCP published M = 1,000 / 4,000 / 16,000 | 52 / 54 / 50 | 60–75 / up to 639 | 0.4 / 1.5 / 6.2 s | 0 |
| IPF + RTS, sampled IPF, K = 2 / 4 / 8 | 5 / 4 / 0 | about 60–100 | 0.2 s | 89 / 90 / 98 |
| IPF + RTS, smoothed IPF, K = 2 / 4 / 8 | 35 / 35 / 35 | 76–103 / up to 1,001 | 0.9–1.3 s | 24–25 |

**Endpoints:**
- **Primary:** both I-POMCP arms reach the CP 0.95 bound (on all 219: 219 / 219, ≥ 0.983). No baseline arm reaches it at any budget.
- **Secondary (exact McNemar on the subset, Holm):** I-POMCP 400/400 is right where a baseline arm is wrong on 22–100 beliefs, and never the converse (0 in every comparison). Holm p ≤ 9.5 × 10⁻⁷ for every arm.

**Findings:**
- **On Wumpus, I-POMCP needs no more than 100 simulations with exact beliefs.** Deterministic percepts keep the exact beliefs small; the cost per decision (73–89 s) is the exact filter and the modeled wumpus.
- **12.58's configuration holds on 219 fresh beliefs (S6):** this closes the "no held-out Wumpus check" gap (HANDOFF 0.0000000000).
- **The sampled IPF fails on about 90% of beliefs** (12.59's failure mode, at reachable histories).
  - Likelihood smoothing repairs the zero-likelihood case, but still fails on about 25%.
  - The remaining failures are nested particle depletion: the wumpus's own update keeps only particles whose successors continue the episode. When every sampled particle ends the episode (the wumpus believes the human is directly ahead), no particle survives, which smoothing cannot repair.
  - Where it does not fail, the smoothed filter is optimal on 35%, with losses up to 1,001 (a death).
- **I-NTMCP improves with M on Wumpus** (tuned 55% → 77%), unlike on UAV (12.94), but stays far from the benchmark, with losses up to 639.
- **Figure 1 (M1)** is final: `docs/paper/latex/compute_benchmark_col.pdf` (column) and `compute_benchmark.pdf` (wide).

### 12.94 S4, cell 1: compute to the Bayes-optimal benchmark on UAV L2 H3 at histories reachable from the common prior. I-POMCP reaches it (100/100, CP ≥ 0.964) at about 217 s per decision with exact beliefs; IPF + RTS levels off at 81% and I-NTMCP at 58%; belief compression, not search, separates I-POMCP's rungs (October 23, 2026)

**Why:** message M1, how much CPU I-POMCP needs to reach Bayes-optimal decisions against the baselines. The gate grids (12.41, 12.43) placed agents at named beliefs I-NTMCP cannot start from. Here every planner decides the same problems, i's own histories from the common prior (I-NTMCP's model class), which are also the beliefs agents reach in play.

**Design** (`results/s4_compute_benchmark_20261022`; `ENDPOINTS.md` with amendments 1–8, each dated and written before the data it governs; `analyze.py` → `uav_analysis.txt`; `escalation.py` → `escalation.txt`; `figure_m1.py`):
- **Problems:** `experiments.reachable_decisions`.
  - i acts uniformly among non-ending actions; the target acts as the exact level-1 model.
  - History lengths 1–5; 1,400 problems with **512 distinct beliefs**.
  - Exact I-POMDP references with point-mass nesting.
- **Arms:** one decision per distinct belief, planner seed 0.
  - I-POMCP's qualified gate configurations on all 512.
  - Every arm on a fixed random subset of 100.
- **Cost:** about 110 CPU-h with the references (UAV cell), above the 100 CPU-h line; flagged (the user granted autonomy while away).

**Results on the subset** (n = 100 distinct beliefs per arm; optimal = loss ≤ 10⁻⁹, CP 95%; CPU = median per decision, the decision's own search):

| Planner, budget | Optimal [95%] | Within ε = 0.05 | Mean / max loss | CPU (s) |
|---|---|---|---|---|
| I-POMCP 100/100, N = 64 | 79 [0.70, 0.87] | 94 | 0.013 / 0.40 | 13.8 |
| I-POMCP 1,000/400, N = 64 | 81 [0.72, 0.88] | 89 | 0.017 / 0.34 | 52 |
| I-POMCP 1,000/400, N = 256 | 80 [0.71, 0.87] | 92 | 0.013 / 0.16 | 102 |
| I-POMCP 1,000/400, N = 1,024 | 97 [0.92, 0.99] | 99 | 0.001 / 0.09 | 128 |
| I-POMCP 1,000/400, exact beliefs | 97 [0.92, 0.99] | 99 | 0.001 / 0.09 | 136 |
| **I-POMCP 1,000/1,600, exact beliefs** | **100 [0.964, 1]** | 100 | 0 / 0 | **217** |
| IPF + RTS K = 2 / 4 / 8 (N = 64) | 75 / 81 / 81 | 83 / 91 / 91 | 0.16 / 0.09 / 0.06; max 8.9 | 137 / 313 / 451 |
| I-NTMCP published M = 1,000 / 4,000 / 16,000 / 64,000 | 45 / 54 / 53 / 37 | 50–62 | 3.1–9.9; max about 48 | 0.8 / 3.1 / 12 / 49 |
| I-NTMCP tuned (same M) | 47 / 58 / 49 / 39 | 46–66 | 3.8–9.0; max about 55 | 0.8 / 3.0 / 12 / 48 |

On all 512 distinct beliefs: N = 64 is optimal on 405 [0.75, 0.83] (466 within ε), and N = 256 on 431 [0.81, 0.87] (488 within ε).

**Endpoints:**
- **Primary:** the only arm whose CP lower bound reaches 0.95 is I-POMCP 1,000/1,600 with exact beliefs (100/100, ≥ 0.964, 217 s). No baseline arm reaches 0.95 at any budget tested.
- **Secondary (exact McNemar, Holm over 11):** that arm is right where a baseline arm is wrong on 19–63 beliefs, and never the other way (0 in every comparison). Holm p ≤ 7.6 × 10⁻⁶ against every IPF + RTS arm and ≤ 1.8 × 10⁻¹² against every I-NTMCP arm.

**Findings:**
- **What separates I-POMCP's rungs is the belief bound, not simulations.**
  - At N = 64, root 100 to 1,000 and target 100 to 400 all give 79–81%.
  - N = 1,024 or exact beliefs give 97%, and a stronger target model (1,600) closes the last 3.
  - Diagnosis 1 (ENDPOINTS): exact beliefs fixed every N = 64 miss tried, while root 4,000 or target 1,600 at N = 64 fixed none.
  - The real belief has only 52–163 atoms. The compression that matters is in the search tree's beliefs, whose support grows by up to 45× per simulated step (as in Tiger, 12.78).
  - **12.41's qualification at history 0 does not carry to reachable histories; the paper must say so** and qualify the configuration where it is used.
- **I-POMCP's errors are small even below the benchmark:** 89–94% of its N = 64 decisions are within ε = 0.05, with a maximum loss of 0.40. IPF + RTS's misses reach 8.9, and I-NTMCP's about 50, the cost of a missed spotting.
- **IPF + RTS levels off at about 81%** from K = 4 to K = 8, at 2–3.5× the CPU of I-POMCP's benchmark configuration.
- **I-NTMCP gets worse with more simulations** (54–58% at M = 4,000; 37–39% at 64,000; same pattern for both variants). This is consistent with its opponent model: it predicts the target by a softmax over visit counts, not by the target's best response. More search then converges toward the answer to a different problem than the I-POMDP the reference solves. A hypothesis; not tested further.
- **Near-tie escalation (amendment 6), calibrated on the subset** (`escalation.txt`):
  - Escalating the N = 64 base to the exact-belief shadow where the base's top-two gap is ≤ 2 matches the shadow (97/100) at 0.67× its mean CPU (143 against 214 s).
  - 10 of the 19 base misses have a scored gap of exactly 0: compression makes distinct actions look tied.
  - It is not cheaper than N = 1,024 (128 s median). So the held-out validation was not run, and escalation is not recommended for UAV.

**Use in the paper:** Figure 1 (M1) panel (a) is `docs/paper/latex/compute_benchmark.pdf`, from `figure_m1.py`. Table 1's UAV rows remain gate evidence at named beliefs. The text states the qualified configuration per use, and that reachable histories need exact (or N ≥ 1,024) beliefs.

### 12.93 S5: exact pair curves for the UAV level matrix. Zero-variance level gains agree in sign with every 12.91 episode gain; over-thinking against random is real but small (UAV −1.3 to −2.0, target −1.2 to −3.3 by steps 8–10); predictability costs the target −22.7 by step 5; the planner pairs contain the exact pair at 31 of 36 agent-steps (October 23, 2026)

**Why:** M2 and M4 on UAV rested on n = 100 episodes per cell (12.91). 12.72's exact pair curves give the Bayes-optimal agents' expected rewards with zero variance.

**Design** (`results/s5_uav_exact_curves_20261023`; `ENDPOINTS.md` fixed before the run; `job.py` = 12.72's job on UAV; `gain_table.py` → `gain_table.txt`; `compare_planner.py` → `compare_planner.txt`):
- Exact agents at levels 0–2 for both, H3, γ 0.95, uniform nesting, ties uniform: the agents of 12.91's replay.
- Up to 20 steps, with a 3 GB cap and 2 × 10⁶ nodes.
- **Windows:** i0_j0 and i1_j0 20 steps; i2_j0 and i2_j1 11; i0_j2, i1_j2 and i2_j2 10; i0_j1 and i1_j1 9 (still running at the time of writing).
- **Cost:** 1.2 CPU-h logged so far.

**Exact one-level gains** (the stepping agent's k → k + 1 against the opponent's level m; cumulative reward by the step):

| Agent, step | By step | m = 0 | m = 1 | m = 2 |
|---|---|---|---|---|
| UAV 0 → 1 | 5 / 8 / 10 | +14.8 / +21.3 / +24.4 | +25.0 / +56.0 / — | +0.1 / **−4.1** / **−7.9** |
| UAV 1 → 2 | 5 / 8 / 10 | **−0.4 / −2.0 / −1.3** | +27.8 / +7.9 / — | +16.1 / +42.9 / +51.2 |
| Target 0 → 1 | 5 / 8 | +14.0 / +17.4 | +3.0 / **−9.6** | **−22.7 / −15.0** (−10.1 by 10) |
| Target 1 → 2 | 5 / 8 | **−1.2 / −3.3** | +21.5 / +41.3 | +32.3 / +12.8 (+10.5 by 10) |

**Findings (zero variance, inside each window):**
- **Every sign agrees with 12.91's 20-step episode gains.**
  - The second level pays against the opponent it newly models (UAV +7.9 against level 1 and +42.9 against level 2 by step 8; target +41.3 against level 1).
  - The first level pays against random.
- **Over-thinking against a random opponent is real and small:** the UAV loses 0.4–2.0, the target 1.2–3.3. That is why the UAV's episode estimate (−5.6 by step 20, n = 100) is not significant: the effect is a few points against a 20-step reward SD of about 30. M4's UAV example should cite these exact values.
- **Predictability is exact and large early:**
  - A level-1 target loses 22.7 by step 5 against a level-2 UAV (shrinking to 10 by step 10), and 9.6 by step 8 against a level-1 UAV.
  - A level-1 UAV loses 4–8 against a level-2 target.
  - In 12.91's episodes the target's version was n.s. at step 20 (−2.5) and the UAV's significant (−23.8). The exact curves show both exist, with different time courses.
- **The planner pairs against the exact pairs** (`compare_planner.txt`; 12.91's episodes at steps 3, 5 and the window's end): the exact value lies inside the planner's 95% interval at 31 of 36 agent-steps.
  - Four of the five exclusions are cell i2_j1, where all 100 planner episodes spot the target at step 5 (an interval of zero width). The exact pair differs by 0.07–0.65 through rare tie-driven branches that 100 trials did not draw.
  - The fifth is i2_j0 at step 11 (planner 7.75 ± 7.45 against exact 15.54, z ≈ 2.05; rows are shared across steps). It is flagged for a fresh-trial check (the S3(c) pattern).
- **Fresh-trial check of i2_j0** (`results/s5_uav_i2j0_fresh_20261023`; `ENDPOINTS.md` fixed before the run; about 3.4 CPU-h):
  - On 300 fresh 11-step episodes (trials 100–399), the level-2 UAV earns **14.32 [10.28, 18.35]** by step 11 against the exact 15.54 (z = −0.59).
  - Trials 0–99 reproduce 12.91's actions over steps 1–11 in 100 of 100.
  - The excursion was the shared 100-trial sample. Apart from the degenerate i2_j1 intervals, the UAV planner pairs match the Bayes-optimal pairs.

### 12.92 S3(c): the level-4 coordination pairs match the exact pair at step 3 on 600 fresh trials (|z| ≤ 0.97); 12.73's excursion (z ≈ 2) was its shared 100-trial sample (October 22, 2026)

**Why:** HANDOFF S3 item 3. In 12.73 the level-4 pairs i4_j1 and i4_j2 sat above the exact pair by step 3 (21.1 against 16.8; z ≈ 2.0–2.1), on the same 100 trials for every agent and cell. Figure 2's level-4 panel needed a fresh check.

**Design** (`results/s3c_level4_step3_20261022`; `ENDPOINTS.md` fixed before the run; `job.py` = 12.73's job at 3 steps; `analyze.py` → `analysis.txt`):
- 12.73's configuration (coordination, H2, 100 simulations, N = 64, uniform level priors), source 198bf5b (the adopted tie rule).
- Trials 0–699 per cell: trials 100–699 are the fresh sample; trials 0–99 are the reproducibility check.
- Rao-Blackwellized cumulative reward at step 3. The forward–backward pass conditions on the 3-step episode's observations, which is unbiased for the same estimand.
- **Cost:** about 8 CPU-h.

| Cell, agent | Fresh n | RB mean by step 3 [95%] | Exact (12.72) | z |
|---|---|---|---|---|
| i4_j1, i (level 4) | 600 | 17.49 [15.84, 19.14] | 16.814 | +0.80 |
| i4_j1, j (level 1) | 600 | 17.07 [15.40, 18.73] | 16.242 | +0.97 |
| i4_j2, i (level 4) | 600 | 18.63 [16.96, 20.31] | 18.224 | +0.48 |
| i4_j2, j (level 2) | 600 | 18.70 [17.03, 20.37] | 18.166 | +0.63 |

**Findings:**
- **Every exact value lies inside the fresh 95% interval** (|z| ≤ 0.97). The pre-fixed precision target (half-width ≤ 1.7, 10% of the exact value) is met.
- **Reproducibility:** trials 0–99 take the same actions at steps 1–3 as 12.73's stored trials in 100 of 100 per cell, under the adopted tie rule.
- **Reading, fixed in advance:** 12.73's excursion was the shared 100-trial sample. Figure 2's level-4 panel stands, and its caption should cite the fresh n = 600 at step 3.

### 12.91 U6b: the UAV level matrix under the exact tie convention. A second level for the UAV is worth +81 against a level-2 target (12.49: +26); over-thinking against random is no longer significant for the UAV (−5.6, Holm 0.09) but is for the target (−6.5, Holm 0.035); a first level loses −23.8 against an opponent two levels up (October 22, 2026)

**Why:** 12.87. Under the adopted tie rule, a level-2 target's episodes change, so every UAV level result of 12.49 is re-measured before the paper uses it.

**Design** (`results/u6b_uav_matrix_ties_20261022`; `ENDPOINTS.md` fixed before the run; `analyze.py` → `analysis.txt`):
- 12.49 exactly: levels 0–2 for both agents, depth 3, root 1,000, N = 64, 20 steps, uniform priors, n = 100 per cell, the same trial seeds. Source 198bf5b.
- The level-2 UAV cells are 12.87's gate run (identical source).
- **Cost:** about 8 CPU-h here, plus 12.87's gate.

| UAV i \ target j | Level 0 | Level 1 | Level 2 |
|---|---|---|---|
| **Level 0** | −15.4 (spotted 73%) | −67.2 (36%) | −52.3 (49%) |
| **Level 1** | 17.7 (96%) | 24.1 (100%) | −76.0 (25%) |
| **Level 2** | 12.2 (95%) | **30.0** (100%, 5.0 steps) | **5.3** (87%) |

(The UAV's mean 20-step reward and the share of episodes in which it spots the target. 12.49's cells were identical where no exact tie arose, for example i1_j0 and i2_j1 = 30.0, and i2_j2 was −45.6 with 48% spotted.)

**Paired one-level differences, Holm over the family of 12** (12.49 in brackets):

| Agent | Step | Against level 0 | Against level 1 | Against level 2 |
|---|---|---|---|---|
| UAV | 0 → 1 | +33.1 (+33.1) | +91.3 (+91.1) | **−23.8 [−36.4, −11.6], Holm 0.002** (−16.4, 0.055) |
| UAV | 1 → 2 | **−5.6 [−12.0, +0.4], Holm 0.09** (−7.2, 0.032) | +6.0 [4.1, 8.2] (+5.8) | **+81.3 [68.3, 93.1]** (+26.0) |
| Target | 0 → 1 | +18.5 (+18.5) | −2.0, Holm 0.09 (−2.0) | −2.5 [−5.0, −0.5], Holm 0.074 (−3.5, 0.032) |
| Target | 1 → 2 | **−6.5 [−11.5, −2.0], Holm 0.035** (−5.0, 0.09) | +37.5 [33.0, 41.5] (+35.0) | **+6.5 [3.5, 10.0]** (+26.0) |

**What changed and what holds:**
- **Holds:**
  - A second level pays against the opponent level it newly models (m = k + 1): UAV +6.0 and target +37.5, both p < 0.001.
  - A first level pays against random and level-1 opponents.
  - The level-2 UAV's anticipation of a level-1 target (5.0 steps, 100% spotted) is unchanged.
- **Changed by the convention:**
  - **Against a level-2 opponent,** the UAV's second level is worth +81.3 and the target's only +6.5 (12.49: +26 and +26). Under exact ties the level-2 target no longer escapes a level-2 UAV (87% spotted). Its 12.49 evasion rested on float tie-breaking (12.87). The 12.49 reading "the target gains more from depth than the UAV" is withdrawn.
  - **Over-thinking against a random opponent:** the UAV's −5.6 is no longer significant (Holm 0.09), but the target's −6.5 now is (Holm 0.035). Both point the same way, and the exact screen's per-decision cost (1.6 at H2, 12.64) does not depend on episodes. The paper's M4 example must say "−5.6, not significant after correction" for the UAV, and can use the target's −6.5.
  - **Predictability (H-CH′, CLAIMS_AUDIT A7):** a first level costs the UAV −23.8 against a level-2 target, now significant. A level-1 UAV best-responds to a random target, and the level-2 target models exactly that UAV and evades it (spotted 25% against 49% for a random UAV). The target's mirror effect (−2.5) is small and not significant.

**Replay against the Bayes-optimal agent** (12.81's design, every intentional agent; `replay.py`, `replay_analysis.txt`; 12 series × 100 trials, about 1.5 CPU-h):

| Agent | Decisions agreeing [95% CP] | Level-posterior TV, mean / max |
|---|---|---|
| UAV, level 1 | 3,116 / 3,116 [0.9988, 1] | 0 / 0 |
| UAV, level 2 | 1,837 / 1,837 [0.9980, 1] | 0.0000 / 0.014 |
| Target, level 1 | 2,859 / 2,859 [0.9987, 1] | 0 / 0 |
| Target, level 2 | 2,579 / 2,579 [0.9986, 1] | 0 / 0 |

- **Every one of 10,391 replayed decisions agrees with the Bayes-optimal agent** (12.81 had 10,582 of 10,652, with 70 tie-convention misses). The UAV level grid is now Bayes-optimal in action at every replayed decision, at both levels, for both agents.

### 12.90 S3(b): root stopping (near-tie root escalation) matches flat root 64,000 at UAV L2 H5 (36/40 each, against 17/40 at 1,000) but saves no CPU, because the modeled target dominates the cost. Not adopted as the recommended setting; the option stays off by default (October 22, 2026)

**Why:** HANDOFF S3 item 2; 12.67. With a uniform level prior at H5, I-POMCP needs root 64,000 to resolve near-ties, and the paper concedes it. The design fix grows the root only where the decision is unresolved.

**Change (code, kept; default off):** `PlannerConfig.root_stopping`.
- The validated modeled-stopping rule (12.11, 12.15) is applied to the agent's own decision, with `simulations` as the cap. The root tree is seeded as the flat search of the cap and grows by doubling; it stops on a separated leading gap or on exact scores.
- Seeds ignore both stopping rules, so every earlier result is reproduced. Tests: the rule at its cap equals the flat search, bounds included; it stops early on clear or exact roots; there is a seed-digest guard. 411 tests pass.

**Gate** (`results/s3b_root_stopping_20261022`, `ENDPOINTS.md` fixed before the run):
- The 8 uniform-prior H5 histories of 12.67 × seeds 0–4 (n = 40 per arm), with stored exact references.
- Target I-POMCP 4,000, N = 64.

| Arm | Optimal [n = 40] | Max loss | CPU per decision, median / mean | Root simulations used |
|---|---|---|---|---|
| Flat root 1,000 | 17 | 0.669 | 42.9 / 64.6 s | 1,000 |
| Flat root 64,000 | **36** | 0.066 | 73.3 / 95.5 s | 64,000 |
| Root stopping (pilot 1,000, 4 trees, α = 0.01, cap 64,000) | **36** | 0.066 | 77.8 / 97.7 s | median 64,000; stopped at 32,000 on 9 of 40 |

**Endpoints:**
- **Endpoint 1 passes:** 36 against 36.
- **Endpoint 3 fails narrowly:** the mean CPU is 1.02× flat 64,000's, so the rule is not adopted as the recommended configuration (pre-fixed reading).

**Findings:**
- **The near-tie misses need the large root, and the rule finds that itself.** On the six histories with gap 0.669, root stopping is 30 of 30 optimal (flat 1,000: 9 of 30). Three of them stop at 32,000 in 3 of 5 seeds; the rest run to the cap.
- **No CPU saving on this problem,** as 12.67 predicted: the modeled target (4,000 per modeled query) dominates the decision's cost, so root simulations are cheap. At the prior, E and N are exactly tied (gap 0), and non-exact scores cannot separate an exact tie, so the rule always runs to the cap there.
- **A residual near-tie the paper did not report:** history [S, 0] (exact gap 0.132) is missed in 4 of 5 seeds at root 64,000 by both arms (loss 0.066, above the ε = 0.05 certification standard). The paper's "root 64,000 resolves the near-ties" holds for the 0.669-gap histories of 12.67's diagnosis, not for every H5 history.
  - **S7:** state "root 64,000 resolves 30 of 30 decisions at the 0.669 near-ties; one further history with gap 0.13 remains missed (loss 0.066)".

### 12.87 S3(a): exact ties within 10⁻⁹ for exactly scored solves. The UAV level-2 replay goes from 70 disagreements with the Bayes-optimal agent to 0 of 1,837, Tiger and Wumpus gates are unchanged apart from 14 exact-tie corrections, and the rule is adopted. It changes the UAV's episodes against a level-2 target (+50.8) (October 22, 2026)

**Why:** HANDOFF 0.0000000000, S3 item 1; 12.81. Every UAV disagreement with the exact agent was the planner's modeled target playing a strict subset of exactly tied maximizers. The exact reference ties values within a relative 10⁻⁹; I-POMCP used strict float equality.

**Change** (`planning/policy.py`, `planning/i_pomcp.py`, `reference/exact_agent.py`):
- `greedy_policy(values, tolerance)`.
- I-POMCP passes `TIE_TOLERANCE` (10⁻⁹, relative, now shared with the exact agent) whenever the solve's chance scores are exact (`ChanceScores.exact`, Proposition 1's condition). This covers real decisions and modeled solves, including sequential stopping.
- Non-exact scores keep strict equality.
- THEORY.md, "Tie convention", is revised. There are 407 tests, and ruff is clean.

**Gates** (`results/s3a_ties_20261022`; `ENDPOINTS.md` fixed before any run, with an interim note):

| Gate | Design | Result |
|---|---|---|
| 1 (UAV, must pass) | 12.49's cells with a level-2 UAV (i2_j0–j2, 100 trials each) rerun under the rule; the UAV replayed through the exact agent as in 12.81 | **1,837 / 1,837 decisions agree** [0.998, 1]; 12.81 had 1,920 / 1,990 (70 misses). Level-posterior TV: mean 0.0000, max 0.014 (12.81: 0.007–0.016, max 0.27) |
| 2a (Tiger) | 12.23's five coordination gates (1,244 decisions), run on HEAD's source and on the new source | Scored values identical in **1,244 of 1,244**. Policies identical except 14, all exact ties (L and OR both −1.95 exactly, differing by 2 × 10⁻¹⁵), where the old rule played OR alone and the new rule plays both, as the exact I-POMDP does. Optimal: 1,244 / 1,244 under both rules |
| 2b (Wumpus) | 12.58's episode-agent gate (L2 H2, 60), old against new source, with the stored exact references | Values and losses identical in **60 / 60**, also equal to the stored 12.58 rows; 60 / 60 optimal |

**Gate 2's wording:** "every policy identical" fails at the 14 Tiger rows above. The pre-written criterion did not exempt exact ties, which are exactly what the change targets. In every one of them the old policy was a strict subset of the exact tie set and the new policy is the set. This is reported, not reclassified quietly.

**Effect on episodes** (same seeds as 12.49; `compare_episodes.py`):

| Cell | Trials changed | First changed step | UAV reward, new against 12.49 [95%] |
|---|---|---|---|
| i2_j0 | 21 / 100 (exactly 12.81's 21 miss trials) | 6 | 12.2 against 10.6: +1.6 [−2.2, 5.4] |
| i2_j1 | 37 / 100 | 2 | 30.0 against 30.0: 0 |
| i2_j2 | 81 / 100 | 2 | 5.2 against −45.6: **+50.8 [38.4, 63.2]** |

- **Mechanism in i2_j2:** the real level-2 target meets exact ties early (for example at step 2, E and N both −0.2166, differing at 10⁻¹⁶). Under the old rule it played whichever was larger in floating point. That deterministic choice happened to evade a level-2 UAV well over 20 steps, beyond its H3 lookahead. Under the exact convention it draws among the tie set and is caught more often.
- **Consequence:** 12.49's UAV level matrix, in particular "a level-2 target gains +26 to +35" and "the target gains more from depth than the UAV", partly rests on floating-point tie-breaking.
  - Every UAV level result must be rerun under the adopted rule before the paper uses it: the 9-cell matrix (about 15 CPU-h, 12.49), the replay, and the screens.
  - The Tiger and Wumpus results are unaffected wherever their gates are representative: no tie at a modeled exact solve changed a decision there.

**Cost:** about 9 CPU-h for gate 1, 1 CPU-h for gate 2a and 4 CPU-h for gate 2b, under the 25 CPU-h kill line.

### 12.89 S2(c): IPF + RTS fidelity against Doshi & Gmytrasiewicz (2009), Fig. 17(a). Our implementation reaches the published plateau at every K. The published losses at small K come from executing an offline tree whose sampled branches miss the real observation, a protocol our planner does not use (October 22, 2026)

**Why:** PLAN.md S2 and CLAIMS_AUDIT A9. IPF + RTS is our reimplementation. Without a reproduction on the authors' setting, any gap to I-POMCP can be called an implementation artifact.

**Design** (`results/s2_ipf_fidelity_20261022`: `ENDPOINTS.md` fixed after a 3-tree timing pilot, with amendment 1 written before the offline arm; `job.py`, `job_offline.py`, `analyze.py` → `analysis.txt`):
- **Target:** Fig. 17(a), the multiagent tiger at H3, IPF + RTS with Np ∈ {50, 100} particles and K sampled observations per node. Values were read off a 300-dpi render (±0.3).
- **The paper's setting, rebuilt:**
  - The original game (−1 / −100 / +10; growls 0.85, creaks 0.9).
  - i is their level 1 (our level 2), with the Fig. 3(a) prior: a uniform tiger and j's level-0 belief uniform. It is discretized to 100 points; each run draws the tiger and j's belief from the prior.
  - Planning γ = 0.9; i's undiscounted 3-step reward; 10 policy trees × 100 runs per point, as the paper.
  - Every modeled level uses RTS at the same K, and the real j uses the same solver at its own belief.
- **Two protocols for i:**
  - **Online:** RTS of depth H − t at every reached history, on the IPF belief. This is how our planner uses RTS everywhere in the paper.
  - **Offline (amendment 1):** the paper's APPROXPOLICY. One sampled tree is built at the root and executed. A real observation the tree did not sample has no subtree, and the paper does not say what i does then, so two fallbacks bracket it: listen, and uniform random.
- **Cost:** about 8.4 CPU-h online and about 4 CPU-h offline, under the 40 CPU-h kill line.

| Np | K | Paper | Online (10 trees; SE) | Offline, listen fallback | Offline, random fallback | Runs leaving the tree |
|---|---|---|---|---|---|---|
| 50 | 2 | −20.5 | −0.27 (0.64) | −1.77 | −39.67 | 74% |
| 50 | 3 | −18.7 | +0.30 (0.58) | −1.95 | −37.89 | 72% |
| 50 | 4 | −9.3 | −0.15 (0.64) | −1.09 | −29.05 | 56% |
| 50 | 6 | −3.5 | +0.36 (0.65) | −0.95 | −18.18 | 39% |
| 50 | 8 | −1.4 | −0.39 (0.81) | −0.83 | −15.03 | 35% |
| 50 | 10 | −0.2 | −0.52 (0.86) | −0.10 | −8.06 | 23% |
| 100 | 2 | −18.4 | −0.27 (0.51) | −1.59 | −36.56 | 72% |
| 100 | 3 | −14.6 | +0.75 (0.65) | −1.61 | −34.16 | 65% |
| 100 | 4 | −4.2 | −0.12 (0.71) | −1.57 | −30.70 | 59% |
| 100 | 6 | −2.9 | +0.38 (0.54) | −0.03 | −16.14 | 39% |
| 100 | 8 | −0.8 | +0.07 (0.58) | −0.08 | −12.97 | 32% |
| 100 | 10 | +0.2 | −0.21 (0.60) | −0.33 | −11.38 | 24% |

**Endpoints:**
- **Endpoint 1 (amended):** the paper's value lies inside the offline bracket at **12 of 12** points. The bracket is wide at small K, so this endpoint alone is weak.
- **Endpoint 2 (shape):**
  - The offline random-fallback curve rises with K at both Np, and K = 2 is below −10.
  - Np = 100 ≥ Np = 50 holds at K = 2 and 3, not at K = 4 (−30.7 against −29.1).
  - The online arm fails the shape endpoint by design: it is flat.

**Findings:**
- **Our IPF + RTS reaches the paper's plateau.** Online, every point is within about 1 of 0. The paper reaches −0.2 to +0.9 only at K ≥ 10, and our K = 10 points (−0.52 and −0.21) agree with its K = 10 points (−0.2 and +0.2).
- **The paper's small-K losses are an execution effect.** An offline tree from K = 2 samples misses the real observation in about 3 of 4 runs. The share falls with K (74% → 23%), and the paper's values fall between the two fallbacks at every point, tracking the off-tree share.
  - Replanning at the reached history (online) removes the effect entirely.
- **What this means for the paper's comparisons:** our IPF + RTS baseline is the published method used online, which is at least as strong as the published protocol. Its decision misses in Table 1 (where every decision is a root decision, with no execution) are not an artifact of a weak reimplementation.
- **Not reproduced:** the paper's exact fallback rule (unpublished), and its Fig. 15 exact lines. This run reproduces Fig. 17(a) only. The 2.7 of Fig. 15 is the single-agent ceiling of 12.48.

### 12.88 S5 input: the exact level match-up matrix from the exact pair curves. Zero-variance one-level gains inside the exact window; level 4 is exactly level 3 through step 3; over-thinking against a random opponent turns from gain to loss after the lookahead (October 22, 2026)

**Why:** M2 needs the level match-ups with zero variance where possible (PLAN.md S5; CLAIMS_AUDIT A6–A7). 12.72's exact pair curves already hold every cell, but only the diagonal and step-3 rows were ever reported.

**Method** (`results/n6_exact_curves_20261019/gain_table.py` → `gain_table.txt`; 0 CPU):
- The value of an agent at level a against level b at step s is i's side of cell i{a}_j{b} when a ≥ b. Otherwise it is j's side of i{b}_j{a}; the games are symmetric.
- The gain from k to k + 1 against level m is value(k + 1, m) − value(k, m): exact expected cumulative reward of the exact (Bayes-optimal) agents, H2, γ = 0.95, uniform level priors.
- Windows: level 4, 3 steps; level 3, 4 steps; level 2, 4–5 steps.

**Coordination game, exact gain of one more level** (rows k → k + 1; columns: the opponent's level m):

| By step | Step | m = 0 | m = 1 | m = 2 | m = 3 | m = 4 |
|---|---|---|---|---|---|---|
| 3 | 1 → 2 | +0.461 | +1.783 | +1.910 | +1.924 | +1.924 |
| 3 | 2 → 3 | +0.921 | +0.198 | +0.213 | +0.214 | +0.214 |
| 3 | 3 → 4 | 0 | 0 | 0 | 0 | 0 |
| 4 | 1 → 2 | **−0.032** | +1.216 | +1.843 | +1.912 | — |
| 4 | 2 → 3 | **−0.063** | +0.135 | +0.204 | +0.212 | — |
| 5 | 1 → 2 | **−0.551** | +0.430 | — | — | — |

**Original game:**

| By step | Step | m = 0 | m = 1 | m = 2 | m = 3 | m = 4 |
|---|---|---|---|---|---|---|
| 3 | 1 → 2 | 0 | 0 | 0 | 0 | 0 |
| 3 | 2 → 3 | −0.517 | +0.515 | +0.515 | +0.515 | +0.515 |
| 3 | 3 → 4 | 0 | 0 | 0 | 0 | 0 |
| 4 | 2 → 3 | −0.709 | +0.350 | +0.350 | +0.385 | — |

**Findings (zero variance, inside the window):**
- **Level 4 is exactly level 3 through step 3,** against every opponent in both games. This is the exact form of "level 4 adds nothing". It does not reconcile 12.22's positive 3 → 4 screen at point chains (S3(d)): those chains are not reachable from the common prior in 3 steps.
- **Coordination:**
  - 1 → 2 is worth about +1.8 to +1.9 by step 3 against every intentional opponent.
  - 2 → 3 is worth +0.20 to +0.21, against level 1 as well. A step can pay against an opponent below the agent's current level, so H-CH's "only if m ≥ k" is too strict: an extra level can refine the posterior over the opponent's lower levels.
  - 2 → 3 is about 1/9 of 1 → 2. That is why 700 episodes cannot resolve it (12.73), and the paper should state it exactly.
- **Over-thinking against a random opponent is a horizon effect:**
  - Coordination 1 → 2 against m = 0 is +0.46 by step 3, −0.03 by step 4 and −0.55 by step 5. Monte Carlo gives −0.89 by step 10 (12.73).
  - Proposition 3 covers one decision at lookahead H. Within the first H steps the deeper agent can even gain, because a receding H2 policy is not optimal for the episode, but its wrong model costs more as the episode runs on.
  - The paper's Section 5 caveat ("receding-horizon episodes carry no such guarantee") now has its exact illustration.
- **Original game:**
  - 1 → 2 is exactly 0 by step 3 (levels 1 and 2 act identically, 12.82's fixed point).
  - 2 → 3 is +0.515 against any intentional opponent and −0.517 against a random one, a clean exact case of H-CH (the paper reports no level results on this game, 12.83).

**Use in the paper (S5, S7):** Figure 2 (M2) becomes this matrix at step 3, with step-4 values where available, and the planner's paired Monte Carlo gains overlaid as sign agreement only. Extending the windows is S5 work; the 3 GB caps of 12.72 bound them.

### 12.86 S2: I-NTMCP fidelity. The authors' runner reproduces their published Runner-Chaser 4 × 4 numbers exactly, and our adapter matches them (65.6 against 65.96, n = 1,000). The adapter costs 1.45× the native CPU per search (October 22, 2026)

**Why:** PLAN.md S2 and CLAIMS_AUDIT item A9. I-NTMCP runs in the paper through our adapter (`baselines.intmcp`). Without a reproduction on the authors' own setting, a reviewer can attribute any gap to the adapter.

**Design** (`results/s2_intmcp_fidelity_20261022`; `ENDPOINTS.md` fixed before both runs):
- **Target:** the authors' published I-POMDP Lite comparison on Runner-Chaser 4 × 4 (`iros_paper/rc/i-pomdp-lite_comparison/4x4/compiled_results.csv`, Schwartz et al., IROS 2022). The runner is I-NTMCP at nesting level 1 with 1,024 simulations, uct_c 175, γ = 0.95, ε = 0.1 and a 20-step limit. Exp 0 plays a uniformly random chaser, exp 1 an I-NTMCP level-0 chaser. n = 1,000 each.
- **(a) The authors' runner, unchanged:** `run_scripts/run_exp.py` on their YAML, seed 0. It ran in a separate venv (Python 3.12) with their requirements; the project venv is untouched.
- **(b) Our adapter:** the new kernel domain `domains/runner_chaser.py`, the cap4 layout. `tests/test_runner_chaser.py` shows it equal to the authors' `RCModel` on every state and joint action (next state, both observations, both rewards, termination).
  - `INTMCPAgent(level=2, simulations=1024, uct_c=175, horizon=46, variant="published")`. Horizon 46 gives the authors' depth limit of 45 for ε = 0.1.
  - The chaser is uniformly random; 1,000 episodes on our own seeds.
  - The authors' initial observation is not represented. Both start cells are fixed and far apart, so it carries no information.
- **Cost:** 0.56 CPU-h for (a) and 0.54 CPU-h for (b).

| Run | Runner's return, mean (SD) | Reaches safety | Steps | Search CPU (s) |
|---|---|---|---|---|
| Published, exp 0 (random chaser) | 65.964 (71.215) | 849 / 1,000 | 4.836 | 0.61 |
| (a) Authors' runner, exp 0 | **65.964 (71.215)** | 849 / 1,000 | 4.836 | 0.27 |
| (b) Our adapter, exp 0's match-up | **65.566 (71.643)** | 847 / 1,000 | 4.834 | 0.38 (median) |
| Published, exp 1 (level-0 chaser) | 95.998 (0.045) | 1,000 / 1,000 | 5.002 | 0.62 |
| (a) Authors' runner, exp 1 | **95.998 (0.045)** | 1,000 / 1,000 | 5.002 | 0.26 |

**Findings:**
- **(a) passes, bit for bit:** with seed 0, the authors' code reproduces every published statistic exactly. Their published numbers are therefore their code's output, and our environment runs it unchanged.
- **(b) passes:** our adapter differs from the published mean by −0.40, inside the pre-fixed two-sample 95% half-width of ±6.26. The share reaching safety and the episode length also agree.
  - The adapter does not change I-NTMCP's play on the authors' own domain.
  - So the paper's I-NTMCP gaps (Wumpus 40/60, UAV episodes; 12.65, 12.67, 12.85) cannot be attributed to the adapter.
- **Caveat for the matched-CPU comparisons:** through the adapter, a search costs 0.38 CPU-s against 0.27 s natively on the same machine (1.45×). The cost comes from wrapping states, actions and observations, and from kernel sampling.
  - The paper's matched-CPU rule therefore charges I-NTMCP up to about 1.45× its native cost.
  - Every UAV lead of 12.85 holds at CPU ratios of 0.22–0.79, inside that factor except own-250 against M = 4,000 at H4 (0.79 × 1.45 ≈ 1.15).
  - S7 should state the overhead, and S4 should report CPU both as measured and divided by 1.45 for I-NTMCP.

### 12.85 S0 claims audit: against the stronger I-NTMCP variant, I-POMCP's UAV lead at matched CPU is +14.0 (n.s. after Holm) at M = 4,000 and +2.5 to +2.7 at M = 16,000; the paper's +42.3 used the weaker variant (October 22, 2026)

**Why:** the S0 claims audit (`docs/paper/CLAIMS_AUDIT.md`, item A1) found that the paper's UAV matched-CPU comparison (12.67, endpoint 2) uses the tuned I-NTMCP variant. The paper calls it the stronger one, but on UAV the released variant earns more at every M ≤ 16,000. This section redoes endpoint 2 against both variants, from stored episodes (0 new episodes).

**Design** (`results/g3_uav_episodes_20261018/analyze_released.py` → `analysis_released.txt`):
- The matching rule is 12.67's: the most expensive point-mass I-POMCP arm with no more median CPU, or the cheapest arm if none qualifies.
- Paired by trial (n = 30). The stronger variant is chosen per (H, M) by mean reward, and Holm runs over those 8 tests.
- **Cost:** under 1 CPU-min.

| H | M | Stronger variant (reward) | I-POMCP arm (reward; CPU ratio) | Difference [95%] | Holm p | Tuned variant, for comparison |
|---|---|---|---|---|---|---|
| 4 | 1,000 | released (17.5) | own-50 (33.5; 1.5, *more* CPU) | +16.0 [8.0, 27.0] | 0.019 | +36.2 |
| 4 | 4,000 | released (20.8) | own-250 (34.8; 0.79) | **+14.0 [4.3, 27.5]** | **0.12** | +42.3 |
| 4 | 16,000 | released (32.2) | own-250 (34.8; 0.22) | **+2.7 [1.7, 3.7]** | **10⁻⁴** | +4.7 |
| 4 | 64,000 | both 35.0 | own-1,000 (35.0; 0.29) | 0 | 1 | 0 |
| 5 | 1,000 | released (9.0) | own-50 (31.7; 6.2, *more* CPU) | +22.7 [7.7, 39.8] | 0.061 | +42.7 |
| 5 | 4,000 | released (28.2) | own-50 (31.7; 1.3, *more* CPU) | +3.5 [0.7, 6.8] | 0.12 | +20.7 |
| 5 | 16,000 | released (32.0) | own-250 (34.5; 0.58) | **+2.5 [1.2, 4.0]** | **0.010** | +3.5 |
| 5 | 64,000 | both 35.0 | own-1,000 (35.0; 0.51) | 0 | 1 | 0 |

**Findings:**
- **Against the stronger variant, I-POMCP still leads wherever the planners differ, with less CPU,** but by less than the paper says:
  - +2.7 (H4) and +2.5 (H5) at M = 16,000, both significant;
  - +14.0 at M = 4,000 (H4), not significant after Holm at n = 30.
- **The paper's "+42.3 [23.7, 63.2]" is against the weaker variant,** and the sentence "the variant as released gives no better results" is false on UAV. On Wumpus the tuned variant is as strong or stronger (12.65), so Table 1's I-NTMCP cell stands.
- **Paper action (S7):**
  - Report the stronger variant per budget.
  - Drop "no better results" from Section 6.3.
  - Raise n for the matched comparison in S4. The own-budget I-POMCP arms and I-NTMCP at M ≤ 16,000 cost under 15 s per decision, so 100 more episodes per arm need no exact references and cost a few CPU-h.
- **Other audit findings** (no new data; recorded in the audit and applied in S7):
  - Table 1's UAV L3 H2 "640" double-counts two modeled arms of 320 problems, so the total is 2,787 decisions, not 3,107.
  - Section 7.6 mixes 12.60's trials 0–99 death rates with the fresh-trial gain.
  - The level-posterior TV range is 0.004–0.011, not 0.005–0.011.
  - The stored UAV level matrix contradicts the plain cognitive-hierarchy hypothesis: 0 → 1 loses against an opponent two levels up (target −3.5, Holm p = 0.032; UAV −16.4, p = 0.055). This is the predictability effect, and S5 tests the refined H-CH′.

### 12.84 Coordination-only splits for the paper: escalation removes 32 of 34 held-out miss histories at 0.41× the larger bound's CPU; the exact pair lies in the I-POMCP interval at 97 of 100 agent-steps (October 21, 2026)

**Why:** the paper reports no level or depth results on the original tiger game (12.83). 12.74, 12.75, 12.80 and 12.73 pooled both games, so their coordination-only counts are recomputed here from stored data (0 new episodes).

**Data and scripts:**
- `results/n12_escalation_heldout_20261020/analyze_coordination.py` → `analysis_coordination.txt` (12.80's analysis restricted to the coordination cells).
- 12.73's `compare_exact.csv`, filtered to the coordination game (|z| ≤ 1.96, the criterion of `compare_exact.py`).
- 12.74's `disagreements.csv` and `diagnosis.jsonl`, filtered to the coordination game.

| Measure (coordination game only) | Value | Pooled source |
|---|---|---|
| Disagreements with the exact agent, levels 2–4 (12.74) | 67 (21 + 39 + 7); median exact gap 0.090, max 5.99 | 200 |
| Diagnosed (12.74) | 61 of 67; all 61 reproduce as configured and pick an exact maximizer with exact beliefs | 148 of 200 |
| Distinct histories with a miss at N = 64 vs the larger N (12.75) | 21 vs 0 over 1,200 cell-trial windows; two-sided sign test p = 9.5 × 10⁻⁷ | 38 vs 0 over 2,400 |
| Held-out escalation (12.80): miss histories, base → escalation | 34 → 2 (L2 9 → 2, L3 22 → 0, L4 3 → 0); no new misses | 62 → 2 |
| Decisions escalated | 337 of 10,400 = 3.2% | 3.3% |
| CPU, escalation / larger N (n = 120 subsample trials) | 0.41 (L2 0.67, L3 0.43, L4 0.39) | 0.38 |
| Exact pair inside the I-POMCP RB 95% interval (12.73) | 97 of 100 intentional agent-steps; level-0 control 42 of 42 | 182 of 200; 76 of 84 |

- The coordination-only numbers tell the same story as the pooled ones. The paper (Section 7.2–7.3) reports these.
- **Cost:** under 1 CPU-min.

### 12.82 P1: whom does each agent believe it faces? Level identification over the level grid, with exact backups (reviewer question, October 21, 2026)

**Why:** the user asked whether the experiments explain why a level-2 j converges on "i is level 1" as i's level rises from 1 to 3 (first answered in 12.47 on the October 7 page's trials). This section redoes it on the definitive grid (12.73: n up to 15,000 per cell, levels 0–4, both games) for the paper section of the same name. There are no new episodes.

**Design** (`results/p1_level_identification_20261021`: `analyze.py` → `analysis.txt`, CSVs, `level_identification.pdf`; `divergence_gaps.py` → `divergence_gaps.txt`; cost under 0.1 CPU-h):
- the posteriors over the other's level from N3's rows;
- the exact replay's posterior TV from n7 (12.74);
- the exact fixed point from 12.47 (`screen_table.csv`);
- behavior paired by trial across cells.

**Findings:**
- **A level-2 observer, after 10 steps:** P(other is level 1) is 0.91–0.95 for any planning other (levels 1–4, both games; n = 100–15,000), and 0.078 [0.071, 0.086] against a random other (published, n = 2,500); coordination 0.077 (n = 15,000).
- **Exact fixed point (published, H2):** level k+1's optimal actions differ from level k's at 29% of chains for 1→2, and at 0% for 2→3, 3→4, 4→5 and 5→6.
- **Episodes (published, against a random opponent):** levels 2 and 3 take identical 10-step action sequences in 90% of 700 trials. Where they diverge, the level-3 agent's top-two gap is ≤ 1 in 88% of cases (near-ties).
- **Episodes (coordination):** levels 1 and 2 diverge in 24% (against level 0) and 62% (against level 1) of trials. At the first divergence the level-2 agent's median gap is 7.4 and 8.3, so the behavior genuinely differs.
- **A level-3 observer at step 10:**
  - 0.90 on level 0 against a random other;
  - its mode is the true level for others at levels 1 and 2 (published 0.49 vs 0.45 and 0.50 vs 0.45; coordination 0.52 vs 0.42 and 0.53 vs 0.43);
  - others at levels 3–4 are assigned to level 2 (0.49–0.59), with level 0 at most 0.06.
- **A level-4 observer (published):** 0.27–0.35 on each of levels 1–3 whatever the truth. Those levels act identically, so they cannot be identified.
- **Exactness:** the planner's level posterior against the exact agent's, over the exact window, has a mean TV of 0.004–0.012 per observer and cell (max 0.27), with TV < 0.01 in 67–83% of decisions.
- **In the paper:** results subsection "Whom Does Each Agent Believe It Faces?", with Figure `level_identification.pdf` and the fixed-point table.

### 12.83 P1b: exact posteriors along the planner's histories; the paper drops level and depth results on the original tiger game (October 21, 2026)

**Exact overlay** (`results/p1_level_identification_20261021/exact_posteriors.py`; `ENDPOINTS.md` section P1b with one amendment, both before any curve was read; 10.0 CPU-h; kill line 15):
- **What:** the exact level-2 observer's belief update along the planner's own histories, steps 0–8. Only belief updates are computed, no planning.
- **Trials:** levels 0–2 use trials 0–999; levels 3 and 4 use 700 and 100.
- **Amendment:** the measured cost was 4.4 CPU-s per trial, so the cap moved from 2,500 to 1,000 trials.

At step 8, the planner's P(other is level 1) minus the exact P(level 1), paired on the same trials (`analysis.txt` §6):

| Game | Paired differences by cell (other at levels 0–4) | Largest magnitude | Mean absolute difference |
|---|---|---|---|
| Coordination | −0.0025, −0.0028, −0.0015, +0.0003, −0.0006 | 0.003 | 0.012–0.018 |
| Original | −0.0019, −0.0007, −0.0007, −0.0005, −0.0013 | 0.002 | 0.017–0.020 |

**Paper scope decision (user and reviewer, October 21):** no level or depth results on the original tiger game.
- **Why:** Table 2 (the exact fixed point from level 2 up, 12.47) and the related text suggest that I-POMDP modeling is wasted compute in such setups.
  - That is a claim about the framework in general.
  - It needs a proof rather than experiments on a few belief points, one horizon and levels up to 6.
- **Removed from the paper:** Table 2, the fixed-point paragraph, the original-game panels of Figure 2, the non-identifiability remark, and "the original game serves as a control".
- **Kept:** the original game, defined in the setup, for single-decision exactness and the baseline comparison.
- **Coordination game only:** the level grid and the level-identification subsection.
- **Code and data are kept:** 12.47, 12.82 and this section's runs remain on disk for later work.

### 12.81 V2: the UAV level grid against the Bayes-optimal agent: 10,582 of 10,652 decisions agree; all 70 others are one decision explained by the tie convention, not by error (October 21, 2026)

**Why:** HANDOFF 0.000000 item 3. 12.74's replay against the Bayes-optimal agent covered only the Tiger games. UAV reconnaissance already had a level grid (12.49), so no new episodes were needed.

**Design** (`results/v2_uav_replay_20261021`; `ENDPOINTS.md` fixed before the run; `job.py`, `analyze.py` → `analysis.txt`):
- **Replay:** every intentional agent of 12.49's grid goes through the exact agent of the same level (horizon 3, γ = 0.95, uniform nesting): the UAV at levels 1–2 against targets 0–2, and the target at levels 1–2 against UAVs 0–2. That is 12 series × 100 trials.
- **Window:** level 1, the whole episode (up to 20 decisions); level 2, up to 10 decisions. No trial exceeded the 4 GB cap.
- **Prerequisite:** this needed the `exact_replay` fix for numeric observations (commit 0f5ce71). Before it, UAV replays silently stopped after the first step.
- **Cost:** 0.8 CPU-h, far below the 13 priced, because histories repeat across trials and the exact agent's caches persist within a chunk. The kill line was 30.

| Agent | Level | Decisions agreeing [95% CP] | Level-posterior TV, mean / max |
|---|---|---|---|
| UAV | 1 | 3,079 / 3,079 [0.9988, 1] | 0 / 0 |
| Target | 1 | 2,850 / 2,850 [0.9987, 1] | 0 / 0 |
| Target | 2 | 2,733 / 2,733 [0.9987, 1] | 0 / 0 |
| UAV | 2 | 1,920 / 1,990 [0.956, 0.973] | 0.007–0.016 / 0.27 |

**The 70 disagreements are one decision:**
- They all occur in cells i2_j0 (21) and i2_j2 (49), every one at step 6 after the same actions (listen, N, N, E, E), with the same exact loss of 1.604.
- They span 57 observation sequences. The UAV's observations while moving are uninformative, so the exact belief does not depend on them.
- The level-2 UAV flies to the corner where a level-1 target would hide. When the target is not there, the UAV listens, while the exact agent moves S.

**Diagnosis** (not part of the fixed endpoints; `diagnose.py`, `diagnose_br.py`, `diagnose_levels.py`, three trials):
- **Not compression, not search:** rebuilt with the trial's seeds, the agent reproduces "listen" as configured, with exact beliefs, and with exact beliefs at root and modeled budgets of 4,000. Its scored values do not move.
- **Optimal for its own model:** the exact best response to the planner's own (search-solved) target models reproduces its values exactly (listen 6.692, S 6.639). The UAV decides optimally for the target it models.
- **Where the difference comes from:**
  - The planner's and the exact agent's posteriors over the target's level agree to 10⁻¹⁵ through step 4.
  - After the step-5 update the planner puts 0 on a level-1 target, while the exact agent keeps 0.064.
  - In that update, 13 of the 159 modeled level-1 target decisions differ from the exact ones. **All 13 are the planner's target playing a strict subset of the exact target's tied maximizers** (for example {E, N} against {E, N, listen}). None is outside the maximizers.
- **Reading:** this is the tie convention of THEORY.md ("Three meanings of exact"; 12.50). The exact I-POMDP's modeled agent plays its ties uniformly; I-POMCP's plays its search's own tie choice. Both are equally rational opponents, but tied actions give the modeling agent different observations, so its best response can differ. No budget closes the gap.
  - In these episodes the real target is itself an I-POMCP agent, so the planner's convention is the one that predicts the real target.
  - Apart from this one convention-driven decision, every replayed UAV and target decision at levels 1–2 is Bayes-optimal.

### 12.80 A3b: on held-out trials, near-tie escalation at gap ≤ 3 removes 60 of 62 miss histories at 0.38× the larger bound's CPU; both pre-fixed bars pass, so escalation becomes the recommended configuration (October 21, 2026)

**Why:** 12.79 missed its miss bar narrowly at gap ≤ 1, and every miss it left had a base gap of 1.01–2.19. So the threshold was fixed at 3 (12.77: 97.5% of misses caught, 3.6% of decisions flagged) and the test was moved to trials that no analysis had used.

**Design** (`results/n12_escalation_heldout_20261020`; `ENDPOINTS.md` with one amendment, both before any escalation result was read; `job.py`, `analyze.py` → `analysis.txt`):
- **Why these trials:** 12.77's gap analysis used every decision of n7, so its trials are not held out.
- **Held-out trials** (N3's cells and configuration):
  - level 2: trials 200–299 (N3's own rows, replayed here for the first time);
  - level 3: trials 400–599 (same);
  - level 4: trials 100–199 (new windows; trials 0–1 reproduce N3's rows exactly).
- **Arms:**
  - base: N = 64;
  - escalation: the shadow at N = 1,024 (level 2) or 256 (levels 3–4) decides when the base top-two gap is ≤ 3;
  - CPU subsample: the first 10 held-out trials per cell, also run plain at N = 64 and at the larger N, in the same run.
- **Replay reuse:** a replay reads only agent i's rows, so an escalation trial whose rows for i match the base's reuses the base replay. That covered 3,088 of 3,200 trials. Each chunk recomputed its first reused trial as a check: 580 of 580 identical.
- **Cost:** 57.9 CPU-h logged, plus at most about 2 CPU-h lost in a restart (the amendment: a watchdog kill stopped GNU xargs; the run was restarted at 12 workers with a signal-safe wrapper). The kill line was 95.

**Miss histories of agent i** (distinct histories, exact McNemar against base):

| Game | Level | Distinct histories | Misses at N = 64 | With escalation | p |
|---|---|---|---|---|---|
| Coordination | 2 | 225 | 9 | 2 | 0.016 |
| Coordination | 3 | 372 | 22 | **0** | 5 × 10⁻⁷ |
| Coordination | 4 | 182 | 3 | **0** | 0.25 |
| Published | 2 | 209 | 9 | **0** | 0.0039 |
| Published | 3 | 370 | 17 | **0** | 1.5 × 10⁻⁵ |
| Published | 4 | 182 | 2 | **0** | 0.5 |
| All | | 1,540 | 62 | **2 (3%)** | |

Escalation created no new miss (0 discordant pairs against it).

**CPU over the window** (both agents; subsample, 10 trials per cell; ratio of means [95% bootstrap]):

| Level | Trials | Escalation / larger N | Escalation / N = 64 |
|---|---|---|---|
| 2 | 60 | 0.68 [0.64, 0.72] | 4.6 [4.4, 4.8] |
| 3 | 80 | 0.43 [0.42, 0.45] | 2.1 [2.0, 2.2] |
| 4 | 100 | **0.35** [0.31, 0.39] | 1.5 [1.3, 1.7] |
| All | 240 | **0.38** | 1.7 |

- 3.3% of decisions were escalated (694 of 20,800), close to 12.77's 3.6%.
- **By the reading fixed in advance** (at most 10% of base's misses left, at most 0.6 of the larger N's CPU), **both bars pass: 3% and 0.38.** Escalation at gap ≤ 3 becomes the recommended configuration.
- **The two remaining misses are both coordination level 2:**
  - **Trial 220, step 7** (the same history in cells j1 and j2): not escalated. Its base gap was 3.25, just over the threshold, and the exact loss is 2.59, so this is not a near-tie: compression made the base confidently wrong.
  - **Trial 263, step 8:** escalated (gap 1.77), but the shadow at N = 1,024 chose the same action (loss 0.32). So the larger bound itself misses one held-out decision, which 12.75's trials (0 of 2,400) never showed.
- **Cost where it lands:** escalation is cheapest relative to the larger bound where the larger bound is dearest (level 4: 0.35×). At level 2 the shadow's real-step updates dominate, so it costs 4.6× N = 64 there.

### 12.79 A3: near-tie escalation removes 34 of 38 miss histories at 0.34× the larger bound's CPU, missing the pre-fixed 10% bar narrowly (11% left); not adopted, and the paper reports the larger N (October 20, 2026)

**Why:** PLAN A3. A shadow agent at the larger bound decides only when the planner's own top-two gap is at most 1 (`planning.escalation`, 12.77–12.78). Does it match the larger bound's agreement at a fraction of its cost?

**Design** (`results/n11_escalation_20261020`; `ENDPOINTS.md` with its amendment, both before any result was read; the kill line was amended from 80 to 95 before it was reached; `job.py`, `job_plain.py`, `analyze.py` → `analysis.txt`):
- **Cells and trials:** N4(b)'s cells, trials 0–99, i's exact window.
- **Escalation:** every agent at level ≥ 2 keeps N = 64 and a shadow at N = 1,024 (level-2 cells) or 256 (levels 3–4).
- **Miss endpoint:** against N = 64 (12.74) and the larger N (12.75).
- **CPU endpoint:** a concurrent subsample, trials 0–19 rerun alongside at N = 64 and at the larger N, so that every arm ran under the same load.
- **Cost:** 82.5 CPU-h.

**Miss histories of agent i** (distinct histories, exact McNemar against N = 64):

| Game | Level | Misses at N = 64 | With escalation | p |
|---|---|---|---|---|
| Coordination | 2 | 5 | 3 | 0.5 |
| Coordination | 3 | 9 | **0** | 0.0039 |
| Coordination | 4 | 7 | 1 | 0.031 |
| Published | 2 | 9 | **0** | 0.0039 |
| Published | 3 | 7 | **0** | 0.016 |
| Published | 4 | 1 | **0** | 1 |
| All | | 38 | 4 (11%) | |

At the larger N there were 0 (12.75). Escalation created no new miss.

**CPU over the window** (both agents, concurrent subsample, trials 0–19 per cell; ratio of means [95% bootstrap]):

| Level | Trials | Escalation / larger N | Escalation / N = 64 |
|---|---|---|---|
| 2 | 120 | 0.77 [0.72, 0.83] | 4.9 [4.6, 5.3] |
| 3 | 160 | 0.45 [0.43, 0.48] | 2.3 [2.2, 2.4] |
| 4 | 200 | **0.30** [0.27, 0.32] | 1.75 [1.63, 1.89] |
| All | 480 | **0.34** | 2.0 |

- 1.4% of decisions were escalated (231 of 16,000).
- **By the reading fixed in advance** (at most 10% of N = 64's misses left, at most 0.6 of the larger N's CPU), **the cost criterion passes but the miss criterion fails narrowly: 11%.** Escalation is not adopted as the recommended configuration. The paper reports the larger N (12.75) and states escalation as a promising, partly validated method.
- **Where the rest lies:** none of the remaining misses was escalated. Their base gaps are 1.01–2.19 (five cell-trials, four distinct histories), so a threshold of 3 would have caught every one. This is consistent with 12.77's catch rates.
  - At level 2, a gap of at most 1 catches 85% of misses; at most 2 catches 90%; at most 3, 94%.
  - A larger threshold, chosen in advance and tested on held-out trials, is the natural next test. From 12.77's data, a threshold of 3 catches 97.5% while flagging 3.6% of decisions.
- **The shadow's real-step updates dominate escalation's cost at level 2.** There escalation costs 4.9× N = 64, against 1.75× at level 4.

### 12.78 A2a: a larger bound on the real-step belief alone fixes only 40% of the miss trials (0 of 24 at level 3), so escalation must re-decide with the larger bound throughout (October 20, 2026)

**Why:** PLAN A2a. The package bounds real-step beliefs (`particles`) and search beliefs (`search_particles`) separately. If the harmful compression were in the real-step belief only, escalation would need just a larger real bound.

**Design** (`results/n10_split_bound_20261020`; `ENDPOINTS.md` with the reading fixed before the run; `job.py`, `analyze.py` → `analysis.txt`):
- **Trials:** every trial 0–99, in N4(b)'s cells, with a miss at N = 64, plus one seeded agreeing control per miss in the same cell.
- **Bounds:** real-step beliefs at N = 1,024 (level 2) or 256 (levels 3–4), search beliefs at S = 64, over the exact window.
- **Replay:** agent i through the exact agent.
- **Cost:** 3.5 CPU-h.

| Level | Miss trials fixed | Controls broken |
|---|---|---|
| 2 | 15 / 23 | 1 / 23 |
| 3 | **0 / 24** | 0 / 24 |
| 4 | 7 / 8 | 0 / 8 |
| All | 22 / 55 (40%) | 1 / 55 |

- **By the reading fixed in advance (under 50% fixed), the search must escalate too.** At level 3 the harmful compression is in the nested beliefs the search builds; at levels 2 and 4 it is mostly in the real-step belief.
- **The split bound was not cheaper:** per trial it cost 55–178 CPU-s, against 48–152 for N = S at the larger bound (one trial per cell, so noisy).
- **Next (A2):** a shadow agent at the larger bound, with the same seeds and the same realized history, decides whenever the base planner's top-two gap is at most 1.

### 12.77 A1b: the planner's own top-two action-value gap flags the misses: AUC 0.996; gap ≤ 1 catches 93.5% of them while flagging 1.5% of decisions (October 20, 2026)

**Why:** 12.76 found that the compression error does not predict the misses, which are near-ties. The planner records its own action values at every decision, so a near-tie trigger can be tested from data on disk.

**Design** (`results/n9_compression_tv_20261020`, `ENDPOINTS.md` section A1b, with the reading fixed before the analysis; `gap_analysis.py` → `gap_analysis.txt`, `gaps.csv`):
- **Decisions:** all 38,000 decisions replayed in 12.74 at levels 2–4, both games (200 misses).
- **Gap:** the planner's largest minus its second-largest recorded action value (N3's `action_values`).
- **Cost:** no CPU.

| Level | Decisions (misses) | AUC [95%] | Median gap, miss / agree | Gap ≤ 0.5: misses caught / decisions flagged | Gap ≤ 1: misses caught / decisions flagged |
|---|---|---|---|---|---|
| Pooled | 38,000 (200) | **0.996** [0.994, 0.997] | 0.16 / 21.4 | 89.5% / 1.2% | **93.5% / 1.5%** |
| 2 | 17,600 (82) | 0.990 [0.985, 0.995] | 0.12 / 14.1 | 79.3% / 1.0% | 85.4% / 1.7% |
| 3 | 16,800 (110) | 0.999 [0.998, 0.999] | 0.16 / 22.8 | 100% / 1.4% | 100% / 1.5% |
| 4 | 3,600 (8) | 0.994 [0.990, 0.998] | 0.47 / 23.0 | 50% / 0.9% | 87.5% / 1.2% |

- **By the reading fixed in advance** (pooled AUC lower bound above 0.85; a threshold that catches at least 90% of misses while flagging at most 20%), **the trigger is worth building.**
- **Gap ≤ 1 qualifies:** 93.5% of misses caught, 1.5% of decisions flagged.
- **Next (PLAN A2):** re-decide flagged decisions with a larger belief bound, then A/B it against N = 64 and N = 256 over the exact windows (A3).

### 12.76 A1: the exact compression error does not predict which decisions miss (AUC 0.50), so no adaptive belief bound is built (October 20, 2026)

**Why:** PLAN, "adaptive belief bound", step A1. N = 64 compression causes every miss of 12.74, and a larger N removes them (12.75). A bound that grows only where a compression drops much mass would be cheaper than a uniformly larger N, but only if the dropped mass predicts the misses.

**Design** (`results/n9_compression_tv_20261020`; `ENDPOINTS.md` with the reading fixed before the run; `measure.py`, `analyze.py` → `analysis.txt`, `summary.csv`):
- **Items:** the 148 diagnosed misses of 12.74, each with one matched agreeing decision: same game, cell, agent and step, another trial agreeing at every step so far, drawn with seed 20261020.
- **Measure:** each agent is rebuilt with its trial's seeds at N = 64 and replays its history. Every compression records its exact TV, between the exact posterior and its compression. All 296 rebuilt decisions reproduce the recorded action.
- **Cost:** 0.6 CPU-h.

| Items | n | AUC of max compression TV, miss vs agree [95%] | Median max TV, miss / agree |
|---|---|---|---|
| Pooled | 296 | **0.50** [0.43, 0.56] | 0.218 / 0.218 |
| Level 2 | 60 | 0.39 [0.25, 0.54] | 0.080 / 0.143 |
| Level 3 | 220 | 0.51 [0.44, 0.59] | 0.226 / 0.222 |
| Level 4 | 16 | 0.25 [0.00, 0.53] | 0.267 / 0.313 |

The summed TV (pooled AUC 0.41) and the TV during the decision alone (0.50) separate no better.

- **By the reading fixed in advance (pooled lower bound above 0.75, no level below 0.6), the signal is not usable. A2–A4 are not run.**
- **Interpretation:** the agreeing decisions see as much compression error as the misses. A decision misses when it is a near-tie (12.74: median exact gap 0.065–0.09), so any perturbation can flip it; how much mass a compression drops does not decide it.
- **The paper reports 12.75 as it stands:** a larger N removes every miss inside the window, at its cost. An adaptive bound would need a different trigger, such as the planner's own action-value gap or its certified bounds; that is stated as future work.

### 12.75 N4(b): every disagreement with the Bayes-optimal agent disappears when the belief bound N is raised (0 of 2,400 trial windows at N = 256 or 1,024, against 38 distinct trial histories with a miss at N = 64) (October 20, 2026)

**Why:** PLAN N4(b). It tests 12.74's diagnosis, that N = 64 belief compression causes the misses, on fixed trials rather than on the misses alone.

**Design** (`results/n8_convergence_20261019`; `ENDPOINTS.md` fixed before the main run, kill line amended from 90 to 100 before it was reached; `job.py`, `analyze.py` → `analysis.txt`, `table.csv`):
- **Cells:** those with i at level 2–4 (i2_j0..2, i3_j0..3, i4_j0..4), both games, trial ids 0–99.
- **Rerun:** each cell for i's exact window (8, 4 and 3 steps at levels 2, 3 and 4), with every intentional agent's belief bound at N = 256. Level-2 cells also run at N = 1,024, because their 8-step window outgrows 256 atoms.
- **Replay:** agent i through the exact agent (`analysis.exact_replay`).
- **The N = 64 arm is N3 itself.** A window-length run reproduces N3's first rows exactly (checked).
- **Cost:** 79.1 CPU-h. One chunk whose replay a hand kill cut short was regenerated; its played columns and the replays it had written were identical to the first run (`superseded/`).

| Game | Level | Distinct trial histories | With a miss at N = 64 | With a miss at the largest N | Exact McNemar p (Holm over 6) |
|---|---|---|---|---|---|
| Coordination | 2 (N = 1,024) | 224 | 5 | **0** | 0.062 (0.12) |
| Coordination | 3 (N = 256) | 185 | 9 | **0** | 0.0039 (0.023) |
| Coordination | 4 (N = 256) | 180 | 7 | **0** | 0.016 (0.064) |
| Published | 2 (N = 1,024) | 209 | 9 | **0** | 0.0039 (0.023) |
| Published | 3 (N = 256) | 185 | 7 | **0** | 0.016 (0.064) |
| Published | 4 (N = 256) | 180 | 1 | **0** | 1 |

(Cells of a row often replay the same history, because j1–j3 behave alike, so each distinct history is counted once.)

- **No replayed decision disagrees with the exact maximizers at the largest N:** 0 of 2,400 cell-trial windows.
  - Pooled: 38 discordant histories against 0, two-sided sign test p = 7 × 10⁻¹².
  - With 12.74's diagnosis (148 of 148), the evidence is consistent and complete: **inside the exact window, I-POMCP's only departure from Bayes-optimal play in the grid is belief compression, and a larger N removes it.**
- **At levels 3–4, N = 256 does not compress i's belief inside the window.** The level-posterior TV against the exact agent is about 10⁻¹⁶: the agent is exactly Bayes-optimal there.
- **At level 2, N = 256 is not enough for an 8-step window.**
  - Published i2_j1 and i2_j2 still miss in 9 trials each at N = 256.
  - At N = 1,024 there are no misses, and the level TV falls to at most 0.012.
- **What N3 keeps:** N3's grid stays at N = 64 (its configuration since 12.37). Its level-3 deviations (12.73) are this compression error, small and in near-ties. The paper reports them as such, with this test as the evidence.

### 12.74 N4(a): N3's agents against the Bayes-optimal agent along their own histories: level 1 is Bayes-optimal on all 96,000 decisions; levels 2–4 agree on 97.7–100% per cell, and every diagnosed miss is belief compression at N = 64 (October 20, 2026)

**Why:** PLAN N4(a), the user's second priority: is the opponent modeling in the grid correct against the Bayes-optimal benchmark?

**Design** (`results/n7_exact_replay_20261019`; `ENDPOINTS.md` fixed before the main run, with two recorded amendments; `job.py`, `analyze.py` → `analysis.txt`, `table.csv`, `disagreements.csv`; `diagnose.py` → `diagnosis.jsonl`):
- Every intentional agent of every N3 cell, both games, is replayed along its own realized actions and observations through the exact agent of the same level (`analysis.exact_replay`). The exact agent uses the same frames, priors and H2 lookahead, with exact beliefs and exact nested models.
- **Window and sample:** level 1 for 10 steps, trials 0–999; level 2 for 8 steps, trials 0–199; level 3 for 4 steps, trials 0–399; level 4 for 3 steps, trials 0–99.
- **Measures at each decision:** whether the action is an exact maximizer; the total variation (TV) of the level posterior; the TV of the predicted opponent action.
- **Amendment:** the published level-4 replays needed a 5 GB cap and ran after N3 (all succeeded).
- **Cost:** 40.5 CPU-h of replays (kill line 60), plus 7.5 CPU-h of diagnosis.

**Agreement with the exact maximizers** (decisions pooled over the 6 cell-agents of each game and level; cells of a row often replay the same history, so the pooled counts are not independent):

| Game | Level | Window | Decisions agreeing | Range over cells | Level-posterior TV, mean / max | Prediction TV, mean / max |
|---|---|---|---|---|---|---|
| Coordination | 1 | 10 | **48,000 / 48,000** | 100% | 0 / 0 | 0 / 0 |
| Coordination | 2 | 8 | 8,779 / 8,800 | 99.6–99.9% | 0.006 / 0.27 | 0.010 / 0.18 |
| Coordination | 3 | 4 | 8,361 / 8,400 | 97.8–100% | 0.005 / 0.08 | 0.002 / 0.08 |
| Coordination | 4 | 3 | 1,793 / 1,800 | 97.7–100% | 0.011 / 0.09 | 0.006 / 0.05 |
| Published | 1 | 10 | **48,000 / 48,000** | 100% | 0 / 0 | 0 / 0 |
| Published | 2 | 8 | 8,739 / 8,800 | 99.0–99.9% | 0.006 / 0.25 | 0.009 / 0.17 |
| Published | 3 | 4 | 8,329 / 8,400 | 98.8–99.3% | 0.005 / 0.09 | 0.002 / 0.06 |
| Published | 4 | 3 | 1,799 / 1,800 | 99.7–100% | 0.011 / 0.08 | 0.007 / 0.06 |

- **Level 1 is exactly Bayes-optimal** on every replayed decision: its beliefs over a uniform level-0 opponent are exact at N = 64.
- **The 200 disagreements are near-ties.**
  - Exact Q gaps have a median of 0.065 (published) and 0.09 (coordination). The largest is 6.0, at level 2 in the late steps.
  - By step: 117 at step 3 (levels 3–4), and the rest at steps 4–8 (level 2).
- **Diagnosis: 148 of 148 diagnosed disagreements have one cause.**
  - Each disagreement is rebuilt with its trial's seeds at the disagreement step and decided three ways: as configured, with exact beliefs, and with exact beliefs at budgets 1,000.
  - Every one is reproduced as configured, and every one picks an exact maximizer with exact beliefs. **Belief compression at N = 64 causes the misses, not search.**
  - Coverage: all 118 at levels 3–4, and 30 of the 82 at level 2 (every one diagnosed before the sample, plus a seeded sample, `diagnose_jobs_sample.txt`). Cost: 7.5 CPU-h logged.
- N4(b) (12.75) tests this on fixed trials, rather than on the disagreements alone.

### 12.73 N3, the definitive Tiger level grid: depth pays once (level 1 → 2) on the coordination Tiger and not again; inside the exact window, the I-POMCP pairs match the Bayes-optimal pairs except where N = 64 compression bites at level 3 on the published Tiger (October 20, 2026)

**Why:** PLAN N3, the user's two priorities. First, low variance at the step the user named (step 3 of the published Tiger, where the user expected about 2.7). Second, a level grid i L0–4 × j L0–4 checked against the Bayes-optimal benchmark. The design (B, 10 steps) is the user's choice of October 19 (12.71).

**Design** (`results/n5_level_grid_20261019`; `ENDPOINTS.md` fixed before the run; `job.py`, `analyze.py` → `analysis.txt`, `gains.csv`, `cells.csv`; `compare_exact.py` → `compare_exact.txt`; `figures.py`):
- 12.37's configuration: H2, 100 simulations, N = 64 compressed beliefs, uniform priors over lower levels at every nesting. Episodes of 10 steps, both games, the 15 cells with i's level ≥ j's (the others are read mirrored).
- **n per cell:** levels ≤ 2 at 15,000 (coordination) and 2,500 (published); the level-3 row at 700; the level-4 row at 100.
- **Rewards are Rao-Blackwellized (RB):** each sampled reward is replaced by its exact expectation given both agents' actions and observations. Same estimand, lower variance (sampled beside it in `cells.csv`).
- **Held 12.37 trials were not reused:** today's code compresses beliefs differently (`gate/`).
- **Cost:** 231.7 CPU-h (price 207; kill line 420). No watchdog kills.

**Precision reached** (RB 95% half-width of a cell mean):

| Cells | Step 3 | Step 10 |
|---|---|---|
| Levels ≤ 2, coordination (n = 15,000) | ±0.35 | ±0.65 |
| Levels ≤ 2, published (n = 2,500) | ±0.35 | ±0.6 |
| Level-3 row (n = 700) | ±0.6 (published), ±1.5 (coordination) | ±1.1 (published), ±2.8 (coordination) |
| Level-4 row (n = 100) | ±1.7 (published), ±4.1 (coordination) | ±3.2 (published), ±8 (coordination; ±4.1 against j0) |

**The user's example, step 3 of the published Tiger** (agent i, RB mean [95%], against the exact pair of 12.72):

| Pair | I-POMCP, N3 | Exact Bayes-optimal |
|---|---|---|
| L1 or L2 against L1 or L2 (n = 2,500) | 2.34 [1.98, 2.67] | 2.148 |
| L3 against L1–L3 (n = 700) | 3.08 [2.43, 3.66] | 2.663 |
| L4 against L1–L4 (n = 100) | 3.88 [2.04, 5.39] | 2.663 |

The single-agent ceiling of 2.72 (12.48) is not the pair's expectation: the other agent's door openings reset the tiger.

**Primary endpoint: paired one-level gains of i at step 10** (RB; Holm within game; n is the number of paired trials):

| Gain | Coordination Tiger | Published Tiger |
|---|---|---|
| 1 → 2 against j0 | −0.89 [−1.11, −0.68], Holm p < 10⁻⁴ (n = 15,000) | −0.14 [−0.26, −0.04], 0.19 |
| 1 → 2 against j1 / j2 | **+7.84 / +7.57**, Holm p < 10⁻⁴ (n = 15,000) | −0.34 [−0.55, −0.11], 0.036 / −0.27, n.s. |
| 1 → 2 against j3 / j4 | **+9.23** (n = 700), p < 10⁻⁴ / +7.02 [2.26, 11.84] (n = 100), Holm 0.060 | +0.01 / +0.17, n.s. |
| 2 → 3 against j0–j4 | −0.84 to +1.39, none significant (smallest Holm p 0.16) | −0.26 to +0.46, none significant |
| 3 → 4 against j0–j4 (n = 100) | −0.36 to +1.62, none significant | −0.40 to −0.07, none significant |

- **Depth pays once, from level 1 to 2, on the coordination Tiger**: by step 10, +7.6 to +9.2 against levels 1–3 (Holm p < 10⁻⁴), and +7.0 against level 4 (n = 100, Holm p = 0.06). It costs about 0.9 against a uniform opponent. Levels 3 and 4 add nothing measurable by step 10, at n = 700 and n = 100. This is 12.37's profile, now at 15,000 trials per cell and with level 4.
- **On the published Tiger, depth buys nothing by step 10**, as 12.37 found. The one significant gain is a small loss: −0.34 for 1 → 2 against j1.
- **At step 3 (secondary),** the coordination Tiger's 1 → 2 gain is +1.88 [1.74, 2.03] against j1, where the exact value is 1.78. The published Tiger's 2 → 3 gain is +0.34 [0.20, 0.48] against j1 or j2, below the exact 0.515 (next bullet).

**Against the exact Bayes-optimal pairs inside their window** (`compare_exact.txt`; every cell, both agents, every step the exact curve reaches):
- The exact expectation lies inside the RB 95% interval for **182 of 200** intentional agent-steps. The control, uniform level-0 agents (exact by construction), is 76 of 84. Both fall below 95% for the same reason: rows repeat across cells, which share trial ids, and across steps.
- **Systematic deviation: the published Tiger's level-3 row.**
  - i3 against j0 earns −1.28 [−1.66, −0.89] by step 3, against the exact −1.893 (z = 3.1; z = 3.3 at step 4).
  - When i is at level 3, j at level 1 or 2 earns 2.97 against the exact 2.148 (z = 2.7).
  - The step-3 gain from 2 to 3 is 0.34 against the exact 0.515.
  - This matches N4(a)'s diagnosis: at step 3, level-3 agents with N = 64 compressed beliefs sometimes listen where the exact agent opens, and with exact beliefs they open. N4(b) (12.75) tests it on fixed trials.
- **Other exclusions are one sample counted several times.** On the published Tiger, j at L1/L2 in i1_j1, i2_j1 and i2_j2 shows identical trajectories (z ≈ 2.2–2.3, steps 3–5). On the coordination Tiger, the level-4 row's step 3 (n = 100, z ≈ 2.0) repeats the same 100 trials. 12.48's 10,000 fresh L1-against-L1 trials (RB 2.23 ± 0.19 at step 3) contain the exact 2.148.

**Figures:** `docs/figures/grid_curves.png` (i's expected cumulative reward by step for L1–L4 diagonal pairs, both games, with the exact pair's curve over its window and the 2.72 ceiling) and `docs/figures/grid_heatmap.png` (i's RB reward by step 10 over the 5 × 5 grid).

### 12.72 N4(c): exact expected reward curves of the Bayes-optimal pairs over the level grid; on the published Tiger a pair at level 3 or above earns exactly 2.663 by step 3 (October 19, 2026)

**Why:** PLAN N4(c). These are the zero-variance benchmark curves that N3's Monte Carlo curves of the I-POMCP pairs are overlaid on (N5).

**Design** (`results/n6_exact_curves_20261019`; `ENDPOINTS.md` fixed before the run, one amendment recorded there; `job.py`, `summarize.py` → `summary.txt`):
- Every cell of the grid (i's level a ≥ j's level b, 0–4), both games. Exact agents throughout: H2, γ = 0.95, uniform priors over lower levels at every nesting, level 0 uniform, ties uniform over the exact maximizers.
- `analysis.exact_curves.exact_pair_curve`, up to 10 steps. A cell stops at 2,000,000 nodes, an exact update's budget of 100,000 branches, or its memory cap.
- **Amendment:** the four cells i4_j3 and i4_j4 (both games) were stopped by the operator during step 4, when the machine's available memory fell to 1 GB beside N3. The remaining cells ran at a 3 GB cap.

**Exact window reached** (steps; why it stopped):

| Pair | Steps | Stop |
|---|---|---|
| L0 against L0; L1 against L0 | 10 | none |
| L1 against L1 | 6 | 2,000,000 nodes |
| L2 against L0–L2 | 4–5 | branch budget or 3 GB |
| L3 against L0–L3 | 4 | 3 GB |
| L4 against L0–L4 | 3 | 3 GB, or the operator at step 4 |

**Published Tiger, cumulative reward at step 3 (i / j):**
- Level 1 or 2 against level 1 or 2: **2.148 / 2.148.**
- Level 3 or 4 against level 1 or 2: **2.663 / 2.148.**
- Level 3 or 4 against level 3 or 4: **2.663 / 2.663.**

So the 2.7 the user expects at step 3 is approached only from level 3 on: 2.663, against the single-agent ceiling of 2.72 (12.48). The other agent's door openings reset the tiger, so a pair cannot reach the ceiling.

**Coordination Tiger, cumulative reward at step 3 (i / j):**

| | j at L1 | j at L2 | j at L3 | j at L4 |
|---|---|---|---|---|
| i at L1 | 14.833 / 14.833 | — | — | — |
| i at L2 | 16.616 / 16.101 | 18.011 / 18.011 | — | — |
| i at L3 | 16.814 / 16.242 | 18.224 / 18.166 | 18.380 / 18.380 | — |
| i at L4 | 16.814 / 16.242 | 18.224 / 18.166 | 18.380 / 18.380 | 18.380 / 18.380 |

- **On the coordination Tiger the exact value of depth is positive and monotone to level 3,** for both agents, in every column.
- **Level 4 adds nothing over level 3** through step 3, in either game. This matches 12.40's finding that level 4 does not pay.
- **Against level 0 on the published Tiger, depth costs a little by step 3:** −1.376 at levels 1–2, −1.893 at levels 3–4. The mechanism is not diagnosed. On the coordination Tiger, depth gains against level 0 (0.933 at level 1, 1.394 at level 2, 2.315 at levels 3–4).
- **Cost:** about 4.6 CPU-h (4.0 logged, plus the four level-4 jobs started before CPU logging was added; their per-step CPU is in their curves). Kill line 30.

### 12.71 Pricing the definitive Tiger level grid from data on disk: the planned half-widths cost about 25,000 CPU-h on the coordination Tiger; paired gains and tiers bring it to 100–800 (October 19, 2026)

**Why:** PLAN N2. Before spending CPU on pricing runs, the 12.37 (levels 0–3, both games, n = 100) and 12.40 (level 4 against 0–4, coordination, n = 40) episodes already give each cell's CPU and, through `analysis.rao_blackwell`, its Rao-Blackwellized (RB) variance. RB means each sampled reward is replaced by its exact expectation given both agents' actions and observations; the estimand is unchanged.

**Scripts:** `results/n4_grid_price_20261019/from_existing.py` (→ `from_existing.txt`) and `options.py` (→ `options.txt`). No new episodes were run. The games are symmetric, so cells with i's level ≥ j's cover the grid.

**Per-cell spread of the cumulative reward** (RB standard deviation over trials, intentional agents only; a level-0 agent's own reward is not an endpoint):

| Game | Step 3 | Step 20 | RB variance ÷ sampled variance | n for a 95% half-width of 0.5 at step 3 and 1.0 at step 20 |
|---|---|---|---|---|
| Published Tiger, levels 1–3 | 8.6–8.7 | 20–26 | 0.2–0.5 | 2,100–2,600 |
| Coordination Tiger, levels 1–4 | 18–21 | 53–64 | 0.6–1.07 | 12,000–16,000 |

- **RB helps much less on the coordination Tiger.** Its spread comes mostly from the agents' own action draws (who opens which door, and whether together), which RB does not condition out. The published Tiger's spread comes mostly from the tiger's position, which RB integrates.
- **At PLAN N3's targets, the cell means cost about 24,500 CPU-h on the coordination Tiger and 330 on the published Tiger (levels 0–3 only).** That is not feasible on the shared machine.

**The grid's endpoint is a paired one-level gain, which is much tighter** (RB, step 20, i at a against a − 1, the same j level, paired by trial):

| Gain | Coordination: SD, n for ±1 | Published: SD, n for ±1 |
|---|---|---|
| 1 → 2 against j0 / j1 | 23 / 55: 2,027 / 11,798 | 1.5 / 11.3: 9 / 490 |
| 2 → 3 against j0 / j1 / j2 | 15 / 33 / 24: 821 / 4,084 / 2,247 | 2.7 / 8.9 / 14.2: 28 / 305 / 770 |
| 3 → 4 against j0–j3 (n = 40) | 13–27: 659–2,707 | no data |

(The 0 → 1 gain is about +570 against an SD of about 150, so it needs no precision.)

**CPU per episode is concentrated late:** on level-4 cells, steps 1–8 take 12–13% of an episode's CPU (12.70's trials). A 10-step episode costs about a fifth of a 20-step one at level 4 (`options.txt`). Held 20-step trials also serve a 10-step design, because their first 10 rows are exactly a 10-step episode.

**Tiered designs** (CPU-h for the trials still missing, quiet-machine process CPU; × about 1.8 at load 20, as 12.68 measured; the published level-4 row is priced at the coordination row × the published/coordination ratio at level 3):

| Design | 20 steps | 10 steps |
|---|---|---|
| A: levels 0–2 at n = 2,500 (published) and 15,000 (coordination); level-3 row n = 700; level-4 row n = 200 | 794 | 243 |
| B: as A, level-4 row n = 100 | 520 | 177 |
| C: as A, level-3 row n = 400, level-4 row n = 40 | 242 | 97 |
| D: every cell n = 1,000 | 3,019 | 772 |

### 12.70 N1(d): a cache shared across trials would save 27–33% of a level-3 or level-4 episode's CPU, so no cache is built (October 19, 2026)

**Why:** PLAN N1(d). Each episode builds fresh solver banks, so modeled solves are recomputed in every trial. The question was whether sharing them across trials would make large n affordable.

**Design** (`results/n3_repeat_work_20261019`; `ENDPOINTS.md` and `KILL_CRITERION.txt` fixed before the run; `job.py`, `analyze.py`, `analysis.txt`):
- Coordination Tiger in the level grid's configuration (12.37: H2, 100 simulations, N = 64 compressed, uniform nested priors, 20 steps). Cells i2_j2, i3_j3, i4_j2 and i4_j4, trials 0–7, one process per episode.
- **One planner seed per cell.** Under the grid's per-trial seeds, no seeded result can be reused bit-identically, because searches and compression draw from the trial's bank seed. So only this regime could profit.
- Every modeled policy (one nested search) and every filter update was logged with its key and CPU. The endpoint for trial k is the inclusive CPU of the calls whose key occurred in trials 0..k − 1 and whose ancestors' keys did not (a hit skips its subtree), as a share of trial k's episode CPU.

| Cell | Episode CPU (min, mean) | Saved with 1 / 3 / 5 / 7 earlier trials | Speedup with 7 |
|---|---|---|---|
| i2_j2 | 0.08 | 39% / 50% / 49% / 60% | 2.5× |
| i3_j3 | 2.1 | 19% / 31% / 31% / 32% | 1.46× |
| i4_j2 | 17 | 14% / 18% / 24% / 32% | 1.46× |
| i4_j4 | 45 | 12% / 22% / 23% / 27% | 1.36× |

- **Decision, by the rule fixed before the run: no cache.** The cells that cost CPU (levels 3–4) plateau at 1.4–1.5× from about four earlier trials on. The only cell above 2× is i2_j2, which costs 5 s per episode.
- **Two further costs would come with a cache.** First, about 1.7 million new keys per i4_j4 episode, so its memory grows without bound unless it is capped, and a cap lowers the saving. Second, one seed per cell correlates trials through shared nested approximations, so trials would no longer be independent draws of the planner.
- Nearly all of the saving is in modeled policies (94–99%), not in filter updates.
- **Cost:** 8.6 CPU-h (kill line 25). Six workers, peak 1.9 GB per level-4 job, and no watchdog kills.

### 12.69 Held-out check: the qualified Tiger configurations are optimal on 960 of 960 random beliefs no qualification used (October 19, 2026)

**Why:** every qualified budget is the smallest rung of a ladder that was 100% optimal on its grid, and the paper then reports that grid. A reviewer can call this selection on the test set. This reruns the qualified configurations, unchanged, on problems no qualification or gate used.

**Design** (`results/g7_heldout_20261019`; `ENDPOINTS.md` fixed before the main run; `job.py`, `analyze.py`, `analysis.txt`):
- **Problems:** beliefs drawn uniformly from [0.02, 0.98] (3 decimals) by a generator seeded with (game, level, seed), on seeds 100 and above, which no earlier run used. History 0. Every decision is scored against the exact I-POMDP.
- **L2 H3:** root 1,000, modeled `400`, exact beliefs (the configuration of 12.3 and 12.23). Per seed, 10 beliefs of i × 3 of j; seeds 100–109, so n = 300 per game.
- **L3 H3:** root 1,000, modeled `6400s/100`, exact beliefs (12.7, 12.23). Per seed, 6 × 3 × 2 beliefs; seeds 100–104, so n = 180 per game.
- **Games:** the published and the coordination Tiger.

| Game, cell | n | Optimal [95%] | ε-correct | Median CPU per decision |
|---|---|---|---|---|
| Published Tiger, L2 H3 | 300 | **300/300** [0.988, 1] | 300/300 | 1.2 s |
| Coordination Tiger, L2 H3 | 300 | **300/300** [0.988, 1] | 300/300 | 1.2 s |
| Published Tiger, L3 H3 | 180 | **180/180** [0.980, 1] | 180/180 | 31 s |
| Coordination Tiger, L3 H3 | 180 | **180/180** [0.980, 1] | 180/180 | 22 s |

- **The qualification carries over to held-out beliefs.** All 960 decisions are optimal. By the reading fixed in advance, the qualified Tiger configurations are not an artefact of selecting on their own grids.
- **UAV's held-out evidence is 12.67.** There, every decision at histories 0–4 reached in H4 episodes was scored exactly (120/120 per I-POMCP arm). Those beliefs were never used for qualification.
- **CPU is higher than in 12.23's L3 gate** (22–31 s against 8.4 s per decision). Random beliefs reach more distinct nested beliefs than the grid's symmetric points, and the machine ran at load 12–14.
- **Cost:** 4.6 CPU-h, against a pilot price of about 3 and a kill line of 8.

### 12.68 IPF + RTS at level 3 on the coordination Tiger: 91–98% optimal against I-POMCP's 162/162, never right where I-POMCP is wrong; significant only at K = 2 (October 19, 2026)

**Why:** PAPER C2's comparison with the published baseline against the exact I-POMDP covered level 2 only. A reviewer could ask whether it holds at level 3 (HANDOFF 0.000, item 2).

**Design** (`results/g6_rts_l3_20261019`; `ENDPOINTS.md` fixed before the main run; `job.py`, `analyze.py`, `analysis.txt`):
- **Problems:** the coordination Tiger (`together_safe = 40`) at level 3 (i-L3 → j-L2 → i-L1), on 12.23's grid: i's belief 0.1–0.9 (9) × j's 0.085/0.5/0.915 × the level-1 i below j 0.085/0.5/0.915, seeds 0–1, so n = 162 per cell. History 0. Every decision is scored against the exact level-3 I-POMDP.
- **IPF + RTS:** K = 2, 4, 8 observation samples per node at every level, 64 particles, with the published filter and with the smoothed filter of 12.62 (ε = 0.01), at H2 and H3.
- **I-POMCP:** 12.23's gate configurations (H3: root 1,000, modeled `6400s/100`, exact beliefs; H2: root 100, modeled `100`, compressed 32).
- **Tests:** paired exact McNemar tests, each RTS cell against I-POMCP at the same horizon, with Holm correction over 12 tests.
- **CPU:** per root decision, nested solves included. Every cell ran in the same run at machine load about 20.

| Planner | H | K | Filter | Optimal [95%] | Mean loss | Max loss | Optimal-action Q error | Median CPU (s) | Pairs only I-POMCP / only RTS right, Holm p |
|---|---|---|---|---|---|---|---|---|---|
| **I-POMCP** | 2 | — | compressed 32 | **162/162** [0.977, 1] | 0 | 0 | 0 | 0.09 | — |
| IPF + RTS | 2 | 2 | published | 148/162 [0.859, 0.952] | 1.73 | 27.0 | 7.05 | 0.42 | 14 / 0, **0.0015** |
| IPF + RTS | 2 | 2 | smoothed | 148/162 [0.859, 0.952] | 1.73 | 27.0 | 6.28 | 0.41 | 14 / 0, **0.0015** |
| IPF + RTS | 2 | 4 | published | 158/162 [0.938, 0.993] | 0.49 | 27.0 | 2.27 | 0.71 | 4 / 0, 0.75 |
| IPF + RTS | 2 | 4 | smoothed | 158/162 [0.938, 0.993] | 0.49 | 27.0 | 1.96 | 0.70 | 4 / 0, 0.75 |
| IPF + RTS | 2 | 8 | published | 158/162 [0.938, 0.993] | 0.49 | 27.0 | 1.80 | 0.93 | 4 / 0, 0.75 |
| IPF + RTS | 2 | 8 | smoothed | 158/162 [0.938, 0.993] | 0.49 | 27.0 | 1.98 | 0.91 | 4 / 0, 0.75 |
| **I-POMCP** | 3 | — | exact | **162/162** [0.977, 1] | 0 | 0 | 0 | **5.3** | — |
| IPF + RTS | 3 | 2 | published | 150/162 [0.874, 0.961] | 0.73 | 21.8 | 7.40 | 25 | 12 / 0, **0.0049** |
| IPF + RTS | 3 | 2 | smoothed | 157/162 [0.929, 0.990] | 0.26 | 21.8 | 6.49 | 24 | 5 / 0, 0.44 |
| IPF + RTS | 3 | 4 | published | 156/162 [0.921, 0.986] | 0.35 | 25.7 | 4.13 | 68 | 6 / 0, 0.25 |
| IPF + RTS | 3 | 4 | smoothed | 155/162 [0.913, 0.982] | 0.16 | 6.58 | 3.95 | 72 | 7 / 0, 0.14 |
| IPF + RTS | 3 | 8 | published | 159/162 [0.947, 0.996] | 0.09 | 6.58 | 4.50 | 146 | 3 / 0, 0.75 |
| IPF + RTS | 3 | 8 | smoothed | 158/162 [0.938, 0.993] | 0.10 | 6.58 | 3.01 | 161 | 4 / 0, 0.75 |

- **I-POMCP reproduces 12.23: 162/162 at both horizons.** Its optimal-action Q error is 0 (below 10⁻¹⁵).
- **IPF + RTS is optimal on 91–98% of decisions, and never right where I-POMCP is wrong** (0 such pairs in all 12 cells). Its 3–14 misses per cell lose up to 27.
- **By the reading fixed in advance, C2 does not fully extend to level 3.** The gap is significant only at K = 2 (H2 with both filters, H3 published; Holm p ≤ 0.005). At K = 4–8 the 3–7 misses per cell are not significant after the correction (Holm p ≥ 0.14). Tiger is the easy domain for this baseline, as at level 2 (12.20: 95–99.5%), so this matches what level 2 showed. The large gaps are on UAV and Wumpus (12.43, 12.46, 12.59, 12.62).
- **At H3, I-POMCP is also 5–30× cheaper** (5.3 s against 24–161 s per decision), so a more accurate baseline costs more, not less.
- **At H2 the misses do not shrink from K = 4 to K = 8, and both filters miss exactly the same problems,** so they are not the zero-likelihood failure that smoothing repairs (12.62).
- **Diagnosis** (`DIAGNOSIS.md`, written before the run; `diagnose.py`, `diagnosis.txt`; 72 extra jobs, under 0.1 CPU-h). The same 162 H2 problems were rerun with RTS on compressed beliefs (N = 64, so these point and small-support beliefs stay exact) at K = 2, 4 and 8, and with IPF + RTS at K = 32.

  | Arm | K = 2 | K = 4 | K = 8 | K = 32 |
  |---|---|---|---|---|
  | IPF + RTS (published) | 148/162 | 158/162 | 158/162 | **162/162** |
  | RTS on compressed (exact) beliefs | 150/162 | **162/162** | **162/162** | — |

  - **At K = 2, the misses are RTS's search error.** On exact beliefs RTS misses 12 of the IPF arm's 14 problems, with the same losses (13 and 27).
  - **At K = 4–8, the misses are the particle filter's.** On exact beliefs RTS makes no miss. The IPF's 4 misses fall on different problems at K = 4 and K = 8 (all seed 1), so they follow the particle draws. At K = 32 they vanish.
  - **Every miss in every arm is on a problem where i and j are both confident about the tiger** (i's belief 0.1–0.2 or 0.8–0.9, j's 0.085 or 0.915, on the same side).
- **Smoothing helps only at H3 with K = 2** (7 problems only the smoothed filter gets right, 0 the other way; uncorrected p = 0.016, a secondary test). It changes nothing elsewhere.
- **Cost:** 25.0 CPU-h (re-priced from the pilot's 14 at a measured 1.8× load factor; kill line amended before it was reached, `KILL_CRITERION.txt`).

### 12.67 UAV episodes beyond exact reach, every planner: I-POMCP leads at matched CPU wherever the planners differ, and is exactly optimal on every scored H4 decision from root 4,000 or own budget 1,000 (October 18–19, 2026)

**Why:** this is the last piece of G3 (PLAN.md, publication gaps). Every exactly checked domain so far is small, so this run tests the planners in episodes, at horizons where exact solutions run out.

**Design** (`results/g3_uav_episodes_20261018`; `ENDPOINTS.md` and `KILL_CRITERION.txt` were fixed after n = 10 pilots and before the main run, with one dated amendment to the pairing rule):
- **Episodes:** UAV, 20 steps, from the common prior (I-NTMCP's model class). Spotting ends an episode; 35 means spotted after 4 steps, the most any episode earned.
- **Real target j:** level-1 I-POMCP at root 4,000, N = 64, depth H, the same agent in every arm.
- **i:** the level-2 UAV at depth H = 4 or 5. Arms pair by trial seed.

| Arm | What it is | n per H |
|---|---|---|
| `ipomcp-<n>`, `ipomcp-pm-<n>` | I-POMCP, root n = 1,000 / 4,000 / 16,000, uniform or point-mass prior on j's level. It models j exactly as configured (4,000), so its cost is set by j's model, not by n | 30 |
| `ipomcp-pm-own-<n>` | I-POMCP, point-mass prior, root n = 50 / 100 / 250 / 1,000, modeling j at its own budget n, as I-NTMCP and RTS do. Added after the pilot, which showed the arm above has no point below about 75 s per decision | 30 |
| `intmcp-tuned-<M>`, `intmcp-published-<M>` | I-NTMCP (12.65's adapter), M = 1,000 / 4,000 / 16,000 / 64,000 | 30 |
| `rts-sampled-k2`, `rts-smoothed-k2` | IPF + RTS, K = 2, on the published and the repaired (12.62) IPF; H4 only, because K = 4–8, or any K at H5, costs 2,000 to over 40,000 s per decision (12.46) | 10 |

- **Exact scoring:** every decision of i whose own history has at most 4 steps (H4) or 1 step (H5) is scored against the exact I-POMDP the arm solves (`oracle.episode_reference_q`: its prior, every lower level exact).
- **Why the caps:** the exact cost about triples per history step: 10, 24, 73, 211, 615 and 1,989 CPU-s at H4 histories 0–5. At H5 it is 165 s at the prior and 490 s after one step, and at H6 3,284 s at the prior.
- **No decision was unscorable:** the real target never played an action the exact target never plays.
- **Cost:** 173 CPU-h: 118 for episodes (32 of them IPF + RTS), 48 for exact references and 7.5 for the diagnosis below. The price fixed before the run was about 95 CPU-h to grow to n = 30, plus about 20 for the pilots. The overrun (the exact references, and IPF + RTS decisions at about twice 12.46's CPU under this load) passed the 120 CPU-h kill line without being caught, because cumulative CPU was not tracked during the run.

**Endpoint 1, anytime curves** (Figure: `docs/figures/anytime_uav.png`; i's mean 20-step reward [bootstrap 95%] at its median CPU per decision; n = 30):

| H | Planner (budget) | Median CPU per decision (s) | Reward |
|---|---|---|---|
| 4 | I-NTMCP tuned (1,000 / 4,000 / 16,000 / 64,000) | 0.9 / 4.2 / 13 / 54 | −2.7 / −7.5 / 30.2 / **35.0** |
| 4 | I-NTMCP published (same) | 0.9 / 3.7 / 13 / 50 | 17.5 / 20.8 / 32.2 / **35.0** |
| 4 | I-POMCP, own budget (50 / 100 / 250 / 1,000) | 1.4 / 1.8 / 2.9 / 15 | 33.5 / 34.7 / 34.8 / **35.0** |
| 4 | I-POMCP, target as configured (6 arms) | 81–94 | **35.0** each |
| 4 | IPF + RTS K = 2 (sampled / smoothed) | 1,344 / 1,130 | 30.0 / 30.0 |
| 5 | I-NTMCP tuned | 1.0 / 4.4 / 15 / 58 | −11.0 / 11.0 / 31.0 / **35.0** |
| 5 | I-NTMCP published | 0.9 / 4.2 / 14 / 56 | 9.0 / 28.2 / 32.0 / **35.0** |
| 5 | I-POMCP, own budget | 5.5 / 5.6 / 8.4 / 29 | 31.7 / 33.3 / 34.5 / **35.0** |
| 5 | I-POMCP, target as configured | 66–78 | 35.0 in 5 arms; 33.8 (uniform, root 1,000) |

**Endpoint 2, matched CPU** (each tuned I-NTMCP arm against the most expensive point-mass I-POMCP arm with no more median CPU; paired by trial; Holm over 8):

| H | I-NTMCP | I-POMCP arm (CPU ratio) | Difference [95%] | Holm p |
|---|---|---|---|---|
| 4 | M = 4,000 | own-250 (0.69) | **+42.3** [23.7, 63.2] | 0.002 |
| 4 | M = 16,000 | own-250 (0.22) | **+4.7** [3.3, 6.3] | < 0.001 |
| 4 | M = 64,000 | own-1,000 (0.29) | 0 (both 35 in all 30) | 1 |
| 5 | M = 16,000 | own-250 (0.57) | **+3.5** [2.2, 4.8] | < 0.001 |
| 5 | M = 64,000 | own-1,000 (0.51) | 0 (both 35 in all 30) | 1 |
| 4 / 5 | M = 1,000 | own-50 (1.6 / 5.8; *more* CPU) | +36.2 / +42.7 | 0.002 / < 0.001 |
| 5 | M = 4,000 | own-50 (1.3; *more* CPU) | +20.7 [8.3, 36.3] | 0.026 |

**Endpoint 3, exact decision quality** (exactly optimal = loss ≤ 10⁻⁹; Wilson 95%):

| Arm | H4 (histories 0–4) | H5 (histories 0–1) |
|---|---|---|
| I-POMCP point mass: root 1,000 / 4,000 / 16,000 with the target as configured, and own budget 1,000 (4 arms) | **120/120** each [0.969, 1] | **60/60** each |
| I-POMCP uniform, root 4,000 / 16,000 | **120/120** each | 40/60, 46/60 (max loss 0.67) |
| I-POMCP uniform, root 1,000 | 106/120 (max loss 0.46) | 37/60 |
| I-POMCP own 50 / 100 / 250 | 115/126, 120/122, 120/121 | 45, 49, 57 of 60 |
| I-NTMCP tuned 1,000 / 4,000 / 16,000 / 64,000 | 96/144, 93/145, 117/143, **120/120** | 57, 60, 60, 60 of 60 |
| I-NTMCP published (same) | 104/143, 112/141, 119/136, **120/120** | 60/60 each |
| IPF + RTS K = 2, sampled / smoothed | 29/47 [0.47, 0.74], 32/46 [0.55, 0.81] | not run |

(Arms differ in the number of scored decisions because weaker planners play longer episodes. The ε = 0.05 counts equal the exact counts in every arm except own-50 and own-100 at H5, at 47 and 51 of 60.)

- **Below about 15 s per decision, I-POMCP earns near the maximum where I-NTMCP does not.**
  - At H4, I-POMCP with 2.9 s per decision earns 34.8. Tuned I-NTMCP earns −7.5 at 4.2 s and 30.2 at 13 s.
  - At matched or lower CPU, I-POMCP's advantage is +42.3 at M = 4,000 and +4.7 at M = 16,000 (H4), and +3.5 at M = 16,000 (H5).
  - At M = 64,000 (about 55 s) both reach 35 in every episode, so the planners tie there.
- **Decision quality explains the reward gap.** At H4, tuned I-NTMCP at M ≤ 16,000 is exactly optimal on 64–82% of scored decisions (losses up to 15.5), against 120/120 for I-POMCP's own-1,000 at a similar CPU (15 s against 13 s for M = 16,000).
- **Published I-NTMCP force-fills 31–49% of its beliefs on average at M ≥ 4,000 (up to 100%), and about 1% at M = 1,000.**
  - The HANDOFF expected this to be rare on UAV, whose senses are noisy, but it is not.
  - It still earns more than the tuned variant at every M ≤ 16,000 (for example 17.5 against −2.7 at M = 1,000, H4), and the same (35) at 64,000. Why was not investigated. Its exact shares at H4 are also higher (73–88% against 64–82%).
- **IPF + RTS at K = 2 costs 1,130–1,340 s per decision and earns 30.0,** below every I-POMCP arm. Its exactly optimal share at H4 is 62–70% (12.46, over 80 decision problems at the prior with K = 2–8: 46–50%).
- **H5 uniform-prior misses: I-POMCP's own search error, diagnosed** (exploratory, `diag_h5.py`). The misses are 37–46/60 at root 1,000–16,000. The diagnosis replayed four missed histories, at 5 seeds per rung:
  - The gap is 0.67, on values near 22: continue North after a first North move, or East after East.
  - **Root 1,000:** optimal on 6/20 as run (64 atoms, target at 4,000). Exact beliefs give 6/20, an exact target 4/20, and both 5/19, so neither removes the miss.
  - **Root 16,000:** fixed after North (10/10), not after East (2/10).
  - **Root 64,000:** **19/19**, at the same CPU per decision (55–82 s), because the modeled target dominates the cost.
  - So, like 12.41's L1 H5 (root 64,000), **L2 H5 with the uniform prior needs root 64,000.** The point-mass arms are exact at root 1,000.
- **What this does and does not show.**
  - It shows the planners beyond exact reach (rewards over 20-step episodes at H4–H5), anchored by exact scoring of every decision in the first five steps at H4.
  - It does not show a gap at large budgets on this domain: once I-NTMCP has about 55 s per decision, UAV is easy enough that both planners spot the target in 4 steps every time.

### 12.66 Prediction is not reward: one more level predicts better everywhere, but pays only where the payoff depends on the opponent (October 18, 2026)

**Why:** contribution 3, sharper (PAPER.md, G5a). Across the three domains a deeper level often predicts better but earns the same (12.27, 12.32, 12.61). This puts every level step on one plot.

**Data:** stored, 0 CPU (`results/g5a_prediction_reward_20261018/assemble.py`, `rows.csv`). One point per (domain, agent, level step k → k + 1 with k ≥ 1, opponent level), 44 points in all, from:
- Tiger, published and coordination (12.37 matrix);
- UAV (12.49);
- Wumpus 3 × 3, the human's steps (12.60–12.61).

The axes:
- **Prediction gain:** Tiger and UAV, the drop in log-loss of the agent's prediction of the opponent's next action, as a share of the uniform log-loss; Wumpus, the rise in the probability given to the wumpus's actual move, as a share of the gap from 1/3 (a level-1 human predicts uniformly) to 1.
- **Reward gain:** each analysis's own paired gain, divided by the agent's one-step reward range (Tiger 110 or 140, UAV 55 or 50, Wumpus 2,000), so the domains share a scale. Significance is each analysis's own Holm p.

**Figure:** `docs/figures/prediction_vs_reward.png`.

| Cluster | Points | Prediction gain | Reward gain | Significant (Holm < 0.05) |
|---|---|---|---|---|
| Tiger (published), 1 → 2 against an intentional opponent | 6 | 0.54–0.57 | −0.04 to +0.02 | 0 of 6 |
| Tiger (coordination), 1 → 2 against an intentional opponent | 6 | 0.52–0.59 | +0.04 to +0.21 | 4 of 6 |
| UAV, 1 → 2 against an intentional opponent | 3 | 0.38–0.44 | +0.10 to +0.70 | 3 of 3 |
| Every other step (level 2 → 3, or against level 0) | 29 | −0.11 to +0.26 | −0.13 to +0.47 | 3 of 29 (UAV 1 → 2 against levels 0 (−0.13) and 2 (+0.47); Wumpus 1 → 2 against level 2 (+0.09)) |

- **The same prediction gain can be worth nothing or a great deal.** In the published Tiger and the coordination Tiger, a second level improves prediction by the same amount (about 55% of the gap). It earns nothing in the first, because an agent's payoff there depends only on its own door, and up to +0.21 reward ranges in the second.
- **So prediction gain is necessary but not sufficient.** It pays only where the payoff depends on the opponent's action (THEORY.md, "When a deeper model can pay"). The exact screen (12.22, 12.64) measures that dependence directly, per decision.
- **A third level adds little prediction anywhere** (at most +0.10) and no significant reward. The one large reward gain with small prediction gain (UAV against a level-2 target, +0.47 at a prediction gain of 0.11) is a level-2 UAV acting on a better model of evasion, not on better next-action prediction.

### 12.65 I-NTMCP, the modern baseline, against the exact I-POMDP: 40/60 at I-POMCP's CPU, against 60/60 (October 18, 2026)

**Why:** I-NTMCP (Schwartz, Zhou & Kurniawati, IROS 2022) is the closest prior planner: a nested MCTS over history particles, with public code (`docs/RELATED_WORK.md`). Its paper never compared it with exact I-POMDP solutions. This does, on the problems where I-POMCP was qualified (PLAN.md, publication gaps, G3).

**Setup** (`results/g3_intmcp_20261018`; `ENDPOINTS.md` fixed before the runs, with three amendments, each recorded before the cells it changed).
- The authors' code, unchanged, is pinned as the submodule `third_party/i-ntmcp` (`63deec3`). The adapter, `baselines.intmcp`, presents our kernels as their POSG and uses their defaults (`uct_c` = r_max − r_min, random rollouts).
- **Level mapping:** our level L is their nesting level L − 1 (their level 0 is a POMDP planner against a random opponent). **Horizon:** H through their discount horizon, ε = γ^(H − 1.5).
- **A decision:** replay the problem's history (search, update, search, …), then one search. Total CPU counts every search, because I-NTMCP builds each belief from its previous search's particles.
- **Two variants, as 12.62 does for the IPF:**
  - *Published:* their code as released, with no reinvigoration function.
  - *Tuned* (the strong baseline): a reinvigoration function from our exact kernels (s′ ∝ T(s′ | s, a) O(o | s′, a) from the parent particle), and a rejection budget of 50 per particle.
- **Problems:** Wumpus 3 × 3 (1 pit), common prior, random-script histories of 0, 2 or 4 steps.

**Adapter gate (the stop rule; added as amendment 1 after the original L1 H2 gate proved unable to separate an adapter bug from search bias).** At L1 H1 there is no deeper backup, so the root values are particle means of the immediate reward. Tuned, M = 64,000, 60 problems: **57/60 optimal** (the 3 misses are 1.5-point near-ties at the prior), median max |Q̂ − Q| 4.7 on a reward range of 2,000, and no force-filled particles. The adapter is sound.

**Force-filling (amendment 2).** I-NTMCP refills a belief after a real step by rejection sampling. After 3,000 rejections it fills the rest with particles whose own observation contradicts the real one. With deterministic senses and large M, this dominates the published variant: at L1 H2 the median force-filled share is 0, 0.16, 0.59 and 0.69 at M = 1,000, 4,000, 16,000 and 64,000, with a maximum of 0.99. The tuned variant force-fills nothing.

**L1 H2** (I-NTMCP is then POMCP against the reference's own uniform wumpus):

| Variant | M = 1,000 | 4,000 | 16,000 | 64,000 (median total CPU) |
|---|---|---|---|---|
| Tuned | 24/60 | 29/60 | 42/60 | 46/60 (53 s) |
| Published | 24/60 | 25/60 | 37/60 | 40/60 (49 s) |

Convergence is slow because UCT backs up *means*. With c = r_max − r_min ≈ 2,000 on Wumpus, depth-1 values average over explored bad actions. At the prior, every action's value is biased down by 20–60 against the exact Q.

**L2 H2, the 60 problems of 12.58** (I-POMCP's qualified cell: 60/60, median 118 CPU-s per decision). Exact McNemar tests, paired by problem, with Holm correction over the 7 cells:

| Variant, M | Optimal [95%] | Max loss | Median CPU: final search / total | Force-filled share (median, max) | Pairs only I-POMCP / only I-NTMCP right | Holm p |
|---|---|---|---|---|---|---|
| Tuned, 1,000 | 17/60 [0.175, 0.414] | 949 | 0.3 s / 5.2 s | 0, 0 | 43 / 0 | 1.6 × 10⁻¹² |
| Tuned, 4,000 | 32/60 [0.400, 0.663] | 122 | 1.1 s / 6.2 s | 0, 0 | 28 / 0 | 3 × 10⁻⁸ |
| Tuned, 16,000 | 34/60 [0.432, 0.694] | 102 | 4.8 s / 25 s | 0, 0 | 26 / 0 | 6 × 10⁻⁸ |
| **Tuned, 64,000** (CPU-matched) | **40/60** [0.533, 0.783] | 102 | 19.6 s / **97 s** | 0, 0 | **20 / 0** | **1.9 × 10⁻⁶** |
| Published, 1,000 | 19/60 [0.203, 0.450] | 949 | 0.3 s / 1.9 s | 0, 0.96 | 41 / 0 | 5.5 × 10⁻¹² |
| Published, 4,000 | 32/60 [0.400, 0.663] | 949 | 1.3 s / 6.4 s | 0.17, 0.98 | 28 / 0 | 3 × 10⁻⁸ |
| Published, 16,000 | 29/60 [0.352, 0.616] | 949 | 5.9 s / 28 s | 0.43, 0.99 | 31 / 0 | 4.7 × 10⁻⁹ |

- **At matched CPU, I-POMCP is right on 20 problems I-NTMCP gets wrong, and I-NTMCP on none.** Tuned at M = 64,000 uses a median 97 CPU-s per decision in total, against I-POMCP's 118 s.
- **More simulations help I-NTMCP slowly** (17 → 32 → 34 → 40 of 60 across a 64-fold budget). The published variant stops improving beyond 4,000, as force-filling grows.
- **Mechanism (descriptive, from the miss rows):** I-NTMCP's root values are nearly flat across actions (−20 to −33), while the exact Q singles out one safe action (about −2) against dangerous ones (−45 to −123). It underrates the wumpus's attack. Its nested wumpus samples actions by a softmax over visit counts, which is near-uniform at thinly visited histories, and its mean backups blur the safe action.
- **Cost:** 17.8 CPU-h for the reported cells. The M = 256,000 cells were dropped before they ran (amendment 3), because they were projected at about 70 CPU-h for the tuned variant.
- **Reading, fixed in advance:** I-NTMCP at its largest budget is significantly worse than I-POMCP, so contribution 2 extends to the modern baseline on this domain. The scale comparison beyond exact reach (UAV episodes from the common prior) is next (HANDOFF 0.0a).
