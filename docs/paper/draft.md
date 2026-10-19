# Exact Where Checkable: Online Planning for Interactive POMDPs with Verified Nested Beliefs

*Draft 1, October 19, 2026. Built from `docs/PAPER.md`. Every number cites its section of `docs/RESULTS.md` in brackets, for example [12.20]; the citations are for the authors and are removed before submission. Target: AAMAS main track.*

## Abstract

Interactive POMDPs (I-POMDPs) let an agent plan against opponents that plan in turn, by holding beliefs over the opponent's beliefs. Exact solutions are intractable beyond toy problems, so practice relies on sampling planners. Their decisions have never been checked against exact I-POMDP solutions. We present I-POMCP, an online Monte Carlo tree search planner whose nested beliefs are exact, or compressed from exact updates, and whose search values are rescored with the exact chance law after search. Each decision carries per-action intervals that contain the exact action values of the searched model. Against exact I-POMDP solutions in three domains (multi-agent Tiger, UAV reconnaissance and a two-agent Wumpus world) and at nesting levels 1–4, I-POMCP's decisions are optimal on every problem of every gate we ran, for example 486 of 486 at level 4. The published baseline (an interactive particle filter with reachability-tree sampling) is optimal on 1.7–99.5% of the same problems. A modern nested MCTS planner (I-NTMCP) is optimal on 40 of 60 at matched CPU. The sampled filters fail outright when sensors are deterministic, and a standard repair does not close the gap. Beyond exact reach, in 20-step UAV episodes, I-POMCP earns more than I-NTMCP at matched CPU up to about 13 s per decision, and both reach the best reward at about 55 s. Finally, we show both by proposition and in paired episodes in four games that one more level of reasoning pays exactly when the opponent's actions depend on its beliefs about the agent, and not against a random opponent.

## 1. Introduction

An agent that shares its world with another planner must predict that planner, and the planner's actions depend on what it believes about the agent. Interactive POMDPs [Gmytrasiewicz and Doshi 2005] make this recursion explicit. An agent of level *k* holds a belief over pairs of a physical state and a model of the opponent at a lower level, and each model holds its own belief in turn. The recursion is what makes I-POMDPs expressive and what makes them expensive. Nested belief supports grow geometrically with time and depth, and every predicted opponent action is itself the output of a planner.

Online sampling planners make I-POMDPs usable at all: the interactive particle filter with reachability-tree sampling (IPF + RTS; Doshi and Gmytrasiewicz 2009), POMCP adapted to I-POMDPs [Hula et al. 2015], and nested MCTS over history particles (I-NTMCP; Schwartz et al. 2022). Their evidence is reward in simulation. None has been checked decision by decision against exact I-POMDP solutions. So it is unknown how often their decisions are wrong, why, or whether more samples fix them.

We take the opposite route: make the planner exact wherever exactness is affordable, and check it against exact solutions wherever they can be computed. Our contributions:

1. **I-POMCP, a planner that is exact where checkable** (Sections 4–5). It searches over the agent's private histories with Bellman backups and exact expected rewards. It holds nested beliefs exactly or compressed from exact updates, and rescores the tree with the exact chance law after search. On every gate grid we ran, across three domains and levels 1–4, its decisions are optimal against the exact I-POMDP. Its decisions carry certified per-action bounds, which contained the exact value in 1,016 of 1,016 decisions checked.
2. **A decision-level comparison with the published and the modern baselines** (Section 6). IPF + RTS misses 0.5–54% of decisions where its filter works. Its interactive particle filter fails on 55–57 of 60 problems when sensors are deterministic, and the standard likelihood-smoothing repair does not close the gap. I-NTMCP, run with the authors' code, is optimal on 40 of 60 Wumpus problems at I-POMCP's CPU. Beyond exact reach, I-POMCP earns more than tuned I-NTMCP in UAV episodes at matched CPU, until both saturate.
3. **When a deeper model pays** (Section 7). A proposition shows that against an opponent whose actions do not depend on its beliefs about the agent, no model beats the level-1 best response. Paired episodes in four games confirm both directions: a second level gains 5.8–179 in reward against an intentional opponent where payoffs couple the agents, and nothing against a random one. The same prediction gain earns nothing when payoffs do not depend on the opponent.

## 2. Related work

**Exact and offline solvers.** Exact solutions use behavioural equivalence to merge opponent models [Rathnasabapathy et al. 2006]. Approximate offline solvers include interactive point-based value iteration [Doshi and Perez 2008], interactive bounded policy iteration [Sonu and Doshi 2015], finite-state controllers of others [Panella and Gmytrasiewicz 2017] and interactive dynamic influence diagrams [Zeng and Doshi 2012]. They are offline, and none checks an online planner against exact solutions. We use full enumeration of the finite-horizon I-POMDP as our reference (Section 5.1).

**Sampling planners.** IPF + RTS [Doshi and Gmytrasiewicz 2009] represents each nesting level by N sampled particles weighted by the agent's observation likelihood, and looks ahead by sampling K observations per node of a reachability tree. It is our published baseline. I-POMDP Lite [Hoang and Low 2013] assumes the opponent observes the state, so it cannot represent the opponent's beliefs about us. Hula et al. [2015] adapted POMCP to I-POMDPs to model human behaviour in a trust game. I-NTMCP [Schwartz et al. 2022] is the closest work. It builds one UCT tree per nesting level over history-state particles, samples the opponent's action by a softmax over visit counts one level down, and fixes one opponent level. It needs only a generative model, while we need explicit kernels. It reports returns, not decision quality against exact solutions; it is our modern baseline. POTMMCP [Schwartz et al. 2023] scales online planning to large partially observable games against fixed-policy types, not agents that model us.

**What is new here.** No prior planner reports decisions scored against exact I-POMDP solutions, certified per-action bounds, or uncertainty over the opponent's level with level-0 mass. No prior work compares a sampled nested filter with exact nested beliefs, or repairs the interactive particle filter's zero-likelihood failure.

## 3. Background

**I-POMDPs.** An agent *i* of level *l* > 0 holds a belief *b_i* over interactive states (*s*, *m_j*), where *s* is the physical state and *m_j* = (frame, *b_j*) is a model of opponent *j* at a lower level. Level 0 acts uniformly at random and holds no belief. Beliefs are joint: correlations between the physical state and *j*'s private belief are kept. After own action *a_i* and observation *o_i*, the update is

*b_i'*(*s'*, *m_j'*) ∝ Σ_{*s*, *m_j*} *b_i*(*s*, *m_j*) Σ_{*a_j*} π_j(*a_j* | *m_j*) *T*(*s'* | *s*, *a_i*, *a_j*) *O_i*(*o_i* | *s'*, *a_i*, *a_j*) Σ_{*o_j*} *O_j*(*o_j* | *s'*, *a_i*, *a_j*) 1[*m_j'* = *U_j*(*m_j*, *a_j*, *o_j*)],

where *U_j* is *j*'s own update and π_j is *j*'s policy under its model. Every modeled policy π_j is itself the solution of *j*'s I-POMDP one level down.

**Level priors.** An agent at level *L* holds a prior over the opponent's levels 0 … *L* − 1, and so does every nested model. The level-0 mass is not cosmetic. With deterministic sensors, a modeled *j* whose model of *i* is a point mass on one level can receive an observation its model gives probability 0, and its update is undefined. Any positive level-0 mass restores it, since level 0 gives every action positive probability (Remark in Section 7).

## 4. Method

### 4.1 Search over private histories

I-POMCP estimates, at the agent's belief *b* and lookahead *H*,

*Q_H*(*b*, *a*) = E[*R* | *b*, *a*] + γ Σ_*o* *P*(*o* | *b*, *a*) max_{*a'*} *Q*_{*H*−1}(*b_ao*, *a'*),

with opponents acting by their modeled policies. Each simulation draws an interactive state from *b* and descends a tree of the agent's private histories, choosing actions by UCB with an exploration width set from the declared reward bounds. Backups are empirical chance-weighted Bellman backups, *Q*(*h*, *a*) = *R̄*(*h*, *a*) + γ Σ_*o* *C*(*h*, *a*, *o*)/*N*(*h*, *a*) · *V*(*hao*), with *V*(*h*) the maximum over evaluated actions. Terminal outcomes stay in the count with zero continuation.

**Exact pieces.** The planner carries the agent's own posterior *b_h* down the tree with the same update it uses for real observations. *R̄*(*h*, *a*) is therefore the exact E[*R* | *b_h*, *a*], and the last decision uses *Q*_1(*b_h*, *a*) = E[*R* | *b_h*, *a*]. Neither uses the sampled state or the opponent's sampled action, so action selection is unchanged and the reward variance is removed.

### 4.2 Nested beliefs: exact, or compressed from exact updates

Each modeled opponent is solved by a bank of solvers, one per level, keyed by the opponent's belief. Every distinct nested belief that a search reaches costs one modeled solve, and modeled policies and belief updates are cached across real steps. Beliefs are held either exactly or by a **compressed filter**: the exact update of a belief with at most *N* atoms, followed, only when the posterior has more than *N* distinct atoms, by a stratified resampling to *N* atoms. Small beliefs stay exact, the evidence *P*(*o_i* | *b*, *a_i*) is exact under the compressed prior, and nested opponents are updated by the same filter. The interactive particle filter, by contrast, holds every belief as *N* equal-weight samples and redraws every term of every update. Nested probabilities then move in steps of 1/*N*. Modeled opponents act greedily, so a step across a decision threshold flips their predicted action.

### 4.3 Exact chance scoring and certified bounds

After search, the tree is rescored bottom-up with the exact observation law instead of sampled frequencies:

*Q**(*h*, *a*) = E[*R* | *b_h*, *a*] + γ Σ_*o* *P*(*o*, alive | *b_h*, *a*) *V**(*hao*).

With two steps left, every child is a last decision whose exact value needs only its posterior, so each such action is scored over all its positive branches and is exact given *b_h*. If every legal action was evaluated and every positive branch has a child, backward induction makes the scored root equal to the exact best response to the modeled opponents (checked by tests to 10⁻⁹). The planner records whether this held. The same pass returns, for each root action, an interval that contains the exact *Q* of the searched model. Unsampled branches and rollout frontiers enter at the bounds implied by the declared reward range. The interval's width is the search error the tree leaves unresolved. It does not cover opponent-model or belief error, which is why we also score against the exact I-POMDP.

The scored values decide both real decisions and modeled-opponent decisions, so an opponent is predicted by the same rule the agent uses.

### 4.4 Modeled solves with a consistent stopping rule

A modeled opponent's solve may stop early when its sampled root value has converged, under a stopping rule with a 1/*n* consistency schedule [12.11, 12.15]. Budgets are written as stacks, one per nested level (for example `6400s/100`: 6,400 simulations with stopping for the level-2 opponent, 100 for the level-1 agent below it).

**Limitation.** I-POMCP needs explicit transition, observation and reward kernels. I-NTMCP needs only a generative model.

## 5. Exactness

### 5.1 Protocol

**Reference.** For each problem, the finite-horizon I-POMDP is solved by full enumeration, with every lower level solved exactly in turn (the *exact I-POMDP*). In the exact I-POMDP a modeled agent with several optimal actions plays them uniformly. Search may return a subset of the tie set instead, which is an equally rational opponent; we count such cases separately.

**Measures.** The *first-action loss* is *V**(*b*) − Σ_*a* π(*a* | *b*) *Q**(*b*, *a*), the value lost by the root decision when every later decision is optimal. A decision is *optimal* when this loss is at most 10⁻⁹, and *ε-correct* when it is at most ε = 0.05 reward units [12.54]. The *optimal rate* is the share of optimal decisions, with exact Clopper–Pearson 95% intervals.

**Problem grids.** On Tiger, *i*'s belief over the tiger's side is crossed with *j*'s belief and, at level 3 and above, with the beliefs of the models below *j* (for example 19 × 5 beliefs × 4 seeds = 380 problems at level 2 [12.3]). On UAV, each level holds one of 4 named laws over the other agent's sector [12.41]. On Wumpus, problems are reachable beliefs from the common prior. A configuration is *qualified* when every decision on its grid is optimal, or ε-correct where stated.

### 5.2 Results

| Domain | Level, horizon | Problems | Optimal | 95% interval | Source |
|---|---|---|---|---|---|
| Tiger | L1, H5 | 220 | 220/220 | [0.983, 1] | 12.1 |
| Tiger | L2, H3–H4 | 380 per horizon | 380/380 | [0.990, 1] | 12.3–12.4 |
| Tiger | L3, H3 | 162 | 162/162 | [0.977, 1] | 12.7 |
| Coordination Tiger | L4, H2 | 486 | 486/486 | [0.992, 1] | 12.23 |
| UAV | L1, H2–H5 | 40 per horizon | 40/40 | [0.912, 1] | 12.41 |
| UAV | L2, H2–H4 | 80 per horizon | 80/80 | [0.955, 1] | 12.41–12.42, 12.46 |
| UAV | L3, H2 | 640 | 640/640 | [0.994, 1] | 12.41 |
| Wumpus 3 × 3 | L2, H2 | 60 | 60/60 | [0.940, 1] | 12.58 |
| Wumpus 3 × 3 | L3, H2 | 55 | 55/55 | [0.935, 1] | 12.58 |

Wumpus 4 × 4 at level 1 is qualified on 52 beliefs with *N* = 4,096 atoms [12.53]. The one residual is Tiger L3 H4, where 4 of 80 decisions miss near-ties with gaps of at most 0.03. All 80 are ε-correct [12.14, 12.54]. The certified bounds contained the exact *Q* of the searched model in 1,016 of 1,016 decisions checked [12.17, 12.41].

**Where the error goes.** Without post-search scoring, 98% of the remaining search error at horizon 3 is sampling noise in observation frequencies [12.24]. Exact chance scoring removes it, which takes the L3 H3 optimal-action *Q* error from 0.06 to 0.0006 [12.14]. At horizon 4, a fifth of the error comes from choices deeper in the tree and 9% from rollouts below it, and a qualified root budget removes the rest from the decisions [12.24, 12.33].

**Figure 1** (`docs/figures/exactness.png`): the optimal rate of every checked cell, I-POMCP against each baseline at its best K, with exact Clopper–Pearson 95% intervals.

## 6. Against the baselines

### 6.1 Decisions against the exact I-POMDP

Each baseline decides on the same problems, scored by the same exact reference. IPF + RTS uses *N* = 64 particles at every level and *K* sampled observations per node. We run it both as published and with likelihood smoothing (each particle weighted by (1 − ε)*O_i* + ε/|Ω_i|, ε = 0.01, at every level), the standard repair for zero-likelihood particles [12.62]. I-NTMCP runs on the authors' code through an adapter, both as published and tuned. Paired differences are tested with exact McNemar tests, Holm-corrected within each comparison family.

**Table 1.** Optimal decisions against the exact I-POMDP.

| Domain, cell | I-POMCP | IPF + RTS, published (K = 2 / 4 / 8) | IPF + RTS, smoothed (K = 2 / 4 / 8) | I-NTMCP, tuned | Source |
|---|---|---|---|---|---|
| Tiger L2 H3 (n = 380) | 380/380, 1.5 s | 98.4 / 98.9 / 99.5%, 1.0–4.0 s | — | — | 12.20 |
| Tiger L2 H4 (n = 380) | 380/380, 16.7 s | 95.3 / 95.3 / 96.8%, 12–92 s | — | — | 12.20 |
| Coordination Tiger L3 H2 (n = 162) | 162/162, 0.09 s | 148 / 158 / 158, 0.4–0.9 s | 148 / 158 / 158 | — | 12.68 |
| Coordination Tiger L3 H3 (n = 162) | 162/162, 5.3 s | 150 / 156 / 159, 25–146 s | 157 / 155 / 158, 24–161 s | — | 12.68 |
| UAV L2 H2 (n = 80) | 80/80, 0.6 s | 46 / 51 / 54 | 49 / 54 / 55 | — | 12.43, 12.62 |
| UAV L2 H3 (n = 80) | 80/80, 11.8 s | 42 / 44 / 48, 24–47 s | — | — | 12.43 |
| UAV L2 H4 (n = 80) | 80/80, 366 s | 40 / 37 / 40, 601–3,980 s | — | — | 12.46 |
| Wumpus 3 × 3 L2 H2 (n = 60) | 60/60, 118 s | 2 / 1 / 2 | 22 / 20 / 26 | 40/60 at 97 s (M = 64,000) | 12.59, 12.62, 12.65 |

(CPU is per root decision, nested solves included, from runs in which the compared cells shared one machine state.)

**Findings.**
- **I-POMCP is optimal on every problem in every cell.** IPF + RTS misses 0.5–4.7% of Tiger decisions at level 2, 1.9–8.6% at level 3, and 32–54% of UAV decisions. It is never right where I-POMCP is wrong. On Tiger the gap is small: at level 3 it is significant only with K = 2 (Holm p ≤ 0.005; at K = 4–8, Holm p ≥ 0.14), while I-POMCP is also 5–30× cheaper at H3 [12.68]. At UAV H4, I-POMCP is 1.6–11× cheaper [12.46].
- **The sampled filter breaks under deterministic sensors.** On Wumpus, where every percept is deterministic, the published IPF fails on 55–57 of 60 problems: every particle is inconsistent with an observation, and the update is undefined [12.59]. Likelihood smoothing removes 54–56 of these failures and lifts the baseline to 20–26/60. That matches RTS on *exact* beliefs (20–28/60), so once the filter is repaired, what remains is RTS's search error [12.59, 12.62]. I-POMCP is right on every problem the smoothed baseline gets right, and on 25–40 more per arm (Holm p ≤ 6 × 10⁻⁸ in all six tests).
- **The error of IPF + RTS is search error.** RTS on compressed exact beliefs makes about the same decisions as on the IPF, at a fraction of the CPU [12.20, 12.43]. On UAV it converges only at K = 128 [12.45].
- **I-NTMCP at matched CPU is optimal on 40 of 60 Wumpus problems,** against 60 of 60. In 20 pairs only I-POMCP is right, and in none only I-NTMCP (Holm p = 1.9 × 10⁻⁶) [12.65]. The published variant force-fills up to 99% of its beliefs, because its history particles run out under deterministic percepts.

### 6.2 Beyond exact reach: anytime episodes

At horizons 4–5 in 20-step UAV episodes from the common prior, no exact solution of the whole episode is affordable. We compare every planner on reward against CPU per decision, n = 30 episodes per arm [12.67].

**Figure 2** (`docs/figures/anytime_uav.png`): *i*'s mean 20-step reward against median CPU per decision, horizons 4 and 5.

- **At matched CPU, I-POMCP earns more than tuned I-NTMCP** by +42.3 (M = 4,000, H4), +4.7 (M = 16,000, H4) and +3.5 (M = 16,000, H5). At M = 64,000, both earn 35 in every episode, the most any episode earned, at about 55 s per decision. This is a tie.
- **IPF + RTS (K = 2) earns 30.0** at 1,130–1,340 s per decision.
- **Exact scoring at H4.** Every decision of *i* at histories 0–4 of each episode was scored exactly. I-POMCP is optimal on 120 of 120 per arm from root 4,000, or from its own budget of 1,000. Tuned I-NTMCP is optimal on 64–82% at M ≤ 16,000 and on 120 of 120 at 64,000. IPF + RTS is optimal on 62–70%.
- **One weakness, diagnosed.** At H5 with the uniform prior, I-POMCP below root 64,000 misses near-ties (37–46 of 60 exactly scored decisions, gap 0.67). Replays of those decisions show root-search error, and root 64,000 fixes them (19 of 19) at the same CPU, because the modeled target dominates the cost. We have confirmed this fix in single-decision replays, not in full episodes.

## 7. The value of depth

### 7.1 When a deeper model can pay

Fix a real opponent *j* and a lookahead *H*. Call two models of *j* *predictively equivalent* on a history when they give *i* the same joint law over the next *H* steps of *j*'s actions and *i*'s observations.

**Proposition 1 (correct prediction is enough).** If *i*'s model is predictively equivalent to the real *j* on every history of the next *H* steps, then *i*'s exact best response maximizes *i*'s true expected *H*-step reward at every decision, and no other model, of any level or prior, does better at that decision.

*Proof.* *i*'s planning problem is a POMDP over the physical state and *j*'s internal state, whose transition and observation laws are fixed by the predictive law. Equivalent laws give the same POMDP, hence the same *Q_H*, which the exact best response maximizes. ∎

**Corollary (depth cannot pay against a non-intentional opponent).** Call *j* *non-intentional* when its action law depends only on its own history, not on any belief about *i*; level 0 is the extreme case. A level-1 *i* that models *j* by that law is the exact best response to the truth. A level-*k* *i* (*k* ≥ 2) with any prior mass on intentional levels predicts differently, so its loss against the truth is at least zero at every decision.

So a deeper level can pay only when the real opponent's actions depend on its belief about *i*, and even then only when the misspecification changes *i*'s action at a belief *i* reaches. The statement is per decision; receding-horizon episodes add no such guarantee, which is why we test it in episodes.

**Remark (level-0 mass keeps nested updates defined).** With deterministic percepts, a modeled *j* whose model of *i* is a point mass on one level can receive an observation that model gives probability 0. Any positive weight on level 0 restores the update whenever the observation is possible under some action of *i* at some state in *j*'s support [12.58].

### 7.2 Episodes

**Design.** In each game, both agents at levels 0–2 (Tiger: 0–3) play 20-step episodes. Every agent of level *L* holds a uniform prior over the opponent's levels 0 … *L* − 1, and every model matches the real opponent's configuration. n = 100 trials per cell, paired across cells by trial seed. The primary endpoint is the paired gain of one more level against the same opponent, Holm-corrected within each game [12.37, 12.49, 12.60].

**Figure 3** (`docs/figures/depth_gain.png`): *i*'s level 1 → 2 gain against a random (level-0) and an intentional (level-1) opponent, one panel per game.

| Game | Gain vs random opponent [95%] | Gain vs intentional opponent [95%] | Source |
|---|---|---|---|
| Coordination Tiger | −2.6 [−9.8, 4.0], Holm p = 1 | **+27.2** [15.9, 38.4], Holm p < 0.001 | 12.37 |
| UAV reconnaissance | −7.2 [−12.7, −2.6], Holm p = 0.032 | **+5.8** [3.9, 8.0], Holm p < 0.001 | 12.49 |
| Wumpus 3 × 3, fresh trials | +20 [−70, 110], p = 0.76 | **+169** [84, 259], p = 0.0004 (levels 1–2 pooled) | 12.60 |
| Published Tiger (control) | −0.2 [−0.6, 0.0], Holm p = 1 | −4.4 [−8.9, 0.1], Holm p = 0.83 | 12.37 |

- **Against an intentional opponent, a second level pays in every game whose payoffs couple the agents.** A level-2 UAV models a level-1 target's flight to the far corner and spots it in 5.0 steps in every episode, against 6.2 for a level-1 UAV [12.49]. A level-2 human in the Wumpus world is killed by the wumpus in 41–44% of episodes, against 57–60% at level 1 [12.60]. The Wumpus result was confirmed on 100 fresh trials, with the test fixed in advance.
- **Against a random opponent, it pays nothing, and can cost.** As the corollary predicts, no second level is significantly better than the first against a level-0 opponent in any game. On UAV, a level-2 UAV loses 7.2 by anticipating a flight that never comes.
- **In the published Tiger, depth pays nothing at all** [12.37]. There an agent is paid for its own door only, so modeling *j* hardly changes *i*'s best response. Levels 2–5 coincide exactly at H2 [12.47].
- **A third level predicts better but earns no more.** On Wumpus, a level-3 human gives the wumpus's actual move probability 0.54–0.55, against 0.48–0.50 for a level-2 human, but its reward gains (+80, +120) are not significant [12.61]. On the coordination Tiger at H3, with a model matched to the opponent's configuration, a third level gains +0.8 [−6.7, 8.6] (n = 50) [12.32].

**Prediction is not reward.** **Figure 4** (`docs/figures/prediction_vs_reward.png`) plots, per level step and opponent, the gain in prediction against the gain in reward. The same prediction gain (about 55% of the gap to the true policy) earns nothing in the published Tiger and up to +0.21 one-step reward ranges with coordination payoffs: it pays only where the payoff depends on the opponent [12.66].

**An exact screen predicts where depth pays.** Before running episodes, the exact value of one more level can be computed on reachable beliefs. On Wumpus 3 × 3 it is worth 19.7 per decision against an intentional wumpus and costs 1.7 per decision against a random one, matching the episodes. On UAV at H2 it is worth 6.0 against 1.6, matching 12.49 [12.22, 12.64].

**Interpretability.** Because nested beliefs are exact or compressed from exact updates, the full hierarchy of beliefs can be read at every step: which level the agent thinks the opponent is, what that model believes about the agent, and what each model believes about the world. One check matters for reading these plots. In the published Tiger, a level-2 *j* converges to "*i* is level 1" whatever *i*'s true level. This is not a bug: levels ≥ 1 of *i* act identically there, which an exact fixed-point computation over levels 2–5 confirms. A level-0 *i* is identified as such (posterior 0.986 by step 20, n = 100) [12.47].

## 8. Ablations

**Table 2.** What each component buys, from paired results [12.63].

| Component | Paired evidence | Source |
|---|---|---|
| Root chance scoring | Turning it off breaks 34 of 560 UAV decisions and fixes none (p = 1.2 × 10⁻¹⁰); it changes none of 1,786 Tiger decisions | 12.63 |
| Exact scoring with two steps left | L3 H3 optimal-action *Q* error 0.06 → 0.0006 | 12.14 |
| Compressed exact filter vs sampled IPF | Wumpus: IPF fails on 55–57/60; RTS on exact beliefs matches the repaired IPF | 12.59 |
| Stratified vs systematic compression | Systematic draws aliased a pit cell on uniform beliefs (12 of 60 problems lost 102.5); stratified draws fix 31 of 34 discordant pairs on that grid and are worse nowhere | 12.57 |
| Consistent 1/*n* stopping rule | Same decisions as flat 6,400 simulations at 0.52× its CPU; the 1/*n* rate uses 14–23% fewer simulations again | 12.11, 12.15 |
| Root budget 1,000 vs 4,000 | L2 H4 optimal-action *Q* error 0.121 → 0.031 at 1.06× CPU | 12.33 |
| Certified bounds | Contain the exact *Q* in 1,016/1,016 decisions | 12.17, 12.41 |
| Level priors with level-0 mass vs a single fixed level | A wrong point prior costs 17.5–30.0 per episode; a uniform prior costs nothing significant (−6.0, n.s.); level-0 mass keeps Wumpus L3 updates defined | 12.39, 12.58 |

## 9. Limitations

- **Two agents.** Every result is for two agents.
- **Explicit kernels.** I-POMCP needs *T*, *O* and *R* explicitly. I-NTMCP needs only a generative model.
- **Exact references are limited to small horizons.** In the UAV episodes, only the first 5 decisions (H4) or 2 decisions (H5) of each 20-step episode are exactly scored. Later decisions are judged by reward alone.
- **H5 with the uniform prior.** Below root 64,000, I-POMCP misses near-ties (37–46 of 60). Root 64,000 fixes the misses at the same CPU in single-decision replays, but has not been run in full episodes.
- **I-NTMCP ties I-POMCP at high CPU on UAV.** At M = 64,000, both earn 35 in every episode (about 55 s per decision). I-POMCP's advantage on UAV is at lower CPU. Its decision-quality advantage at matched CPU is on Wumpus.
- **Scale.** Wumpus at level 2 on a 4 × 4 grid is out of reach of the exact reference [12.55–12.56].
- **Declared parameters.** The UAV domain's sensor and reward values follow Zeng and Doshi [2012] where the original paper is silent; the coordination Tiger's payoff was chosen by an exact screen [12.22].

## 10. Conclusion and future work

Online I-POMDP planning can be checked, and when it is checked, it can be exact. I-POMCP's decisions are optimal against exact I-POMDP solutions on every gate we ran, at levels 1–4 and in three domains. The published and modern sampling baselines miss up to 54% (IPF + RTS, where its filter works) and 33% (I-NTMCP at matched CPU) of the same decisions, and the sampled nested filter fails outright when sensors are deterministic. One more level of reasoning pays exactly when the opponent has a mind to model and the payoffs depend on it.

**Future work.** (1) Equilibria: the episodes record each agent's predictions of the other, and a measure of how far the joint play is from an equilibrium is defined (`docs/EQUILIBRIA.md`), but no result reports it yet. (2) More than two agents. (3) A generative-model variant of exact chance scoring, which would remove the kernel requirement.

## References (to be completed)

- Doshi, P., and Gmytrasiewicz, P. J. 2009. Monte Carlo sampling methods for approximating interactive POMDPs. JAIR 34.
- Doshi, P., and Perez, D. 2008. Generalized point based value iteration for interactive POMDPs. AAAI.
- Gmytrasiewicz, P. J., and Doshi, P. 2005. A framework for sequential planning in multi-agent settings. JAIR 24.
- Hoang, T. N., and Low, K. H. 2013. Interactive POMDP Lite. IJCAI.
- Hula, A., Montague, P. R., and Dayan, P. 2015. Monte Carlo planning method estimates planning horizons during interactive social exchange. PLOS Computational Biology.
- Panella, A., and Gmytrasiewicz, P. 2017. Interactive POMDPs with finite-state models of other agents. JAAMAS.
- Rathnasabapathy, B., Doshi, P., and Gmytrasiewicz, P. 2006. Exact solutions of interactive POMDPs using behavioral equivalence. AAMAS.
- Schwartz, J., Zhou, R., and Kurniawati, H. 2022. Online planning for interactive-POMDPs using nested Monte Carlo tree search. IROS.
- Schwartz, J., Kurniawati, H., and Hutter, M. 2023. Combining a meta-policy and Monte-Carlo planning for scalable type-based reasoning in partially observable environments. AAMAS / arXiv 2306.06067.
- Sonu, E., and Doshi, P. 2015. Scalable solutions of interactive POMDPs using generalized and bounded policy iteration. JAAMAS.
- Zeng, Y., and Doshi, P. 2012. Exploiting model equivalences for solving interactive dynamic influence diagrams. JAIR 43.

*Several entries are known only through other papers' related-work sections (RELATED_WORK.md, "cited"). Open each before citing it for anything specific.*
