# Plan: publication gaps (October 18, 2026)

**Why:** the user asked on October 17 whether the suite is publication-worthy. Claude's assessment: a strong AAMAS or workshop paper as it stands, borderline for a main track. The gaps reviewers would attack, most serious first:
- every exactly checked domain is tiny;
- the baseline can be called a strawman (IPF's zero-likelihood failure; no modern baseline);
- there is no ablation table;
- contribution 3 may be judged obvious.

**Status (October 18):** planned; nothing run. Details and the order are in HANDOFF 0.0a, item 2.
- **G1 done** (`docs/RELATED_WORK.md`):
  - I-NTMCP (Schwartz, Zhou & Kurniawati, IROS 2022; MIT code) is the closest prior work, a nested MCTS over history particles. It is not checked against exact I-POMDPs, and it **joins G3 as a modern baseline** through an adapter.
  - The name "I-POMCP" is free; Hula et al. (2015) adapted POMCP without naming it.
  - Contribution 1 is reframed as *verified* exactness, not "first online planner".
- **G2 done** (`docs/PAPER.md`): claims → RESULTS → figures, with open slots marked `MISSING`.
- **G3, step 1 done (RESULTS 12.62):**
  - The IPF with likelihood smoothing (`particle_filter="smoothed"`, ε = 0.01) passed the gate: at ε = 0, 420/420 rows were identical to before the change.
  - It removes 54–56 of the 55–57 Wumpus failures, yet still loses to I-POMCP: Wumpus 20–26/60, UAV 49–55/80, against 60/60 and 80/80.
- **G3, step 2 done (RESULTS 12.65):** I-NTMCP against the exact I-POMDP on Wumpus 3 × 3 L2 H2. At matched CPU (tuned, M = 64,000) it is 40/60, against I-POMCP's 60/60 (Holm p = 1.9 × 10⁻⁶).
- **G3, step 3 done (October 19, RESULTS 12.67):** UAV episodes from the common prior at H4–H5 for every planner, in `results/g3_uav_episodes_20261018`. `ENDPOINTS.md` and `KILL_CRITERION.txt` were fixed after the n = 10 pilots and before the main run.
  - **Arms:**
    - I-POMCP at root 1,000, 4,000 and 16,000, uniform and point-mass priors, modeling the target as configured;
    - I-POMCP modeling the target at its own budget n = 50–1,000 (`ipomcp-pm-own-<n>`; added after the pilot);
    - I-NTMCP tuned and published, M = 1,000 to 64,000, at n = 30;
    - IPF + RTS at K = 2, sampled and smoothed, at H4 only, n = 10.
  - **Exact scoring:** at H4 up to history 4, at H5 up to history 1 (`oracle.episode_reference_q`).
  - **Price:** about 95 CPU-h.
  - **Pilot:**
    - I-POMCP scored 35 in all 120 episodes.
    - Tuned I-NTMCP reached 35 only at M = 64,000.
    - Published I-NTMCP force-fills up to 99.9% of its beliefs on UAV.
    - I-POMCP's cost does not depend on its root budget here, because the modeled target dominates it. That is why the own-budget arms were added.
- **G4 done:** RESULTS 12.63 (ablation table).
- **G5b, Wumpus part done:** RESULTS 12.64 (the exact depth screen).

| Step | What | CPU | Gate or deliverable |
|---|---|---|---|
| G1 | Literature check: modern online I-POMDP planners, and whether the name "I-POMCP" is taken | 0 | `docs/RELATED_WORK.md`; whether a modern baseline joins G3 |
| G2 | Paper outline: claims → RESULTS → figures, gaps marked | 0 | `docs/PAPER.md` |
| G3 | Scale beyond exact reach: anytime reward and decision quality against CPU, I-POMCP vs IPF + RTS (published and with a zero-likelihood fix) and I-NTMCP (G1), UAV L2 H4–H5 and/or Wumpus 4 × 4 L1 | price first; ask above 100 CPU-h | IPF fix with ε = 0 reproduces today's rows bit for bit; `ENDPOINTS.md` fixed before runs |
| G4 | Ablation table of the planner's components, from stored paired results; small paired runs only where none exist | low | one table in `PAPER.md` and RESULTS |
| G5 | Contribution 3, sharper: prediction gain vs reward gain across domains; the exact screen as an a-priori predictor (UAV, Wumpus 3 × 3 at H2); a figure for THEORY's proposition | low | screen predictions recorded before comparing with 12.49 and 12.60 |
| (opt.) | Wumpus 3 × 3 at H3, third level | likely > 100 CPU-h | ask the user |

# Plan: Wumpus 3 × 3, value of depth in an adversarial domain (October 17, 2026)

**Chosen by Claude** (the user delegated the direction on October 17 while away). Reasoning:
- Wumpus 3 × 3 is now qualified against the exact I-POMDP at L2 H2 (60/60), L3 H2 (55/55) and, pending, L2 H3 (12.58).
- Contribution 3 (the value of modeling) rests on two domains so far:
  - Tiger, where depth pays only with joint payoffs (12.37);
  - UAV, where a second level pays (12.49).
- Wumpus is the first adversarial pair: the wumpus hunts the human. Do deeper humans survive more and find more gold, and does a deeper wumpus kill more?
- The step after this (12.59 or later) is I-POMCP against IPF + RTS on the same board (contribution 2), if the matrix shows depth matters.

**Status (October 17, 22:30): phase complete.**
- Steps 1–2: gate 60/60; the matrix at n = 100 (RESULTS 12.60).
- Decisions against IPF + RTS (12.59).
- Step 3: confirmed on fresh trials, +169 [84, 259], p = 0.0004; control +20, not significant (12.60).
- Step 4: a third level for the human, gate 55/55 (12.58). Not significant against any wumpus level: +80 and +120 against an intentional wumpus, −19 against a random one. It predicts better, 0.54–0.55 against 0.48–0.50 (12.61).
- Write-up: the reviewer page has a Wumpus section (version 25), the Claude Doc has a cross-domain section on the value of depth, and THEORY.md has "When a deeper model can pay".

**Next, Claude's recommendation (October 17; HANDOFF 0.0a):** the publication gaps. Check the literature for modern online I-POMDP baselines; run a scale experiment beyond exact reach, CPU-matched against IPF + RTS (including IPF with a standard zero-likelihood fix); build one ablation table of the planner's components; write a paper outline.

**Design:** `results/c15_wumpus3_episodes_20261017/ENDPOINTS.md` (fixed before any run).
1. Pilot: 9 cells × 2 trials, for cost.
2. Qualification of the episode agents' own configuration: uniform level priors, 400 simulations.
3. The matrix at an n sized to stay under 100 CPU-h.

# Plan: the UAV reconnaissance domain of Doshi & Gmytrasiewicz (2009) (October 12, 2026)

## Status (October 16, 2026)
**October 15–16 follow-ups (HANDOFF 0.0a):** tie convention (12.51); exact-update profile, no change (12.52); Wumpus L1 qualified over 52 beliefs at N = 4,096 (12.53); the ε = 0.05 certification standard adopted by the user, and the L3 H4 residual 80/80 ε-correct (12.54). Level 4 at H3 (5b) is deferred by the user. Wumpus L2 priced (12.55): over 8 GB per decision after 4 steps, so it needs a memory-lean nested belief before any grid.

**Planned by Claude (the user delegated the choice of direction on October 12 while away). U1–U3 done. U4 gates done (RESULTS 12.41): I-POMCP is optimal at a qualified budget at L1 H2–H5 (root up to 64,000 at H5), L2 H2–H4 (root 16,000 at H4 with exact opponent models) and L3 H2; certified bounds held in 1,016/1,016. U4b/U4c done (RESULTS 12.42): at L2 H4 the target needs 6,400 with stopping (32/32 decisions at root 16,000; an exact target is about 10× cheaper on UAV). U5 step 1 done (RESULTS 12.43): on UAV L2 decisions, I-POMCP is 80/80 at H2 and H3 against 53–68% for IPF + RTS (K = 2–8), and cheaper at H3. U5 steps 2–3 done: episodes at H3 earn the same reward (the target flees to a corner beyond the horizon) at 8–18× less CPU for I-POMCP (12.44); against particles, I-POMCP loses less than IPF + RTS at every N, and small sampled filters fail on UAV (12.45). H4 done (12.46): I-POMCP 80/80 against 46–50% for IPF + RTS, and 1.6–11× cheaper (12.42's CPU corrected to about 6 min per decision). Reviewer concerns C1–C3 done (12.47–12.48, page version 22). U6 done (12.49): a second level pays for both agents on UAV. U7 done (page version 23, Claude Doc). The depth-4 arm of U6 was not triggered: the depth-3 UAV does not stall (a level-1 UAV spots a level-1 target in 6.2 steps, 100%). Details: HANDOFF 0.2, item 2.**

## Context
The Tiger suite is complete (RESULTS 12.37–12.40, and level 4 of 12.39). On Tiger the evidence for all three contributions is in hand: exact
optimality wherever exact references exist, a win over IPF + RTS in decisions (12.20) and in episodes (12.38), and a
settled value-of-depth profile with its control (12.37, 12.40) and prior sensitivity (12.39). The reviewers' remaining
weakness is that every result is on Tiger (2 states). The user chose UAV reconnaissance as the next domain on
October 11 (HANDOFF 0.2, item 2); this plan makes it concrete.

**Why this is the right next step (Claude's judgement, recorded for the user).** Alternatives considered: level 4 at
H3 or Wumpus breadth (both stay inside settled questions or a domain without an opponent), the L3 H4 residual (4/80
near-ties, gaps ≤ 0.03; needs a PAC standard that is the user's call), a solver-level speedup (worth it only once a
domain needs it). UAV is the only item that answers the open reviewer question, it is the benchmark of the paper whose
method we beat on Tiger, and exact references for it are something that paper did not have.

**Parameters.** The 2009 paper gives the structure but no values (HANDOFF 0.2). The user could not be asked
(away), so per HANDOFF the search came next: Zeng & Doshi, "Exploiting Model Equivalences for Solving Interactive
Dynamic Influence Diagrams", JAIR 43 (2012), Section 7 and Appendix B.3 (arXiv:1401.4600), state values for the same
group's UAV reconnaissance domain: listening reports the target's relative position correctly with probability 0.8,
other reports equiprobable, and moving gives random reports; the UAV gets 50 for spotting the target and −5 for any
other action; the CPTs themselves are "available upon request". Declared model (to `DOMAINS.md`, "the paper's
structure, with declared values where it is silent"):

| Item | 2009 paper | Declared here | Source of the value |
|---|---|---|---|
| Grid | 3×3 sectors; UAV i, target T | same | 2009 |
| State | 36 (sides merged by symmetry) | **81**: (i's sector, T's sector), spotted terminal | the 36-state collapse cannot tell whether two agents in side columns share a sector, so it cannot express "spotted"; the full grid can (the mirror symmetry is tested, not used) |
| Actions | N, S, E, W, listen for both ("5 actions", stated twice) | same; moves deterministic, clamped at the edge | 2009 (its "or diagonal" remark contradicts its 5 actions; 5 kept) |
| i's observations | TR, CR, BR: T's row | same 3; listen: correct row 0.8, others 0.1; moving: uniform | Zeng & Doshi 2012 |
| i's own position | known | i starts in a fixed, commonly known sector (bottom-left); moves are deterministic, so i always knows it | 2009 ("aware of its own location") |
| T's observations | "informed … with high uncertainty" | (own sector, i's row); listen: i's row correct 0.6, others 0.2; moving: uniform | declared ("high uncertainty" < i's 0.8) |
| T's prior position | any of 9 | uniform over the 8 sectors other than i's start | 2009 |
| Spotting | i moves into T's sector; game ends | same sector after the joint move; terminal | 2009 |
| i's reward | in [0, 1] (yet Fig. 16 plots up to ~12) | +50 on spotting, −5 per step otherwise | Zeng & Doshi 2012 |
| T's reward | conflicting objectives | −50 when spotted, 0 otherwise | declared (zero-sum on the spotting event) |
| Discount | — | 0.95 (the package default) | declared |

T's own sector enters T's observation because the package's prior is common and never conditions on the true state
(`bootstrapper` docstring); T therefore learns its sector at its first observation, and its first decision is made
without it. Recorded as a deviation.

## Summary, cheapest and most informative first

| Order | Workstream | CPU | Kill criterion |
|---|---|---|---|
| U1 | Model + tests (kernels sum to 1, terminal handling, mirror symmetry, contract tests); replace `domains/uav.py` and its tests (no shim); `DOMAINS.md` | tests | any contract test fails → fix before U2 |
| U2 | Generic problem construction for oracle/panel: a problem gives i's physical belief and, per state, the opponent's belief (the target knows its own sector, so its model depends on the state; Tiger's chain of point models is the special case of a constant map). `sample_history` takes a domain's history policy | tests (Tiger panels reproduce bit-identically) | any Tiger oracle/panel row changes → fix |
| U3 | Exact references: L1 by alpha-vector VI at H2–H4; time one exact L2 I-POMDP problem before any grid | < 1 CPU-h | one L2 H3 problem > 10 min → L2 gates at H2 only |
| U4 | Gates as on Tiger: L1 vs VI, L2 vs exact I-POMDP; Clopper–Pearson; `optimal_q_error`; certified bounds; modeled-budget qualification | sized by U3 | > 100 CPU-h → cut problems |
| U5 | Benchmark: I-POMCP vs IPF + RTS at matched CPU (decisions vs exact, and `planner_comparison` episodes); the paper's reward-vs-particles shape at L1 H2–H3 | sized by U3–U4 | as U4 |
| U6 | Episodes with the suite drivers unchanged (level matrix 0–2, prediction misses, sunbursts) if U4 passes | pilot first | > 100 CPU-h → flag |
| U7 | Write-up: RESULTS, DOMAINS, reviewer page, Claude Doc | — | — |

Profile the hardest case first (the efficient-experiments rule; template `results/s0_nested_profile_20261011/profile_episode.py`).

### U6 design (Claude, October 15, written before any U6 run)
- **Question:** does reasoning one level deeper pay on UAV, a pursuit game where the opponent's model matters by
  construction? (12.44: a level-1 target flees to the far corner; a level-1 UAV models the target as random, a level-2
  UAV models it as level 1.)
- **Suite:** `suite.level_matrix("uav", levels=(0, 1, 2), depth=3, simulations=1000, particles=64, steps=20)`: i is the
  UAV, j the target; each agent holds a uniform prior over the other's lower levels; every model matches the real
  agent's configuration (12.30). Depth 3 with root 1,000 is qualified at L1 and L2 (12.41); modeled agents use 1,000,
  above the qualified 400.
- **Endpoints (fixed now):** the UAV's episode reward and steps to spot (primary: the paired one-level differences
  along each row and column, Holm-corrected, as 12.37); the target's reward; prediction log-loss and misses; level
  identification at the last step, where inside the prior. n = 100 per cell; pilot 2 trials per cell first.
- **Depth-4 arm:** only if the pilot shows the depth-3 UAV stalls on the far corner (12.44) and a depth-4 cell costs
  less than 30 CPU-h at n = 50 (a qualified decision costs about 6 CPU-min, 12.46).
- **Kill:** over 100 CPU-h projected from the pilot → cut n to 50 or levels to (1, 2).

# Plan: the Tiger experiment suite (October 11, 2026)

## Status (October 11, 2026)
**All runs complete (RESULTS 12.37–12.40); W done October 13 (reviewer page v21, Claude Doc).** E: suite packaged (`df8803d`), Gate E 18/18. M1:
levels 0–3 on both games, n = 100 (12.37). M2: the level-4 row (user-approved), n = 40: 3 → 4 gains nothing (12.40).
P: I-POMCP beats IPF + RTS by 28–38 on the coordination Tiger at H3 (Holm p = 0.003); no difference on the
published Tiger (12.38). R: at level 3 a wrong point prior costs 17.5–30.0; at level 4 nothing (12.39). 

## Context
The user wants a complete, packaged Tiger suite before UAV (HANDOFF 0.2, item 1). The three suites that existed
before the September 30 restructure (a level matrix including L0, I-POMCP against IPF + RTS in episodes, prior
sensitivity) ran on the retired estimator, so their numbers are void. Their replacements (12.20, 12.21, 12.27, 12.32)
cover parts only, and their drivers (`grid.py`, `endpoint.py`, `analyze_grid.py`, `h3_matched.py`) live in git-ignored
`results/` folders. This phase packages three domain-generic suite drivers in `src/ipomdp/` (reused unchanged for
UAV), fills the gaps with endpoints fixed before each run, and reports from RESULTS 12.37 on.

**Configuration that binds** (HANDOFF 0.2): compressed beliefs N = 64; every model of the opponent matches the real
opponent's whole configuration (budgets, depth, N, nested priors; 12.30); `prediction_misses` reported for every
run; H2 for any grid containing level 4 (chance scoring makes H2 decisions exact for the searched model at any
budget, so 100 root simulations as in 12.21 and 12.27, which keeps their seeds and pairing); root 1,000 where H ≥ 3;
level 5 out (Section 6). Domains: the coordination Tiger (`tiger-joint`, `together_safe = 40`) and the published
Tiger as the control in which depth cannot pay.

## Summary, in the user's order (engineering first: it gates the rest)

| Order | Workstream | CPU (post-12.36 estimates) | Kill criterion |
|---|---|---|---|
| E | Package: `experiments/suite.py` (multi-cell runner), `analysis/endpoints.py`, three suite builders, tests | tests + ~1 CPU-h gate | the packaged driver does not reproduce 12.27's trial CSVs exactly on 9 cells × 2 trials → fix before any run |
| M | Level matrix L0–L4 × L0–L4, both domains | M1 ~12 CPU-h; M2 sized by its pilot | M2 pilot: level-4 median > 30 CPU-min per episode at N = 64 → n = 40; any suite > 100 CPU-h → ask the user |
| P | I-POMCP against IPF + RTS in episodes, at matched CPU | ~25 CPU-h | first 10 H4 RTS episodes > 2× estimate → H4 n = 30 |
| R | Prior sensitivity at levels 3 and 4 | R3 ~6 CPU-h; R4 from M2's pilot | as M2 |
| W | Write-up: RESULTS 12.37+, HANDOFF, PLAN status, reviewer page, Claude Doc report | — | — |

## Workstreams

**E. Engineering** (reuse; no shims).
- `experiments/suite.py`: `SuiteSpec(name, cells: {label: StudySpec}, trials)`; `run_suite(spec, output, workers)`
  schedules trials round-robin across cells (all cells at equal n whenever stopped), one supervised process per
  episode through `infra.process_supervisor.supervise_jobs` (per-job kill, source lock, memory cap; never a
  `ProcessPoolExecutor`), writes `trials/<cell>/<trial>.csv` atomically, skips finished files, and records a
  manifest of the specification and source hashes as `runner.run_study` does. Episodes are `episodes.run_episode`,
  unchanged (same seeds as 12.21 and 12.27).
- Suite builders (pure functions returning a `SuiteSpec`; the domain is a parameter): `level_matrix` (the `grid.py`
  agents: uniform priors over lower levels at every nesting; level 0 is uniform random), `planner_comparison`,
  `prior_sensitivity` (the real opponent's configuration copied into i's nested priors and stack).
- `analysis/endpoints.py`, from `endpoint.py` and `analyze_grid.py`: final rewards, paired differences (paired t-test
  and trial bootstrap), Holm, the one-level endpoint, per-cell tables (rewards, welfare, log-loss, misses with
  Clopper–Pearson intervals), joint safe openings.
- Tests: builders (matched nested configurations asserted), suite resume and round-robin with a stub episode,
  endpoint functions on synthetic frames.
- **Gate E (fixed now):** rerun 12.27's cells L1–3 × L1–3, trials 0–1, through `suite.py` from a snapshot. Every row
  must equal `results/p4c_joint_level_grid_n64_20261007/trials/*.csv`, CPU, wall and memory columns excepted. If it
  passes, those 900 stored episodes are imported as M1's coordination L1–3 cells instead of being rerun.

**M. Level matrix** (fixed now).
- M1: levels 0–3, n = 100, H2, root and modeled 100, N = 64. Coordination: 7 new cells with level 0, plus the 9
  imported. Published Tiger: all 16 cells (12.21 was N = 32 and is superseded).
- M2: the level-4 cells, coordination only (the published control at level 4 stays 12.21's, under its caveat). Pilot
  4 episodes each of (4, 4) and (4, 2) at N = 64 for CPU and memory per step; then 9 cells, n sized to at most
  100 CPU-h (target 40).
- **Primary:** the paired one-level endpoint (the agent at k + 1 minus the same agent at k, against the same
  opponent level, on the same seeds), now including 0 → 1, for both agents; Holm over the family per domain. M1 at
  n = 100; M2 as its own family at its final n.
- **Secondary:** per cell, i's and j's reward, joint welfare, joint safe openings, log-loss, level posterior at
  step 20, and the miss rate with Clopper–Pearson intervals; heatmaps.
- 8 workers for level ≥ 3 cells, with the watchdog and memory log.

**P. Planner comparison** (fixed now).
- i at level 2 against a real j at level 1 (I-POMCP; the cell where modeling pays 27–30 on the coordination Tiger),
  20 steps, n = 100.
- Arms for i: I-POMCP (root 1,000; modeled 400 at H3 and `6400s` at H4; compressed N = 64); IPF + RTS (sampled IPF,
  N = 64) at K = 2 and K = 4, which bracket I-POMCP's CPU per decision (12.20). H3 on both domains; H4 on the
  coordination Tiger only, n = 50.
- **Primary:** i's 20-step reward, I-POMCP minus each RTS arm, paired by trial, Holm over arms × (domain, horizon).
  **Secondary:** the paired CPU ratio per decision, j's reward, misses. IPF + RTS models j with RTS, as in the paper.

**R. Prior sensitivity** (fixed now).
- Coordination Tiger, H2, N = 64, 100 simulations. i at level L, the real j at level L − 2, so misspecification
  exists in both directions: L = 3 against j at level 1, and L = 4 against j at level 2.
- Arms for i's prior over j's level: correct {L − 2: 1}, over-estimated {L − 1: 1}, under-estimated {L − 3: 1},
  uniform {0 … L − 1}. The uniform arm is M's cell (L, L − 2), imported (same seeds and configuration).
- **Primary:** i's reward, each misspecified arm minus correct, paired, Holm over 3 arms × 2 levels.
  **Secondary:** log-loss, the level-posterior trajectory (does i recover?), misses. R4's n comes from M2's pilot.

**W. Write-up.** RESULTS 12.37 on; HANDOFF; this status; the reviewer page (HANDOFF 0.6) and the Claude Doc report.
Commit and push after each workstream.

# Plan: cheaper nested solves (October 11, 2026)

## Status (October 11, 2026)
**Phase complete (12.36).** S1 + S2 (`661526c`) and S3 (`7183339`; filter cache 65,536 and one-pass order and digest;
belief interning not needed after the cache fix) passed every gate bit-identically. Net, paired on 10-step episodes:
L3 0.50–0.52× CPU and peak memory 1.9 → 0.65 GB; L2 0.57×. S4 not started: the remaining profile is the exact update's
own enumeration (~1/3) with nothing else above ~11%, so a further gain needs a solver-level change (planned separately).

## Context
The root-budget phase (below; 12.33–12.35) measured that nested (modeled) solves are almost all of a decision's CPU:
16–23 s of a gate decision and 101 s of an L3 episode decision, against 0.2–0.35 ms per root simulation. L3 episodes
take 114 CPU-min each (12.32); level 4 and larger belief supports (12.31) are blocked by the same cost. Every study
gets faster if nested solves do, so this phase looks for result-preserving speedups first (W1's standard, 12.12:
bit-identical outputs), then solver-level changes only if those run out.

**S0 profile** (`results/s0_nested_profile_20261011`; cProfile of 12.32's L3 condition, trial 0, 4 steps, 462 s
profiled, of which 445 s is inside modeled solves):

| Hotspot | Cumulative | Why it is waste |
|---|---|---|
| Canonical sort keys and digests (`canonical.atom_row` via `canonical_mass` in every compression, and `belief_digest`) | ~150 s (32%) | `atom_row` re-encodes the opponent's physics (`stable_value(vars(physics))`, then JSON) for every atom: 5.4M times for a handful of frames |
| `TigerModel.update_rollout_memory` | 63 s (14%) | 2M physical Bayes updates over a small finite set of (memory, action, observation) |
| `FiniteBelief.__eq__` (recursive, via frozenset equality of equal-but-distinct beliefs) | 34 s (7%) | equal beliefs built along different paths are compared atom by atom |
| Tiger kernels (`observation_distribution`, `transition_distribution`) | ~60 s | memoizing them gave no gain before (HANDOFF 6) |

## Summary, in order of value per CPU-hour

| Order | Workstream | CPU | Kill criterion |
|---|---|---|---|
| S1 | Canonical keys from cached per-frame and per-state fragments: byte-identical sort keys and digests | tests + ~1 CPU-h gates | any key, digest or gate output differs → revert |
| S2 | Memoized rollout-memory update (exact key, same arithmetic) | with S1's gates | any gate output differs → revert |
| S3 | Re-profile; then belief interning (equal beliefs share one object) if `__eq__` is still ≥ 5% | ~1 CPU-h | not bit-identical, or < 3% saved → drop (HANDOFF 6's threshold) |
| S4 | Only if S1–S3 leave the same hot spots in the solver itself: a solver-level change, gated like 12.15 (same decisions on replayed solves) | planned then | — |
| S5 | Write-up: RESULTS 12.36+, HANDOFF, Claude Doc cost paragraph | — | — |

**Gates for S1–S3 (fixed now):**
1. Unit tests, including new ones: the new sort key equals `json.dumps(atom_row(...))` and the new digest equals
   the old digest over beliefs from L1–L3 models; full suite and ruff clean.
2. **Bit-identical outputs:** 20 R0 problems (10 L2 H4, 10 L3 H3) rerun with the new code: every checkpoint's scored
   values and bounds equal R0's exactly; 12.32's L3 and L2 trial 0, 4 steps: every episode row equal to the stored
   CSV (actions, values, bounds, beliefs' support sizes, predictions).
3. **Speed:** paired CPU, same machine state, old and new snapshots alternating on the same jobs (L3 episode 4 steps,
   3 trials; 10 R0 problems). Report the ratio with its range.

# Plan: root adaptive budgets (October 11, 2026)

## Status (October 11, 2026)
**Phase complete.** R0–R1 (12.33): the root's CPU is a fixed cost, all paid within the first 125 simulations (solving
every reachable modeled policy); a further root simulation costs 0.2–0.35 ms. **No stopping rule qualifies** (the most
any can save is 3%), so R2 as written was void. Flat 4,000 cuts L2 H4's optimal-action Q error 3.9× (0.121 → 0.031) at
1.06× CPU, 380/380 and 162/162 (each replay checkpoint is a flat search, so this is the gate). R2' (12.34): the same
holds for one decision in episodes (1.015× at L3), but with a matched model of j every root rises and L3 episodes cost
1.32–1.42×; episodes keep 1,000. R3 (12.35): at root 4,000 the deeper-choice and rollout error is about 0.009 of a
0.026 scored error, so no deeper-node redesign. Adopted: root 4,000 for gates and Q-error reporting.

## Context
The bounded-belief phase (below; RESULTS 12.30–12.32) is complete. Contribution 1's open cost is at the root: every
real decision runs a flat 1,000 simulations, whether the decision is clear or not. The root budget does not decide
correctness (HANDOFF Section 3, row 5: the modeled budget does), so the target is **Q accuracy per CPU-second at
real decisions**, and lower mean CPU on easy decisions. At L2 H4 the optimal action's Q error after chance scoring
is 0.117–0.132 (12.14, 12.24), of which 22% is deeper greedy choices and 9% rollouts below the tree; at H3 scoring
leaves almost nothing (L3 H3 0.0006; 83% of scored roots exact).

What exists: one growing tree with a pilot-calibrated test at the 1/n rate (`planning.stopping`, modeled solves only;
12.11, 12.15); certified per-action bounds at every real decision (12.17); the exact error split
(`results/p5_error_attribution_20261007/attribution.py`); and the replay method of 12.15
(`results/stopping_v2_20261006/`), which tests rules on recorded checkpoints instead of rerunning.

## Summary, in order of value per CPU-hour

| Order | Workstream | Contribution | CPU | Kill criterion |
|---|---|---|---|---|
| R0 | Replay dataset: one root tree per gate problem, scored with bounds at 125–4,000 simulations, with per-checkpoint CPU, against the exact reference (L2 H4 380, L3 H3 162) | 1, 2 | ~6–10 CPU-h | a 10-problem pilot > 3× estimate → checkpoints to 2,000 |
| R1 | Offline: flat budgets vs rules (a) growing-tree test at the root, (b) certified separation, (a ∨ b); pick one by the pre-fixed rule below | 1, 2 | < 0.5 CPU-h | no rule beats the flat Pareto front → skip R2, report the flat front, go to R3 |
| R2 | Implement the chosen rule (`PlannerConfig.root_stopping`); online gates L2 H4 380 and L3 H3 162, paired with flat 1,000 | 1 | ~4–6 CPU-h | any wrong decision → do not adopt |
| R3 | Error split at the adopted rule; if deep choices and rollouts still dominate L2 H4, a literature-backed deeper-node change, A/B gated (planned then) | 1 | planned after R2 | — |
| R4 | Write-up: RESULTS 12.33+, HANDOFF, Claude Doc report | — | — | — |

**Deferred, with reasons:** the L3 H4 residual (a PAC-style standard is the user's call, not asked); Wumpus breadth,
UAV paths and level 4 on the coordination Tiger (no reviewer question waits on them); cheaper nested solves (only if
compression misses are shown to cost reward; 12.30, 12.32 say they do not); housekeeping (the user's call).

## Workstreams

**R0. Dataset** (`results/r0_root_replay_20261011/`, one snapshot, one process per problem, JSON per problem).
- Problems and stacks of the current gates: L2 H4, 19 × 5 beliefs × seeds 0–3, modeled `6400s` (12.14's
  `g3_decisions`); L3 H3 k = 0, 9 × 3 × 3 × seeds 0–1, stack `6400s/100`.
- Per problem: one root tree seeded as the flat search of the cap (4,000), grown and scored with bounds at 125, 250,
  500, 1,000, 2,000, 4,000; plus 3 pilot trees of 125 (the stopping rule's calibration). Recorded per checkpoint:
  scored values, bounds, exactness, and process CPU split into growth and scoring (bank caches shared as online); the
  exact reference Q once. A flat arm costs its growth plus one scoring; a rule costs growth plus every scoring up to
  its stop, plus its pilots.
- Every flat arm is a checkpoint of the same tree (a flat search of n with another seed), so all arms are paired.
  Check: the 1,000 checkpoint reproduces 12.14's rates (380/380, 162/162) and its mean `optimal_q_error` within noise.

**R1. Replay** (fixed before R0 runs).
- Arms: flat 125–4,000; (a) the `ModeledStopping` test at the root (pilot 125, trees 4, α 0.01, cap 4,000);
  (b) stop at the first checkpoint whose certified intervals separate the leading action (lo_best ≥ hi of every
  other action), else the cap; (a ∨ b).
- Endpoint: per problem, `optimal_q_error` and CPU at the stopping checkpoint. Wrong decisions counted with
  Clopper–Pearson intervals.
- Adopt a rule if, on both sets, it makes no wrong decision **and** either (i) mean CPU ≤ 0.8× flat 1,000 with
  paired `optimal_q_error` no worse by more than 0.01 (upper 95% bootstrap bound of the paired mean difference), or
  (ii) lower `optimal_q_error` than flat 1,000 at ≤ 1.0× its mean CPU (paired Wilcoxon, p < 0.05). If both rules
  qualify, take the lower mean CPU.

**R2. Engineering and gates.** Generalise `stopping.ModeledStopping` to a stopping rule used for the root as well
(`PlannerConfig.root_stopping`, no shim for the old name); certified separation as a rule option if (b) is chosen;
real-decision rows record the simulations spent. Tests: the root rule at the cap equals the flat cap search;
separation never stops on overlapping intervals. Online gates as in R0's problem sets, paired with flat 1,000 in
the same run.

**R3.** Planned after R2, from the error split (`attribution.py`) at the adopted rule.

# Plan: the bounded-belief phase (October 9, 2026)

## Status (October 9, 2026)
Q0 done. Q1 done (12.30): 17 of 19 misses are young, light branches lost at the last one or two compressions
(14 below 1/N); 2 H3 misses were a nested-budget mismatch (i modeled j's model of i with 100 simulations, the
real j uses 400), and those 2 trials carry almost all of 12.29's deficit. So Q4 must match j's configuration.
Q2 done on branch `bounded-belief/compressors`; only the miss metric (`analysis.metrics.predictions`,
`prediction_misses`) is on `main`. Q3 killed (12.31): every candidate costs 1.7-3.8x the control or fails G1
(coverage 0.1), so episodes keep systematic N = 64. Q4 as written (conditional on Q3) is void.
**Q4' done (12.32):** H3 with i's model of j matched to j's configuration (stack `1000/400`): L3 − L2
+0.8 [−6.7, 8.6], p = 0.84, n = 50 paired; 12.29's deficit was the model mismatch. Phase complete; next: Q5 (root
adaptive budgets), HANDOFF 0.2.

## Context
The coordination-Tiger phase (below; RESULTS 12.22–12.29) found the binding constraint at level 3: **the bounded
belief drops the opponent's realized branch.** Compression to N atoms sometimes removes every j-model on the branch
that happened; i then gives j's real action probability 0, and the 5 of 50 H3 trials with such a miss carry level 3's
whole deficit (12.29). Optimal resampling (12.28, branch `resampling/optimal-fc`) is far more accurate per atom but
lost at matched N and CPU. The belief gate (level-posterior TV, 12.16) cannot see these misses.

This phase builds a bounded belief that keeps the realized branch at acceptable cost, makes the zero-probability
rate a standard metric and part of the belief gate, and re-runs the H3 depth test only if that works.

## Summary, in order of value per CPU-hour

| Order | Workstream | Contribution | CPU | Kill criterion |
|---|---|---|---|---|
| Q1 | Diagnose the misses: replay i's filter along logged histories; realized-branch mass at the dropping step vs 1/N | 1, 3 | ~1–2 CPU-h | replay does not reproduce logged support sizes → fix the replay first |
| Q2 | `analysis.metrics.zero_probability`; compressor as a strategy (systematic, optimal, coverage-ε, and stratified by j-model if Q1 calls for it) | 1, 3 | tests only | — |
| Q3 | A/B at matched cost: zero-probability trial rate on the 12.26 probe (pilot 20 seeds per arm; main 200 paired seeds), plus G1/G3 non-inferiority | 1, 2 | pilot ~8, main ≤ ~40 CPU-h | no pilot arm has fewer misses at ≤ 1.2× the control's CPU → keep systematic N = 64, go to Q5 |
| Q4 | H3 depth test re-run with the adopted compressor, pre-registered endpoint unchanged | 3 | ~115 CPU-h (n = 50) | Q3 adopts nothing; 4-episode pilot > 3× estimate → halve n |
| Q5 | Root adaptive budgets (12.24) | 1 | planned after Q3 | — |

**Deferred, with reasons:** Wumpus beyond three beliefs and level 4 on the coordination Tiger (no reviewer question
waits on them; level 4 needs the bounded belief fixed first); a PAC-style certification standard for L3 H4 and
housekeeping (the user's call, not asked).

## Workstreams

**Q1. Miss diagnosis** (`results/q1_miss_diagnosis_20261009/`). Replay i's `CompressedInteractiveFilter` along
the logged (a_i, o_i) of each miss trial (12.29's 5; 12.26's 3 at N = 32; the 12.27 N = 64 grid misses), with the
same seed; the filter is a pure function of (belief, seed), checked against the logged `belief_support_i`. At each
compression: atoms before and after, distinct atoms kept, the j-model marginal, and the mass of the j-models that
predict j's later realized action. Question: was the dropped branch lighter than 1/N (any N-bounded compressor
loses it sometimes) or heavier (lost to draw luck and slots spent on heavy atoms, which optimal resampling fixes)?

**Q2. Engineering.**
- `analysis.metrics.zero_probability(frame)`: per-row miss flag (`prediction.get(action) < 1e-9`), per-trial miss
  rate, and the minimum probability given to the realized opponent action; replaces the check in `compare_n.py`.
- A compressor strategy passed to `CompressedInteractiveFilter`, so A/B arms run paired in one process:
  `systematic` (control); `optimal` (only `optimal_resampling` and its tests ported from `resampling/optimal-fc`,
  whose docs are stale); `coverage(ε, N_min, N_max)`, optimal resampling with the smallest count whose expected
  dropped mass Σ_light w(1 − c·w) is ≤ ε, which bounds the per-update probability of dropping the realized branch
  under a correct prior by ε; and, if Q1 finds the branch spread over many tiny atoms, `stratified` (compress the
  j-model marginal, keep each surviving model's state conditional exact).
- Losing strategies are deleted after Q3. Tests: unbiasedness (Monte Carlo), exact counts, the ε bound,
  determinism given (belief, seed), pickling.

**Q3. Gate** (`results/q3_compressor_gate_20261009/`, one snapshot, JSONL rows, ≤ 8 workers, memory watchdog).
- **Endpoint (fixed before the run):** zero-probability trial rate on the 12.26 probe condition (level-3 i with a
  correct prior, coordination Tiger, H2, 20 steps). Secondary: realized-action log-loss, minimum realized
  probability, i's CPU per episode (paired, same process).
- Pilot: 20 seeds × {systematic N = 64, 96; optimal N = 32, 48; coverage ε = 0.01, 0.003}. Main: 200 paired
  seeds for the control and the 1–2 candidates with pilot CPU ≤ 1.2× the control.
- Non-inferiority for the chosen arm: G1 (`belief_qualification`, levels 2–3) and G3 (12.28's `g3.py`, 190/190).
- Adopt if fewer misses (one-sided paired test, α = 0.05) at ≤ 1.2× CPU and G1/G3 pass.

**Q4. H3 depth re-run** (only if Q3 adopts): `results/p7_h3_depth_20261008/h3_depth.py` with the new compressor,
endpoint and kill criterion unchanged; report the zero-probability rate per condition.

**Write-up:** RESULTS 12.30 (diagnosis), 12.31 (gate), 12.32 (H3, if run); HANDOFF Sections 0 and 6; the Claude Doc
report's "Bounded-belief caveat"; the reviewer page only if Q4 runs.

# Plan: the next phase (October 7, 2026, evening)

## Status (October 7, 2026, evening)
Phase complete (next: the bounded-belief phase, above). P1 done; P2 (12.22: `together_safe = 40`); P3 (12.23: 1,244/1,244, L4 486/486); P5 (12.24); P4 (12.25 at N = 32, superseded by 12.27 at N = 64 after the compression finding, 12.26); P6 (reviewer page, Claude Doc). Next phase: HANDOFF Section 0.1.

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
