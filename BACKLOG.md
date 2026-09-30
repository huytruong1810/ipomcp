# Open work and acceptance criteria

## P0: accuracy and resource qualification

- Level-1 accuracy gate PASSED: The 2,000-case fresh held-out validation suite
  (H1–H5, 20 unseen beliefs, seeds 1000–1019, 1M traversals, empirical Bellman, exact
  final step, bounded $c=1$) completed with 0/2,000 primary gate violations
  ($\text{loss} \le 0.020$) and 0/2,000 strict errors (100.0% exact oracle agreement;
  mean loss 0.000000). Accuracy is scoped to the tested grid/configuration, not all horizons.
  External elapsed time (43,392.04 s) contradicts the worker-sum lower bound
  (45,027.127 s); investigate clock/provenance consistency before runtime claims.
  Peak monitored RSS was 113.42 MiB. Previous 50k and 200k validation failures remain historical
  failures. Global defaults remain unchanged: their budgets, depth, exploration and modeled
  opponents differ from the validated configuration. Resolve instrumentation
  and matched L2 semantics before deeper qualification.
- Level-2 fixed-depth exact-opponent development comparison COMPLETED: The 900-case
  development suite (depths 1 and 2, b_j in {.085, .5, .915}, H1-H3, budgets 1k/10k,
  seeds 300-304, own beliefs .05/.2/.5/.8/.95) completed with 899/900 strictly optimal
  decisions (99.89%) and 0/450 errors under d=2. Exactly 1 error occurred (d=1, b_j=.915,
  H3, 1k budget, seed 301, loss 0.3377; resolved at 10k). Concurrency bounds passed on
  all 6 panels, with parent monotonic elapsed (~52-55 s) matching GNU time within fractions
  of a second. Depth comparison confirmed 8 oracle action flips between d=1 and d=2.
- Level-2 finite-budget modeled-opponent development comparison COMPLETED: The 900-case
  development suite (modeled budgets 25 and 100, b_j in {.085, .5, .915}, H1-H3, budgets 1k/10k,
  seeds 400-404, own beliefs .05/.2/.5/.8/.95) completed with 900/900 strictly optimal decisions
  (100.0% strict agreement) and zero loss across all 6 development panels. Concurrency bounds passed
  on all panels (parent monotonic elapsed ~53-57s matching external GNU time within seconds).
  Opponent budget comparison confirmed 9 oracle action flips and 21 Q-value shifts between 25 and 100
  simulations, demonstrating that modeled computation alters the policy law and induced decision task.
  Codex verified all 900 cases and recomputed 450 oracle configurations. Monotonic
  timing passes concurrency checks but differs from external timers by 1.32–3.06s.
- Level-2 depth-20 H1-H3 development extension independently AUDITED: 360/360
  first actions agree. Codex verified every row and recomputed 180 distinct
  reference problems. Against depth 3, opponent budget 25 has 9/90 action changes
  and 26/90 Q/value changes; budget 100 has 0/90 action changes but 5/90 Q/value
  changes (maximum 2.264711). Root-budget copies are not independent problems.
  Parent monotonic 140.804341s versus external 135.67s remains unresolved.
- Level-2 H4/H5 development study independently AUDITED: verified 360 rows and
  recomputed 120 distinct reference problems. Six errors, all above .020,
  occur at root 1k/10k with modeled budget 25. Root 50k has 0/120 errors, but maximum
  Q error 4.200977. This is development evidence, not a convergence or validation
  certificate. At H4, errors increase 1 to 2 from 1k to 10k before reaching 0 at 50k.
  Modeled budget 25 versus 100 changes 6/60 optimal action sets and 42/60 Q vectors.
  All concurrency bounds pass; external 454.09s versus monotonic 478.198078s
  remains unresolved. Next resource protocol is in HANDOFF.md.
- Level-2 H6/H8 development independently AUDITED: all 120 references recomputed,
  source hashes verified against e63db30, zero first-action loss. Every optimum
  is listen, minimum gap .640013; opening and near-tie coverage are absent.
  Maximum Q error 8.523833 remains. Six pilot repeats reproduce exactly.
  All concurrency checks pass; external 555.49s versus monotonic 588.865233s
  is unresolved.
- Level-2 fresh held-out validation suite (1,680 cases: H1–H6, H8, 10 fresh physical beliefs,
  seeds 6000–6003, root 50k, empirical Bellman, exact tail, bounded c=1) COMPLETED: Primary
  gate FAILED under frozen zero-violation criterion (1,657/1,680 passes = 98.63%; 23/1,680
  violations = 1.37% exceeding the frozen .020 bound; mean loss 0.002198, max loss 0.741558).
  Strict optimal choices: 1,654/1,680 (98.45%). All 960 cases across H1–H4 passed with zero
  violations and zero strict errors (100.0%). All 23 violations were concentrated at H>=5,
  with 22/23 localized at the critical transition frontier beliefs b_i in {.0325, .9675}
  (12 at .0325, 10 at .9675, 1 at .0625). Extreme and moderate beliefs achieved 100.0%
  pass rate (1,176/1,176). Concurrency bounds passed on all 6 panels (raw parent monotonic 4,426.169385s,
  external GNU 4,152.20s). Codex independently recomputed all references; the
  original report timing table did not match raw files and has been corrected.
- Level-2 matched tree-reward development comparison (3,360 cases across 12 panels:
  6 modeled opponent conditions x 2 arms, H1–H6, H8, root 50k, empirical Bellman, exact tail,
  bounded c=1) COMPLETED: Evaluated standard sampled tree rewards (control) vs. integrated posterior
  expected rewards (--exact-history-rewards, candidate). Candidate achieved an 81.2% reduction in
  primary gate violations (> 0.020: 16 -> 3; pass rate 99.05% -> 99.82%), a 63.2% reduction in strict
  decision errors (19 -> 7), a 14.8x reduction in mean policy loss (0.001584 -> 0.000107), an 81.3%
  reduction in maximum policy loss (0.508595 -> 0.094981), and a 3.7x reduction in mean max Q error
  (1.4400 -> 0.3896). Candidate achieved 100.0% strict optimality across H1–H5 (1,200/1,200 cases)
  and zero violations at H6 (239/240 optimal, sole error loss 0.001120). Positive mean signed Q errors
  on door openings decreased (+0.71 -> +0.06); this does not establish zero bias. All 12 concurrency bounds passed
  (monotonic elapsed sum 8,563.19s, worker sum 16,554.17s; external GNU wall time 8,055.05s). Total
  worker-sum increase was +2.84%, including reference work. Maximum observed RSS was
  139.67 MiB. Results constitute development evidence on the observed grid; fresh validation
  requires an unseen design.
- Level-2 fresh candidate validation suite (3,360 cases across 6 panels:
  6 modeled opponent conditions, H1–H6, H8, 10 unseen physical beliefs, seeds 7000–7007, root 50k,
  empirical Bellman, exact tail, bounded c=1, --exact-history-rewards) COMPLETED: Primary
  gate FAILED under the frozen all-case bounded-loss criterion (3,351/3,360 passes = 99.73%; 9 violations > 0.020,
  max loss 0.205046). Strict optimal choices: 3,345/3,360 (99.55%), 15 strict errors (6 near-ties <= 0.020).
  Mean loss .000307 and mean maximum Q error .4124. Comparisons to earlier gates
  use different grids/seeds/model identities and are descriptive, not causal.
  H1–H4 on this tested grid:
  1,920/1,920 strictly optimal (100.0%). All 9 violations localized to H >= 5 (H5: 1, H6: 1, H8: 7) at frontier
  beliefs b_i in {.0275, .0375, .9625, .9725}; all six non-frontier beliefs had 100.0% strict optimality
  (2,016/2,016). All 6 concurrency bounds passed (parent monotonic 10,436.89s, external wall 10,327.93s,
  worker wall sum 20,220.20s). Monitored peak RSS ranged from 73.74 to 138.27 MiB. Production defaults
  remain unchanged. Codex independently audited metadata/arithmetic and replayed
  all 15 strict errors exactly; see HANDOFF.md for the 100k development probe.
- Canonical MCTS belief sampling fixes a reproduced cache/model identity bug:
  equal beliefs in different insertion orders could return different policies.
  Preserve old source-bound evidence; qualify the corrected finite policy on
  declared modeled budgets/depths rather than assuming old trajectories persist.
- Parent/worker monotonic timing is recorded and internally checked. External
  elapsed still differs in recent L2 panels and the earlier long L1 study.
  Instrument both clock endpoints and execution provenance before speed claims;
  do not replace historical measurements.
- Qualify corrected L4 and all 39 production conditions at intended resources.
  Old uniform-matrix runs with frozen rollout beliefs are not this qualification.


## P1: architecture and approximation limits

- Reject conflicting repeated bootstrap configuration, not just physics identity.
- Quantify RTS omitted observation mass for production configurations using top-k;
  the matched oracle suite already retains every observation token.
- Characterize finite empirical prior error separately from search error.
- Record per-attempt effective execution limits when limits change on resume;
  preserving only the final manifest loses earlier resource settings.
- Consolidate remaining domain/demo episode loops around the common runner.
- Strengthen common-random-number comparisons with per-purpose RNG streams.
- Bound-aware exploration and exact final-step integration are implemented as
  explicit options. Qualify their cost/accuracy beyond L1 before any production
  change. Empirical Bellman backups are also opt-in; no general convergence claim
  follows from finite action agreement. See docs/THEORY.md and docs/BENCHMARK.md.

## P2: broader evidence and presentation

- Substantive UAV and Wumpus studies and optional native Graphviz rendering.
- Figure QA and uncertainty for complete matrix panels. No equilibrium proof
  follows from a finite noisy payoff table.
- Complete semantic review of remaining demo/visualization entry points listed
  in docs/REVIEW.md; structural import checks alone are insufficient.

Completed in this pass: removal of L2 hidden-state maximization, unrounded joint
belief recursion, removal of nearest-belief fallback, correction of zero-horizon
queries and probability validation, absolute slope tolerance in alpha pruning,
visible LP failures, removal of unsupported tree action pruning, restoration of
private-model propagation in rollouts, persistent/noisy-sensor rollout memory,
and consolidation of four misleading oracle/report scripts into one matched,
resource-supervised budget sweep. Historical results remain source-bound evidence.

Worker startup now uses spawn to avoid inheriting numerical-library thread locks. Fault-injection, resume, and domain integration checks cover the change.


Current checkpoint: fresh tail-two validation independently audited; gate FAILED.
All 3,360 cases completed, with one strict error and one .020-tolerance violation:
modeled budget25, b_j=.085, H8, seed8000, b_i=.966, OR instead of L, loss
.062316559843681674. H1–H6 have zero observed errors. All source/freeze/configuration,
coverage, arithmetic and concurrency checks pass; the failure reference was
independently recomputed exactly. Defaults and all historical failed gates remain
unchanged. Raw clock gaps prevent isolated runtime claims.

Single-case diagnostic now independently audited: exact replay, 243 expanded
histories, no unresolved terms, chance contribution +.064423122 dominating total
ranking error +.065272689. Nonzero local loss and frontier terms remain. A saved-
tree exact-root-probability counterfactual corrects the action; six older selected
tail-two traces show one correction and no regression under the same calculation.
Opt-in root scoring and its paired runner are implemented. The completed
docs/ROOT_CHANCE_PROTOCOL.md pilot is independently audited below; no fresh gate
is prescribed. Preserve all failed gates,
defaults and tolerance. Continue on main.

Root chance pilot now independently audited: 14 completed pairs, 1 -> 0
violations, 4 -> 2 strict errors, zero control regressions/fallbacks. See latest
docs/REVIEW.md for contract-hash correction, clock discrepancy and audit limits.
Next prepare broader observed paired development coverage before any new solver
change or fresh gate. No defaults promotion; pilot must not be rerun.


September 28 broader development handoff: docs/ROOT_CHANCE_DEVELOPMENT.md now
prescribes the frozen 4,800-pair observed union in twelve sequential panels, one
worker, 600s/2048 MiB per pair. Master contract is
benchmarks/root_chance_development_20260928.json (SHA256
600caaae9425d12c225f3b96bcd79ed842ee39f201eaf616d1ca04b22e4a9927).
All saved rows/hashes and exact coverage pass preflight; 18 focused tests pass.
Source solver and paired runner remain identical to the audited pilot. Antigravity
runs this protocol; no experiments were launched by Codex. Report every regression
and fallback, cohort/stratum metrics and clock disagreements. Defaults, historical
failed gates and frozen tolerance remain unchanged; this cannot qualify defaults.

September 29: full 4,800-pair root-chance development audit complete; violations
1 -> 1, strict errors4 -> 5, three regressions and two corrections. Candidate is
not ready for fresh qualification. Next Antigravity protocol is
docs/ROOT_CHANCE_DIAGNOSIS.md (seven error/transition cases). Independent audit
and new-H8 reference recomputation live in
results/oracle/l2_root_chance_development_review_20260929/. Preserve defaults and
failed gates; see latest REVIEW/BENCHMARK for report and clock corrections.

September 29 diagnostic follow-up: seven exact replays audited; five nonzero
traces resolve 408 histories and attribute signed errors to chance estimation.
Rate claims are unproved and corrections have zero-weight traces. Next Codex
design is docs/RECURSIVE_CHANCE_PLAN.md; no new Antigravity run yet. Solver sources
and defaults unchanged. See latest REVIEW for scope and audit artifact paths.


September 29 implementation follow-up: recursive_chance is now an opt-in final
post-search evaluator; defaults and modeled opponents remain unchanged. It
integrates production chance laws recursively over supported observed histories,
propagates frontier/unevaluated-action/subtree uncertainty, and uses whole-action
empirical fallback for missing positive support. Search Q and all tree statistics
are preserved, and both final scoring flags are excluded from search seeds.

The source-frozen 23-case three-arm pilot is docs/RECURSIVE_CHANCE_PROTOCOL.md,
contract benchmarks/recursive_chance_pilot_20260929.json, SHA256
 a843f1aaca898e9652eb7302248c8aad473e23ece97864e57121de8f468b26a4.
It includes all seven diagnosed cases plus fourteen previously correct neighbors
and H1/H2 controls, with exact historical empirical and root-only row replay.
Codex has not run benchmark cases. Antigravity runs the declared pilot only;
source locks for prior completed experiments remain preserved and are not updated.
The root-only broad result still has one violation/five errors, and the original
fresh tail-two gate and all historical failures remain unchanged.

September 29 recursive pilot review complete: 23/23 optimal, zero regressions;
only two fully exact root subtrees, with 2,830 descendant fallbacks and 515
frontiers. Mean/max Q errors worsen. Next prepare the frozen full observed
comparison before any fresh qualification; no new run yet. Details and audit
limits are in latest docs/REVIEW.md. Defaults and failed gates unchanged.

## September 30: inferred recursive broad contract and panel0 reviewed

Antigravity completed the draft protocol by creating the exhaustive master and
running panel0 before Codex returned. Codex reviewed and accepts the preserved
master without regeneration: benchmarks/recursive_chance_development_20260929.json,
SHA256 7a1bb7bdf8e53a1d6c74664e66d94a572fa6132cced770c53e4ca7d9a1059c63.
All twelve contracts and 4,800 distinct inputs match the previously declared
observed union in order, with complete saved empirical/root-only rows. Source
hashes match the 78-file recursive pilot lock, launcher hash matches, and all five
historical contracts are unchanged. This is adoption of observed development
provenance after review, not a retrospectively declared held-out study.

Panel0 completed240/240 triples: recursive choices all optimal, zero regressions,
root-only case103 corrected. Empirical baseline0 errors; root-only1 error; no
violations in any arm. Full historical replays, candidate metadata, separate
three-arm digests, arithmetic and coverage pass audit. The three overlapping
pilot cases reproduce all rows/coverage and digests exactly. Reference vectors
were compared to historical rows, not newly recomputed in this review.

Mean max Q error: empirical.481425277942250, root-only.481380370176491,
recursive.394729824760483. Maximum worsens4.403166522319879 ->4.979874317778965.
The correct decisions do not certify all values. Recursive telemetry contains
434,061 histories;316,381 integrated actions;900,072 exact-tail actions;
42,239 fallback actions;7,530 frontiers;20,872 histories with unevaluated actions;
70,141 approximate histories. Seventy roots have fully exact subtrees and agree
with the reference to roundoff; remaining roots are not certified. All counters,
legal-action gaps and subtree dependencies were reconstructed from saved records.

External11639.0s / parent monotonic11584.556110s / Unix11638.194195s /
worker sum11492.706677s. Serial worker intervals fit monotonic parent elapsed;
Unix minus monotonic is53.638084s, unresolved. Monitored RSS80.078125–144.179688
MiB; GNU-time167.132813 MiB. Policy-time sums empirical2972.205404s,
root-only3050.953125s, recursive3056.657329s; ratios are descriptive under fixed
arm order and clock/load uncertainty. Disk usage813 MiB for panel0; WSL available
space926 GiB at review, not reserved capacity.

Codex review scripts/results: results/oracle/l2_recursive_chance_development_p00_codex_review_20260930/.
Antigravity's original audit and raw artifacts are preserved. No solver/source
change or benchmark launch by Codex. Launcher and recursive-focused tests24 passed
in .92s, lint and whitespace checks pass; previous full implementation suite292
passed, and the eight launcher tests are additional focused checks.

Next Antigravity instructions: docs/RECURSIVE_CHANCE_DEVELOPMENT.md now continues
from panel1 through11 only (4,560 triples). Do not rerun panel0, refresh contracts,
overwrite telemetry, or tune after seeing outcomes. Complete fixed coverage unless
an operational/provenance failure requires stopping. Preserve all regression and
coverage evidence; no default promotion or fresh qualification follows. All
historical failed gates and .020 +1e-8 remain unchanged.
