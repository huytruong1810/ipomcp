# Current review handoff

## Workflow

Work in /home/andyj1810/projects/ipomcp on main; do not clone. Codex owns coding,
mathematics and independent audits. Antigravity runs the declared experiments.
Preserve raw artifacts and failed gates. Competing CPU/GPU workloads confound
isolated timing claims. Current changes are in the working tree based on 62d6cd8;
source hashes, not that base commit alone, identify the candidate implementation.

## Current checkpoint: recursive broad panel0 reviewed, continue panels1–11

The existing Antigravity-created master was reviewed and accepted unchanged.
All 4,800 inputs/source hashes match the intended exhaustive observed comparison.
Panel0 is complete and audited:240/240 optimal recursive choices, no regressions,
root-only case103 corrected;70 fully exact roots, with substantial descendant
fallback/frontier coverage elsewhere. Three overlapping pilot cases reproduce
exactly. See latest docs/REVIEW.md for coverage, Q errors and timing discrepancy.

Antigravity now follows docs/RECURSIVE_CHANCE_DEVELOPMENT.md, starting at panel1,
then2–11 sequentially (4,560 triples). Preserve panel0 and all its telemetry;
do not rerun it. Use the existing frozen master hash
7a1bb7bdf8e53a1d6c74664e66d94a572fa6132cced770c53e4ca7d9a1059c63.
One worker,600s/2048 MiB per triple; no source/configuration changes or silent
retries. Defaults, all historical failed gates and tolerance remain unchanged.
Codex review: results/oracle/l2_recursive_chance_development_p00_codex_review_20260930/.

## Active result: fresh tail-two gate FAILED (September 28)

All 3,360 declared cases completed. Independent audit verified the original
frozen contract hash, launcher/source hashes, all six freeze sidecars, exact
Cartesian coverage, full modeled/root settings, greedy policies, loss/Q-error
arithmetic, summary consistency, per-worker intervals and sequential panels.
The single error's reference Q vector was independently recomputed exactly.
Other reference vectors were checked for arithmetic/configuration consistency,
not independently recomputed in this audit.

The unchanged zero-violation criterion FAILED: one loss .062316559843681674
exceeds .020 +1e-8. There is also exactly one strict error. H1–H6 have 2,880/2,880
optimal choices; H8 has 479/480. No global-default promotion follows.

Failure: panel0/case486, modeled budget25, b_j=.085, H8, seed8000, b_i=.966.
The planner chooses OR, oracle best is L. Since b_i denotes P(Tiger Left),
.966 is strong belief in Tiger Left, not Tiger Right as the walkthrough says.
Oracle L=10.206540834891277, OR=10.144224275047595. Estimated OR-minus-L margin
is +.002956129650402; estimation distorts that margin by +.065272689494083.
The large OL error -2.521602510 does not determine the competing OR/L ranking.
The completed deeper trace is summarized below; root errors alone were not
sufficient to establish that attribution.

Overall mean loss .000018546595, mean maximum Q error .222224762, maximum Q
error 5.418270410. This fresh design changes beliefs and finite-policy seeds;
comparisons with older gates are descriptive, not matched causal improvements.

Raw external / parent monotonic / worker totals are
46,744.00 / 14,031.381507 / 27,464.300930 seconds. Unix endpoint differences sum
to 46,741.252978s and track external clocks more closely. All interval/concurrency
checks pass. Panel2 and panel5 have large clock gaps (3,216.41 / 29,227.05s).
Antigravity reports host sleep/load, but the artifacts do not independently
establish causes. Preserve raw timings; no isolated runtime claim. Worker RSS
ranges 73.79–128.19 MiB; 121.96–128.19 is only the range of panel maxima.

## Completed single-case diagnostic and next code direction

Antigravity's replay exactly matches the full original row and frozen source.
All 243 relevant histories were expanded, with zero pending/unresolved terms.
The OR-minus-L estimation error .065272689494 decomposes as:
chance +.064423122423, signed local action loss +.001444464425,
rollout frontier -.000594897354, immediate/tail terms at numerical roundoff.
All 121 H2 boundaries match reference within 1.78e-15. Chance dominates this
saved-tree attribution; do not call it the sole remaining failure mechanism.

Root counts L=49,752 and OR=49,781 exclude gross root allocation starvation
here. Largest incremental chance terms are at remaining H6/H5
(+.022947233 / +.022218160); root chance contributes +.011548169.
One six-visit H3 decision has local loss .722849, contributing +.001444464.
A separate H3 rollout frontier contributes -.000594897. These are signed
algebraic contributions, not a statistical variance or universal causal proof.

Replacing root frequencies by exact probabilities while holding all saved child
estimates fixed gives L=10.171595006 versus OR=10.163081977: the post-hoc choice
would be correct. The same calculation on the six older tail-two diagnostic
trees corrects their one residual near tie and preserves five correct choices.
All seven are selected cases, not qualification or an implemented solver arm.

The opt-in exact_root_chance candidate and paired replay runner are implemented.
The completed pilot protocol is docs/ROOT_CHANCE_PROTOCOL.md, with contract
benchmarks/root_chance_pilot_20260928.json. Do not rerun it or start a new
qualification or broad sweep from those completed instructions. It checks exact full control-row replay and full
search-tree identity before interpreting candidate recommendations. Missing
positive continuation mass triggers explicit whole-action empirical fallback.
Search Q and recommendation Q remain separate. Defaults remain unchanged.
The old tail-two frozen source lock intentionally no longer matches this new
implementation; preserve its contract and failed result without refreshing it.

Diagnostic timing: external30.31s / parent30.066639s / worker29.405550s;
monitored worker peak RSS130.636719 MiB. Diagnostic costs include references.
Audit: results/oracle/l2_tail2_fresh_failure_review_20260928.json.
Counterfactuals: results/oracle/l2_tail2_root_chance_counterfactual_20260928.json.

## Independently audited tail-two development result

All 1,440 H5/H6/H8 rows are complete. Verified Cartesian coverage, every source
hash, full settings, policy normalization/greedy ties, loss/Q-error arithmetic,
paired oracle Q and modeled-policy metadata, summary consistency and worker
intervals. All three residual-error references were independently recomputed
and agree exactly. Other references were checked for paired equality, not all
independently recomputed. The six deeper control rows reproduce their originals;
all six diagnostic candidate rows equal the corresponding full-panel rows.

| Metric | 100k tail one | 100k tail two |
| --- | ---: | ---: |
| Strict errors /1,440 | 6 | 3 |
| Loss >.020 +1e-8 | 3 | 0 |
| Mean first-action loss | .000083521 | .000011368 |
| Maximum loss | .048061971 | .006057959 |
| Mean maximum Q error | .771836 | .491868 |
| Maximum Q error | 4.447722 | 6.225103 |

Five old errors become optimal. Two previously optimal cases regress to small
H6 errors; one old H6 error persists. H5 and H8 each have 480/480 optimal choices.
These are development results on the observed grid, not a fresh validation pass.
The worse maximum Q error precludes a blanket value-accuracy improvement claim.

All residual errors choose L when OR is optimal:

| Modeled budget | b_j | H | Seed | b_i | Loss |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | .085 | 6 | 7000 | .9625 | .006057959 |
| 100 | .5 | 6 | 7003 | .9725 | .005155991 |
| 100 | .5 | 6 | 7004 | .9725 | .005155991 |

The last estimated L-minus-OR margin is .020159332, not the oracle loss .005155991.
Keep true loss and estimated action margin distinct.

## Deeper mechanism and implemented candidate

Read docs/CONTINUATION_PROTOCOL.md and the latest docs/BENCHMARK.md section.
In the expanded control-tree attribution, chance error dominates cases 0–4.
Case 5 is dominated by local action loss: an H3 node has only L/OL visited once
each and chooses OL after L receives a -100.95 H2 frontier estimate. Its actual
local loss is 48.228569, contributing +.036951 to the root margin error. Replacing
that saved H2 estimate by its reference value would reverse that local ranking;
this is a post-hoc diagnostic, not an unchanged-trajectory causal experiment.

The opt-in exact_tail_depth=2 boundary uses production reward and observation
kernels and full recursive history posteriors. It optimizes after conditioning,
never on sampled hidden state. It removes estimator and rollout errors inside
two remaining decisions, not all higher-tree errors. Candidate traces contain
311 exact H2 boundaries, maximum absolute reference discrepancy 1.78e-15.
Baseline H8 traces stop at 256 expansions with zero remaining signed/absolute
value residual, but still have unexpanded zero-error histories; internal
cancellations inside those histories are not resolved. See audit limitations.

Default tail depth one is omitted from canonical seed/config serialization to
preserve historical finite policies. The candidate's nondefault depth is explicit
and changes protagonist seeds/trajectories. This is not a common-random-number
or nested-prefix comparison. Opponent budgets/depth/policies are preserved.

## Development resources, provenance and report corrections

Raw external / parent monotonic / worker totals:
9,711.66 / 9,708.448361 / 19,059.712159 seconds. All six two-worker concurrency
bounds and interval checks pass. Worker peak RSS ranges 76.55–127.30 MiB.
The raw report's 126.8/129.8 MiB summaries are inconsistent with these records.
Measured elapsed is lower than the old run; concurrent workloads and reference
work in worker durations preclude an isolated 13.5% solver speedup claim.
The applicable concurrency bound is worker sum <= 2 * parent elapsed, with
actual interval overlap <=2, not worker sum >= parent elapsed.

Antigravity's walkthrough reports base 62d6cd8 and unmodified source; the actual
candidate is a hashed modified working tree. Every recorded source hash matches
the current source. Preserve this distinction when committing or reporting it.

## Frozen validation record

The completed protocol is docs/TAIL2_VALIDATION.md and the unchanged machine
contract benchmarks/tail2_validation_20260927.json. Contract SHA256:
f47c2375cfb6787c2b629291c83f44a6a4dfa33f82c41b8cbeea16c1dd050e85.
Preserve all six output directories, freeze sidecars and external logs/timers.
This failed gate joins the historical ac3df1c/f4d2aef failures; it does not replace
them. Defaults, the .020 tolerance, numerical slack and zero-violation rule remain
unchanged. The pre-run novelty audit remains evidence about protocol selection,
not permission to reuse the now-observed cases as held out.

## Artifacts and engineering checks

Fresh audit: results/oracle/l2_tail2_fresh_review_20260928/ (audit.py, audit.json,
reference_checks.json). Fresh panels and freeze sidecars:
results/oracle/l2_tail2_fresh_n{25,100}_b{0.085,0.5,0.915}_20260927[.freeze.json].

Development raw panels: results/oracle/l2_rewards_100k_tail2_n{25,100}_b{0.085,0.5,0.915}_20260926/.
Deep controls/candidates: results/oracle/l2_deeper_{control,tail2}_20260926/.
Independent audit: results/oracle/l2_tail2_review_20260927/ (coverage/arithmetic
script and audit, diagnostic reconstruction, three reference checks, novelty).
Raw evidence is local and Git-ignored; documentation does not archive it.

Full implementation suite: 254 passed in 558.80s. Four additional regression
tests were then added; all 17 tail/diagnostic focused tests passed, covering those
additions (258 tests now collected). Ruff lint/format and diff whitespace passed.
The frozen launcher help/preflight configuration were checked without launching
validation. No broad test repetition is needed unless code changes again.

September 28 audit changed only notes and added independent audit artifacts.
Solver, frozen contract and launcher are unchanged; no test-suite rerun or
protagonist planner replay was performed. Whitespace and source-lock checks pass.

September 28 implementation follow-up: final root chance scoring now uses the
production continuation law after search, with no tree mutation or oracle input.
See docs/ROOT_CHANCE_PROTOCOL.md for the operational handoff and audit checklist.
The pilot is now audited at the current checkpoint above; the earlier saved-tree
corrections remain distinct post-hoc evidence.

Engineering validation for the root-chance implementation: 268 tests passed in
397.70s, including ten new coverage, terminal-mass, tie, inference-failure,
configuration, unchanged-tree/RNG and paired-L2 replay checks. Ruff lint/format
and git diff whitespace checks pass. All 14 saved pilot input hashes and rows,
and its 76-file Python source lock, were verified without running the panel.
The original failed tail-two contract SHA256 remains
f47c2375cfb6787c2b629291c83f44a6a4dfa33f82c41b8cbeea16c1dd050e85.
These are engineering checks, not a candidate accuracy qualification.

Engineering validation, September 29: full suite 292 passed in 391.76s, including
16 new recursive integration/coverage/fallback/configuration/tree-invariance
and three-arm L2 checks. Ruff lint/format and git diff whitespace checks pass.
The 23-case pilot input rows, source lock and four preserved historical contract
hashes were verified without running the pilot. Both CLI help and no-experiment
preflight were checked. Benchmark accuracy and resource costs remain pending
Antigravity's declared run; passing unit/integration tests is not qualification.
