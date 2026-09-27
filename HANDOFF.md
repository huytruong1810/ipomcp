# Current review handoff

## Workflow

Work in /home/andyj1810/projects/ipomcp on main. The prior fix branch was merged,
pushed and deleted locally/remotely. Create branches only for separately
justified side work. Codex owns coding/math; Antigravity runs declared studies.
Preserve raw artifacts and failed gates. Competing CPU/GPU workloads from
another codebase confound timing and resource failures.

## Audited 100k development result

The 1,440-case H5/H6/H8 study at source aa8963b completed and is independently
audited. Exact coverage, source hashes, complete configurations, probabilities,
loss/Q-error arithmetic and worker intervals are verified. All paired modeled
policies and oracle Q vectors match the 50k baseline. Independently recomputed
all six remaining error references; other references were checked for matching
and arithmetic, not independently rerun in this audit.

| Metric | 50k | 100k |
| --- | ---: | ---: |
| Strict errors /1,440 | 15 | 6 |
| Loss >.020 +1e-8 | 9 | 3 |
| Mean loss | .000716953 | .000083521 |
| Maximum loss | .205046 | .048062 |
| Mean maximum Q error | .821262 | .771836 |

All 15 old errors meet tolerance at 100k; 13 become strictly optimal. Four
previously optimal cases become errors, including all three current violations.
All seven old H8 violations were corrected; the current H8 violation is new.

| Modeled budget | b_j | H | Seed | b_i | Loss |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 25 | .085 | 5 | 7001 | .0375 | .031063194 |
| 100 | .5 | 8 | 7003 | .0275 | .020122447 |
| 100 | .915 | 6 | 7004 | .9625 | .048061971 |

The fourth new error is H8, budget 100, b_j=.915, seed 7005, b_i=.0275,
loss .009166912. Two old H6 near ties remain at .005156 and .006699.
Do not round the .020122447 violation into a pass.

H8 mean loss improves substantially, but H6 mean/max loss increases.
The results do not establish resolved continuation variance or a universal
boundary-margin cutoff. Changing traversal budget also changes deterministic
root seeds, so these solves are not nested prefixes or common-random-number
comparisons.

## Timing and interpretation

Raw external/parent-monotonic/worker totals:
11,227.72 /11,224.411573 /22,031.377033 seconds. All six two-worker consistency
checks pass. External and parent clocks differ by 3.308427s in total.
This does not prove isolated execution or resolve historical timing issues.
Peak RSS ranges 77.93–156.41 MiB rather than remaining invariant.

The user reports concurrent CPU/GPU demand. Preserve raw costs without
attributing cross-run ratios solely to the algorithm. Worker time includes
reference calculation. Full-panel external timers cannot provide an exact
external elapsed time for the H5/H6/H8 subset of the older baseline.

## Decision and next work

Keep exact_history_rewards opt-in and retain current defaults.
100k improves the tested distribution but still misses the all-case .020
criterion. The 50k-shallow/100k-deep rule therefore is not qualified either.
This development study cannot repair the historical ac3df1c or f4d2aef gates.

Next Codex analysis should reproduce and decompose the three new violations,
checking root chance terms and continuation-value errors before proposing
another estimator, exploration setting or resource rule. Do not adapt depth
to change the benchmark objective, select only successful reruns, or raise
the tolerance. No new large experiment or fresh gate is prescribed by this
documentation update. Antigravity should preserve evidence pending the next
concrete development protocol.

Raw runs: results/oracle/l2_rewards_100k_n{25,100}_b{0.085,0.5,0.915}_20260926/.
Audit: results/oracle/l2_rewards_100k_review_20260926.json.
Full verified summary and distributions: docs/BENCHMARK.md.
Raw files are local and Git-ignored; documentation commits do not archive them.

No solver changes in this checkpoint. Last implementation verification remains
241 passing tests. Ruff and documentation whitespace checks passed for this
update; the implementation suite was not redundantly rerun. All 15 overlapping
100k cases from the earlier targeted probes reproduced identical result rows.
