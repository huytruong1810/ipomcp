# September 23 correctness and resource review

The full experiment suite is not scientifically qualified. Corrected inference/search semantics expose substantial finite-budget planning error.

## Thirty-seed Tiger comparison

L3 versus L2, 80% L2 prior; 20 decisions; depth 5; real budgets 20,000/15,000; initial samples 2,000/1,500. All 30 trials completed on each side. The changes restore opponent updates and remove unsupported action pruning; this is a corrected-model comparison, not a pure speed optimization.

| Implementation | Mean I return | Mean J return | Mean trial wall s | Mean trial CPU s | Max RSS MiB |
|---|---:|---:|---:|---:|---:|
| before-planner | 21.43 | 18.13 | 38.14 | 42.36 | 192.75 |
| after-planner | -10.47 | -8.27 | 79.36 | 88.16 | 178.61 |

Paired corrected-minus-before differences (exploratory reused seeds):

- reward_i: -31.90; 95% t interval [-58.06, -5.74].
- reward_j: -26.40; 95% t interval [-55.38, 2.58].

No equivalence margin was prespecified. An interval containing zero does not establish equivalence. Action-dependent random draw consumption weakens per-step pairing. Concurrent verification jobs make wall times unsuitable for an isolated speed ratio.

## Matched oracle budget sweep

L1 against uniform L0; exact starting P(TL) in {0.1, 0.5, 0.9}; seeds 0–2; gamma .95; horizons 1–3. The two planners and oracle share the same model. All 189 cases completed. RTS enumerates all nine observation tokens. MCTS simulations and RTS particles per branch are distinct work units.

| Planner | Horizon | Budget | Cases | Mean first-action loss | Maximum loss | Mean max Q error | Mean wall s |
|---|---:|---:|---:|---:|---:|---:|---:|
| mcts | 1 | 1000 | 9 | 0.000000 | 0.000000 | 0.0000 | 0.028 |
| mcts | 1 | 10000 | 9 | 0.000000 | 0.000000 | 0.0000 | 0.068 |
| mcts | 1 | 50000 | 9 | 0.000000 | 0.000000 | 0.0000 | 0.255 |
| mcts | 2 | 1000 | 9 | 0.243833 | 2.194500 | 24.0279 | 0.031 |
| mcts | 2 | 10000 | 9 | 0.975333 | 2.194500 | 20.4747 | 0.270 |
| mcts | 2 | 50000 | 9 | 0.000000 | 0.000000 | 19.9763 | 1.428 |
| mcts | 3 | 1000 | 9 | 2.227785 | 3.341678 | 39.6917 | 0.059 |
| mcts | 3 | 10000 | 9 | 2.227785 | 3.341678 | 41.2894 | 0.556 |
| mcts | 3 | 50000 | 9 | 2.227785 | 3.341678 | 38.9635 | 2.694 |
| mcts | 3 | 100000 | 9 | 2.227785 | 3.341678 | 40.1394 | 6.209 |
| mcts | 3 | 500000 | 9 | 2.227785 | 3.341678 | 37.3118 | 30.538 |
| mcts | 3 | 1000000 | 9 | 2.227785 | 3.341678 | 36.4386 | 53.776 |
| rts | 1 | 100 | 9 | 0.000000 | 0.000000 | 3.5444 | 0.008 |
| rts | 1 | 1000 | 9 | 0.000000 | 0.000000 | 0.5867 | 0.016 |
| rts | 1 | 5000 | 9 | 0.000000 | 0.000000 | 0.5133 | 0.048 |
| rts | 2 | 100 | 9 | 0.243833 | 2.194500 | 3.0556 | 0.017 |
| rts | 2 | 1000 | 9 | 0.000000 | 0.000000 | 0.6382 | 0.130 |
| rts | 2 | 5000 | 9 | 0.000000 | 0.000000 | 0.5299 | 0.507 |
| rts | 3 | 100 | 9 | 0.000000 | 0.000000 | 3.3430 | 0.082 |
| rts | 3 | 1000 | 9 | 0.000000 | 0.000000 | 1.3933 | 0.745 |
| rts | 3 | 5000 | 9 | 0.000000 | 0.000000 | 0.4962 | 4.391 |

First-action loss is V*(b) − Σa π(a|b)Q*(b,a), with optimal continuation. It is not full policy regret. Zero observed loss on this small grid cannot certify global optimality. Persistent positive loss at high budgets blocks any claim that current default budgets reliably match the oracle.

## Corrected defects and remaining limits

- L2 hidden-state Bellman maximization replaced by observation-conditioned joint-belief recursion; uniform two-step value corrected from +8.5 to −1.95.
- Nearest-belief substitution and rounded L2 belief grid removed; H0 and invalid probability queries checked.
- Unsupported confidence-based tree pruning removed; all legal tree actions remain.
- Opponent private beliefs advance in every nonterminal rollout transition; final reward-only steps omit unnecessary propagation.
- Four misleading oracle/report scripts consolidated into one source-bound, supervised, matched comparison.
- L2 production/reference opponent horizon and computational models are not yet matched. L3/L4 exact optimality is not established.
- Full current 39-condition qualification remains pending, as do remaining demo/visualization semantic reviews.

Raw case/trial JSON and manifests are retained under this directory. Earlier interrupted `baseline` output is invalid qualification evidence. Historical favorable runs used different planner semantics and cannot validate the correction.

## Provenance and final verification

Planning checkpoint: 4e9ff4c; the after-run source hashes match every Python source
file at that Git commit. Before-run checkpoint: 35303a5, likewise hash-verified.
Engineering checkpoint 9e40997 additionally fixes clean worker startup and audit
budget/import wiring without changing the default planning model tested above.
The old audit manifest's modeled budget field says 10, but captured source uses
25; separate METADATA_ERRATUM.md files correct this without rewriting evidence.
The current driver records and executes the requested budget consistently.

Final verification: 165 tests passed (161 unit, 4 integration); lint/format,
source imports, clean wheel contents, prior/comparison audit CLI smoke and MCTS/RTS
spawn-worker smoke checks pass. Historical source lives in Git. There is only one
working checkout. The scientific accuracy and full-suite gates remain open.

## Exploration diagnosis, September 24

Antigravity's `results/oracle/diagnostic_boundary_{mcts,rts}_20260924`
contains 315 completed cases per planner, all source hashes matching the
pre-change checkout. At horizon three and budget 50,000, MCTS has positive
first-action loss in 12/35 cases; RTS has none at 5,000 particles per branch.
These are different work units, not a compute-matched superiority claim.
No separate Antigravity prose report was found in the checkout during this pass.

The new `--exploration normalized|standard|bounded` and `--exploration-const`
options apply only to the matched oracle runner's MCTS planner. `normalized`
retains the existing global empirical-return scale; `standard` uses raw reward
units; `bounded` uses the physical reward interval times the discounted remaining
horizon. The Tiger interval is exhaustively derived from both states, every joint
action, and transition support. It is not fitted to the oracle or sampled states.
Configuration is recorded in the manifest and actual strategy parameters in each
MCTS row. Production defaults and mean-return backups are unchanged.

A 70-case calibration panel is in `results/exploration-20260924`: horizon three,
10,000 simulations, seeds 0 and 1, beliefs .02/.1/.15/.5/.85/.9/.98, gamma .95,
two workers, 120 seconds and 1,024 MiB per case. All completed. The default replay
matches all original result fields exactly in its 14 overlapping Antigravity cases.

| Exploration | Coefficient | Wrong / 14 | Mean first-action loss |
| --- | ---: | ---: | ---: |
| Empirical range | 1 | 8 | 3.371483 |
| Empirical range | 0.1 | 2 | 0.676106 |
| Raw reward units | 10 | 4 | 1.518977 |
| Horizon reward bounds | 0.1 | 4 | 1.320253 |
| Horizon reward bounds | 1 | 4 | 0.954765 |

No setting passed every case. In particular, empirical coefficient .1 introduced
errors at beliefs .02/.98. These are reused calibration beliefs/seeds, not held-out
validation. Do not promote a new default or launch the full study from this table.

Why retain mean backups: Algorithm 1 of Silver and Veness (2010),
https://proceedings.neurips.cc/paper_files/paper/2010/file/edfbe1afcf9246bb0d40eb4d8027d90f-Paper.pdf,
uses incremental mean simulated returns. Poor finite-budget results do not alone
identify that estimator as an implementation defect. A Bellman backup using
maximal child action estimates is a different estimator and requires explicit
observation probabilities, history-conditioned values, and treatment of noisy
maximization. Maximizing over sampled hidden states or chance outcomes would
solve a different information problem. A principled Bellman variant remains a
possible future design, not an accepted one-line correction.
