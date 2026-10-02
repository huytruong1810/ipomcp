# Root chance scoring: paired development pilot (September 28)

Status: COMPLETED and independently audited. Preserve the instructions below as
the historical protocol; do not rerun into the completed output directory.
See latest docs/REVIEW.md for findings and reporting corrections. The tail-two fresh gate remains FAILED. This pilot
cannot qualify defaults or replace any historical gate.

## Frozen selection and execution

Read HANDOFF.md, docs/ROOT_CHANCE_PLAN.md and this protocol. Stay in the existing
/home/andyj1810/projects/ipomcp checkout on main. Do not clone or refresh the old
validation contract. The new contract benchmarks/root_chance_pilot_20260928.json
contains 14 complete saved rows and raw-file hashes: the fresh H8 failure, all
three earlier H6 errors, two adjacent belief cases for each error (all previously
correct at the same seed/horizon/opponent), and H1/H2 exact-boundary controls.
Cases were selected before running this implemented candidate. They are observed
development cases, not held out. Root budget remains 100,000; all task, tail-two,
opponent, horizon, reward, discount, exploration and seed settings come from each
unchanged saved row. Only exact_root_chance differs between paired arms.

From the WSL checkout, run once into the new output directory:

```bash
cd /home/andyj1810/projects/ipomcp
/usr/bin/time -v -o results/oracle/l2_root_chance_pilot_20260928.time \
  .venv/bin/python -m examples.experiments.root_chance_experiment \
  --input benchmarks/root_chance_pilot_20260928.json \
  --output results/oracle/l2_root_chance_pilot_20260928
```

The runner verifies every input hash and full row, and the locked current source
hashes, before creating outputs. It refuses an existing output directory. Preserve
the contract, manifest, raw pairs, logs, summary and external timer. Source mismatch
requires Codex review; do not regenerate hashes to bypass it. One supervised worker,
600 seconds per pair, 2048 MiB RSS. Each pair runs fresh control and candidate banks,
control first, with separate planner-policy timings. No broad sweep is authorized
by this protocol. Stop and report failed replay, tree identity, reference/task
identity, timeout, memory failure or other failed assertion. Do not overwrite or
silently retry partial evidence.

## Required audit and return

- Confirm all 14 pairs completed, exact full historical control-row identity,
  identical search tree digests (all statistics and ordered particle reservoirs),
  unchanged opponent and oracle vectors, and search_estimated_q equal to control Q.
- Recompute greedy ties, reference loss and Q error from each recommendation Q.
  Report strict errors (loss >1e-8 for continuity with prior audits), violations
  (loss > .020 +1e-8), corrected errors and newly regressed controls, case by case.
  Retain both raw Q vectors. Report margins between oracle-best and chosen actions.
- Tabulate every action's status, continuation probability, missing probability
  and missing branches; never call an empirical fallback exactly weighted. Report
  any fallback on either candidate's chosen action or the oracle-best action.
- Report both policy timings per pair, supervisor wall/RSS, total parent monotonic
  and Unix elapsed, and external timing. Policy timings exclude the reference and
  tree digest, but include candidate scoring. Control-first ordering, cache/load
  effects and this selected small panel prevent an isolated speedup claim.
- Return the results directory and timer location to Codex for audit. No defaults
  change or new qualification follows from even a perfect pilot. A broader observed
  development comparison requires review; a later qualification requires fresh
  unseen cases and the unchanged tolerance/zero-violation rule.

## Estimator contract

MCTSConfig.exact_root_chance defaults False and requires empirical_bellman. Its
True value is recorded in metadata but excluded from search seed settings. Default
serialization and default rows retain their prior shape. After search, each fully
covered action receives expected_reward + gamma * sum P(obs,alive) * saved_child_V.
The law is unconditional: terminal mass contributes zero, and survivor mass is
never renormalized. A missing child or child without action/rollout value causes
whole-action empirical fallback with explicit missing mass. Inference failures
propagate; no oracle is called by scoring. Exact root tail boundaries bypass this
step; H1 or gamma=0 needs only the expected reward.

The tree keeps empirical Q and all search statistics. Evaluator estimated_q and
policy describe the recommendation; search_estimated_q retains the search Q.
Tree diagnostics explicitly label empirical_search_q and do not attribute the
recommendation margin to empirical backups. This removes only the direct root
frequency discrepancy for fixed child estimates, not deeper chance error or the
adaptive estimation bias of those child values.

Engineering validation for the root-chance implementation: 268 tests passed in
397.70s, including ten new coverage, terminal-mass, tie, inference-failure,
configuration, unchanged-tree/RNG and paired-L2 replay checks. Ruff lint/format
and git diff whitespace checks pass. All 14 saved pilot input hashes and rows,
and its 76-file Python source lock, were verified without running the panel.
The original failed tail-two contract SHA256 remains
f47c2375cfb6787c2b629291c83f44a6a4dfa33f82c41b8cbeea16c1dd050e85.
These are engineering checks, not a candidate accuracy qualification.
