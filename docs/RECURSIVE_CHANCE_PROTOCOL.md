# Recursive chance integration: three-arm pilot — September 29

COMPLETED and reviewed September 29: 23/23 optimal, with incomplete deeper
coverage. Preserve this historical protocol; do not rerun. Read latest
docs/REVIEW.md before preparing the broader comparison. This is not a fresh validation gate. Do not promote defaults.
Stay in the existing /home/andyj1810/projects/ipomcp checkout on main; do not clone.

## Frozen selection

Contract: benchmarks/recursive_chance_pilot_20260929.json
SHA256: a843f1aaca898e9652eb7302248c8aad473e23ece97864e57121de8f468b26a4

Twenty-three cases, selected before recursive benchmark outcomes: all seven
previously diagnosed cases, their two adjacent saved belief cases at identical
seed/horizon/opponent, and H1/H2 boundary controls. All fourteen neighbors and two
boundary controls were optimal in both historical arms. No cases are removed or
substituted based on new outcomes. Each entry freezes the broad-comparison raw
file/hash and full empirical and root-only rows. The empirical baseline has four
strict errors and one violation; root-only has five strict errors and one violation.

The three arms are empirical tail-two, root-only exact chance, and recursive
post-search chance. Search settings, n_sims100k, horizon, beliefs, seeds, gamma,
full-history rewards, exact-tail boundary and modeled opponent remain unchanged.
The scorer option alone differs and is excluded from search seed derivation.
All three searches must have identical full tree/reservoir digests. Both historical
rows must reproduce exactly before interpreting the recursive arm. The runner
records a separate digest for each arm, not the full trees.

## Antigravity commands

Preflight reads all inputs and source hashes without launching an experiment:

```bash
cd /home/andyj1810/projects/ipomcp
.venv/bin/python -m examples.experiments.recursive_chance_experiment \
  --check-only --input benchmarks/recursive_chance_pilot_20260929.json \
  --output results/oracle/l2_recursive_chance_pilot_20260929
sha256sum benchmarks/recursive_chance_pilot_20260929.json
```

Run once only when the output directory, timer and console log do not exist:

```bash
cd /home/andyj1810/projects/ipomcp
set -e
base=results/oracle/l2_recursive_chance_pilot_20260929
if [ -e "$base" ] || [ -e "${base}.time" ] || [ -e "${base}.log" ]; then
  echo 'Preserve the existing attempt; stop for review' >&2
  exit 1
fi
/usr/bin/time -v -o "${base}.time" \
  .venv/bin/python -m examples.experiments.recursive_chance_experiment \
  --input benchmarks/recursive_chance_pilot_20260929.json --output "$base" \
  > "${base}.log" 2>&1
```

One supervised worker, 600 seconds and 2048 MiB per three-arm case (69 root
solves total). Limits include searches, references and digest serialization.
The recursive pass can be expensive because it conditions full private-history
beliefs across the supported observed tree; no large-budget runtime result exists
yet. Do not increase limits or exclude a timeout silently. The runner stops on
failed status/assertion; preserve all logs and partial artifacts and return to
Codex. No retry, overwrite, broad sweep or fresh gate is prescribed here.

The preflight verifies source hashes against this NEW contract. Earlier pilot,
broad-comparison and diagnostic source locks intentionally no longer match this
implementation. Preserve those contracts; never refresh their frozen hashes.

## Scoring and coverage contract

MCTSConfig.recursive_chance defaults False, requires empirical_bellman, and is
mutually exclusive with exact_root_chance. Both flags are absent from search seed
settings. The enabled scoring mode is recorded in policy metadata. Existing
empirical and root-only default row shapes remain unchanged.

The recursive pass computes expected rewards and integrates unconditional
P(observation,alive) against recursively scored saved child values. It uses only
production finite kernels and full Bayesian history models. Terminal mass is
not renormalized. It does not mutate tree counts, Q, particles, or random state,
add search visits, or query a reference solver. The memoization cache exists only
for one scoring pass and keys on tree node, immutable history model and remaining
horizon; scores from unrelated histories or solves are not reused.

A missing positive-probability branch or child without an estimate causes a
whole-action fallback to its original empirical Q. The pass does not partially
blend rescored branches into that action. It skips descendant scoring under that
fallback action. A saved rollout frontier remains an approximate value.
Unevaluated legal actions stay unevaluated and make the subtree approximate.
Inference-budget/conditioning failures propagate as failures, not repaired beliefs.
Existing exact-tail Q vectors are reused; H1 or gamma0 at evaluated action nodes
requires only expected reward. Do not label an unevaluated frontier exact.

Candidate estimated_q and policy describe final recommendations. Search Q is
stored separately as search_estimated_q. The recursive_chance report contains
per-history values, scores, legal-action gaps and per-action support/fallback
status, missing mass, approximate_child_probability and subtree_exact flags.
Its counts describe histories/actions actually evaluated by this pass, not every
node in the search tree. approximate_child_probability is immediate successor
mass reaching an uncertified subtree, not an error bound or total missing mass;
it is null for whole-action fallbacks that skip child scoring. Missing masses
at different histories must not be added as a root probability. A root is marked
exact only when all legal actions and all their positive branches recursively
reach exact boundaries. No dominance pruning is used to certify partial coverage.

## Required audit and return

Return the contract/manifest, all 23 raw case files, worker logs, summary, console
log and external timer, plus an audit script/report. Verify:

- Full control and root-only replay identity, three equal tree digests, unchanged
  oracle/opponent/task settings, and recursive search_estimated_q equal to both
  baseline search vectors. No oracle values enter the solver scoring pass.
- Greedy ties and reference loss/Q-error arithmetic for all three arms. Report
  errors >1e-8 and violations > .020 +1e-8, all corrections/regressions, mean/max
  loss and Q error, action margins, and each of the seven known difficult cases.
  Preserve every previously correct control in the comparison.
- Coverage by node/action and remaining horizon, every fallback and frontier,
  unevaluated-action histories, approximate subtrees, and root certification.
  Distinguish local exact probability integration from a fully exact subtree.
  If root_subtree_exact=true, check final Q against the declared finite reference
  up to numerical roundoff; a material disagreement is an implementation issue.
- Serial intervals and supervisor limits; external, Unix, parent monotonic and
  worker times; RSS measures; three per-arm policy timings (exclude reference
  and digest work). Fixed arm order and load prevent a causal overhead claim.
  Report clock disagreements without editing recorded times.

Even zero violations in this selected panel do not qualify defaults. Codex reviews
before extending to all observed cases or designing fresh validation. All failed
gates, default settings and the frozen tolerance remain unchanged.

Engineering validation, September 29: full suite 292 passed in 391.76s, including
16 new recursive integration/coverage/fallback/configuration/tree-invariance
and three-arm L2 checks. Ruff lint/format and git diff whitespace checks pass.
The 23-case pilot input rows, source lock and four preserved historical contract
hashes were verified without running the pilot. Both CLI help and no-experiment
preflight were checked. Benchmark accuracy and resource costs remain pending
Antigravity's declared run; passing unit/integration tests is not qualification.
