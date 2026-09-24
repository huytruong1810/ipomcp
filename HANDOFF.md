# Current review handoff

## Decision and ownership

Antigravity runs experiments; Codex owns algorithm, math and code changes.
Use the single checkout `/home/andyj1810/projects/ipomcp`, branch
`fix/tiger-policy-inversion`. Do not edit source while a run is active, create
source copies, or pool runs from different source/configuration manifests.

Current action: run the fixed **new held-out L1 protocol** below. Candidate:
bounded c=1, exact final-step evaluation, 200,000 traversals. This is a candidate
at a newly declared resource budget, not a retrospective pass of the failed
50,000-traversal gate. Production defaults remain unchanged. Do not launch the
full experiment suite.

## Verified evidence and implementation

- Source checkpoint 582bf1a: 191 non-integration plus 4 domain integration tests
  passed; Ruff lint/format passed. Subsequent 32a01ce changed documentation only.
- All 180 development budget cases independently verified against coverage,
  source hashes, settings and recomputed first-action losses. H3 exact-tail had
  0/15 errors at 200k and 1M; sampled-tail had 10/15 at 200k and 0/15 at 1M.
- Those three beliefs (.08/.5/.92) all have **listen** as the reference-optimal
  action. This is evidence of progress on listening decisions, not validation of
  opening decisions, an exact value function, or an asymptotic convergence proof.
- H3 mean max Q error remains 15.238546 at 200k and 14.486203 at 1M for exact tail.
  Sampled mean backups are standard POMCP; the integrated boundary is a deliberate
  finite-model variant. See docs/THEORY.md and docs/BENCHMARK.md.
- Raw cases/manifests are local under `results/oracle/tail_budget_*_20260924`.
  Git tracks the documentation, **not** these ignored raw directories. Preserve
  the raw evidence; do not claim it has been committed or remotely archived.

Exact-tail mode uses full private-history joint posteriors and integrates only
the final decision. No benchmark oracle, sampled hidden state, actual opponent
action, or heuristic rollout belief enters that maximization. Earlier tree nodes
retain sampled mean backups. Finite-model enumeration may be costly at deeper
levels; L1 cost measurements do not qualify L2/L3/L4.

## Frozen new validation protocol

Candidate and control differ only in exact-final-step mode. Both use bounded c=1,
L1 against uniform L0, gamma .95, horizons 1/2/3 and **200,000 traversals**.
The fresh beliefs below and seed indices 200-219 were checked against local oracle
manifests: neither the belief points nor the seed range appeared there. Do not
inspect results and then change this selection, budget, coefficient or gate.

Run the two panels sequentially, with two supervised workers within each:

```bash
cd /home/andyj1810/projects/ipomcp
git status --short
git rev-parse HEAD
uv sync --frozen --group dev

uv run python -m examples.experiments.planner_oracle_experiment \
  --out results/oracle/tail_validation_exact_200k_20260924 \
  --planners mcts --exploration bounded --exploration-const 1 --exact-final-step \
  --horizons 1 2 3 --budgets 200000 --seed-start 200 --seeds 20 \
  --beliefs 0.03 0.07 0.075 0.085 0.11 0.25 0.75 0.89 0.915 0.925 0.93 0.97 \
  --workers 2 --timeout 2400 --max-rss-mb 4096

uv run python -m examples.experiments.planner_oracle_experiment \
  --out results/oracle/tail_validation_sampled_200k_20260924 \
  --planners mcts --exploration bounded --exploration-const 1 \
  --horizons 1 2 3 --budgets 200000 --seed-start 200 --seeds 20 \
  --beliefs 0.03 0.07 0.075 0.085 0.11 0.25 0.75 0.89 0.915 0.925 0.93 0.97 \
  --workers 2 --timeout 2400 --max-rss-mb 4096
```

Require a clean tree, pin HEAD and keep source fixed during both runs. Use fresh
output paths if these exist. Each panel has 720 cases (240 per horizon), 1,440
total. Record actual panel elapsed time separately from summed case wall time.
Two 4-GiB worker ceilings need additional memory headroom for parent/OS. RSS is
sampled, not an instantaneous allocation cap; worker startup counts toward limits.

Gate, fixed before execution: every requested case completes, and every candidate
case has first-action loss <=1e-8. The tolerance is numerical, not a scientific
reward-equivalence margin or a population guarantee. Keep all failures. Report
per-belief/per-horizon decisions, exact optimal action sets, first-action losses,
all Q errors, wall/RSS cost and raw paths; explicitly separate opening and
listening decisions. Do not average away a failing belief or replace 200k by a
higher-budget result. The control is reported in full, not used to relax the gate.

A pass qualifies only this observed L1 action panel. It does not certify Q-value
accuracy, optimal episode returns, a general convergence rate or higher levels.
A failure returns the candidate to development; these points then cease to be an
untouched validation set. Do not tune repeatedly on them.

## Remaining gates and maintenance

The previous 50k held-out test failed (80/720 candidate errors); that outcome
stands. Before a full study we still need matched L2 opponent horizon/computation
semantics, long-horizon/deep Tiger before/after evidence, L4 and all 39 production
conditions at intended resources. A 200k L1 candidate does not validate current
10k-25k real-agent or default 25-simulation modeled-agent budgets.

Configuration is part of the deterministic search seed. In particular, adding
the exact_final_step field changed streams even when false. Compare both modes
at the same source checkpoint; same seed index is not a promise of common random
numbers across configurations. Historical source lives in Git, not copied trees.
Old prior audit manifests have a documented 10-versus-25 modeled-budget erratum;
consult their METADATA_ERRATUM.md rather than rewriting captured evidence.

Update current claims in this file, BACKLOG.md and docs/BENCHMARK.md together.
Keep HANDOFF.md as the current protocol; historical instructions remain in Git.
Never describe finite decision agreement as exact values or a convergence proof.

## Antigravity 200k validation execution report, September 24

Both 200k validation panels (1,440 total cases, 0 failures) completed under `results/oracle/tail_validation_exact_200k_20260924` (Candidate) and `results/oracle/tail_validation_sampled_200k_20260924` (Control) at commit `ccf99b6`.

Decision gate outcome: **FAILED (candidate returns to development)**.
- At 200,000 traversals (N=720 cases: 240/horizon), Candidate achieved 0 errors across Horizons 1 and 2 (0/480 wrong, loss 0.0000, mean Q error 0.01 at H2).
- However, at Horizon 3, Candidate failed 20/20 at each of the four razor-thin inflection beliefs ($P \in \{0.070, 0.075, 0.925, 0.930\}$), yielding 80 / 720 overall wrong choices (11.11%, mean loss 0.062899).
- Control had 85 / 720 wrong choices (11.81%, mean loss 0.075592), additionally failing in 5 cases across .085 and .915 where Candidate was 100% correct.
- Measured panel elapsed times: 2,870.3s (Candidate) and 3,000.1s (Control); peak monitored RSS was 76.5 MiB.
- Detailed tables are committed in `docs/BENCHMARK.md` and `results/oracle/TAIL_VALIDATION_200K_REPORT.md`.

