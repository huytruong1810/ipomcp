# ipomdp: online planning for finitely nested interactive POMDPs

An I-POMDP agent plans while reasoning about another agent who reasons about it,
down to a finite nesting level. This package implements:

1. **An online planner (I-POMCP)** for finitely nested I-POMDPs. It searches the
   agent's own observation histories; the other agent is simulated by its own
   modeled planner at every level below. Nested beliefs are exact, or compressed
   from exact updates to a bounded support. After search the tree is scored with
   the exact observation law, and certified search keeps, at every history, an
   interval that contains the exact value of the searched model. The cost of
   searching futures and the cost of modeling minds are separate budgets.
2. **Exact references** for measuring planning error: the exact best response to
   declared opponent models (any level, any finite domain), exact I-POMDP agents
   built from it, and alpha-vector value iteration for two-state level-1 problems.
3. **Baselines:** the interactive particle filter with reachability-tree sampling
   (IPF + RTS, Doshi & Gmytrasiewicz 2009), and I-NTMCP (Schwartz, Zhou &
   Kurniawati 2022), whose authors' code is a pinned submodule.
4. **Interpretability and error measurement:** nested belief hierarchies (who
   believes what about whom, at what level), exact expected-reward curves,
   exact replays of recorded episodes, and Rao-Blackwellized episode rewards.

Domains: two-agent Tiger and its joint-payoff variants (Gmytrasiewicz & Doshi
2005), UAV reconnaissance (Doshi & Gmytrasiewicz 2009), a two-agent Wumpus world,
and Runner-Chaser (the I-NTMCP authors' domain). Specifications and sources are
in [docs/DOMAINS.md](docs/DOMAINS.md).

## Setup

Linux (including WSL) with Python 3.12+ and [uv](https://docs.astral.sh/uv/):

```bash
git submodule update --init          # third_party/i-ntmcp, the I-NTMCP baseline
uv sync --frozen --group dev
uv run pytest -m "not slow" -q       # fast tier, about 40 s
uv run pytest -q                     # full suite, about 9 minutes
uv run ruff check src tests
```

The tests check every layer against something that shares no code with it: the
domains against the published tables and exhaustive enumeration, the belief
filters against closed forms and brute-force Bayes updates, the exact references
against independent brute-force solvers, the planner's intervals against the
exact values at every history it expands, and the package's dependency rules
against [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Single decisions against the exact I-POMDP

A problem is an agent's history reachable from the common prior, stored with the
exact I-POMDP's action values; a decision row replays the history through one
planner and scores its first action. One process per job, one JSON file per
result:

```bash
uv run python -m ipomdp.experiments.reachable_decisions reference --domain uav --level 2 \
    --horizon 3 --seed 0 --steps 2 --out refs/uav_l2_h3_s0_k2.json
uv run python -m ipomdp.experiments.reachable_decisions decide --problem refs/uav_l2_h3_s0_k2.json \
    --planner mcts --budget 1000 --modeled 400 --seed 0 --profile certified --out rows/s0_k2.json
```

`--planner rts` runs IPF + RTS and `--planner intmcp` runs I-NTMCP on the same problem.

## Episodes

A study is a declarative specification (TOML, see `studies/`): domain, both
agents' levels, planners, belief representation and priors, trials and steps.
Suites of paired cells (level matrices, planner comparisons, prior sensitivity)
are built by `ipomdp.experiments.suite`.

```bash
uv run python -m ipomdp.experiments.runner studies/tiger_l3_vs_l2.toml --output results/tiger_l3_vs_l2
```

## Running many jobs on a shared machine

`ipomdp.infra.job_queue` runs a tab-separated list of one-process jobs with a
bounded number of workers and a budget on their total memory, skips jobs whose
output exists, stops starting jobs when a stop file appears, and records each
finished job's CPU, wall time and peak memory:

```bash
uv run python -m ipomdp.infra.job_queue jobs.tsv --workers 4 --log jobs.jsonl \
    --stop-file STOP --rss-cap-mb 6000 --min-available-mb 8000
```

## Layout

```
src/ipomdp/
  model/        the domain contract: exact kernels, rewards with bounds; sampling derived from them
  domains/      tiger, uav, wumpus, runner_chaser
  belief/       interactive states, canonical identity, exact, compressed and particle filters
  planning/     the I-POMCP planner, certified bounds and scoring, stopping rules, solver bank
  reference/    exact best response, exact I-POMDP agents, two-state alpha-vector iteration
  baselines/    IPF + RTS, and the adapter that runs I-NTMCP on these domains
  analysis/     hierarchies, metrics, endpoints, statistics, exact curves and replays, figures
  experiments/  study specs, episodes, suites, single-decision drivers, qualification drivers
  infra/        process supervision, job queue, source locks, resource measurement, paths
studies/        declarative study specifications (TOML)
tests/          contract, reference, exactness and architecture tests
third_party/    the I-NTMCP authors' code (submodule, pinned)
docs/           architecture, theory, domains, related work, equilibria
```

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) explains how the parts fit and why;
[docs/THEORY.md](docs/THEORY.md) states what is computed exactly, what is
estimated, and what "exact" means in each error measure.

## Results directory

Experiment outputs go to `results/`, which is not tracked. A run directory holds
the source it ran from or the hashes of that source, and runners skip outputs
that already exist, so an interrupted run resumes and a finished one is never
overwritten.
