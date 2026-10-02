# ipomdp: online planning for finitely nested interactive POMDPs

An I-POMDP agent plans while reasoning about another agent who reasons about it,
down to a finite nesting level. This package implements:

1. **An online sampling-based I-POMDP planner (I-POMCP).** It combines nested
   Monte Carlo tree search, where opponents are simulated by their own modeled
   planners, with either exact interactive belief updates or the interactive
   particle filter (IPF). The cost of searching futures and the cost of
   modeling minds are separate, measured budgets.
2. **Exact references** for measuring planning error: the exact best response
   to declared opponent models (any level, any finite domain), exact I-POMDP
   agents built from it, and alpha-vector value iteration for level 1.
3. **Interpretability data:** nested belief hierarchies (who believes what about
   whom, at what level), recorded every step for sunburst views.

Domains: two-agent Tiger (Gmytrasiewicz & Doshi, 2005), UAV pursuit, and a
two-agent Wumpus world. Their specifications are in [docs/DOMAINS.md](docs/DOMAINS.md).

## Setup

Linux (including WSL) with Python 3.12+ and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --frozen --group dev
uv run pytest -q          # about 45 s; includes the L1 convergence gate
uv run ruff check src tests && uv run ruff format --check src tests
```

## Qualification first, scale second

No study runs at scale before its small-scale behaviour is verified against an
exact reference. The gates, in order:

```bash
# Planner vs exact value iteration (L1 Tiger, horizons 1-5)
uv run python -m ipomdp.experiments.l1_qualification --output results/l1_qualification
# IPF beliefs vs exact beliefs along the same histories (L2/L3 Tiger)
uv run python -m ipomdp.experiments.ipf_qualification --output results/ipf_qualification
# Planner vs exact best response or exact I-POMDP on matched problems (L1/L2 Tiger)
uv run python -m ipomdp.experiments.oracle --out results/oracle --budgets 1000 --horizons 3 --seeds 5 --beliefs 0.2 0.5
# Root search vs opponent modeling at matched cost, vs the exact I-POMDP (L2 Tiger)
uv run python -m ipomdp.experiments.allocation_study --output results/allocation_study
```

Current results, each with its sample size, are in [docs/RESULTS.md](docs/RESULTS.md).

## Layout

```
src/ipomdp/
  model/        the domain contract: exact kernels, rewards with bounds; sampling derived
  domains/      tiger, uav, wumpus
  belief/       interactive states, canonical identity, exact filter, particle filter
  planning/     I-POMCP planner, exploration, history nodes, solver bank
  reference/    exact best response, exact I-POMDP agents, L1 alpha-vector iteration
  experiments/  hierarchy builder, episode runner, qualification and study drivers
  analysis/     statistics, error attribution, figures
  infra/        process supervision, source locks, resource measurement, paths
tests/          contract, unit, reference and qualification tests
docs/           architecture, theory, domains, results, handoff; review/ records the
                review that preceded the current design
```

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) explains how the parts fit and why;
[docs/THEORY.md](docs/THEORY.md) states what is computed exactly, what is
estimated, and what "exact" means in each error measure.

## Results directory

Experiment outputs go to `results/` (git-ignored). Each run directory holds a
manifest with the source hashes it ran under; a run never overwrites an
existing directory. Back up `results/` outside the working machine.
