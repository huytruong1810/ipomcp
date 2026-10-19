"""Experiment suites: many paired study cells run as one supervised, resumable job.

A ``SuiteSpec`` is a set of named cells, each a ``StudySpec``, run on the same
trial ids. Episodes are ``episodes.run_episode``, whose environment and planner
seeds depend only on the trial id, so every cell sees the same streams and
cells pair by trial (until their actions diverge).

``run_suite(spec, output)``:

1. writes ``manifest.json`` (the cells and the hashes of every source file); an
   existing output is resumed only if both match. Only ``trials`` may grow
   between runs, so a suite sized by a pilot extends in place;
2. schedules trials round-robin over cells (every cell at trial t before any at
   t + 1), so all cells hold about the same n whenever the run is stopped;
3. runs each episode in its own supervised process (wall, memory and source
   lock per episode; a killed episode loses only itself) and writes it
   atomically to ``trials/<cell>/<trial>.csv``; finished files are skipped.

Three suite builders cover the packaged Tiger suite (docs/PLAN.md, "Tiger
experiment suite"); each takes the domain as a parameter:

- ``level_matrix``: every pair of levels (i, j), level 0 included; each agent
  holds a uniform prior over the opponent's lower levels, at every nesting;
- ``planner_comparison``: i at level 2 with I-POMCP or IPF + RTS against the
  same real opponent;
- ``prior_sensitivity``: i's prior over the opponent's level, correct or
  misspecified, against one real opponent;
- ``anytime_comparison``: i at level 2 with every planner (I-POMCP, IPF + RTS,
  I-NTMCP) at several budgets against one real level-1 opponent, for anytime
  curves (RESULTS 12.67).

Every model of the opponent matches the real opponent's whole configuration
(depth, budgets, belief bound, nested priors; RESULTS 12.30), except where the
baseline under test itself models the opponent with its own method (RTS).

Command line: ``python -m ipomdp.experiments.suite <builder> --output DIR [options]``.
"""

import argparse
import json
import logging
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path
from uuid import uuid4

import pandas as pd

from ipomdp.belief import canonical
from ipomdp.experiments.episodes import run_episode
from ipomdp.experiments.runner import _read, _write_atomic, valid_panel
from ipomdp.experiments.study import AgentSpec, StudySpec
from ipomdp.infra.paths import REPOSITORY_ROOT
from ipomdp.infra.process_supervisor import supervise_jobs
from ipomdp.infra.source_lock import source_hashes

logger = logging.getLogger(__name__)

COORDINATION = {"together_safe": 40.0}


@dataclass(frozen=True)
class SuiteSpec:
    """Attributes:
    name: Suite name (recorded in the manifest).
    cells: {label: StudySpec}; each cell's own ``trials`` is ignored.
    trials: Trials per cell (ids 0 .. trials - 1, shared by every cell).
    """

    name: str
    cells: dict
    trials: int

    def __post_init__(self):
        if not self.cells:
            raise ValueError("A suite needs at least one cell")
        if type(self.trials) is not int or self.trials < 1:
            raise ValueError("trials must be a positive integer")
        for label in self.cells:
            if not label or "/" in label or label.startswith("."):
                raise ValueError(f"Cell label {label!r} is not a plain directory name")

    def to_dict(self):
        """Canonical form without ``trials`` (the one field a resumed suite may change)."""
        return {
            "name": self.name,
            "cells": {label: _without_trials(cell) for label, cell in sorted(self.cells.items())},
        }


def run_suite(spec: SuiteSpec, output, workers, episode=run_episode):
    """Run (or resume) every pending (cell, trial) of ``spec`` into ``output``.

    Returns {label: frame of that cell's finished trials}. Raises after the
    whole schedule if any episode failed; finished episodes stay on disk.
    """
    root = Path(output)
    lock = source_hashes(REPOSITORY_ROOT)
    manifest = {"suite": spec.to_dict(), "source_sha256": lock}
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != json.loads(json.dumps(manifest)):
            raise ValueError(
                "Existing output was produced by a different suite or source; use a new directory"
            )
    elif root.exists() and any(root.iterdir()):
        raise ValueError("Output directory is not empty and has no manifest")
    for label in spec.cells:
        (root / "trials" / label).mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))

    schedule = [
        (label, trial)
        for trial in range(spec.trials)
        for label in spec.cells
        if not trial_path(root, label, trial).exists()
    ]
    logger.info("%d episodes to run", len(schedule))
    jobs = {
        index: partial(episode, spec.cells[label], trial)
        for index, (label, trial) in enumerate(schedule)
    }
    limits = list(spec.cells.values())
    failures = []
    attempt = root / "attempts" / uuid4().hex
    for index, outcome in supervise_jobs(
        jobs,
        workers=min(workers, max(len(jobs), 1)),
        timeout_seconds=max(cell.trial_timeout_seconds for cell in limits),
        max_rss_mb=max(cell.max_trial_rss_mb for cell in limits),
        log_directory=attempt,
        source_lock=(REPOSITORY_ROOT, lock),
    ):
        label, trial = schedule[index]
        cell = spec.cells[label]
        rows = outcome.pop("rows", None)
        if outcome["status"] == "complete":
            frame = pd.DataFrame(rows)
            if valid_panel(frame, [trial], cell.steps):
                _write_atomic(frame, trial_path(root, label, trial))
            else:
                outcome.update(status="failed", reason="incomplete step panel")
        (attempt / f"{label}-{trial}.json").write_text(json.dumps(outcome, indent=2, default=str))
        if outcome["status"] != "complete":
            failures.append((label, trial, outcome["reason"]))
            logger.error("Episode %s/%s failed: %s", label, trial, outcome["reason"])
    if failures:
        raise RuntimeError(f"{len(failures)} episodes failed; finished ones are kept: {failures}")
    return load_suite(root)


def trial_path(root, label, trial):
    return Path(root) / "trials" / label / f"{trial:06}.csv"


def load_suite(output):
    """{label: frame} of every finished trial under ``output``, ordered by trial and step."""
    cells = {}
    for directory in sorted((Path(output) / "trials").iterdir()):
        paths = sorted(directory.glob("*.csv"))
        if paths:
            frame = pd.concat([_read(path) for path in paths], ignore_index=True)
            cells[directory.name] = frame.sort_values(["trial", "step"], ignore_index=True)
    return cells


def load_manifest(output):
    """The suite's cells as recorded: {label: StudySpec dict}."""
    return json.loads((Path(output) / "manifest.json").read_text())["suite"]["cells"]


# Builders ------------------------------------------------------------------------------------


def ipomcp(simulations, depth):
    """An I-POMCP planner entry whose own modeled queries use the same budget."""
    return {
        "kind": "ipomcp",
        "simulations": simulations,
        "depth": depth,
        "modeled_simulations": simulations,
    }


def hierarchy_agent(level, planner, modeled, particles, level_weights=None):
    """An agent at ``level`` with uniform priors over lower levels at every nesting.

    ``level_weights`` overrides the agent's own prior over the opponent's level;
    the nested models keep uniform priors, as every real agent of a level matrix
    holds, so a model of the opponent at level l matches the matrix's agent at l.
    """
    if level == 0:
        return AgentSpec(level=0)
    own = level_weights or {k: 1 / level for k in range(level)}
    if level == 1:
        return AgentSpec(level=1, particles=particles, planner=planner, level_weights=own)
    return AgentSpec(
        level=level,
        particles=particles,
        planner=planner,
        modeled=modeled,
        level_weights=own,
        nested_level_weights={n: {k: 1 / n for k in range(n)} for n in range(1, level)},
    )


def level_matrix(
    domain="tiger-joint",
    levels=(0, 1, 2, 3),
    *,
    depth=2,
    simulations=100,
    particles=64,
    steps=20,
    trials=100,
    domain_options=None,
    pairs=None,
    timeout_seconds=14400.0,
    max_rss_mb=3072.0,
):
    """Every (i, j) level pair (or only ``pairs``); cell labels ``i<level>_j<level>``.

    Defaults are RESULTS 12.27's design: horizon 2, where chance scoring makes
    every decision exact for the searched model at any budget, so a budget of
    100 only needs to try every action; modeled levels solve the same way.
    """
    planner = ipomcp(simulations, depth)
    pairs = pairs or [(li, lj) for li in levels for lj in levels]
    # Expensive cells first within each trial round, so stragglers start early.
    pairs = sorted(pairs, key=lambda pair: (-max(pair), -min(pair), pair))
    cells = {
        f"i{li}_j{lj}": StudySpec(
            name=f"{domain}-matrix-i{li}-j{lj}",
            domain=domain,
            domain_options=dict(domain_options or {}),
            trials=1,
            steps=steps,
            agents={
                "i": hierarchy_agent(li, planner, str(simulations), particles),
                "j": hierarchy_agent(lj, planner, str(simulations), particles),
            },
            trial_timeout_seconds=timeout_seconds,
            max_trial_rss_mb=max_rss_mb,
        )
        for li, lj in pairs
    }
    return SuiteSpec(name=f"{domain}-level-matrix", cells=cells, trials=trials)


def planner_comparison(
    domain="tiger-joint",
    *,
    depth=3,
    simulations=1000,
    opponent_simulations=1000,
    rts_samples=(2, 4),
    particles=64,
    steps=20,
    trials=100,
    domain_options=None,
    timeout_seconds=14400.0,
    max_rss_mb=3072.0,
):
    """i at level 2 with I-POMCP (cell ``ipomcp``) or IPF + RTS (``rts-k<K>``) against j at level 1.

    The real j is a level-1 I-POMCP agent with ``opponent_simulations``. I-POMCP's
    i models j with exactly that solve; IPF + RTS models every level with RTS at
    its own K, on the sampled IPF, as in Doshi & Gmytrasiewicz (2009).
    """
    opponent = hierarchy_agent(1, ipomcp(opponent_simulations, depth), None, particles)
    arms = {
        "ipomcp": hierarchy_agent(
            2, ipomcp(simulations, depth), str(opponent_simulations), particles
        )
    }
    for samples in rts_samples:
        rts = hierarchy_agent(
            2,
            {"kind": "rts", "depth": depth, "observation_samples": samples},
            str(samples),
            particles,
        )
        arms[f"rts-k{samples}"] = replace(rts, particle_filter="sampled")
    cells = {
        label: StudySpec(
            name=f"{domain}-planners-{label}-h{depth}",
            domain=domain,
            domain_options=dict(domain_options or {}),
            trials=1,
            steps=steps,
            agents={"i": agent, "j": opponent},
            trial_timeout_seconds=timeout_seconds,
            max_trial_rss_mb=max_rss_mb,
        )
        for label, agent in arms.items()
    }
    return SuiteSpec(name=f"{domain}-planner-comparison-h{depth}", cells=cells, trials=trials)


def anytime_comparison(
    domain="uav",
    *,
    depth=4,
    opponent_simulations=4000,
    ipomcp_simulations=(1000, 4000, 16000),
    rts_samples=(2, 4, 8),
    rts_filters=("sampled", "smoothed"),
    intmcp_simulations=(1000, 4000, 16000, 64000),
    intmcp_variants=("tuned",),
    particles=64,
    steps=20,
    trials=100,
    domain_options=None,
    timeout_seconds=14400.0,
    max_rss_mb=3072.0,
):
    """i at level 2 with every planner and budget against one real level-1 I-POMCP j.

    Cells:
    - ``ipomcp-<n>``: I-POMCP at root n with the matrix's uniform prior over j's
      level; ``ipomcp-pm-<n>`` with a point mass on level 1, I-NTMCP's model
      class. Both model j with exactly the real j's solve.
    - ``rts-<filter>-k<K>``: IPF + RTS on the sampled (published) or smoothed
      (repaired, 12.62) IPF, every level with RTS at K.
    - ``intmcp-<variant>-<M>``: I-NTMCP, which models j with its own nested trees.
    """
    opponent = hierarchy_agent(1, ipomcp(opponent_simulations, depth), None, particles)
    modeled = str(opponent_simulations)
    arms = {}
    for n in ipomcp_simulations:
        arms[f"ipomcp-{n}"] = hierarchy_agent(2, ipomcp(n, depth), modeled, particles)
        arms[f"ipomcp-pm-{n}"] = hierarchy_agent(
            2, ipomcp(n, depth), modeled, particles, level_weights={1: 1.0}
        )
    for particle_filter in rts_filters:
        for samples in rts_samples:
            rts = hierarchy_agent(
                2,
                {"kind": "rts", "depth": depth, "observation_samples": samples},
                str(samples),
                particles,
            )
            arms[f"rts-{particle_filter}-k{samples}"] = replace(
                rts, particle_filter=particle_filter
            )
    for variant in intmcp_variants:
        for m in intmcp_simulations:
            arms[f"intmcp-{variant}-{m}"] = AgentSpec(
                level=2,
                planner={"kind": "intmcp", "simulations": m, "depth": depth, "variant": variant},
            )
    cells = {
        label: StudySpec(
            name=f"{domain}-anytime-{label}-h{depth}",
            domain=domain,
            domain_options=dict(domain_options or {}),
            trials=1,
            steps=steps,
            agents={"i": agent, "j": opponent},
            trial_timeout_seconds=timeout_seconds,
            max_trial_rss_mb=max_rss_mb,
        )
        for label, agent in arms.items()
    }
    return SuiteSpec(name=f"{domain}-anytime-comparison-h{depth}", cells=cells, trials=trials)


def prior_arms(level, opponent_level):
    """{arm: i's prior over j's level}: correct, over- and under-estimated, uniform."""
    arms = {"correct": {opponent_level: 1.0}}
    if opponent_level + 1 < level:
        arms["over"] = {opponent_level + 1: 1.0}
    if opponent_level > 0:
        arms["under"] = {opponent_level - 1: 1.0}
    arms["uniform"] = {k: 1 / level for k in range(level)}
    return arms


def prior_sensitivity(
    domain="tiger-joint",
    level=3,
    opponent_level=1,
    *,
    depth=2,
    simulations=100,
    particles=64,
    steps=20,
    trials=100,
    domain_options=None,
    timeout_seconds=14400.0,
    max_rss_mb=3072.0,
):
    """i at ``level`` with each prior of ``prior_arms`` against the matrix's j at ``opponent_level``.

    The real j and every model of it are the level matrix's agents, so the
    ``uniform`` arm is the matrix cell (level, opponent_level) on the same seeds.
    """
    if not 0 <= opponent_level < level:
        raise ValueError("The opponent's level must lie below i's")
    planner = ipomcp(simulations, depth)
    opponent = hierarchy_agent(opponent_level, planner, str(simulations), particles)
    cells = {
        arm: StudySpec(
            name=f"{domain}-prior-l{level}-vs-l{opponent_level}-{arm}",
            domain=domain,
            domain_options=dict(domain_options or {}),
            trials=1,
            steps=steps,
            agents={
                "i": hierarchy_agent(level, planner, str(simulations), particles, weights),
                "j": opponent,
            },
            trial_timeout_seconds=timeout_seconds,
            max_trial_rss_mb=max_rss_mb,
        )
        for arm, weights in prior_arms(level, opponent_level).items()
    }
    return SuiteSpec(
        name=f"{domain}-prior-sensitivity-l{level}-vs-l{opponent_level}", cells=cells, trials=trials
    )


BUILDERS = {
    "level-matrix": level_matrix,
    "planner-comparison": planner_comparison,
    "anytime-comparison": anytime_comparison,
    "prior-sensitivity": prior_sensitivity,
}


def _without_trials(cell):
    data = asdict(cell)
    data.pop("trials")
    return canonical.stable_value(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("builder", choices=sorted(BUILDERS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--domain", default="tiger-joint")
    parser.add_argument("--coordination", action="store_true", help="together_safe = 40")
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--levels", type=int, nargs="+", help="level-matrix: levels of both agents")
    parser.add_argument("--pairs", nargs="+", help="level-matrix: only these i,j pairs, e.g. 4,2")
    parser.add_argument("--depth", type=int)
    parser.add_argument("--simulations", type=int)
    parser.add_argument("--opponent-simulations", type=int, help="planner-comparison")
    parser.add_argument("--rts-samples", type=int, nargs="+", help="planner-comparison")
    parser.add_argument("--level", type=int, help="prior-sensitivity: i's level")
    parser.add_argument("--opponent-level", type=int, help="prior-sensitivity")
    args = parser.parse_args()
    options = {"trials": args.trials, "domain": args.domain}
    if args.coordination:
        options["domain_options"] = COORDINATION
    for name in ("levels", "depth", "simulations", "opponent_simulations", "level"):
        if getattr(args, name) is not None:
            options[name] = tuple(args.levels) if name == "levels" else getattr(args, name)
    if args.rts_samples:
        options["rts_samples"] = tuple(args.rts_samples)
    if args.opponent_level is not None:
        options["opponent_level"] = args.opponent_level
    if args.pairs:
        options["pairs"] = [tuple(int(x) for x in pair.split(",")) for pair in args.pairs]
    logging.basicConfig(level=logging.INFO)
    spec = BUILDERS[args.builder](**options)
    cells = run_suite(spec, args.output, args.workers)
    print({label: int(frame["trial"].nunique()) for label, frame in cells.items()})


if __name__ == "__main__":
    main()
