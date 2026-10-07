"""Wumpus L1 qualification: compressed search beliefs x simulations, vs the exact best response.

The human (L1) faces a uniform-random wumpus (L0) on the 4 x 4 board with its
exact 94,500-state prior (RESULTS.md Section 10). Real beliefs are exact
(filtering is cheap: deterministic sensors prune the support fast); each search
starts from the belief compressed to ``search_particles`` atoms. The Section 10
pilot lost 1.8-7.7 in 13 of 18 decisions without separating the two suspects:
root compression (N) and the simulation budget over 5 actions x 32
observations. This grid separates them.

Problems: a scripted history (turn, step, turn, step, step; a step that would
end the episode is replaced by a turn) cut after 0, 2 or 4 steps, per seed. The
exact H-step best response at the exact belief (``BestResponseReference``,
520-600 s at the prior) is computed once per problem and reused by every cell.
Each row records the decision's loss against it, the optimal-action and max Q
errors, the width of the planner's certified bounds (certified for the
*searched*, compressed belief, so not guaranteed to contain this reference),
and CPU. One JSON line per (problem, cell); restarts skip finished rows.

Usage:
    python -m ipomdp.experiments.wumpus_qualification --output DIR \\
        [--seeds 3] [--steps 0 2 4] [--search-particles 64 256 1024 4096] \\
        [--simulations 1000 3000 10000] [--horizon 3] [--workers 6]
"""

import argparse
import itertools
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ipomdp.domains.wumpus.model import WumpusModel
from ipomdp.experiments.bootstrapper import HierarchyBuilder
from ipomdp.experiments.oracle import REFERENCE_BRANCHES, optimal_q_error
from ipomdp.planning.config import PlannerConfig
from ipomdp.planning.solver_bank import SolverBank
from ipomdp.reference.best_response import BestResponseReference

HISTORY = ("TurnLeft", "Forward", "TurnRight", "Forward", "Forward")
GAMMA = 0.95


def history(physics, steps, seed):
    """The human's first ``steps`` (action, observation) pairs; the wumpus acts at random."""
    rng = random.Random(f"wumpus/{seed}")
    state = rng.choices(*zip(*physics.initial_distribution()))[0]
    out = []
    for action in HISTORY[:steps]:
        joint = {"i": action, "j": rng.choice(physics.get_all_actions("j"))}
        following = physics.sample_transition(state, joint, rng)
        if physics.is_terminal(following):
            # History generation only: turn instead of ending the episode.
            action = joint["i"] = "TurnLeft"
            following = physics.sample_transition(state, joint, rng)
        state = following
        out.append((action, physics.sample_observation(state, joint, "i", rng)))
    return out


def human(physics, seed, simulations, horizon, search_particles):
    """The L1 human with exact real beliefs and searches on ``search_particles`` atoms."""
    config = PlannerConfig(
        simulations=simulations, depth=horizon, modeled_simulations=simulations, gamma=GAMMA
    )
    bank = SolverBank(seed, search_particles=search_particles)
    return HierarchyBuilder(bank).build("i", 1, physics, "j", config=config, level_weights={0: 1.0})


def reference(seed, steps, horizon):
    """Exact Q of the problem: the H-step best response at the exact belief."""
    physics = WumpusModel()
    planner = human(physics, seed, 1, horizon, None)
    for action, observation in history(physics, steps, seed):
        planner.update_root(action, observation)
    started = time.process_time()
    exact = BestResponseReference(physics, SolverBank(seed), GAMMA, max_branches=REFERENCE_BRANCHES)
    q = exact.q_values(planner.model(), horizon)
    return (
        {str(a): v for a, v in q.items()},
        time.process_time() - started,
        len(planner.belief.mass),
    )


def evaluate(seed, steps, search_particles, simulations, horizon, oracle):
    physics = WumpusModel()
    planner = human(physics, seed, simulations, horizon, search_particles)
    for action, observation in history(physics, steps, seed):
        planner.update_root(action, observation)
    started = time.process_time()
    planner.policy_for(planner.model())
    cpu = time.process_time() - started
    values = {str(a): v for a, v in planner.recommendation_values.items()}
    best = max(oracle.values())
    chosen = max(values, key=values.get)
    bounds = planner.scores.bounds
    return dict(
        seed=seed,
        steps=steps,
        search_particles=search_particles,
        simulations=simulations,
        horizon=horizon,
        loss=best - oracle[chosen],
        optimal_q_error=optimal_q_error(values, oracle),
        max_abs_q_error=max(abs(values[a] - oracle[a]) for a in oracle),
        bound_width=max(hi - lo for lo, hi in bounds.values()),
        exact_root=planner.scores.exact,
        cpu_seconds=cpu,
        values=values,
        oracle_q=oracle,
    )


def _reference_job(args):
    return args, reference(*args)


def _evaluate_job(args):
    return evaluate(*args)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, nargs="+", default=[0, 2, 4])
    parser.add_argument("--search-particles", type=int, nargs="+", default=[64, 256, 1024, 4096])
    parser.add_argument("--simulations", type=int, nargs="+", default=[1000, 3000, 10000])
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    references, rows = args.output / "references.jsonl", args.output / "rows.jsonl"
    problems = list(itertools.product(range(args.seeds), args.steps))
    known = {}
    if references.exists():
        for line in references.read_text().splitlines():
            r = json.loads(line)
            known[(r["seed"], r["steps"])] = r["q"]
    done = set()
    if rows.exists():
        for line in rows.read_text().splitlines():
            r = json.loads(line)
            done.add((r["seed"], r["steps"], r["search_particles"], r["simulations"]))
    with ProcessPoolExecutor(args.workers, max_tasks_per_child=1) as pool:
        todo = [(s, k, args.horizon) for s, k in problems if (s, k) not in known]
        with open(references, "a") as log:
            for (seed, steps, _), (q, cpu, support) in pool.map(_reference_job, todo):
                known[(seed, steps)] = q
                log.write(
                    json.dumps(dict(seed=seed, steps=steps, q=q, cpu_seconds=cpu, support=support))
                    + "\n"
                )
                log.flush()
        cells = [
            (s, k, n, m, args.horizon, known[(s, k)])
            for (s, k), n, m in itertools.product(problems, args.search_particles, args.simulations)
            if (s, k, n, m) not in done
        ]
        with open(rows, "a") as log:
            for row in pool.map(_evaluate_job, cells):
                log.write(json.dumps(row) + "\n")
                log.flush()


if __name__ == "__main__":
    main()
