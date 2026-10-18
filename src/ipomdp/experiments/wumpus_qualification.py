"""Wumpus qualification: compressed search beliefs x simulations, vs an exact reference.

The human (level L) models the wumpus at level L - 1, down to uniform L0, on
a ``width x height`` board with ``n_pits`` pits (``--board``; default the
4 x 4 board with 2 pits and its exact 94,500-state prior, RESULTS.md Section
10). Real beliefs are exact (filtering is cheap: deterministic sensors prune
the support fast); each search starts from the belief compressed to
``search_particles`` atoms. At level 1 the Section 10 pilot lost 1.8-7.7 in 13
of 18 decisions without separating the two suspects: root compression (N) and
the simulation budget over 5 actions x 32 observations. This grid separates
them. At level 2 and above, the modeled wumpus is solved by I-POMCP with
``modeled`` simulations at every level below the human (RESULTS.md 12.56: on a
3 x 3 board with 1 pit the exact I-POMDP is tractable at L2 H3 and L3 H2).

Problems: a scripted history (turn, step, turn, step, step; a step that would
end the episode is replaced by a turn) cut after 0, 2 or 4 steps, per seed.
That script gives every seed the same observations, so 3 seeds held 3 distinct
beliefs (RESULTS.md 12.19). ``--script random`` draws each step's action
uniformly per seed instead; references record the belief's digest so analyses
count distinct beliefs; ``--script modeled`` (level 2 and above) draws the
human's actions from the wumpus's model of the human, which level 3 needs
(``history``). At level 1 the world's wumpus acts uniformly at
random, as the human models it. Above level 1 it acts as the reference's own
exact model of it (``ModeledAgent``), so every history has positive evidence
under the exact reference; the history is stored with the reference and
replayed to the planner. The planner's own update can still find zero
evidence (its modeled wumpus is a finite search); such a row records the
failure instead of a decision.

References: at level 1 the exact H-step best response at the exact belief
(``BestResponseReference``, 520-600 s at the 4 x 4 prior); above level 1 the
exact I-POMDP (``reference.exact_agent``, every level exact) after the same
history. Each is computed once per problem and reused by every cell. Each row
records the decision's loss against it, the optimal-action and max Q errors,
the width of the planner's certified bounds (certified for the *searched*,
compressed belief, so not guaranteed to contain this reference), and CPU. One
JSON line per (problem, cell); restarts skip finished rows.

Usage:
    python -m ipomdp.experiments.wumpus_qualification --output DIR \\
        [--board 4 4 2] [--level 1] [--modeled 0] [--search-particles ... (0: exact)] \\
        [--seeds 3] [--steps 0 2 4] [--search-particles 64 256 1024 4096] \\
        [--simulations 1000 3000 10000] [--horizon 3] [--script fixed|random|modeled] [--workers 6]
"""

import argparse
import itertools
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ipomdp.baselines.rts import RTSConfig
from ipomdp.belief import canonical
from ipomdp.belief.finite_filter import UnsupportedObservation
from ipomdp.domains.wumpus.model import WumpusModel
from ipomdp.experiments.bootstrapper import HierarchyBuilder
from ipomdp.experiments.oracle import REFERENCE_BRANCHES, optimal_q_error
from ipomdp.planning.config import PlannerConfig
from ipomdp.planning.solver_bank import SolverBank
from ipomdp.reference.best_response import BestResponseReference
from ipomdp.reference.exact_agent import build_exact_agent

HISTORY = ("TurnLeft", "Forward", "TurnRight", "Forward", "Forward")
GAMMA = 0.95
BOARD = (4, 4, 2)


def nested_levels(level, l0=0.0):
    """Each intentional level models the other agent one level below, with weight ``l0`` on level 0.

    {1: {0: 1}}, and {k: {k - 1: 1 - l0, 0: l0}} for k >= 2; ``l0`` "uniform" gives
    {k: uniform over 0 .. k - 1}. A level-0 share gives
    every action of the modeled agent positive probability. With deterministic
    percepts a near-deterministic modeled policy otherwise leaves some of the
    modelling agent's counterfactual actions with zero-evidence nested updates,
    and the exact I-POMDP is undefined there (Wumpus L3; docs/RESULTS.md 12.58).
    """
    if l0 == "uniform":
        # Uniform over every lower level, as the episode suite's agents hold (suite.hierarchy_agent).
        return {k: {m: 1 / k for m in range(k)} for k in range(1, level + 1)}
    levels = {1: {0: 1.0}}
    for k in range(2, level + 1):
        levels[k] = {k - 1: 1.0 - l0, 0: l0} if l0 else {k - 1: 1.0}
    return levels


class HistoryEnded(ValueError):
    """The modeled human's every likely action ends the episode: the problem has no decision."""


class ModeledAgent:
    """An agent of the world acting as an exact model of it, and advancing that model's belief.

    ``bank`` holds the exact policies (``reference.exact_agent``); ``model`` is
    the agent's mental model in the modelling agent's prior. A level-0 model acts
    uniformly at random.
    """

    def __init__(self, bank, model):
        self.bank, self.model = bank, model

    def act(self, rng, allowed=None):
        """An action drawn from the model's policy, restricted to ``allowed`` when given."""
        frame = self.model.frame
        if frame.level == 0:
            actions = frame.pomdp_model.get_all_actions(frame.agent_id)
            policy = dict.fromkeys(actions, 1.0)
        else:
            policy = self.bank.policy(self.model)
        actions = sorted((a for a in policy if allowed is None or a in allowed), key=str)
        if not actions:
            raise HistoryEnded("The modeled agent has no allowed action with positive probability")
        return rng.choices(actions, weights=[policy[a] for a in actions])[0]

    def observe(self, action, observation):
        if self.model.frame.level:
            self.model = self.bank.filter.advance(self.model, action, observation, terminal=False)


def history(physics, steps, seed, script="fixed", wumpus=None, human=None):
    """The human's first ``steps`` (action, observation) pairs.

    ``script`` "fixed" plays ``HISTORY``; "random" draws each action uniformly
    from the human's actions, from a stream separate from the world's;
    "modeled" draws it from ``human`` (a ``ModeledAgent``: the wumpus's own
    model of the human), among the actions that do not end the episode. The
    wumpus acts uniformly at random, or as ``wumpus`` (a ``ModeledAgent``).

    "modeled" keeps every history possible under the exact I-POMDP at level 3
    and above: there the modeled wumpus expects a near-deterministic human, and
    a scripted human's actions can give it observations of zero probability.
    """
    rng = random.Random(f"wumpus/{seed}")
    state = rng.choices(*zip(*physics.initial_distribution()))[0]
    if script == "fixed":
        actions = HISTORY[:steps]
    elif script == "random":
        choose = random.Random(f"wumpus/script/{seed}")
        actions = [choose.choice(physics.get_all_actions("i")) for _ in range(steps)]
    elif script == "modeled":
        if human is None:
            raise ValueError("The modeled script needs the human's model")
        actions = [None] * steps
    else:
        raise ValueError(f"Unknown script {script!r}")
    out = []
    for action in actions:
        other = rng.choice(physics.get_all_actions("j")) if wumpus is None else wumpus.act(rng)
        if action is None:
            safe = [
                a
                for a in physics.get_all_actions("i")
                if not physics.is_terminal(
                    physics.sample_transition(state, {"i": a, "j": other}, random.Random(0))
                )
            ]
            action = human.act(rng, allowed=safe)
        joint = {"i": action, "j": other}
        following = physics.sample_transition(state, joint, rng)
        if physics.is_terminal(following):
            # History generation only: turn instead of ending the episode.
            action = joint["i"] = "TurnLeft"
            following = physics.sample_transition(state, joint, rng)
        state = following
        observation = physics.sample_observation(state, joint, "i", rng)
        out.append((action, observation))
        if wumpus is not None:
            wumpus.observe(other, physics.sample_observation(state, joint, "j", rng))
        if human is not None:
            human.observe(action, observation)
    return out


def human(
    physics,
    seed,
    simulations,
    horizon,
    search_particles,
    level=1,
    modeled=0,
    l0=0.0,
    planner="mcts",
    particles=0,
    particle_filter="sampled",
):
    """The level-``level`` human with exact real beliefs and searches on ``search_particles`` atoms.

    ``search_particles`` 0 or None searches exact beliefs at every level (no
    compression; feasible on small boards). Every modeled intentional level
    runs I-POMCP with ``modeled`` simulations (0: the human's own budget).

    ``planner`` "rts" is the IPF + RTS baseline instead: every level, real and
    modeled, runs reachability-tree sampling with ``simulations`` sampled
    observations per node, on beliefs of ``particles`` samples by the
    interactive particle filter ``particle_filter``: "sampled" (published) or
    "smoothed" (with the zero-likelihood repair). ``particles`` 0 is exact
    beliefs, which isolates RTS's search.
    """
    modeled = modeled or simulations
    if planner == "rts":
        config = lower = RTSConfig(depth=horizon, observation_samples=simulations, gamma=GAMMA)
        bank = SolverBank(
            seed,
            particles=particles or None,
            particle_filter=particle_filter,
            max_branches=REFERENCE_BRANCHES,
        )
    elif planner == "mcts":
        config = PlannerConfig(
            simulations=simulations, depth=horizon, modeled_simulations=modeled, gamma=GAMMA
        )
        lower = PlannerConfig(
            simulations=modeled, depth=horizon, modeled_simulations=modeled, gamma=GAMMA
        )
        # The exact real filter needs more than the default branch guard when the
        # human moves at the 94,500-atom prior (x 3 wumpus actions).
        bank = SolverBank(
            seed, search_particles=search_particles or None, max_branches=REFERENCE_BRANCHES
        )
    else:
        raise ValueError(f"Unknown planner {planner!r}")
    nested = nested_levels(level, l0)
    return HierarchyBuilder(bank).build(
        "i",
        level,
        physics,
        "j",
        config=config,
        modeled_configs=dict.fromkeys(range(1, level), lower),
        level_weights=nested[level],
        nested_level_weights=nested,
    )


def reference(seed, steps, horizon, script="fixed", board=BOARD, level=1, l0=0.0):
    """Exact Q of the problem, the history it conditions on, and the belief's size and digest.

    Level 1: the H-step best response at the exact belief. Above level 1: the
    exact I-POMDP after a history in which the wumpus acts as its exact model.
    """
    physics = WumpusModel(*board)
    if level == 1:
        planner = human(physics, seed, 1, horizon, None)
        steps_taken = history(physics, steps, seed, script)
        for action, observation in steps_taken:
            planner.update_root(action, observation)
        started = time.process_time()
        exact = BestResponseReference(
            physics, SolverBank(seed), GAMMA, max_branches=REFERENCE_BRANCHES
        )
        q = exact.q_values(planner.model(), horizon)
        belief = planner.belief
    else:
        agent = build_exact_agent(
            "i", "j", level, physics, nested_levels(level, l0), horizon, GAMMA
        )
        model = _model_at(agent.belief, level - 1)
        wumpus = ModeledAgent(agent.solver_bank, model)
        # The wumpus's own model of the human (one model at the common prior).
        believed = ModeledAgent(agent.solver_bank, _model_at(model.belief, level - 2))
        steps_taken = history(
            physics, steps, seed, script, wumpus, believed if script == "modeled" else None
        )
        for action, observation in steps_taken:
            agent.update_root(action, observation)
        started = time.process_time()
        q = agent.q_values(agent.model())
        belief = agent.belief
    return (
        {str(a): v for a, v in q.items()},
        time.process_time() - started,
        len(belief.mass),
        canonical.belief_digest(belief),
        steps_taken,
    )


def _model_at(belief, level):
    """The (single) opponent model of ``level`` in a prior belief."""
    (model,) = {atom.opponent for atom, _ in belief.mass if atom.opponent.frame.level == level}
    return model


def evaluate(
    seed,
    steps,
    search_particles,
    simulations,
    horizon,
    oracle,
    script="fixed",
    board=BOARD,
    level=1,
    modeled=0,
    steps_taken=None,
    l0=0.0,
    planner_kind="mcts",
    particles=0,
    particle_filter="sampled",
):
    """One decision after the problem's history, scored against ``oracle`` (the exact Q).

    ``steps_taken`` is the stored history (required above level 1, where it
    depends on the exact wumpus); at level 1 it is regenerated when omitted.
    """
    physics = WumpusModel(*board)
    if planner_kind == "intmcp":
        return _evaluate_intmcp(
            physics,
            seed,
            steps,
            simulations,
            horizon,
            oracle,
            script,
            board,
            level,
            steps_taken,
            l0,
        )
    planner = human(
        physics,
        seed,
        simulations,
        horizon,
        search_particles,
        level,
        modeled,
        l0,
        planner_kind,
        particles,
        particle_filter,
    )
    if steps_taken is None:
        steps_taken = history(physics, steps, seed, script)
    row = dict(
        seed=seed,
        steps=steps,
        script=script,
        search_particles=search_particles,
        simulations=simulations,
        horizon=horizon,
    )
    if (board, level, modeled) != (BOARD, 1, 0):
        row |= dict(board=list(board), level=level, modeled=modeled)
    if l0:
        row["l0"] = l0
    if planner_kind != "mcts":
        row |= dict(planner=planner_kind, particles=particles)
        if particle_filter != "sampled":
            row["particle_filter"] = particle_filter
    try:
        for action, observation in steps_taken:
            planner.update_root(action, observation)
    except UnsupportedObservation as error:
        return row | dict(failure=str(error))
    started = time.process_time()
    try:
        planner.policy_for(planner.model())
    except UnsupportedObservation as error:
        # A sampled belief inside the search can lose every world that explains an observation.
        return row | dict(failure=str(error), cpu_seconds=time.process_time() - started)
    cpu = time.process_time() - started
    values = {str(a): v for a, v in planner.recommendation_values.items()}
    best = max(oracle.values())
    chosen = max(values, key=values.get)
    scored = dict(
        loss=best - oracle[chosen],
        optimal_q_error=optimal_q_error(values, oracle),
        max_abs_q_error=max(abs(values[a] - oracle[a]) for a in oracle),
        cpu_seconds=cpu,
        values=values,
        oracle_q=oracle,
    )
    if planner_kind == "mcts":
        bounds = planner.scores.bounds
        scored |= dict(
            bound_width=max(hi - lo for lo, hi in bounds.values()), exact_root=planner.scores.exact
        )
    return row | scored


def _evaluate_intmcp(
    physics, seed, steps, simulations, horizon, oracle, script, board, level, steps_taken, l0
):
    """One I-NTMCP decision after the problem's history (``baselines.intmcp``), scored as ``evaluate``.

    I-NTMCP's model class is one opponent level under a common prior, so only
    point-mass levels qualify (``l0`` 0). It builds each step's beliefs from its
    own search's particles, so the history is replayed as search, update, ...;
    ``cpu_seconds`` is the final search, ``history_cpu_seconds`` every search
    and update before it. Actions its search never tried have no value.
    """
    from ipomdp.baselines.intmcp import INTMCPAgent

    if l0:
        raise ValueError("I-NTMCP models one opponent level: --l0 must be 0")
    if steps_taken is None:
        steps_taken = history(physics, steps, seed, script)
    row = dict(seed=seed, steps=steps, script=script, search_particles=0)
    row |= dict(simulations=simulations, horizon=horizon, board=list(board), level=level)
    row |= dict(modeled=0, planner="intmcp")
    started = time.process_time()
    agent = INTMCPAgent(physics, level, simulations, horizon, GAMMA, seed)
    agent.search()
    for action, observation in steps_taken:
        agent.update_root(action, observation)
        agent.search()
    total = time.process_time() - started
    chosen = str(max(agent.values, key=agent.values.get))
    values = {str(a): v for a, v in agent.values.items()}
    best = max(oracle.values())
    scored = dict(
        loss=best - oracle[chosen],
        cpu_seconds=agent.search_cpu_seconds,
        history_cpu_seconds=total - agent.search_cpu_seconds,
        values=values,
        oracle_q=oracle,
    )
    if set(values) == set(oracle):
        scored |= dict(
            optimal_q_error=optimal_q_error(values, oracle),
            max_abs_q_error=max(abs(values[a] - oracle[a]) for a in oracle),
        )
    return row | scored


def _reference_job(args):
    try:
        return args, reference(*args)
    except HistoryEnded:
        return args, None


def _evaluate_job(args):
    return evaluate(*args)


def _decoded(steps_taken):
    """A history read back from JSON: observations are tuples of tokens."""
    return [(action, tuple(observation)) for action, observation in steps_taken]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--board", type=int, nargs=3, default=list(BOARD))
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--modeled", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--l0",
        type=lambda v: v if v == "uniform" else float(v),
        default=0.0,
        help="level-0 weight at every level >= 2, or 'uniform' over all lower levels",
    )
    parser.add_argument("--planner", choices=("mcts", "rts", "intmcp"), default="mcts")
    parser.add_argument(
        "--particles", type=int, default=0, help="rts: IPF sample count (0: exact beliefs)"
    )
    parser.add_argument(
        "--particle-filter",
        choices=("sampled", "smoothed"),
        default="sampled",
        help="rts: the published IPF, or the IPF with likelihood smoothing",
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, nargs="+", default=[0, 2, 4])
    parser.add_argument("--search-particles", type=int, nargs="+", default=[64, 256, 1024, 4096])
    parser.add_argument("--simulations", type=int, nargs="+", default=[1000, 3000, 10000])
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--script", choices=("fixed", "random", "modeled"), default="fixed")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    board = tuple(args.board)
    args.output.mkdir(parents=True, exist_ok=True)
    references, rows = args.output / "references.jsonl", args.output / "rows.jsonl"
    problems = list(itertools.product(range(args.seeds), args.steps))
    known, taken, ended = {}, {}, set()
    if references.exists():
        for line in references.read_text().splitlines():
            r = json.loads(line)
            if r.get("ended"):
                ended.add((r["seed"], r["steps"]))
                continue
            known[(r["seed"], r["steps"])] = r["q"]
            if "history" in r:
                taken[(r["seed"], r["steps"])] = _decoded(r["history"])
    done = set()
    if rows.exists():
        for line in rows.read_text().splitlines():
            r = json.loads(line)
            done.add(
                (
                    r["seed"],
                    r["steps"],
                    r["search_particles"],
                    r["simulations"],
                    r.get("modeled", 0),
                )
            )
    with ProcessPoolExecutor(args.workers, max_tasks_per_child=1) as pool:
        todo = [
            (s, k, args.horizon, args.script, board, args.level, args.l0)
            for s, k in problems
            if (s, k) not in known and (s, k) not in ended
        ]
        with open(references, "a") as log:
            for (seed, steps, *_), result in pool.map(_reference_job, todo):
                if result is None:
                    # No decision exists after this history (the modeled human ends the episode).
                    log.write(json.dumps(dict(seed=seed, steps=steps, ended=True)) + "\n")
                    continue
                q, cpu, support, digest, steps_taken = result
                known[(seed, steps)], taken[(seed, steps)] = q, steps_taken
                row = dict(seed=seed, steps=steps, q=q, cpu_seconds=cpu, support=support)
                row |= dict(digest=digest, script=args.script, history=steps_taken)
                if (board, args.level) != (BOARD, 1):
                    row |= dict(board=list(board), level=args.level)
                if args.l0:
                    row["l0"] = args.l0
                log.write(json.dumps(row) + "\n")
                log.flush()
        cells = [
            (
                s,
                k,
                n,
                m,
                args.horizon,
                known[(s, k)],
                args.script,
                board,
                args.level,
                d,
                taken.get((s, k)),
                args.l0,
                args.planner,
                args.particles,
                args.particle_filter,
            )
            for (s, k), n, m, d in itertools.product(
                problems, args.search_particles, args.simulations, args.modeled
            )
            if (s, k) in known and (s, k, n, m, d) not in done
        ]
        with open(rows, "a") as log:
            for row in pool.map(_evaluate_job, cells):
                log.write(json.dumps(row) + "\n")
                log.flush()


if __name__ == "__main__":
    main()
