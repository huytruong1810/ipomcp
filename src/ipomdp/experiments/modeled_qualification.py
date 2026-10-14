"""Modeled-agent qualification: a modeled level's solve, judged where it is used.

Why
---
A level-L decision is only as good as its model of the level-(L-1) opponent,
and that model is solved by a finite search at every belief the level-L search
reaches. The L1 gate (``l1_qualification``) scores decisions at the real
agent's beliefs, which are not those beliefs: it passed at 1,000 simulations
while L2 H4 decisions failed with modeled budgets up to 1,000 (RESULTS.md
Sections 5 and 12.4). This driver scores the modeled agent where it is
actually queried.

What it does
------------
For each problem of level L = 2 or 3 (``problems(level)``, from
``experiments.problems``: by default Tiger's point beliefs of i, of j and, at
level 3, of the i-L1 that j models; ``uav_problems`` for UAV), seed and
history k, on the run's ``physics``:

1. Solve i's decision as ``panel`` does (I-POMCP, root ``budget``, the levels
   below i solved as the ``modeled`` stack says, at depth equal to the
   horizon; ``modeled_levels``), and record every j-(L-1) model whose policy
   i's search requests (``oracle.evaluate_case(opponent_observer=...)``) with
   the policy it got.
2. Re-solve j-(L-1) at each recorded model with every stack of a ``ladder``
   and score its policy against the exact level-(L-1) agent of the same depth
   (``reference.best_response``, every level below exact). A rung names j's
   own solve and, at level 3, the solve of the i-L1 that j models, so a rung
   is a whole modeled stack (``400``, ``6400s``, ``1600s/400``). Re-solving
   uses the search's seeds and settings, so the rung equal to ``modeled`` must
   reproduce the policy i used; the driver checks this on every query, which
   guards against the ladder silently solving a different problem.

The driver runs in two supervised phases. Phase 1 runs one job per problem and
saves every problem's queries to ``queries.pkl``; a later ladder can reuse them
(``queries_from``). Phase 2 runs one job per *distinct* (horizon, seed, queried
belief). A modeled solve is a pure function of those, so scoring each one once
is exact, and it keeps each job's memory to one belief. At L3 H4, 633 queries
held 211 distinct beliefs, and one shared reference cache per problem grew to
9.6 GB. Each rung is solved in a fresh bank, so the recorded CPU is the cold
cost of one modeled solve.

Each row is one (problem, queried model, rung) with the modeled agent's
first-action loss and that CPU. The summary gives, per rung, the
number of wrong modeled decisions with an exact 95% Clopper-Pearson interval
on the correct rate, split by whether i's own decision was wrong (when it is
scored: ``score_decision=False`` skips level L's exact reference); the number
of distinct queried beliefs and of distinct wrong ones (queries repeat across
problems, and mistakes cluster on a few beliefs); and the mean CPU per query.
The **qualifying** rung is the cheapest one, by mean CPU, with no wrong
modeled decision.

Why zero wrong decisions, not a loss tolerance: the modeled agent's own loss
does not separate mistakes that change i's decision from those that do not
(L2 H4: the harmful ones lose 0.66, harmless ones 0.2-1.0; RESULTS.md Section
12.5). Zero wrong is sufficient for i to be unaffected and needs only the
modeled level's exact reference, never level L's, which is what makes it
usable where level L's exact solution is out of reach (L3 beyond H3 k = 0;
12.7). It is conservative (H4: it certifies 6,400 where i's decisions are
already 380/380 at 1,600). Scope: exact beliefs.
"""

import argparse
import itertools
import json
import pickle
import time
from functools import partial
from pathlib import Path

import pandas as pd

from ipomdp.analysis.statistics import clopper_pearson
from ipomdp.belief import canonical
from ipomdp.domains.tiger import TigerModel
from ipomdp.domains.uav import UAVModel
from ipomdp.experiments.modeled_levels import (
    parse_stack,
    planner_config,
    register_levels,
    stack_label,
)
from ipomdp.experiments.oracle import OPTIMALITY_TOLERANCE, REFERENCE_BRANCHES, evaluate_case
from ipomdp.experiments.problems import tiger_problems, uav_problems
from ipomdp.infra.paths import REPOSITORY_ROOT
from ipomdp.infra.process_supervisor import supervise_jobs
from ipomdp.infra.source_lock import source_hashes
from ipomdp.planning.i_pomcp import IPOMCPPlanner
from ipomdp.planning.solver_bank import SolverBank
from ipomdp.planning.solver_types import SolverKey
from ipomdp.reference.best_response import BestResponseReference

# RESULTS.md Section 12.3's grid: 19 beliefs of i x 5 private beliefs of j.
BELIEFS = tuple(round(0.05 * k, 2) for k in range(1, 20))
OPPONENT_BELIEFS = (0.085, 0.3, 0.5, 0.7, 0.915)
NESTED_BELIEFS = (0.5,)
LADDER = ("100", "400", "1600", "4000")
GROUP = ("level", "horizon", "budget", "modeled", "history")
PROBLEM = ("horizon", "history", "seed", "problem")
TIGER_GRID = partial(
    tiger_problems,
    beliefs=BELIEFS,
    opponent_beliefs=OPPONENT_BELIEFS,
    nested_beliefs=NESTED_BELIEFS,
)


def collect_queries(
    level,
    horizon,
    budget,
    modeled,
    history,
    seed,
    problem,
    score_decision=True,
    gamma=0.95,
    physics=None,
):
    """Phase 1 for one problem: i's decision and every j-(level-1) model its search queries.

    Returns the problem (its name and labels), i's decision loss (None when
    ``score_decision`` is False, which skips level L's exact reference) and the
    queries in request order as (canonical belief digest, model, policy i used).
    """
    queried = []
    decision = evaluate_case(
        "mcts",
        horizon,
        budget,
        seed,
        problem.case.get("belief_p"),
        gamma=gamma,
        level=level,
        opponent_depth=horizon,
        modeled=modeled,
        reference="exact" if score_decision else "none",
        history_steps=history,
        physics=physics,
        opponent_observer=lambda model, policy: queried.append(
            (canonical.belief_digest(model.belief), model, policy)
        ),
        **{k: v for k, v in problem.case.items() if k != "belief_p"},
    )[0]
    return dict(
        horizon=horizon,
        history=history,
        seed=seed,
        problem=problem.name,
        labels=problem.labels,
        decision_loss=decision.get("first_action_loss"),
        queries=queried,
    )


def score_belief(level, horizon, seed, model, used_stack, used_policy, ladder, gamma=0.95):
    """Phase 2 for one distinct queried model: exact Q and every rung's policy, loss and CPU.

    Each rung is solved in its own fresh bank, so the CPU is the cold cost of
    one modeled solve. The rung equal to ``used_stack`` must reproduce
    ``used_policy``, the policy i's search got for this model.
    """
    physics = model.frame.pomdp_model
    exact = _exact_reference(physics, level, horizon, seed, gamma).q_values(model, horizon)
    best = max(exact.values())
    results = []
    for label in ladder:
        rung = parse_stack(label, level)
        solver = _queried_solver(physics, level, rung, horizon, seed, gamma)
        started = time.process_time()
        policy = solver.policy_for(model, modeled=True)
        cpu_seconds = time.process_time() - started
        if rung == used_stack and policy != used_policy:
            raise RuntimeError("Re-solving did not reproduce the modeled policy i used")
        results.append(
            dict(
                rung=stack_label(rung),
                policy={str(a): w for a, w in policy.items()},
                loss=best - sum(w * exact[a] for a, w in policy.items()),
                cpu_seconds=cpu_seconds,
            )
        )
    return dict(
        exact_q={str(a): q for a, q in exact.items()},
        physical_belief=_physical_marginal(model),
        rungs=results,
    )


def _exact_reference(physics, level, horizon, seed, gamma):
    """Exact Q of a j agent at ``level`` whose every lower level is exact."""
    bank = SolverBank(seed)
    reference = BestResponseReference(physics, bank, gamma, max_branches=REFERENCE_BRANCHES)
    register_levels(bank, physics, "j", level, (None,) * (level - 1), horizon, reference)
    return reference


def _queried_solver(physics, level, rung, horizon, seed, gamma):
    """j at ``level`` solved as ``rung`` says, in a bank seeded as in ``oracle.evaluate_case``.

    The bank seed, the configuration and the registered lower levels equal
    those inside i's search, so equal stacks give equal search streams.
    """
    bank = SolverBank(seed)
    reference = BestResponseReference(physics, bank, gamma)
    register_levels(bank, physics, "j", level, rung[1:], horizon, reference, gamma=gamma)
    solve = rung[0]
    config = planner_config("mcts", solve.budget, horizon, gamma, 1.0, solve.stopping)
    planner = IPOMCPPlanner(SolverKey("j", level), physics, bank, config)
    bank.register_solver(SolverKey("j", level), planner)
    return planner


def _physical_marginal(model):
    marginal = {}
    for atom, mass in model.belief.mass:
        marginal[str(atom.state)] = marginal.get(str(atom.state), 0.0) + mass
    return marginal


def summarize(rows):
    """Per (group, rung): wrong modeled decisions, overall and by i's outcome, and CPU."""
    frame = pd.DataFrame(rows)
    frame["wrong"] = frame["loss"] > OPTIMALITY_TOLERANCE
    frame["scored"] = frame["decision_loss"].notna()
    frame["decision_wrong"] = frame["decision_loss"].astype(float) > OPTIMALITY_TOLERANCE
    records = []
    for key, group in frame.groupby([*GROUP, "rung"], sort=False):
        wrong, n = int(group["wrong"].sum()), len(group)
        low, high = clopper_pearson(n - wrong, n)
        records.append(
            dict(
                zip([*GROUP, "rung"], key),
                queries=n,
                wrong=wrong,
                beliefs=int(group["belief_digest"].nunique()),
                wrong_beliefs=int(group.loc[group["wrong"], "belief_digest"].nunique()),
                correct_rate_low=low,
                correct_rate_high=high,
                mean_loss=float(group["loss"].mean()),
                max_loss=float(group["loss"].max()),
                wrong_where_decision_wrong=_split(group, group["decision_wrong"]),
                wrong_where_decision_right=_split(group, ~group["decision_wrong"]),
                mean_cpu_seconds=float(group["cpu_seconds"].mean()),
            )
        )
    return pd.DataFrame(records)


def _split(group, decisions):
    """Wrong modeled queries among problems with the given i outcome (None if unscored)."""
    if not group["scored"].all():
        return None
    return int(group.loc[decisions, "wrong"].sum())


def qualifying_rungs(summary):
    """Cheapest rung (mean CPU per query) with no wrong modeled decision, per group (or None)."""
    result = {}
    for key, group in summary.groupby(list(GROUP), sort=False):
        passing = group.loc[group["wrong"] == 0].sort_values("mean_cpu_seconds")
        result[key] = passing["rung"].iloc[0] if len(passing) else None
    return result


def run_qualification(
    output,
    *,
    level=2,
    horizons,
    budget,
    modeled,
    ladder=LADDER,
    histories=(0,),
    seeds=(0, 1, 2, 3),
    problems=TIGER_GRID,
    physics=None,
    score_decision=True,
    queries_from=None,
    workers=4,
    timeout_seconds=7200,
    max_rss_mb=4096,
):
    """Search every problem, score each distinct queried model once, then expand to rows.

    ``problems(level)`` lists the problems (it must pickle: a module function
    or a ``functools.partial`` of one); ``physics`` is their model (None: the
    published Tiger). ``queries_from`` names an earlier run's ``queries.pkl``
    (same settings except the ladder) whose phase 1 is reused, so new rungs
    cost phase 2 only.
    """
    if level not in (2, 3):
        raise ValueError("Qualification scores the level below a level-2 or level-3 agent")
    used_stack = parse_stack(modeled, level - 1)
    if any(parse_stack(label, level - 1)[0] is None for label in ladder):
        raise ValueError("A rung must solve the queried level by search; exact is the reference")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    listed = problems(level)
    lock = source_hashes(REPOSITORY_ROOT)
    problem_settings = dict(
        level=level,
        horizons=list(horizons),
        budget=budget,
        modeled=stack_label(used_stack),
        histories=list(histories),
        seeds=list(seeds),
        problems=[problem.name for problem in listed],
        physics=canonical.physics_identity(TigerModel() if physics is None else physics),
        score_decision=score_decision,
    )
    (output / "manifest.json").write_text(
        json.dumps(
            dict(
                settings=problem_settings | dict(ladder=list(ladder)),
                queries_from=str(queries_from) if queries_from else None,
                source_sha256=lock,
            ),
            indent=2,
        )
    )
    supervision = dict(
        workers=workers,
        timeout_seconds=timeout_seconds,
        max_rss_mb=max_rss_mb,
        source_lock=(REPOSITORY_ROOT, lock),
    )
    failures = []
    if queries_from:
        saved = pickle.loads(Path(queries_from).read_bytes())
        if saved["settings"] != problem_settings:
            raise ValueError("queries_from was searched with different problem settings")
        searched = saved["problems"]
    else:
        index = list(itertools.product(horizons, histories, seeds, range(len(listed))))
        jobs = {
            i: partial(
                collect_queries,
                level,
                horizon,
                budget,
                modeled,
                history,
                seed,
                listed[k],
                score_decision=score_decision,
                physics=physics,
            )
            for i, (horizon, history, seed, k) in enumerate(index)
        }
        searched = []
        for i, outcome in supervise_jobs(
            jobs, log_directory=output / "logs" / "search", **supervision
        ):
            if outcome["status"] == "complete":
                searched.append(outcome["rows"])
            else:
                horizon, history, seed, k = index[i]
                job = dict(horizon=horizon, history=history, seed=seed, problem=listed[k].name)
                failures.append(dict(job=job, reason=outcome["reason"]))
        order = {problem.name: k for k, problem in enumerate(listed)}
        searched = sorted(
            searched,
            key=lambda p: (p["horizon"], p["history"], p["seed"], order[p["problem"]]),
        )
        (output / "queries.pkl").write_bytes(
            pickle.dumps(
                dict(settings=problem_settings, problems=searched), pickle.HIGHEST_PROTOCOL
            )
        )
    # Phase 2: a modeled solve is a pure function of (depth, bank seed, belief),
    # so each distinct queried model is scored once (L3 H4: 633 queries, 211 distinct).
    distinct = {}
    for found in searched:
        for digest, model, used in found["queries"]:
            key = (found["horizon"], found["seed"], digest)
            if key in distinct and distinct[key][1] != used:
                raise RuntimeError("One belief got two modeled policies inside i's searches")
            distinct.setdefault(key, (model, used))
    keys = sorted(distinct)
    jobs = {
        i: partial(
            score_belief,
            level - 1,
            h,
            s,
            distinct[(h, s, d)][0],
            used_stack,
            distinct[(h, s, d)][1],
            tuple(ladder),
        )
        for i, (h, s, d) in enumerate(keys)
    }
    scored = {}
    for i, outcome in supervise_jobs(jobs, log_directory=output / "logs" / "score", **supervision):
        if outcome["status"] == "complete":
            scored[keys[i]] = outcome["rows"]
        else:
            failures.append(dict(belief=list(keys[i]), reason=outcome["reason"]))
    rows = [
        dict(
            {k: found[k] for k in PROBLEM},
            **found["labels"],
            level=level,
            budget=budget,
            modeled=stack_label(used_stack),
            decision_loss=found["decision_loss"],
            query=q,
            belief_digest=digest,
            physical_belief=result["physical_belief"],
            exact_q=result["exact_q"],
            **rung,
        )
        for found in searched
        for q, (digest, _, _) in enumerate(found["queries"])
        if (result := scored.get((found["horizon"], found["seed"], digest)))
        for rung in result["rungs"]
    ]
    (output / "rows.json").write_text(json.dumps(rows))
    (output / "failures.json").write_text(json.dumps(failures, indent=2))
    summary = summarize(rows) if rows else pd.DataFrame()
    summary.to_csv(output / "summary.csv", index=False)
    qualifying = qualifying_rungs(summary) if rows else {}
    (output / "qualifying.json").write_text(
        json.dumps([dict(zip(GROUP, k), qualifying=v) for k, v in qualifying.items()], indent=2)
    )
    return summary, qualifying, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--level", type=int, choices=[2, 3], default=2)
    parser.add_argument("--horizons", type=int, nargs="+", default=[4])
    parser.add_argument("--budget", type=int, default=1000, help="root simulations of i")
    parser.add_argument("--modeled", default="100", help="modeled stack inside i's search")
    parser.add_argument("--ladder", nargs="+", default=list(LADDER), help="modeled stacks to score")
    parser.add_argument("--history", type=int, nargs="+", default=[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument(
        "--domain",
        choices=["tiger", "uav"],
        default="tiger",
        help="tiger: point-belief problems; uav: experiments.problems.uav_problems",
    )
    parser.add_argument("--beliefs", type=float, nargs="+", default=list(BELIEFS))
    parser.add_argument("--opponent-beliefs", type=float, nargs="+", default=list(OPPONENT_BELIEFS))
    parser.add_argument("--nested-beliefs", type=float, nargs="+", default=list(NESTED_BELIEFS))
    parser.add_argument(
        "--no-decision-score",
        action="store_true",
        help="skip i's exact reference (intractable at L3 beyond H3)",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.domain == "uav":
        physics = UAVModel()
        problems = partial(uav_problems, physics)
    else:
        physics = None
        problems = partial(
            tiger_problems,
            beliefs=tuple(args.beliefs),
            opponent_beliefs=tuple(args.opponent_beliefs),
            nested_beliefs=tuple(args.nested_beliefs),
        )
    summary, qualifying, failures = run_qualification(
        args.output,
        level=args.level,
        horizons=args.horizons,
        budget=args.budget,
        modeled=args.modeled,
        ladder=args.ladder,
        histories=args.history,
        seeds=args.seeds,
        problems=problems,
        physics=physics,
        score_decision=not args.no_decision_score,
        workers=args.workers,
    )
    print(summary.round(4).to_string(index=False))
    for key, value in qualifying.items():
        print(dict(zip(GROUP, key)), "qualifying modeled stack:", value)
    if failures:
        raise SystemExit(f"{len(failures)} jobs failed; see failures.json")


if __name__ == "__main__":
    main()
