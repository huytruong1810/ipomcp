"""Modeled-agent qualification: a modeled level's budget, judged where it is used.

Why
---
A level-2 decision is only as good as its model of the level-1 opponent, and
that model is solved by a finite search of ``modeled`` simulations at every
belief the level-2 search reaches. The L1 gate (``l1_qualification``) scores
decisions at the real agent's beliefs, which are not those beliefs: it passed
at 1,000 simulations while L2 H4 decisions failed with modeled budgets up to
1,000 (RESULTS.md Sections 5 and 12.4). This driver scores the modeled agent
where it is actually queried.

What it does
------------
For each L2 Tiger problem (belief of i, private belief of j, seed, history k):

1. Solve i's decision as ``l2_panel`` does (I-POMCP, root ``budget``, j-L1 an
   I-POMCP agent of ``modeled`` simulations at depth equal to the horizon), and
   record every j-L1 model whose policy i's search requests
   (``oracle.evaluate_case(opponent_observer=...)``) with the policy it got.
2. Re-solve j-L1 at each recorded model with every budget of a ``ladder`` and
   score its policy against the exact L1 agent of the same depth
   (``reference.best_response``). Re-solving uses the search's seed and
   settings (``oracle.planner_config``), so the rung equal to ``modeled`` must
   reproduce the policy i used; the driver checks this on every query, which
   guards against the ladder silently solving a different problem.

Each row is one (problem, queried model, ladder budget) with the modeled
agent's first-action loss. The summary gives, per ladder budget, the number of
wrong modeled decisions with an exact 95% Clopper-Pearson interval on the
correct rate, split by whether i's own decision was wrong, the number of
distinct queried beliefs and of distinct wrong ones (queries repeat across
problems, and mistakes cluster on a few beliefs), and the **qualifying
budget**: the smallest rung with no wrong modeled decision.

Why zero wrong decisions, not a loss tolerance: the modeled agent's own loss
does not separate mistakes that change i's decision from those that do not
(L2 H4: the harmful ones lose 0.66, harmless ones 0.2-1.0; RESULTS.md Section
12.5). Zero wrong is sufficient for i to be unaffected and needs no
higher-level reference, which is what makes it usable where the higher
level's exact solution is out of reach; it is conservative (H4: it certifies
6,400 where i's decisions are already 380/380 at 1,600).

Scope: level 1 modeled under level 2, with exact beliefs. A level-3 version
needs an exact j-L2 reference in the oracle (HANDOFF next steps).
"""

import argparse
import itertools
import json
import random
from functools import partial
from pathlib import Path

import pandas as pd

from ipomdp.analysis.statistics import clopper_pearson
from ipomdp.belief import canonical
from ipomdp.experiments.oracle import OPTIMALITY_TOLERANCE, evaluate_case, planner_config
from ipomdp.infra.paths import REPOSITORY_ROOT
from ipomdp.infra.process_supervisor import supervise_jobs
from ipomdp.infra.source_lock import source_hashes
from ipomdp.planning.i_pomcp import IPOMCPPlanner
from ipomdp.planning.random_planner import RandomPlanner
from ipomdp.planning.solver_bank import SolverBank
from ipomdp.planning.solver_types import SolverKey
from ipomdp.reference.best_response import BestResponseReference

# RESULTS.md Section 12.3's grid: 19 beliefs of i x 5 private beliefs of j.
BELIEFS = tuple(round(0.05 * k, 2) for k in range(1, 20))
OPPONENT_BELIEFS = (0.085, 0.3, 0.5, 0.7, 0.915)
LADDER = (100, 400, 1600, 4000)
GROUP = ("horizon", "budget", "modeled", "history")


def evaluate_problems(
    horizon, budget, modeled, ladder, history, seed, belief_p, opponent_beliefs, gamma=0.95
):
    """Rows for every queried j-L1 model of the problems (belief_p, each opponent belief)."""
    rows = []
    for opponent_belief in opponent_beliefs:
        queried = []  # (model, policy) pairs, in request order
        decision = evaluate_case(
            "mcts",
            horizon,
            budget,
            seed,
            belief_p,
            gamma=gamma,
            level=2,
            opponent_depth=horizon,
            opponent_belief=opponent_belief,
            opponent_budget=modeled,
            reference="exact",
            history_steps=history,
            opponent_observer=lambda model, policy: queried.append((model, policy)),
        )[0]
        if not queried:
            continue
        # Frames must carry the very physics instance the reference is built on.
        physics = queried[0][0].frame.pomdp_model
        reference = BestResponseReference(physics, _opponent_bank(physics, seed), gamma)
        for index, (model, used) in enumerate(queried):
            exact = reference.q_values(model, horizon)
            best = max(exact.values())
            for rung in ladder:
                policy = _modeled_policy(model, rung, horizon, seed, gamma)
                if rung == modeled and policy != used:
                    raise RuntimeError("Re-solving did not reproduce the modeled policy i used")
                rows.append(
                    dict(
                        horizon=horizon,
                        budget=budget,
                        modeled=modeled,
                        history=history,
                        seed=seed,
                        belief_p=belief_p,
                        opponent_belief=opponent_belief,
                        decision_loss=decision["first_action_loss"],
                        query=index,
                        belief_digest=canonical.belief_digest(model.belief),
                        physical_belief=_physical_marginal(model),
                        exact_q={str(a): q for a, q in exact.items()},
                        ladder_budget=rung,
                        policy={str(a): w for a, w in policy.items()},
                        loss=best - sum(w * exact[a] for a, w in policy.items()),
                    )
                )
    return rows


def _opponent_bank(physics, seed):
    """A bank for solving j-L1: i is uniform L0, seeded as in ``oracle.evaluate_case``."""
    bank = SolverBank(seed)
    rng = random.Random(bank.stream_seed("actions", "i_L0"))
    bank.register_solver(SolverKey("i", 0), RandomPlanner(physics.get_all_actions("i"), rng))
    return bank


def _modeled_policy(model, simulations, depth, seed, gamma):
    """j-L1's modeled policy at ``model``, with the settings the L2 search uses for it."""
    physics = model.frame.pomdp_model
    bank = _opponent_bank(physics, seed)
    config = planner_config("mcts", simulations, depth, gamma, exploration=1.0)
    planner = IPOMCPPlanner(SolverKey("j", 1), physics, bank, config)
    bank.register_solver(SolverKey("j", 1), planner)
    return planner.policy_for(model, modeled=True)


def _physical_marginal(model):
    marginal = {}
    for atom, mass in model.belief.mass:
        marginal[str(atom.state)] = marginal.get(str(atom.state), 0.0) + mass
    return marginal


def summarize(rows):
    """Per (group, ladder budget): wrong modeled decisions, overall and by i's outcome."""
    frame = pd.DataFrame(rows)
    frame["wrong"] = frame["loss"] > OPTIMALITY_TOLERANCE
    frame["decision_wrong"] = frame["decision_loss"] > OPTIMALITY_TOLERANCE
    records = []
    for key, group in frame.groupby([*GROUP, "ladder_budget"]):
        wrong, n = int(group["wrong"].sum()), len(group)
        low, high = clopper_pearson(n - wrong, n)
        records.append(
            dict(
                zip([*GROUP, "ladder_budget"], key),
                queries=n,
                wrong=wrong,
                beliefs=int(group["belief_digest"].nunique()),
                wrong_beliefs=int(group.loc[group["wrong"], "belief_digest"].nunique()),
                correct_rate_low=low,
                correct_rate_high=high,
                mean_loss=float(group["loss"].mean()),
                max_loss=float(group["loss"].max()),
                wrong_where_decision_wrong=int(group.loc[group["decision_wrong"], "wrong"].sum()),
                wrong_where_decision_right=int(group.loc[~group["decision_wrong"], "wrong"].sum()),
            )
        )
    return pd.DataFrame(records)


def qualifying_budgets(summary):
    """Smallest ladder budget with no wrong modeled decision per group (None if no rung qualifies)."""
    result = {}
    for key, group in summary.groupby(list(GROUP)):
        passing = group.loc[group["wrong"] == 0, "ladder_budget"]
        result[key] = int(passing.min()) if len(passing) else None
    return result


def run_qualification(
    output,
    *,
    horizons,
    budget,
    modeled,
    ladder=LADDER,
    histories=(0,),
    seeds=(0, 1, 2, 3),
    beliefs=BELIEFS,
    opponent_beliefs=OPPONENT_BELIEFS,
    workers=4,
    timeout_seconds=7200,
    max_rss_mb=4096,
):
    """One supervised job per (horizon, history, seed, belief of i); write rows and summary."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    lock = source_hashes(REPOSITORY_ROOT)
    settings = dict(
        horizons=list(horizons),
        budget=budget,
        modeled=modeled,
        ladder=list(ladder),
        histories=list(histories),
        seeds=list(seeds),
        beliefs=list(beliefs),
        opponent_beliefs=list(opponent_beliefs),
    )
    (output / "manifest.json").write_text(
        json.dumps(dict(settings=settings, source_sha256=lock), indent=2)
    )
    jobs_index = list(itertools.product(horizons, histories, seeds, beliefs))
    jobs = {
        i: partial(
            evaluate_problems,
            horizon,
            budget,
            modeled,
            tuple(ladder),
            history,
            seed,
            p,
            tuple(opponent_beliefs),
        )
        for i, (horizon, history, seed, p) in enumerate(jobs_index)
    }
    rows, failures = [], []
    for index, outcome in supervise_jobs(
        jobs,
        workers=workers,
        timeout_seconds=timeout_seconds,
        max_rss_mb=max_rss_mb,
        log_directory=output / "logs",
        source_lock=(REPOSITORY_ROOT, lock),
    ):
        if outcome["status"] == "complete":
            rows.extend(outcome["rows"])
        else:
            failures.append(
                dict(job=dict(zip(("horizon", "history", "seed", "belief_p"), jobs_index[index])))
                | dict(reason=outcome["reason"])
            )
    (output / "rows.json").write_text(json.dumps(rows))
    (output / "failures.json").write_text(json.dumps(failures, indent=2))
    summary = summarize(rows) if rows else pd.DataFrame()
    summary.to_csv(output / "summary.csv", index=False)
    qualifying = qualifying_budgets(summary) if rows else {}
    (output / "qualifying.json").write_text(
        json.dumps([dict(zip(GROUP, k), qualifying=v) for k, v in qualifying.items()], indent=2)
    )
    return summary, qualifying, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--horizons", type=int, nargs="+", default=[4])
    parser.add_argument("--budget", type=int, default=1000, help="root simulations of i")
    parser.add_argument("--modeled", type=int, default=100, help="j-L1 budget inside i's search")
    parser.add_argument("--ladder", type=int, nargs="+", default=list(LADDER))
    parser.add_argument("--history", type=int, nargs="+", default=[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--beliefs", type=float, nargs="+", default=list(BELIEFS))
    parser.add_argument("--opponent-beliefs", type=float, nargs="+", default=list(OPPONENT_BELIEFS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary, qualifying, failures = run_qualification(
        args.output,
        horizons=args.horizons,
        budget=args.budget,
        modeled=args.modeled,
        ladder=args.ladder,
        histories=args.history,
        seeds=args.seeds,
        beliefs=args.beliefs,
        opponent_beliefs=args.opponent_beliefs,
        workers=args.workers,
    )
    print(summary.round(4).to_string(index=False))
    for key, value in qualifying.items():
        print(dict(zip(GROUP, key)), "qualifying modeled budget:", value)
    if failures:
        raise SystemExit(f"{len(failures)} jobs failed; see failures.json")


if __name__ == "__main__":
    main()
