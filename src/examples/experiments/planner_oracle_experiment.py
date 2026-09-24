"""Matched L1 Tiger oracle comparison with explicit computational budget sweeps.

Both planners and the reference solve the same finite-horizon problem: the same
two-state physical belief, uniform L0 opponent, sensor law, discount and horizon.
No sampled initialization is substituted for the stated reference belief. This
suite measures L1 only; it does not certify deeper intentional hierarchies.

MCTS simulations and RTS particles per branch are different work units. Report
both raw budget and wall/RSS cost, never call equal numbers equal compute. RTS
enumerates all nine observation tokens so top-k truncation cannot confound this
comparison. Policy loss is V*(b)-sum_a pi(a|b)Q*(b,a), which handles ties and
measures first-action loss followed by optimal continuation, not episode regret.
"""

import argparse
import hashlib
import itertools
import json
from functools import partial
from pathlib import Path

from core.config import IPOMCPConfig, MCTSConfig, RTSConfig
from examples.tiger.model.tiger_model import TIGER_LEFT, TIGER_RIGHT, TigerModel
from ipomdp.finite_belief import FiniteBelief, InteractiveState, MentalModel
from ipomdp.frame import AgentFrame
from solvers.exact.pomdp_exact_vi import ExactPOMDPSolver
from solvers.exploration import HorizonBoundUCB, NormalizedUCB, StandardUCB
from solvers.solver_bank import SolverBank
from utils.bootstrapper import I_POMDP_Bootstrapper
from utils.process_supervisor import supervise_jobs


def evaluate_case(
    planner_kind,
    horizon,
    budget,
    seed,
    belief_p,
    gamma,
    exploration="normalized",
    exploration_const=1.0,
):
    """One independent solve. The supervisor owns time/RSS measurement."""
    model = TigerModel()
    oracle = ExactPOMDPSolver(model, horizon=horizon, gamma=gamma)
    oracle_q = oracle.q_values(belief_p, horizon)
    bank = SolverBank(seed=seed)
    bootstrap = I_POMDP_Bootstrapper(bank)
    if planner_kind == "mcts":
        planner = bootstrap.create_level1_solver(
            "i",
            model,
            ["j"],
            n_particles=2,
            config=IPOMCPConfig(
                mcts=MCTSConfig(
                    gamma=gamma,
                    n_sims=budget,
                    max_depth=horizon,
                    node_capacity=200,
                    exploration_const=exploration_const,
                )
            ),
        )
        if exploration == "normalized":
            strategy = NormalizedUCB(exploration_const)
        elif exploration == "standard":
            strategy = StandardUCB(exploration_const)
        elif exploration == "bounded":
            # Exhaust the complete Tiger state/joint-action/transition support.
            # These are physics bounds, never oracle Q-values or sample extrema.
            rewards = [
                model.get_reward(state, {"i": own, "j": other}, following, "i")
                for state in (TIGER_LEFT, TIGER_RIGHT)
                for own in model.get_all_actions("i")
                for other in model.get_all_actions("j")
                for following, probability in model.transition_distribution(
                    state, {"i": own, "j": other}
                )
                if probability > 0
            ]
            strategy = HorizonBoundUCB(min(rewards), max(rewards), exploration_const)
        else:
            raise ValueError("Unknown exploration strategy")
        planner.exploration_strategy = strategy
    elif planner_kind == "rts":
        planner = bootstrap.create_level1_rts_solver(
            "i",
            model,
            ["j"],
            n_particles=2,
            config=RTSConfig(gamma=gamma, max_depth=horizon, obs_branching=9, num_particles=budget),
        )
    else:
        raise ValueError("Unknown planner")
    opponent = MentalModel(AgentFrame("j", 0, model))
    belief = FiniteBelief(
        (
            (InteractiveState(TIGER_LEFT, opponent), belief_p),
            (InteractiveState(TIGER_RIGHT, opponent), 1 - belief_p),
        )
    )
    planner.set_initial_belief(belief, sample_count=0)
    policy = planner.policy_for(planner.model())
    estimates = (
        dict(planner.root.action_values) if planner_kind == "mcts" else planner.get_action_values()
    )
    if set(estimates) != set(oracle_q):
        raise ValueError("Planner must evaluate every legal action")
    value = max(oracle_q.values())
    loss = value - sum(prob * oracle_q[action] for action, prob in policy.items())
    return [
        {
            "planner": planner_kind,
            "horizon": horizon,
            "budget": budget,
            "seed": seed,
            "belief_p": belief_p,
            "gamma": gamma,
            "exploration": (
                {"strategy": exploration, **vars(planner.exploration_strategy)}
                if planner_kind == "mcts"
                else None
            ),
            "oracle_q": oracle_q,
            "estimated_q": estimates,
            "policy": policy,
            "first_action_loss": loss,
            "max_abs_q_error": max(abs(estimates[a] - oracle_q[a]) for a in oracle_q),
        }
    ]


def run_oracle_comparison(
    out,
    *,
    budgets=(1000, 10000, 50000),
    horizons=(1, 2, 3),
    seeds=10,
    seed_start=0,
    beliefs=(0.02, 0.15, 0.5, 0.85, 0.98),
    planners=("mcts", "rts"),
    gamma=0.95,
    exploration="normalized",
    exploration_const=1.0,
    workers=2,
    timeout=2400,
    max_rss_mb=4096,
):
    """Persist every outcome and source fingerprint before interpreting results.

    There is no success threshold picked after seeing results and no automatic
    'optimal' certification. Increasing budgets supplies an error/cost curve;
    sufficient resources depend on the accuracy required for a scientific claim.
    """
    # Validate before workers start so invalid settings cannot create a partial panel.
    MCTSConfig(exploration_const=exploration_const)
    if exploration not in {"normalized", "standard", "bounded"}:
        raise ValueError("Unknown exploration strategy")
    if type(seed_start) is not int or seed_start < 0:
        raise ValueError("seed_start must be a nonnegative integer")
    if type(seeds) is not int or seeds < 1 or not budgets or any(b < 3 for b in budgets):
        raise ValueError("Need positive seed count and budgets >= 3")
    if not horizons or any(h < 1 for h in horizons):
        raise ValueError("Need positive planning horizons")
    if any(not 0 <= p <= 1 for p in beliefs) or not 0 <= gamma <= 1:
        raise ValueError("Beliefs and discount must lie in [0,1]")
    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("Use an empty output directory")
    source = Path(__file__).resolve().parents[2]
    settings = dict(
        budgets=budgets,
        horizons=horizons,
        seeds=seeds,
        seed_start=seed_start,
        beliefs=beliefs,
        planners=planners,
        gamma=gamma,
        exploration=exploration,
        exploration_const=exploration_const,
        workers=workers,
        timeout=timeout,
        max_rss_mb=max_rss_mb,
    )
    manifest = {
        "settings": settings,
        "scope": "L1 vs uniform L0 only",
        "source_sha256": {
            str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(source.rglob("*.py"))
        },
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    # Seed indices are bank inputs, independent of panel position. A fresh range
    # allows validation without replaying seeds used for candidate selection.
    cases = list(
        itertools.product(
            planners, horizons, budgets, range(seed_start, seed_start + seeds), beliefs
        )
    )
    jobs = {
        i: partial(evaluate_case, *case, gamma, exploration, exploration_const)
        for i, case in enumerate(cases)
    }
    summaries = []
    for identifier, outcome in supervise_jobs(
        jobs,
        workers=workers,
        timeout_seconds=timeout,
        max_rss_mb=max_rss_mb,
        log_directory=directory / "logs",
    ):
        outcome["case"] = cases[identifier]
        (directory / f"case-{identifier}.json").write_text(
            json.dumps(outcome, indent=2, allow_nan=False)
        )
        summary = {k: v for k, v in outcome.items() if k not in {"rows", "traceback"}}
        summaries.append(summary)
        (directory / "summary.json").write_text(json.dumps(summaries, indent=2, allow_nan=False))
        print(json.dumps(summary), flush=True)
    if any(row["status"] != "complete" for row in summaries):
        raise RuntimeError("Oracle panel has failed cases; inspect saved outcomes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[1000, 10000, 50000])
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--seeds", type=int, default=10, help="Number of seed indices")
    parser.add_argument("--seed-start", type=int, default=0, help="First seed index (inclusive)")
    parser.add_argument("--beliefs", type=float, nargs="+", default=[0.02, 0.15, 0.5, 0.85, 0.98])
    parser.add_argument("--planners", choices=["mcts", "rts"], nargs="+", default=["mcts", "rts"])
    parser.add_argument(
        "--exploration",
        choices=["normalized", "standard", "bounded"],
        default="normalized",
        help="MCTS only: empirical range, raw reward units, or remaining-horizon reward bounds",
    )
    parser.add_argument("--exploration-const", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=2400)
    parser.add_argument("--max-rss-mb", type=float, default=4096)
    run_oracle_comparison(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
