"""Matched Tiger oracle comparisons with explicit computational budget sweeps.

Each comparison binds the same physical belief, sensor law, discount, horizon
and opponent model. L1 uses uniform L0; L2 uses a declared fixed-depth L1.
No sampled initialization is substituted for the stated reference belief. This
suite also supports L2 against an explicitly fixed-depth exact L1 policy,
with a point prior on its private belief. That contract is shared by both
planners. Supplying --opponent-budget selects a finite-computation MCTS L1
policy instead, with an exhaustive L2 response to that same immutable-model
policy. Such a reference is optimal relative to the declared opponent, not a
claim that the finite opponent itself is Bayes-optimal.

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
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

from core.config import IPOMCPConfig, MCTSConfig, OpponentPolicyConfig, RTSConfig
from examples.tiger.model.tiger_model import TIGER_LEFT, TIGER_RIGHT, TigerModel
from ipomdp.finite_belief import FiniteBelief, InteractiveState, MentalModel
from ipomdp.frame import AgentFrame
from solvers.exact.finite_policy_l2 import FinitePolicyL2Reference
from solvers.exact.fixed_tiger_opponent import FixedTigerL1Policy
from solvers.exact.ipomdp_exact_vi import ExactIPOMDPSolver
from solvers.exact.pomdp_exact_vi import ExactPOMDPSolver
from solvers.exploration import HorizonBoundUCB, NormalizedUCB, StandardUCB
from solvers.i_pomcp import IPOMCPPlanner
from solvers.random_planner import RandomPlanner
from solvers.solver_bank import SolverBank
from solvers.solver_types import SolverKey
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
    exact_final_step=False,
    backup="sampled",
    level=1,
    opponent_depth=None,
    opponent_belief=0.5,
    opponent_budget=None,
    opponent_backup="sampled",
    opponent_exact_final_step=False,
    opponent_exploration="normalized",
    opponent_exploration_const=1.0,
    exact_history_rewards=False,
):
    """One independent solve. The supervisor owns time/RSS measurement."""
    _validate_level_contract(level, opponent_depth, opponent_belief, (planner_kind,))
    modeled_config = _modeled_config(
        level,
        opponent_depth,
        opponent_budget,
        gamma,
        opponent_backup,
        opponent_exact_final_step,
        opponent_exploration,
        opponent_exploration_const,
    )
    model = TigerModel()
    if level == 1:
        oracle = ExactPOMDPSolver(model, horizon=horizon, gamma=gamma)
        oracle_q = oracle.q_values(belief_p, horizon)
    elif modeled_config is None:
        oracle = ExactIPOMDPSolver(
            model, horizon=horizon, gamma=gamma, opponent_horizon=opponent_depth
        )
        oracle_q = oracle.q_values(belief_p, opponent_belief, horizon)
    bank = SolverBank(seed=seed)
    bootstrap = I_POMDP_Bootstrapper(bank)
    if planner_kind == "mcts":
        config = IPOMCPConfig(
            mcts=MCTSConfig(
                gamma=gamma,
                n_sims=budget,
                max_depth=horizon,
                node_capacity=200,
                exploration_const=exploration_const,
                exact_final_step=exact_final_step,
                exact_history_rewards=exact_history_rewards,
                backup=backup,
            )
        )
        if level == 1:
            planner = bootstrap.create_level1_solver(
                "i", model, ["j"], n_particles=2, config=config
            )
        else:
            # Build the exact two-state prior below; no empirical bootstrap or
            # unused modeled MCTS solver participates in this reference problem.
            planner = IPOMCPPlanner(
                SolverKey("i", 2), model, model.get_all_actions("i"), bank, config=config
            )
            bank.register_solver(SolverKey("i", 2), planner)
            bank.register_solver(SolverKey("i", 0), RandomPlanner(model.get_all_actions("i")))
            if modeled_config is None:
                provider = FixedTigerL1Policy(model, opponent_depth, gamma)
            else:
                provider = IPOMCPPlanner(
                    SolverKey("j", 1),
                    model,
                    model.get_all_actions("j"),
                    bank,
                    config=modeled_config,
                    exploration_strategy=_exploration(
                        model, "j", opponent_exploration, opponent_exploration_const
                    ),
                )
            bank.register_solver(SolverKey("j", 1), provider)
        planner.exploration_strategy = _exploration(model, "i", exploration, exploration_const)
    elif planner_kind == "rts":
        if exact_final_step or exact_history_rewards or backup != "sampled":
            raise ValueError("Reward, tail and backup ablations are MCTS-only")
        planner = bootstrap.create_level1_rts_solver(
            "i",
            model,
            ["j"],
            n_particles=2,
            config=RTSConfig(gamma=gamma, max_depth=horizon, obs_branching=9, num_particles=budget),
        )
    else:
        raise ValueError("Unknown planner")
    if level == 1:
        opponent = MentalModel(AgentFrame("j", 0, model))
    else:
        l0 = MentalModel(AgentFrame("i", 0, model))
        private = FiniteBelief(
            (
                (InteractiveState(TIGER_LEFT, l0), opponent_belief),
                (InteractiveState(TIGER_RIGHT, l0), 1 - opponent_belief),
            )
        )
        opponent = MentalModel(AgentFrame("j", 1, model), private)
    belief = FiniteBelief(
        (
            (InteractiveState(TIGER_LEFT, opponent), belief_p),
            (InteractiveState(TIGER_RIGHT, opponent), 1 - belief_p),
        )
    )
    planner.set_initial_belief(belief, sample_count=0)
    policy = planner.policy_for(planner.model())
    if modeled_config is not None:
        # Evaluate the exhaustive response after the sampled choice. Shared
        # cached opponent policies define the environment, never i's oracle Q.
        oracle = FinitePolicyL2Reference(model, bank, gamma)
        oracle_q = oracle.q_values(planner.model(), horizon)
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
            "level": level,
            "opponent_model": (
                {"policy": "uniform_l0"}
                if level == 1
                else {
                    "policy": "finite_mcts_l1",
                    "config": asdict(modeled_config),
                    "exploration": {
                        "strategy": opponent_exploration,
                        **vars(provider.exploration_strategy),
                    },
                    "bank_seed": seed,
                    "initial_belief_p": opponent_belief,
                    "tie_rule": "uniform_exact_maximum",
                }
                if modeled_config is not None
                else {
                    "policy": "exact_l1_fixed_horizon",
                    "depth": opponent_depth,
                    "initial_belief_p": opponent_belief,
                    "tie_atol": 1e-10,
                }
            ),
            "bank_seed": seed,
            "planner_config": asdict(planner.config),
            "horizon": horizon,
            "budget": budget,
            "seed": seed,
            "belief_p": belief_p,
            "gamma": gamma,
            "exact_final_step": exact_final_step,
            "exact_history_rewards": exact_history_rewards,
            "backup": backup if planner_kind == "mcts" else None,
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


def _exploration(physics, agent_id, strategy, constant):
    """Physics-derived bounds, shared by real and declared modeled planners."""
    if strategy == "normalized":
        return NormalizedUCB(constant)
    if strategy == "standard":
        return StandardUCB(constant)
    if strategy != "bounded":
        raise ValueError("Unknown exploration strategy")
    other = "j" if agent_id == "i" else "i"
    rewards = [
        physics.get_reward(state, {agent_id: own, other: action}, following, agent_id)
        for state in (TIGER_LEFT, TIGER_RIGHT)
        for own in physics.get_all_actions(agent_id)
        for action in physics.get_all_actions(other)
        for following, probability in physics.transition_distribution(
            state, {agent_id: own, other: action}
        )
        if probability > 0
    ]
    return HorizonBoundUCB(min(rewards), max(rewards), constant)


def _modeled_config(level, depth, budget, gamma, backup, tail, exploration, constant):
    """A budget explicitly selects finite MCTS; no silent model substitution."""
    if budget is None:
        if backup != "sampled" or tail or exploration != "normalized" or constant != 1.0:
            raise ValueError("Modeled MCTS options require an explicit opponent budget")
        return None
    if level != 2 or type(budget) is not int or budget < 3:
        raise ValueError("Opponent budget requires L2 and at least three simulations")
    if exploration not in {"normalized", "standard", "bounded"}:
        raise ValueError("Unknown modeled exploration strategy")
    return IPOMCPConfig(
        mcts=MCTSConfig(
            max_depth=depth,
            n_sims=budget,
            gamma=gamma,
            backup=backup,
            exact_final_step=tail,
            exploration_const=constant,
        ),
        opponent=OpponentPolicyConfig(n_sims=budget),
    )


def _validate_level_contract(level, opponent_depth, opponent_belief, planners):
    """Reject ambiguous opponent semantics before creating any run artifacts."""
    if type(level) is not int or level not in (1, 2):
        raise ValueError("Reference level must be 1 or 2")
    if not 0 <= opponent_belief <= 1:
        raise ValueError("Opponent belief must lie in [0,1]")
    if level == 1:
        if opponent_depth is not None or opponent_belief != 0.5:
            raise ValueError("L1 has uniform L0, not an intentional opponent depth or belief")
    elif type(opponent_depth) is not int or opponent_depth < 1:
        raise ValueError("L2 requires an explicit positive fixed opponent depth")
    elif any(kind != "mcts" for kind in planners):
        raise ValueError("The matched L2 runner currently evaluates MCTS only")


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
    exact_final_step=False,
    backup="sampled",
    workers=2,
    timeout=2400,
    max_rss_mb=4096,
    level=1,
    opponent_depth=None,
    opponent_belief=0.5,
    opponent_budget=None,
    opponent_backup="sampled",
    opponent_exact_final_step=False,
    opponent_exploration="normalized",
    opponent_exploration_const=1.0,
    exact_history_rewards=False,
):
    """Persist every outcome and source fingerprint before interpreting results.

    There is no success threshold picked after seeing results and no automatic
    'optimal' certification. Increasing budgets supplies an error/cost curve;
    sufficient resources depend on the accuracy required for a scientific claim.
    """
    _validate_level_contract(level, opponent_depth, opponent_belief, planners)
    modeled_config = _modeled_config(
        level,
        opponent_depth,
        opponent_budget,
        gamma,
        opponent_backup,
        opponent_exact_final_step,
        opponent_exploration,
        opponent_exploration_const,
    )
    # Validate before workers start so invalid settings cannot create a partial panel.
    MCTSConfig(
        exploration_const=exploration_const,
        exact_final_step=exact_final_step,
        exact_history_rewards=exact_history_rewards,
        backup=backup,
    )
    if (exact_final_step or exact_history_rewards or backup != "sampled") and any(
        kind != "mcts" for kind in planners
    ):
        raise ValueError("Reward, tail and backup ablations are MCTS-only")
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
        exact_final_step=exact_final_step,
        exact_history_rewards=exact_history_rewards,
        backup=backup,
        workers=workers,
        timeout=timeout,
        max_rss_mb=max_rss_mb,
        level=level,
        opponent_depth=opponent_depth,
        opponent_belief=opponent_belief,
        opponent_budget=opponent_budget,
        opponent_backup=opponent_backup,
        opponent_exact_final_step=opponent_exact_final_step,
        opponent_exploration=opponent_exploration,
        opponent_exploration_const=opponent_exploration_const,
    )
    manifest = {
        "settings": settings,
        "modeled_config": asdict(modeled_config) if modeled_config is not None else None,
        "bank_seeds": list(range(seed_start, seed_start + seeds)),
        "scope": "L2 response to declared finite MCTS L1 policy"
        if modeled_config is not None
        else "L1 vs uniform L0"
        if level == 1
        else "L2 vs exact fixed-horizon L1; not finite-budget MCTS opponents",
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
        i: partial(
            evaluate_case,
            *case,
            gamma,
            exploration,
            exploration_const,
            exact_final_step,
            backup,
            level,
            opponent_depth,
            opponent_belief,
            opponent_budget,
            opponent_backup,
            opponent_exact_final_step,
            opponent_exploration,
            opponent_exploration_const,
            exact_history_rewards,
        )
        for i, case in enumerate(cases)
    }
    summaries = []
    started = time.monotonic()
    started_unix = time.time()
    try:
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
            (directory / "summary.json").write_text(
                json.dumps(summaries, indent=2, allow_nan=False)
            )
            print(json.dumps(summary), flush=True)
    finally:
        # Use the same clock as the supervisor. An external elapsed timer may
        # use a different clock domain; retain both rather than overwriting one.
        elapsed = time.monotonic() - started
        worker_sum = sum(row["wall_seconds"] for row in summaries)
        consistent = worker_sum <= workers * elapsed + 1e-6
        timing = {
            "clock": time.get_clock_info("monotonic").implementation,
            "started_unix": started_unix,
            "finished_unix": time.time(),
            "elapsed_monotonic_seconds": elapsed,
            "worker_wall_sum_seconds": worker_sum,
            "workers": workers,
            "requested_cases": len(cases),
            "recorded_cases": len(summaries),
            "worker_concurrency_bound_satisfied": consistent,
        }
        (directory / "timing.json").write_text(json.dumps(timing, indent=2, allow_nan=False))
    if not consistent:
        raise RuntimeError(
            "Worker durations exceed the panel concurrency bound; inspect timing.json"
        )
    if len(summaries) != len(cases) or any(row["status"] != "complete" for row in summaries):
        raise RuntimeError("Oracle panel has failed or missing cases; inspect saved outcomes")


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
    parser.add_argument(
        "--exact-final-step",
        action="store_true",
        help="MCTS only: integrate final-step rewards over the full private-history belief",
    )
    parser.add_argument(
        "--exact-history-rewards",
        action="store_true",
        help="MCTS only: integrate immediate rewards at tree histories; rollouts stay sampled",
    )
    parser.add_argument(
        "--backup",
        choices=["sampled", "empirical_bellman"],
        default="sampled",
        help="MCTS only: mean trajectory returns or empirical chance-weighted Bellman values",
    )
    parser.add_argument("--level", type=int, choices=[1, 2], default=1)
    parser.add_argument(
        "--opponent-depth", type=int, help="Required for L2: fixed L1 replanning depth"
    )
    parser.add_argument(
        "--opponent-belief", type=float, default=0.5, help="L2 point prior on j's P(TL)"
    )
    parser.add_argument(
        "--opponent-budget", type=int, help="L2: finite modeled MCTS simulations; omit for exact L1"
    )
    parser.add_argument(
        "--opponent-backup", choices=["sampled", "empirical_bellman"], default="sampled"
    )
    parser.add_argument("--opponent-exact-final-step", action="store_true")
    parser.add_argument(
        "--opponent-exploration",
        choices=["normalized", "standard", "bounded"],
        default="normalized",
    )
    parser.add_argument("--opponent-exploration-const", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=2400)
    parser.add_argument("--max-rss-mb", type=float, default=4096)
    run_oracle_comparison(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
