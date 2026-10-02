"""Level-3 i versus level-2 j in two-agent Tiger, with both agents' opponent models.

Declared design (September 30):
- Physics: growl accuracy .85 for both agents, creak accuracy .90 under this
  repository's sensor law (a listening opponent never creaks; an opening one
  creaks correctly .90, wrongly .05, silently .05). A heard creak therefore rules
  out every hypothesis that predicted listening.
- i is L3 with prior over j's level {L1: .8, L2: .1, L0: .1}. Its modeled j/L2
  holds {L1: .5, L0: .5} over i, matching the real j's prior (an assumption that
  i knows j's level prior). Every modeled L1 holds uniform L0 opponents.
- j is L2 with prior {L1: .5, L0: .5} over i; the real i (L3) lies outside that
  support, so j's L0 hypothesis absorbs unexplained actions.
- Every physical prior at every nesting level is exactly 50/50 (no sampling).
- Both agents plan a fixed receding lookahead; the episode length is not known
  to the planners. Reported cumulative rewards are undiscounted.

Level posteriors are exact Bayesian filtering relative to finite modeled
policies, so they measure the predictive fit of those finite models rather than
an opponent's true reasoning depth.
"""

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

from ipomdp.domains.tiger import TIGER_LEFT, TIGER_RIGHT, TigerModel
from ipomdp.experiments.batch_runner import (
    GenericBatchRunner,
    _get_opponent_level_distribution,
    extract_nested_belief_hierarchy,
)
from ipomdp.experiments.bootstrapper import I_POMDP_Bootstrapper
from ipomdp.planning.config import (
    DEFAULT_SIM_SCHEDULE,
    ExperimentConfig,
    IPOMCPConfig,
    MCTSConfig,
    OpponentPolicyConfig,
)
from ipomdp.planning.exploration import HorizonBoundUCB, NormalizedUCB
from ipomdp.planning.solver_bank import SolverBank

GROWL = {"i": 0.85, "j": 0.85}
CREAK = 0.90
PRIOR_I = {1: 0.8, 2: 0.1, 0: 0.1}
PRIOR_J = {1: 0.5, 0: 0.5}
NESTED_I = {3: PRIOR_I, 2: PRIOR_J, 1: {0: 1.0}}
NESTED_J = {2: PRIOR_J, 1: {0: 1.0}}
PROFILES = ("default", "validated")


def tiger():
    return TigerModel(growl_accuracy=GROWL, creak_accuracy=CREAK)


def _exploration(profile):
    if profile == "default":
        # The repository's episode runners: sampled backups, normalized UCB.
        return NormalizedUCB(exploration_const=2**0.5)
    # Tiger rewards lie in [-100, 10] for every joint action.
    return HorizonBoundUCB(-100.0, 10.0, 1.0)


def _config(profile, n_sims, depth, gamma, modeled_sims):
    extra = (
        {}
        if profile == "default"
        else dict(
            backup="empirical_bellman",
            exact_final_step=True,
            exact_history_rewards=True,
        )
    )
    return IPOMCPConfig(
        mcts=MCTSConfig(n_sims=n_sims, max_depth=depth, gamma=gamma, **extra),
        opponent=OpponentPolicyConfig(n_sims=modeled_sims),
    )


def build_agent(agent_id, level, profile, n_sims, depth, gamma, modeled_sims, max_support=None):
    """One isolated bank per real agent with an exact balanced physical prior.

    ``max_support`` bounds every interactive posterior in this bank (all nesting
    levels) by deterministic systematic resampling; None keeps exact filtering.
    """
    if profile not in PROFILES:
        raise ValueError(f"Unknown planner profile {profile!r}")
    physics = tiger()
    bank = SolverBank(max_support=max_support)
    bank.set_physical_prior(physics, (TIGER_LEFT, TIGER_RIGHT))
    other = "j" if agent_id == "i" else "i"
    nested = NESTED_I if agent_id == "i" else NESTED_J
    planner = I_POMDP_Bootstrapper(bank).create_solver(
        agent_id=agent_id,
        level=level,
        model=physics,
        other_agent_ids=[other],
        level_weights=nested[level],
        nested_level_weights=nested,
        n_particles=2,
        config=_config(profile, n_sims, depth, gamma, modeled_sims),
        exploration_strategy=_exploration(profile),
    )
    return planner


def predicted_opponent_actions(planner):
    """Per opponent level: that hypothesis's predictive distribution of the next action.

    Uses the bank's own policy law (the one the filter conditions on) at the
    current belief; mixtures within a level are weighted by their belief mass.
    Cached policies make this reuse the next search's queries.
    """
    by_level = {}
    for atom, weight in planner.belief.mass:
        level = atom.opponent.frame.level
        row = by_level.setdefault(level, {"mass": 0.0, "actions": {}})
        row["mass"] += weight
        for action, probability in planner.solver_bank.filter.action_distribution(
            atom.opponent, atom.state
        ):
            row["actions"][str(action)] = (
                row["actions"].get(str(action), 0.0) + weight * probability
            )
    return {
        str(level): {a: p / row["mass"] for a, p in sorted(row["actions"].items())}
        for level, row in sorted(by_level.items())
        if row["mass"] > 0
    }


def hierarchy_values(planner, agent_id):
    """Nested level hierarchy as {path id: joint mass}; ids are stable paths."""
    tree = extract_nested_belief_hierarchy(planner, agent_id)
    return dict(zip(tree["ids"], tree["values"]))


def physical_belief(planner):
    return math.fsum(w for atom, w in planner.belief.mass if atom.state == TIGER_LEFT)


class TigerL3vsL2Runner(GenericBatchRunner):
    def __init__(
        self,
        config,
        log_dir=None,
        profile="default",
        sims_i=DEFAULT_SIM_SCHEDULE[3],
        sims_j=DEFAULT_SIM_SCHEDULE[2],
        modeled_sims=25,
        depth=5,
        gamma=0.95,
        max_support=None,
    ):
        super().__init__(config, log_dir)
        self.max_support = max_support
        self.profile = profile
        self.sims_i, self.sims_j, self.modeled_sims = sims_i, sims_j, modeled_sims
        self.depth, self.gamma = depth, gamma
        self.design = dict(
            growl=GROWL,
            creak=CREAK,
            prior_i={str(k): v for k, v in PRIOR_I.items()},
            prior_j={str(k): v for k, v in PRIOR_J.items()},
            physical_prior="exact 50/50 at every nesting level",
        )

    def _setup_domain(self):
        env = tiger()
        planner_i = build_agent(
            "i",
            3,
            self.profile,
            self.sims_i,
            self.depth,
            self.gamma,
            self.modeled_sims,
            self.max_support,
        )
        planner_j = build_agent(
            "j",
            2,
            self.profile,
            self.sims_j,
            self.depth,
            self.gamma,
            self.modeled_sims,
            self.max_support,
        )
        # The generic loop times only i; record j's decision time on the instance.
        original = planner_j.get_action

        def timed_get_action(*args, **kwargs):
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                planner_j.last_plan_seconds = time.perf_counter() - start

        planner_j.last_plan_seconds = 0.0
        planner_j.get_action = timed_get_action
        return env, planner_i, planner_j, env.get_initial_state()

    def _get_custom_metrics(self, true_state, next_state, env, planner_i, planner_j):
        """Post-update beliefs of both agents; predictions are for the next step."""
        metrics = dict(
            tiger_state=next_state,
            planning_time_j=planner_j.last_plan_seconds,
            p_tl_i=physical_belief(planner_i),
            p_tl_j=physical_belief(planner_j),
            hierarchy_i=json.dumps(hierarchy_values(planner_i, "i")),
            hierarchy_j=json.dumps(hierarchy_values(planner_j, "j")),
            predicted_j_by_level=json.dumps(predicted_opponent_actions(planner_i)),
            predicted_i_by_level=json.dumps(predicted_opponent_actions(planner_j)),
        )
        # i's posterior over j's level is added by the generic loop; add j's over i.
        metrics.update(_get_opponent_level_distribution(planner_j, "i"))
        return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--workers", type=int, default=21)
    parser.add_argument("--profile", choices=PROFILES, default="validated")
    parser.add_argument("--modeled-sims", type=int, default=25)
    parser.add_argument("--sims-i", type=int, default=DEFAULT_SIM_SCHEDULE[3])
    parser.add_argument("--sims-j", type=int, default=DEFAULT_SIM_SCHEDULE[2])
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--max-support", type=int)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--max-rss-mb", type=int, default=3072)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path(
        "results/tiger",
        f"l3_vs_l2_N{args.trials}_T{args.steps}_{datetime.now():%Y%m%d_%H%M%S}",
    )
    runner = TigerL3vsL2Runner(
        ExperimentConfig(
            n_trials=args.trials,
            max_steps=args.steps,
            max_workers=args.workers,
            trial_timeout_seconds=args.timeout,
            max_trial_rss_mb=args.max_rss_mb,
            export_trees=False,
            verbose=False,
        ),
        log_dir=str(output),
        profile=args.profile,
        sims_i=args.sims_i,
        sims_j=args.sims_j,
        modeled_sims=args.modeled_sims,
        depth=args.depth,
        max_support=args.max_support,
    )
    start = time.monotonic()
    df = runner.run_batch()
    print(f"{len(df)} rows in {time.monotonic() - start:.1f}s -> {output}", flush=True)


if __name__ == "__main__":
    main()
