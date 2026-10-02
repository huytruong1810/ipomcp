"""Level-3 i versus level-2 j in two-agent Tiger, with both agents' opponent models.

Declared design:
- Physics: the published two-agent Tiger (growl .85, creak .90, false creaks).
- i is L3 with prior over j's level {L1: .8, L2: .1, L0: .1}. Its modeled j/L2
  holds {L1: .5, L0: .5} over i, matching the real j's prior (an assumption that
  i knows j's level prior). Every modeled L1 holds uniform L0 opponents.
- j is L2 with prior {L1: .5, L0: .5} over i; the real i (L3) lies outside that
  support, so j's L0 hypothesis absorbs unexplained actions.
- Every physical prior at every nesting level is the domain's exact 50/50.
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

from ipomdp.domains.tiger import TIGER_LEFT, TigerModel
from ipomdp.experiments.batch_runner import (
    GenericBatchRunner,
    extract_nested_belief_hierarchy,
    opponent_level_distribution,
)
from ipomdp.experiments.bootstrapper import HierarchyBuilder
from ipomdp.experiments.config import ExperimentConfig
from ipomdp.planning.config import PlannerConfig
from ipomdp.planning.solver_bank import SolverBank

GROWL = {"i": 0.85, "j": 0.85}
CREAK = 0.90
PRIOR_I = {1: 0.8, 2: 0.1, 0: 0.1}
PRIOR_J = {1: 0.5, 0: 0.5}
NESTED_I = {3: PRIOR_I, 2: PRIOR_J, 1: {0: 1.0}}
NESTED_J = {2: PRIOR_J, 1: {0: 1.0}}


def tiger():
    return TigerModel(growl_accuracy=GROWL, creak_accuracy=CREAK)


def build_agent(agent_id, level, n_sims, depth, gamma, modeled_sims, seed, max_support=None):
    """One isolated, seeded bank per real agent.

    ``max_support`` bounds every interactive posterior in this bank (all nesting
    levels) by deterministic stratified resampling; None keeps exact filtering.
    """
    physics = tiger()
    bank = SolverBank(seed, max_support=max_support)
    other = "j" if agent_id == "i" else "i"
    nested = NESTED_I if agent_id == "i" else NESTED_J
    config = PlannerConfig(
        simulations=n_sims, depth=depth, modeled_simulations=modeled_sims, gamma=gamma
    )
    return HierarchyBuilder(bank).build(
        agent_id,
        level,
        physics,
        other,
        config=config,
        level_weights=nested[level],
        nested_level_weights=nested,
    )


def predicted_opponent_actions(planner):
    """Per opponent level: that hypothesis's predictive distribution of the next action.

    Uses the bank's own policy law (the one the filter conditions on) at the
    current belief; mixtures within a level are weighted by their belief mass.
    """
    by_level = {}
    for atom, weight in planner.belief.mass:
        level = atom.opponent.frame.level
        row = by_level.setdefault(level, {"mass": 0.0, "actions": {}})
        row["mass"] += weight
        for action, probability in planner.solver_bank.filter.action_distribution(atom.opponent):
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
        *,
        sims_i,
        sims_j,
        modeled_sims,
        depth=5,
        gamma=0.95,
        max_support=None,
    ):
        super().__init__(config, log_dir)
        self.sims_i, self.sims_j, self.modeled_sims = sims_i, sims_j, modeled_sims
        self.depth, self.gamma, self.max_support = depth, gamma, max_support
        self.design = dict(
            growl=GROWL,
            creak=CREAK,
            prior_i={str(k): v for k, v in PRIOR_I.items()},
            prior_j={str(k): v for k, v in PRIOR_J.items()},
            physical_prior="the domain prior (50/50) at every nesting level",
        )

    def _setup_domain(self, seed):
        env = tiger()
        common = dict(depth=self.depth, gamma=self.gamma, modeled_sims=self.modeled_sims)
        planner_i = build_agent(
            "i", 3, self.sims_i, seed=seed, max_support=self.max_support, **common
        )
        planner_j = build_agent(
            "j", 2, self.sims_j, seed=seed + 1, max_support=self.max_support, **common
        )
        return env, planner_i, planner_j

    def _get_custom_metrics(self, true_state, next_state, env, planner_i, planner_j):
        """Post-update beliefs of both agents; predictions are for the next step."""
        metrics = dict(
            tiger_state=next_state,
            p_tl_i=physical_belief(planner_i),
            p_tl_j=physical_belief(planner_j),
            hierarchy_i=json.dumps(hierarchy_values(planner_i, "i")),
            hierarchy_j=json.dumps(hierarchy_values(planner_j, "j")),
            predicted_j_by_level=json.dumps(predicted_opponent_actions(planner_i)),
            predicted_i_by_level=json.dumps(predicted_opponent_actions(planner_j)),
        )
        metrics.update(opponent_level_distribution(planner_j, "i"))
        return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--sims-i", type=int, required=True)
    parser.add_argument("--sims-j", type=int, required=True)
    parser.add_argument("--modeled-sims", type=int, required=True)
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
        ),
        log_dir=str(output),
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
