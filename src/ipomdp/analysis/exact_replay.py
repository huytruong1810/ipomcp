"""Replay an agent's realized episode through the exact Bayes-optimal agent (PLAN N4(a)).

At every step of the exact window, the exact agent of the same level, frames,
priors and lookahead holds the exact belief given the agent's own realized
actions and observations, and is compared with what the agent recorded:

- ``agree``: the agent's action is one of the exact maximizers;
- ``level_tv``: total variation between the agent's posterior over the
  opponent's level and the exact one (both before the decision);
- ``prediction_tv``: total variation between the agent's predicted opponent
  action and the exact prediction (``episodes.predicted_opponent_action``).

The exact agent's policies and updates are pure functions of its models, so
one ``ExactReplay`` serves any number of trials and its caches only save work.
The window ends where an exact update exceeds its enumeration budget
(``InferenceBudgetExceeded``), at ``max_steps``, or when the episode ends.
"""

import json
import math

from ipomdp.analysis.exact_curves import uniform_nesting
from ipomdp.analysis.hierarchy import level_posterior
from ipomdp.belief.finite_filter import InferenceBudgetExceeded
from ipomdp.experiments.episodes import predicted_opponent_action
from ipomdp.reference.exact_agent import TIE_TOLERANCE, build_exact_agent


class ExactReplay:
    def __init__(
        self,
        physics,
        agent,
        level,
        horizon,
        gamma=0.95,
        nested=None,
        max_steps=None,
        max_branches=100_000,
    ):
        if level < 1:
            raise ValueError("Only intentional agents (level >= 1) have an exact counterpart")
        self.agent, self.level, self.max_steps = agent, level, max_steps
        self.other = "j" if agent == "i" else "i"
        self.planner = build_exact_agent(
            agent,
            self.other,
            level,
            physics,
            nested or uniform_nesting(level),
            horizon,
            gamma,
            max_branches=max_branches,
        )
        self.prior = self.planner.belief
        self.actions = {str(a): a for a in physics.get_all_actions(agent)}
        self.observations = {str(o): o for o in physics.get_all_observations(agent)}

    def clear_caches(self):
        """Drop the exact agent's caches between trials (memory only, never results)."""
        self.planner.reference.clear_caches()
        self.planner.solver_bank.clear_caches()

    def replay(self, frame):
        """Per decision step of one trial's rows: step, agree, level_tv, prediction_tv, support."""
        rows = frame.sort_values("step").to_dict("records")
        planner, agent, other = self.planner, self.agent, self.other
        planner.belief, out = self.prior, []
        for before, row in zip(rows, rows[1:]):
            action = row.get(f"action_{agent}")
            if row.get("status") == "Window" or not isinstance(action, str):
                break
            if self.max_steps is not None and row["step"] > self.max_steps:
                break
            q = planner.q_values(planner.model())
            best = max(q.values())
            maximizers = {
                str(a) for a, v in q.items() if v >= best - TIE_TOLERANCE * max(1.0, abs(best))
            }
            exact_levels = level_posterior(planner.belief)
            recorded = json.loads(row[f"prediction_{agent}"])
            out.append(
                {
                    "step": row["step"],
                    "agree": action in maximizers,
                    "unique_maximizer": len(maximizers) == 1,
                    "level_tv": _tv(
                        {k: before[f"prob_l{k}_{other}"] for k in range(self.level)}, exact_levels
                    ),
                    "prediction_tv": _tv(recorded, predicted_opponent_action(planner)),
                    "support": len(planner.belief.mass),
                }
            )
            observation = row.get(f"observation_{agent}")
            if not isinstance(observation, str):
                break  # the episode ended
            try:
                planner.update_root(self.actions[action], self.observations[observation])
            except InferenceBudgetExceeded:
                break
        return out


def _tv(p, q):
    keys = set(p) | set(q)
    return 0.5 * math.fsum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)
