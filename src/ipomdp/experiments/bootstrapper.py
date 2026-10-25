"""Construct finitely nested agent hierarchies in one solver bank.

What a hierarchy is
-------------------
An agent at level l > 0 holds a belief over (physical state, opponent model),
where every opponent model has a level strictly below l. Building one agent
therefore builds, recursively, the solvers for both agents at every lower level
in the same bank, because the bank answers "what does this modeled opponent
do?" by asking the registered solver for that (agent, level).

Prior construction (declared assumptions)
-----------------------------------------
- The physical prior is the domain's exact ``initial_distribution``, shared by
  every nesting level.
- The prior over opponent levels is ``level_weights`` (default uniform over
  lower levels), independent of the physical state.
- Each opponent level is represented by exactly one model: the registered
  solver's own initial belief. "i's model of j/L1" is therefore j/L1's own
  prior, a common-prior assumption made explicit here.
- ``nested_level_weights`` maps level -> level weights for the models nested
  inside, so a study can declare, e.g., what i believes j believes about i.

Level 0 is uniformly random over the full action set, with its own seeded
action stream.
"""

import math
import random

from ipomdp.baselines.rts import RTSConfig, RTSPlanner
from ipomdp.belief.finite_belief import InteractiveState, MentalModel
from ipomdp.belief.frame import AgentFrame
from ipomdp.planning.config import PlannerConfig
from ipomdp.planning.i_pomcp import IPOMCPPlanner
from ipomdp.planning.random_planner import RandomPlanner
from ipomdp.planning.solver_types import SolverKey


class HierarchyBuilder:
    def __init__(self, solver_bank):
        self.bank = solver_bank

    def build(
        self,
        agent_id,
        level,
        model,
        opponent_id,
        *,
        config=None,
        modeled_configs=None,
        level_weights=None,
        nested_level_weights=None,
    ):
        """Build (or return the registered) solver for ``agent_id`` at ``level``.

        Args:
            config: PlannerConfig (I-POMCP) or RTSConfig (IPF + RTS baseline) of
                this agent. Required for level > 0.
            modeled_configs: {level: configuration} of the lower intentional
                models, one per level for both agents (levels are queried at
                different beliefs and need different budgets, RESULTS.md 12.9;
                ``experiments.modeled_levels`` names such stacks). A level not
                listed uses the configuration of the level that models it, so a
                hierarchy is homogeneous unless a study states otherwise.
            level_weights: Prior over the opponent's levels for this agent.
            nested_level_weights: {level: weights} for nested models.
        """
        key = SolverKey(agent_id, level)
        if self.bank.has_solver(key):
            return self.bank.get_solver(key)
        if type(level) is not int or level < 0:
            raise ValueError("Reasoning level must be a nonnegative integer")
        if level == 0:
            rng = random.Random(self.bank.stream_seed("actions", str(key)))
            solver = RandomPlanner(model.get_all_actions(agent_id), rng)
            self.bank.register_solver(key, solver)
            return solver
        if opponent_id == agent_id:
            raise ValueError("Exactly one distinct opponent is supported")
        if not isinstance(config, (PlannerConfig, RTSConfig)):
            raise TypeError("An intentional agent needs a PlannerConfig or RTSConfig")
        modeled_configs = modeled_configs or {}
        nested = nested_level_weights or {}
        for who, other in ((agent_id, opponent_id), (opponent_id, agent_id)):
            for lower in range(level):
                self.build(
                    who,
                    lower,
                    model,
                    other,
                    config=modeled_configs.get(lower, config),
                    modeled_configs=modeled_configs,
                    level_weights=nested.get(lower),
                    nested_level_weights=nested,
                )
        weights = level_weights or nested.get(level) or dict.fromkeys(range(level), 1 / level)
        _validate_prior(weights, level)
        total = math.fsum(weights.values())
        opponents = {}
        for lower, mass in weights.items():
            if mass > 0:
                solver = self.bank.get_solver(SolverKey(opponent_id, lower))
                physics = solver.pomdp_model if lower else model
                belief = solver.initial_belief if lower else None
                opponents[lower] = MentalModel(AgentFrame(opponent_id, lower, physics), belief)
        # The bank's filter decides the representation: exact rows, or particles.
        belief = self.bank.filter.initial_belief(
            (InteractiveState(state, opponents[lower]), p_state * mass / total)
            for state, p_state in self.bank.physical_prior(model)
            for lower, mass in weights.items()
            if mass > 0
        )
        planner_class = RTSPlanner if isinstance(config, RTSConfig) else IPOMCPPlanner
        solver = planner_class(key, model, self.bank, config)
        solver.set_initial_belief(belief)
        self.bank.register_solver(key, solver)
        return solver


def _validate_prior(weights, level):
    if not weights or any(type(k) is not int or not 0 <= k < level for k in weights):
        raise ValueError("Prior support must contain only levels 0 <= k < agent level")
    if any(not math.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError("Prior weights must be finite and nonnegative")
    total = math.fsum(weights.values())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Prior weights must have finite positive total")
