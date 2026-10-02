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
- The physical prior is shared by every nesting level of the bank: either the
  domain's declared prior (``SolverBank.set_physical_prior``) or ``prior_samples``
  draws from ``get_initial_state``. Sharing it prevents a nested model from
  ruling out an observation only because its own sample missed a state.
- The prior over opponent levels is ``level_weights`` (default uniform over
  lower levels), applied independently of the physical state.
- Each opponent level is represented by exactly one model: the registered
  solver's own initial belief. "i's model of j/L1" is therefore j/L1's own
  prior, a common-prior assumption made explicit here.
- ``nested_level_weights`` maps level -> level weights for the models nested
  inside, so a study can declare, e.g., what i believes j believes about i.

Level 0 is uniformly random over the full action alphabet.
"""

import math

from ipomdp.baselines.rts import RTSConfig, RTSPlanner
from ipomdp.belief.finite_belief import FiniteBelief, InteractiveState, MentalModel
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
        modeled_config=None,
        level_weights=None,
        nested_level_weights=None,
        prior_samples=None,
    ):
        """Build (or return the registered) solver for ``agent_id`` at ``level``.

        Args:
            config: PlannerConfig or RTSConfig of this agent. Required for level > 0.
            modeled_config: PlannerConfig for every lower intentional model.
                Defaults to ``config`` when that is a PlannerConfig; an RTS root
                above level 1 must state how its intentional opponents are modeled.
            level_weights: Prior over the opponent's levels for this agent.
            nested_level_weights: {level: weights} for nested models.
            prior_samples: Size of the sampled physical prior, when the bank has
                no declared prior for this domain.
        """
        key = SolverKey(agent_id, level)
        if self.bank.has_solver(key):
            return self.bank.get_solver(key)
        if type(level) is not int or level < 0:
            raise ValueError("Reasoning level must be a nonnegative integer")
        if level == 0:
            solver = RandomPlanner(model.get_all_actions(agent_id))
            self.bank.register_solver(key, solver)
            return solver
        if opponent_id == agent_id:
            raise ValueError("Exactly one distinct opponent is supported")
        if not isinstance(config, (PlannerConfig, RTSConfig)):
            raise TypeError("An intentional agent needs a PlannerConfig or RTSConfig")
        if modeled_config is None and isinstance(config, PlannerConfig):
            modeled_config = config
        if modeled_config is None and level > 1:
            raise ValueError(
                "An RTS agent above level 1 must declare how its opponents are modeled"
            )
        nested = nested_level_weights or {}
        for who, other in ((agent_id, opponent_id), (opponent_id, agent_id)):
            for lower in range(level):
                self.build(
                    who,
                    lower,
                    model,
                    other,
                    config=modeled_config,
                    modeled_config=modeled_config,
                    level_weights=nested.get(lower),
                    nested_level_weights=nested,
                    prior_samples=prior_samples,
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
        states = self.bank.physical_prior(model, prior_samples)
        belief = FiniteBelief(
            tuple(
                (InteractiveState(state, opponents[lower]), mass / total / len(states))
                for state in states
                for lower, mass in weights.items()
                if mass > 0
            )
        )
        if isinstance(config, RTSConfig):
            solver = RTSPlanner(key, model, self.bank, config)
        else:
            solver = IPOMCPPlanner(key, model, self.bank, config)
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
