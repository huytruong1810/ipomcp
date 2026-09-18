"""Construct immutable finite hierarchies with uniform-random L0.

Physical initial states are empirical samples. Level priors are assigned as exact
weights, not accidentally clipped by a search reservoir. Every hypothesis keeps
its private belief intact; equivalent immutable initial beliefs may safely share.
The bank is one fixed model experiment, and supports exactly two agents.
"""

import math

from core.config import IPOMCPConfig, RTSConfig
from ipomdp.finite_belief import FiniteBelief, InteractiveState, MentalModel
from ipomdp.frame import AgentFrame
from solvers.i_pomcp import IPOMCPPlanner
from solvers.random_planner import RandomPlanner
from solvers.rts_planner import RTSPlanner
from solvers.solver_types import SolverKey


class I_POMDP_Bootstrapper:
    def __init__(self, solver_bank):
        self.bank = solver_bank

    def create_level0_solver(self, agent_id, model):
        self.create_solver(agent_id, 0, model, [])
        return SolverKey(agent_id, 0)

    def create_level1_solver(self, agent_id, model, other_agent_ids, **kwargs):
        return self.create_solver(
            agent_id, 1, model, other_agent_ids, level_weights={0: 1}, **kwargs
        )

    def create_level2_solver(
        self, agent_id, model, other_agent_ids, l1_probability=0.9, level_weights=None, **kwargs
    ):
        weights = (
            level_weights
            if level_weights is not None
            else {1: l1_probability, 0: 1 - l1_probability}
        )
        return self.create_solver(
            agent_id, 2, model, other_agent_ids, level_weights=weights, **kwargs
        )

    def create_level1_rts_solver(
        self,
        agent_id,
        model,
        other_agent_ids,
        n_particles=50,
        max_depth=3,
        obs_branching=3,
        config=None,
    ):
        return self.create_rts_solver(
            agent_id,
            1,
            model,
            other_agent_ids,
            n_particles=n_particles,
            level_weights={0: 1},
            config=config
            or RTSConfig(
                max_depth=max_depth, obs_branching=obs_branching, num_particles=n_particles
            ),
        )

    def create_level2_rts_solver(
        self,
        agent_id,
        model,
        other_agent_ids,
        l1_probability=1.0,
        level_weights=None,
        n_particles=50,
        max_depth=3,
        obs_branching=3,
        config=None,
    ):
        weights = (
            level_weights
            if level_weights is not None
            else {1: l1_probability, 0: 1 - l1_probability}
        )
        return self.create_rts_solver(
            agent_id,
            2,
            model,
            other_agent_ids,
            level_weights=weights,
            n_particles=n_particles,
            config=config
            or RTSConfig(
                max_depth=max_depth, obs_branching=obs_branching, num_particles=n_particles
            ),
        )

    def create_solver(
        self,
        agent_id,
        level,
        model,
        other_agent_ids,
        level_weights=None,
        nested_level_weights=None,
        n_particles=100,
        config=None,
        exploration_strategy=None,
    ):
        return self._create(
            agent_id,
            level,
            model,
            other_agent_ids,
            level_weights,
            nested_level_weights,
            n_particles,
            config or IPOMCPConfig(),
            exploration_strategy,
            False,
        )

    def create_rts_solver(
        self,
        agent_id,
        level,
        model,
        other_agent_ids,
        level_weights=None,
        n_particles=50,
        config=None,
    ):
        return self._create(
            agent_id,
            level,
            model,
            other_agent_ids,
            level_weights,
            None,
            n_particles,
            config or RTSConfig(num_particles=n_particles),
            None,
            True,
        )

    def _create(
        self, agent_id, level, model, others, weights, nested, count, config, exploration, rts
    ):
        if type(level) is not int or level < 0:
            raise ValueError("Reasoning level must be a nonnegative integer")
        if level and (type(count) is not int or count <= 0):
            raise ValueError("Intentional models require a positive initial sample count")
        if weights is not None:
            _validate_prior(weights, level)
        key = SolverKey(agent_id, level)
        if self.bank.has_solver(key):
            return self.bank.get_solver(key)
        if level == 0:
            solver = RandomPlanner(model.get_all_actions(agent_id))
            self.bank.register_solver(key, solver)
            return solver
        if len(others) != 1 or others[0] == agent_id:
            raise ValueError("Exactly one distinct opponent is supported")
        other = others[0]
        for who, opponent in [(agent_id, other), (other, agent_id)]:
            for lower in range(level):
                self._create(
                    who,
                    lower,
                    model,
                    [opponent],
                    nested.get(lower) if nested else None,
                    nested,
                    count,
                    config,
                    exploration,
                    rts,
                )
        weights = (
            weights
            if weights is not None
            else (
                nested.get(level)
                if nested and level in nested
                else dict.fromkeys(range(level), 1 / level)
            )
        )
        _validate_prior(weights, level)
        total = sum(weights.values())
        candidates = {}
        for lower, mass in weights.items():
            if mass <= 0:
                continue
            solver = self.bank.get_solver(SolverKey(other, lower))
            candidates[lower] = MentalModel(
                AgentFrame(other, lower, model), solver.initial_belief if lower else None
            )
        physical = self.bank.initial_states(model, count)
        belief = FiniteBelief(
            tuple(
                (InteractiveState(state, candidates[lower]), mass / total / count)
                for state in physical
                for lower, mass in weights.items()
                if mass > 0
            )
        )
        if rts:
            solver = RTSPlanner(
                key, model, model.get_all_actions(agent_id), self.bank, config=config
            )
        else:
            solver = IPOMCPPlanner(
                key,
                model,
                model.get_all_actions(agent_id),
                self.bank,
                config=config,
                exploration_strategy=exploration,
            )
        solver.set_initial_belief(belief, count)
        self.bank.register_solver(key, solver)
        return solver


def _validate_prior(weights, level):
    if not weights or any(type(k) is not int or not 0 <= k < level for k in weights):
        raise ValueError("Prior support must contain only levels 0 <= k < agent level")
    if any(not math.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError("Prior weights must be finite and nonnegative")
    total = sum(weights.values())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Prior weights must have finite positive total")
