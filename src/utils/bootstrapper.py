"""
bootstrapper.py — Factory and dependency-injection pipeline for I-POMDP agent hierarchies.

Instantiates and recursively registers multi-level intentional agent models into a SolverBank:
- Arbitrary recursive reasoning levels (L0 to Lk) for I-POMCP (MCTS with JIT mental models).
- Exact Reachability Tree Sampling (RTS) Oracle baselines reproducing Doshi & Gmytrasiewicz (JAIR 2009).
- Unified DRY implementation delegating specialized convenience factories to recursive root constructors.
"""

import random
from typing import List, Optional, Dict

from core.pomdp_model import POMDPModel, AgentID
from core.config import IPOMCPConfig, RTSConfig
from solvers.solver_bank import SolverBank
from solvers.solver_types import SolverKey, AgentFrame
from solvers.exploration import ExplorationStrategy
from solvers.planner import Planner
from solvers.i_pomcp import IPOMCPPlanner
from solvers.rts_planner import RTSPlanner
from solvers.random_planner import RandomPlanner
from ipomdp.belief import InteractiveParticle
from solvers.node import POMCPNode


class I_POMDP_Bootstrapper:
    """Factory class to instantiate and register agents into a centralized SolverBank."""

    def __init__(self, solver_bank: SolverBank):
        self.bank = solver_bank

    def create_level0_solver(self, agent_id: AgentID, model: POMDPModel) -> SolverKey:
        """Instantiates and registers a Level-0 uniform random baseline agent."""
        key = SolverKey(agent_id, 0)
        self.create_solver(agent_id, 0, model, [])
        return key

    def create_level1_solver(self, agent_id: AgentID, model: POMDPModel, other_agent_ids: List[AgentID],
                             n_particles: int = 100,
                             config: Optional[IPOMCPConfig] = None,
                             exploration_strategy: Optional[ExplorationStrategy] = None) -> IPOMCPPlanner:
        """Instantiates and registers a Level-1 I-POMCP solver (modeling opponents as Level-0)."""
        return self.create_solver(
            agent_id=agent_id,
            level=1,
            model=model,
            other_agent_ids=other_agent_ids,
            level_weights={0: 1.0},
            n_particles=n_particles,
            config=config,
            exploration_strategy=exploration_strategy
        )

    def create_level2_solver(self, agent_id: AgentID, model: POMDPModel, other_agent_ids: List[AgentID],
                             l1_probability: float = 0.9,
                             level_weights: Optional[Dict[int, float]] = None,
                             n_particles: int = 100,
                             config: Optional[IPOMCPConfig] = None,
                             exploration_strategy: Optional[ExplorationStrategy] = None) -> IPOMCPPlanner:
        """Instantiates and registers a Level-2 I-POMCP solver."""
        weights = level_weights if level_weights is not None else {1: l1_probability, 0: max(0.0, 1.0 - l1_probability)}
        return self.create_solver(
            agent_id=agent_id,
            level=2,
            model=model,
            other_agent_ids=other_agent_ids,
            level_weights=weights,
            n_particles=n_particles,
            config=config,
            exploration_strategy=exploration_strategy
        )

    def create_level1_rts_solver(self, agent_id: AgentID, model: POMDPModel, other_agent_ids: List[AgentID],
                                 n_particles: int = 50,
                                 max_depth: int = 3, obs_branching: int = 3,
                                 config: Optional[RTSConfig] = None) -> RTSPlanner:
        """Instantiates and registers a Level-1 Reachability Tree Sampling (RTS) exact branching baseline solver."""
        cfg = config if config is not None else RTSConfig(max_depth=max_depth, obs_branching=obs_branching, num_particles=n_particles)
        return self.create_rts_solver(
            agent_id=agent_id,
            level=1,
            model=model,
            other_agent_ids=other_agent_ids,
            level_weights={0: 1.0},
            n_particles=n_particles,
            config=cfg
        )

    def create_level2_rts_solver(self, agent_id: AgentID, model: POMDPModel, other_agent_ids: List[AgentID],
                                 l1_probability: float = 1.0,
                                 level_weights: Optional[Dict[int, float]] = None,
                                 n_particles: int = 50,
                                 max_depth: int = 3, obs_branching: int = 3,
                                 config: Optional[RTSConfig] = None) -> RTSPlanner:
        """Instantiates and registers a Level-2 Reachability Tree Sampling (RTS) exact branching baseline solver."""
        cfg = config if config is not None else RTSConfig(max_depth=max_depth, obs_branching=obs_branching, num_particles=n_particles)
        weights = level_weights if level_weights is not None else {1: l1_probability, 0: max(0.0, 1.0 - l1_probability)}
        return self.create_rts_solver(
            agent_id=agent_id,
            level=2,
            model=model,
            other_agent_ids=other_agent_ids,
            level_weights=weights,
            n_particles=n_particles,
            config=cfg
        )

    def create_rts_solver(self,
                          agent_id: AgentID,
                          level: int,
                          model: POMDPModel,
                          other_agent_ids: List[AgentID],
                          level_weights: Optional[Dict[int, float]] = None,
                          n_particles: int = 50,
                          config: Optional[RTSConfig] = None) -> Planner:
        """
        Instantiates and registers an RTS exact branching Oracle solver for any level >= 0.
        Faithfully implements Doshi & Gmytrasiewicz (JAIR 2009).
        """
        key = SolverKey(agent_id, level)
        if self.bank.has_solver(key):
            return self.bank.get_solver(key)

        actions = model.get_all_actions(agent_id)
        if level == 0:
            solver = RandomPlanner(actions)
            self.bank.register_solver(key, solver)
            return solver

        # Ensure prerequisite lower-level opponent solvers exist in bank
        for other in other_agent_ids:
            for sub_lvl in range(level):
                sub_key = SolverKey(other, sub_lvl)
                if not self.bank.has_solver(sub_key):
                    if sub_lvl == 0:
                        self.create_solver(other, 0, model, [agent_id])
                    else:
                        self.create_rts_solver(other, sub_lvl, model, [agent_id], n_particles=n_particles, config=config)

        if level_weights is not None:
            raw_weights = level_weights
        else:
            raw_weights = {k: 1.0 / level for k in range(level)}

        total = sum(raw_weights.values())
        weights = {k: v / total for k, v in raw_weights.items()}
        levels = list(weights.keys())
        probs = list(weights.values())

        # Canonical root search trees for each active lower level
        candidate_roots: Dict[AgentID, Dict[int, POMCPNode]] = {}
        for other in other_agent_ids:
            candidate_roots[other] = {}
            for lvl in levels:
                if lvl > 0:
                    opp_key = SolverKey(other, lvl)
                    opp_solver = self.bank.get_solver(opp_key)
                    iso_root = POMCPNode()
                    if hasattr(opp_solver, 'root') and hasattr(opp_solver.root, 'belief_particles'):
                        for p in opp_solver.root.belief_particles:
                            iso_root.add_particle(p)
                    elif hasattr(opp_solver, 'belief') and opp_solver.belief:
                        for p in opp_solver.belief:
                            iso_root.add_particle(p)
                    candidate_roots[other][lvl] = iso_root

        particles = []
        for _ in range(n_particles):
            s = model.get_initial_state()
            models_map = {}
            for other in other_agent_ids:
                chosen_level = random.choices(levels, weights=probs, k=1)[0]
                frame = AgentFrame(other, chosen_level, model)
                if chosen_level == 0:
                    models_map[other] = (frame, None)
                else:
                    models_map[other] = (frame, candidate_roots[other][chosen_level])

            particles.append(InteractiveParticle(state=s, models=models_map))

        rts_cfg = config if config is not None else RTSConfig(num_particles=n_particles)
        rts_planner = RTSPlanner(key, model, actions, self.bank, config=rts_cfg)
        rts_planner.set_initial_belief(particles)
        self.bank.register_solver(key, rts_planner)
        return rts_planner

    def create_solver(self,
                      agent_id: AgentID,
                      level: int,
                      model: POMDPModel,
                      other_agent_ids: List[AgentID],
                      level_weights: Optional[Dict[int, float]] = None,
                      nested_level_weights: Optional[Dict[int, Dict[int, float]]] = None,
                      n_particles: int = 100,
                      config: Optional[IPOMCPConfig] = None,
                      exploration_strategy: Optional[ExplorationStrategy] = None) -> Planner:
        """
        Recursively instantiates and registers an arbitrary Level-L I-POMCP solver.

        Theoretical Model (Finitely Nested I-POMDP):
        IS_{i,l} = S x prod_{j != i} Theta_j^{<l}, where Theta_j^{<l} = union_{k=0}^{l-1} Theta_j^k.

        Guarantees topological instantiation:
        1. If level == 0, registers and returns a RandomPlanner.
        2. If level >= 1, recursively bootstraps all prerequisite models for opponents
           at levels 0, ..., level - 1 into SolverBank.
        3. Samples interactive belief particles b_{agent_id, level} over S x Theta_{-i}^{<level}.
        4. Registers the resulting IPOMCPPlanner under SolverKey(agent_id, level).
        """
        key = SolverKey(agent_id, level)
        if self.bank.has_solver(key):
            return self.bank.get_solver(key)

        actions = model.get_all_actions(agent_id)

        # Base case: Level 0 (Sub-intentional)
        if level == 0:
            solver = RandomPlanner(actions)
            self.bank.register_solver(key, solver)
            return solver

        # Step 1: Recursively ensure all prerequisite sub-levels exist in SolverBank
        all_agents = [agent_id] + list(other_agent_ids)
        for ag in all_agents:
            opponents = [o for o in all_agents if o != ag]
            for sub_lvl in range(level):
                sub_key = SolverKey(ag, sub_lvl)
                if not self.bank.has_solver(sub_key):
                    sub_weights = None
                    if nested_level_weights and sub_lvl in nested_level_weights:
                        sub_weights = nested_level_weights[sub_lvl]
                    self.create_solver(
                        agent_id=ag,
                        level=sub_lvl,
                        model=model,
                        other_agent_ids=opponents,
                        level_weights=sub_weights,
                        nested_level_weights=nested_level_weights,
                        n_particles=n_particles,
                        config=config,
                        exploration_strategy=exploration_strategy
                    )

        # Step 2: Determine level distribution over Theta_j^{<level}
        if level_weights is not None:
            raw_weights = level_weights
        elif nested_level_weights is not None and level in nested_level_weights:
            raw_weights = nested_level_weights[level]
        else:
            # Default uniform prior over all lower levels 0 ... level - 1
            raw_weights = {k: 1.0 / level for k in range(level)}

        total = sum(raw_weights.values())
        weights = {k: v / total for k, v in raw_weights.items()}
        levels = list(weights.keys())
        probs = list(weights.values())

        # Step 2.5: Build canonical root search trees for each active lower level
        candidate_roots: Dict[AgentID, Dict[int, POMCPNode]] = {}
        cap = config.mcts.node_capacity if (config and hasattr(config, 'mcts') and hasattr(config.mcts, 'node_capacity')) else 500
        for other in other_agent_ids:
            candidate_roots[other] = {}
            for lvl in levels:
                if lvl > 0:
                    opp_key = SolverKey(other, lvl)
                    opp_solver = self.bank.get_solver(opp_key)
                    iso_root = POMCPNode(capacity=cap)
                    if hasattr(opp_solver, 'root') and hasattr(opp_solver.root, 'belief_particles'):
                        for p in opp_solver.root.belief_particles:
                            iso_root.add_particle(p)
                    candidate_roots[other][lvl] = iso_root

        # Step 3: Sample interactive belief particles
        particles = []
        for _ in range(n_particles):
            s = model.get_initial_state()
            models_map = {}

            for other in other_agent_ids:
                chosen_level = random.choices(levels, weights=probs, k=1)[0]
                frame = AgentFrame(other, chosen_level, model)

                if chosen_level == 0:
                    models_map[other] = (frame, None)
                else:
                    models_map[other] = (frame, candidate_roots[other][chosen_level])

            particles.append(InteractiveParticle(state=s, models=models_map))

        # Step 4: Instantiate planner, populate root particles, and register
        solver = IPOMCPPlanner(key, model, actions, self.bank,
                               config=config,
                               exploration_strategy=exploration_strategy)

        for p in particles:
            solver.root.add_particle(p)

        solver.initial_particles = list(solver.root.belief_particles)
        solver.prior_level_weights = dict(weights)

        self.bank.register_solver(key, solver)
        return solver