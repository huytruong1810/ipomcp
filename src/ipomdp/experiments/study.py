"""Declarative study specifications: one frozen object says everything a study is.

A ``StudySpec`` names the domain, each agent's level, planner, belief
representation and priors, and the trial design. ``build_agents`` turns it into
planners; ``experiments.runner`` executes it. Earlier versions encoded every
condition as a hand-written runner subclass with literal budgets, which made
conditions hard to compare and easy to misconfigure; a specification is data,
is recorded verbatim in the run manifest, and is the only thing that must
match for a run to resume.

Specifications load from TOML:

    name = "tiger-l2-vs-l1"
    domain = "tiger"
    trials = 50
    steps = 20

    [agents.i]
    level = 2
    planner = { kind = "ipomcp", simulations = 1000, depth = 4, modeled_simulations = 100 }
    modeled = "6400s"
    level_weights = { 1 = 0.8, 0 = 0.2 }

    [agents.j]
    level = 1
    planner = { kind = "ipomcp", simulations = 1000, depth = 4, modeled_simulations = 100 }

Seeds: each agent's bank is seeded from the trial's planner seed and the agent
id, so the two agents never share randomness and a trial is reproducible.
"""

import math
import random
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional

from ipomdp.baselines.rts import RTSConfig
from ipomdp.belief import canonical
from ipomdp.domains.tiger import JointTigerModel, TigerModel
from ipomdp.domains.uav import UAVModel
from ipomdp.domains.wumpus.model import WumpusModel
from ipomdp.experiments.bootstrapper import HierarchyBuilder
from ipomdp.experiments.modeled_levels import parse_stack, planner_config
from ipomdp.planning.config import PlannerConfig
from ipomdp.planning.random_planner import RandomPlanner
from ipomdp.planning.solver_bank import FILTERS, SolverBank

DOMAINS = {
    "tiger": TigerModel,
    "tiger-joint": JointTigerModel,
    "uav": UAVModel,
    "wumpus": WumpusModel,
}
AGENTS = ("i", "j")


@dataclass(frozen=True)
class INTMCPConfig:
    """I-NTMCP (``baselines.intmcp``): root ``simulations`` M, lookahead ``depth``, and its variant."""

    simulations: int
    depth: int
    gamma: float = 0.95
    variant: str = "tuned"

    def __post_init__(self):
        for name in ("simulations", "depth"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 < self.gamma < 1:
            raise ValueError("I-NTMCP needs a discount strictly between 0 and 1")
        if self.variant not in ("published", "tuned"):
            raise ValueError("variant must be 'published' or 'tuned'")


@dataclass(frozen=True)
class AgentSpec:
    """One real agent.

    Attributes:
        level: Reasoning level (0 = uniform random).
        planner: ``{"kind": "ipomcp", ...PlannerConfig fields}``,
            ``{"kind": "rts", ...RTSConfig fields}`` or
            ``{"kind": "intmcp", ...INTMCPConfig fields}``; unused at level 0.
            I-NTMCP models one opponent level from the common prior with its
            own nested trees, so it takes no modeled stack, no belief bound and
            no level prior other than a point mass on ``level - 1``.
        modeled: How each intentional level below this agent is solved, as a
            stack label of ``experiments.modeled_levels`` (top-down, e.g.
            ``"6400s/100"`` at level 3: every level-2 model with stopping capped
            at 6,400, every level-1 model with a flat 100), for both agents at
            each level. The modeled levels share the planner's kind, depth,
            discount and exploration. None: every level uses ``planner``.
            Exact levels are not allowed: their cost is unbounded in an episode.
        particles: None for exact beliefs, or the bound N on belief supports.
        particle_filter: ``"compressed"`` (exact updates compressed to N
            atoms; the planner's default) or ``"sampled"`` (the published IPF).
        search_particles: None, or the bound on the beliefs each search works
            on, separately from the real-step beliefs (e.g. exact real updates
            with deterministic sensors, bounded search).
        level_weights: Prior over the opponent's levels ({level: weight}).
        nested_level_weights: {level: {level: weight}} for nested models.
    """

    level: int
    planner: Optional[Dict] = None
    modeled: Optional[str] = None
    particles: Optional[int] = None
    particle_filter: str = "compressed"
    search_particles: Optional[int] = None
    level_weights: Optional[Dict[int, float]] = None
    nested_level_weights: Optional[Dict[int, Dict[int, float]]] = None

    def __post_init__(self):
        if type(self.level) is not int or self.level < 0:
            raise ValueError("level must be a nonnegative integer")
        if self.particle_filter not in FILTERS:
            raise ValueError(f"particle_filter must be one of {sorted(FILTERS)}")
        if self.level > 0:
            config = _planner_config(self.planner)
            if isinstance(config, INTMCPConfig):
                _check_intmcp(self)
            elif self.modeled is not None:
                _modeled_configs(self)


@dataclass(frozen=True)
class StudySpec:
    name: str
    domain: str
    trials: int
    steps: int
    agents: Dict[str, AgentSpec]
    domain_options: Dict = field(default_factory=dict)
    record_hierarchies: bool = False
    workers: int = 2
    trial_timeout_seconds: float = 2400.0
    max_trial_rss_mb: float = 4096.0

    def __post_init__(self):
        if self.domain not in DOMAINS:
            raise ValueError(f"Unknown domain {self.domain!r}; expected one of {sorted(DOMAINS)}")
        if set(self.agents) != set(AGENTS):
            raise ValueError("A study specifies exactly agents i and j")
        for name in ("trials", "workers"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.steps) is not int or self.steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        for name in ("trial_timeout_seconds", "max_trial_rss_mb"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

    def physics(self):
        return DOMAINS[self.domain](**self.domain_options)

    def to_dict(self):
        """Canonical, JSON-ready form (recorded in run manifests)."""
        return canonical.stable_value(asdict(self))

    @classmethod
    def from_toml(cls, path):
        data = tomllib.loads(Path(path).read_text())
        agents = {
            agent: AgentSpec(
                level=entry["level"],
                planner=entry.get("planner"),
                modeled=entry.get("modeled"),
                particles=entry.get("particles"),
                particle_filter=entry.get("particle_filter", "compressed"),
                search_particles=entry.get("search_particles"),
                level_weights=_int_keys(entry.get("level_weights")),
                nested_level_weights=(
                    {int(k): _int_keys(v) for k, v in entry["nested_level_weights"].items()}
                    if "nested_level_weights" in entry
                    else None
                ),
            )
            for agent, entry in data.pop("agents").items()
        }
        return cls(agents=agents, **data)


def build_agents(spec: StudySpec, seed: int):
    """(physics, {agent: planner}) for one trial; each agent has its own seeded bank."""
    physics = spec.physics()
    planners = {}
    for agent in AGENTS:
        other = "j" if agent == "i" else "i"
        agent_spec = spec.agents[agent]
        bank_seed = int(canonical.digest([seed, agent])[:15], 16)
        if agent_spec.level == 0:
            rng = random.Random(bank_seed)
            planners[agent] = RandomPlanner(physics.get_all_actions(agent), rng)
            continue
        config = _planner_config(agent_spec.planner)
        if isinstance(config, INTMCPConfig):
            # Imported here: the baseline needs the third_party/i-ntmcp submodule.
            from ipomdp.baselines.intmcp import INTMCPAgent, INTMCPPlanner

            planners[agent] = INTMCPPlanner(
                INTMCPAgent(
                    physics,
                    agent_spec.level,
                    config.simulations,
                    config.depth,
                    config.gamma,
                    bank_seed,
                    variant=config.variant,
                    agents=(agent, other),
                )
            )
            continue
        bank = SolverBank(
            bank_seed,
            particles=agent_spec.particles,
            particle_filter=agent_spec.particle_filter,
            search_particles=agent_spec.search_particles,
        )
        planners[agent] = HierarchyBuilder(bank).build(
            agent,
            agent_spec.level,
            physics,
            other,
            config=config,
            modeled_configs=_modeled_configs(agent_spec),
            level_weights=agent_spec.level_weights,
            nested_level_weights=agent_spec.nested_level_weights,
        )
    return physics, planners


def _planner_config(entry):
    if not entry or "kind" not in entry:
        raise ValueError("An intentional agent needs a planner with a 'kind'")
    options = {k: v for k, v in entry.items() if k != "kind"}
    if entry["kind"] == "ipomcp":
        return PlannerConfig(**options)
    if entry["kind"] == "rts":
        return RTSConfig(**options)
    if entry["kind"] == "intmcp":
        return INTMCPConfig(**options)
    raise ValueError(f"Unknown planner kind {entry['kind']!r}")


def _check_intmcp(agent_spec):
    """Refuse what I-NTMCP's model class cannot express (one opponent level, common prior)."""
    unsupported = [
        name
        for name in ("modeled", "particles", "search_particles", "nested_level_weights")
        if getattr(agent_spec, name) is not None
    ]
    if agent_spec.particle_filter != "compressed":
        unsupported.append("particle_filter")
    if unsupported:
        raise ValueError(f"I-NTMCP agents take no {', '.join(unsupported)}")
    weights = agent_spec.level_weights
    if weights is not None and weights != {agent_spec.level - 1: 1.0}:
        raise ValueError("I-NTMCP models exactly one opponent level: level - 1")


def _modeled_configs(agent_spec):
    """{level: configuration} of the intentional levels below the agent, from its stack label."""
    if agent_spec.modeled is None:
        return None
    levels = agent_spec.level - 1
    if levels < 1:
        raise ValueError("A level-1 agent models no intentional level; drop 'modeled'")
    stack = parse_stack(agent_spec.modeled, levels)
    if any(solve is None for solve in stack):
        raise ValueError("Exact modeled levels are not supported in episode studies")
    base = _planner_config(agent_spec.planner)
    kind = "mcts" if isinstance(base, PlannerConfig) else "rts"
    return {
        level: planner_config(
            kind,
            solve.budget,
            base.depth,
            base.gamma,
            getattr(base, "exploration", None),
            solve.stopping,
        )
        for level, solve in zip(range(levels, 0, -1), stack)
    }


def _int_keys(mapping):
    return None if mapping is None else {int(k): float(v) for k, v in mapping.items()}
