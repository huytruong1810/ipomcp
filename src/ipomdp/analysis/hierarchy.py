"""Nested belief hierarchies: who is believed to be at which level, and what they believe.

A hierarchy flattens an agent's interactive belief into paths of models. The
root is the agent itself. Its children are the opponent levels it entertains,
each opponent level's children are the levels that opponent entertains about
the agent, and so on down to level 0. For every path the hierarchy records:

- ``mass``: the joint probability of the path (product of conditional masses);
- ``physical``: what the models at that path believe about the physical state,
  as a mass-weighted average of their marginals (absent for level 0, which
  holds no belief).

The physical belief is the *content* that a level-only view hides: two j/L1
models, one confident the tiger is left and one confident it is right, carry
the same level label but predict opposite actions. ``feature`` maps a physical
state to the quantity shown (the state itself for Tiger; a projection such as
the target's cell for larger domains).

``hierarchy_error`` compares two hierarchies (an approximation against the
exact belief) per nesting depth, which is how belief error is attached to the
interpretability views.
"""

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple


@dataclass(frozen=True)
class HierarchyNode:
    path: str
    parent: str
    agent: str
    level: int
    mass: float
    physical: Optional[Dict[str, float]] = field(default=None, hash=False)

    @property
    def depth(self) -> int:
        return self.path.count("/")

    def to_dict(self):
        return dict(
            path=self.path,
            parent=self.parent,
            agent=self.agent,
            level=self.level,
            mass=self.mass,
            physical=self.physical,
        )


def nested_hierarchy(
    belief, agent_id, level, depth=8, feature: Callable = lambda state: state
) -> Tuple[HierarchyNode, ...]:
    """Flatten ``belief`` (agent ``agent_id`` at ``level``) into hierarchy nodes."""
    root = f"{agent_id}_L{level}"
    nodes = [
        HierarchyNode(root, "", str(agent_id), level, 1.0, _physical([(belief, 1.0)], feature))
    ]

    def descend(weighted_beliefs, path, remaining):
        if remaining == 0:
            return
        groups = {}
        for nested, mass in weighted_beliefs:
            for atom, weight in nested.mass:
                frame = atom.opponent.frame
                groups.setdefault((str(frame.agent_id), frame.level), []).append(
                    (atom.opponent.belief, mass * weight)
                )
        for (agent, lvl), members in sorted(groups.items()):
            child = f"{path}/{agent}_L{lvl}"
            total = math.fsum(m for _, m in members)
            held = [(b, m) for b, m in members if b is not None]
            nodes.append(
                HierarchyNode(
                    child, path, agent, lvl, total, _physical(held, feature) if held else None
                )
            )
            if held:
                descend(held, child, remaining - 1)

    descend([(belief, 1.0)], root, depth)
    return tuple(nodes)


def level_posterior(belief) -> Dict[int, float]:
    """The belief's marginal over the opponent's level."""
    masses = {}
    for atom, weight in belief.mass:
        masses[atom.opponent.frame.level] = masses.get(atom.opponent.frame.level, 0.0) + weight
    return masses


def hierarchy_error(approximate, exact) -> Dict[int, Dict[str, float]]:
    """Per nesting depth: TV of path masses and mass-weighted physical-belief TV.

    Physical TV at a path is weighted by the smaller of the two path masses, so
    a path present in only one hierarchy counts toward the mass TV only.
    """
    a = {node.path: node for node in approximate}
    e = {node.path: node for node in exact}
    result = {}
    for path in set(a) | set(e):
        depth = path.count("/")
        row = result.setdefault(depth, {"mass_tv": 0.0, "physical_tv": 0.0, "weight": 0.0})
        mass_a = a[path].mass if path in a else 0.0
        mass_e = e[path].mass if path in e else 0.0
        row["mass_tv"] += 0.5 * abs(mass_a - mass_e)
        if (
            path in a
            and path in e
            and a[path].physical is not None
            and e[path].physical is not None
        ):
            weight = min(mass_a, mass_e)
            row["physical_tv"] += weight * _tv(a[path].physical, e[path].physical)
            row["weight"] += weight
    return {
        depth: {
            "mass_tv": row["mass_tv"],
            "physical_tv": row["physical_tv"] / row["weight"] if row["weight"] else 0.0,
        }
        for depth, row in sorted(result.items())
    }


def _physical(weighted_beliefs, feature):
    total = math.fsum(m for _, m in weighted_beliefs)
    marginal = {}
    for belief, mass in weighted_beliefs:
        for atom, weight in belief.mass:
            key = str(feature(atom.state))
            marginal[key] = marginal.get(key, 0.0) + mass * weight / total
    return dict(sorted(marginal.items()))


def _tv(p, q):
    return 0.5 * math.fsum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in set(p) | set(q))
