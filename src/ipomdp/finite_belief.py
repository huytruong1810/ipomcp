"""Immutable finite-support subjective models, independent of search trees.

This is the representation consumed by the recursive reference filter. A mass
function keeps physical states and private opponent beliefs JOINTLY distributed.
Two histories may share a value only when their complete resulting distributions
agree; having the same reasoning level is not sufficient. No rounding is used in
equality or cache keys. Finite support is exact relative to the supplied prior,
not a claim that a sampled initial prior equals the environment's true prior.

The current application supports two-agent domains. Encoding exactly one opponent
makes that restriction explicit instead of silently factoring a multi-agent belief.
"""

import math
from dataclasses import dataclass, field

from core.pomdp_model import State
from ipomdp.frame import AgentFrame


@dataclass(frozen=True, slots=True)
class MentalModel:
    """Static frame and private belief; random L0 deliberately has no belief.

    Planning rules belong to the filter's fixed policy provider. A provider must
    specify preferences, horizon, tie rule and computational approximation for
    each frame. Reusing a filter after changing any of those invalidates its model.
    A higher-level model's support may contain only strictly lower-level opponents.
    """

    frame: AgentFrame
    belief: "FiniteBelief | None" = None

    def __post_init__(self):
        if type(self.frame.level) is not int or self.frame.level < 0:
            raise ValueError("Reasoning level must be a nonnegative integer")
        if self.frame.level == 0:
            if self.belief is not None:
                raise ValueError("Uniform-random L0 has no subjective belief")
            return
        if not isinstance(self.belief, FiniteBelief):
            raise TypeError("An intentional model requires an immutable FiniteBelief")
        opponents = set()
        for atom, _ in self.belief.mass:
            other = atom.opponent.frame
            opponents.add(other.agent_id)
            if other.agent_id == self.frame.agent_id or other.level >= self.frame.level:
                raise ValueError("Opponent nesting must change agent and strictly decrease level")
        if len(opponents) != 1:
            raise ValueError("A two-agent belief must refer to one opponent identity")


@dataclass(frozen=True, slots=True)
class InteractiveState:
    """One indivisible hypothesis about physical state and private opponent model."""

    state: State
    opponent: MentalModel

    def __post_init__(self):
        if not isinstance(self.opponent, MentalModel):
            raise TypeError("Opponent must be an immutable MentalModel, never a search node")
        hash(self.state)


@dataclass(frozen=True, slots=True, eq=False)
class FiniteBelief:
    """Normalized positive masses with duplicate hypotheses combined exactly.

    Inputs are defensively copied. Equality ignores insertion order, and a cached
    hash makes repeated nested cache queries cheap. Zero masses are discarded;
    invalid, nonfinite or empty measures are rejected instead of repaired. Normalizing
    by the largest weight avoids overflow of otherwise valid relative weights.
    """

    mass: tuple[tuple[InteractiveState, float], ...]
    _key: frozenset = field(init=False, repr=False)
    _hash: int = field(init=False, repr=False)

    def __post_init__(self):
        entries = tuple(self.mass)
        if not entries:
            raise ValueError("A belief cannot be empty")
        if any(not isinstance(atom, InteractiveState) for atom, _ in entries):
            raise TypeError("Belief support must contain InteractiveState values")
        if any(not math.isfinite(w) or w < 0 for _, w in entries):
            raise ValueError("Belief masses must be finite and nonnegative")
        scale = max(w for _, w in entries)
        if scale == 0:
            raise ValueError("A belief must have positive total mass")
        grouped = {}
        for atom, weight in entries:
            if weight:
                grouped.setdefault(atom, []).append(weight / scale)
        totals = {atom: math.fsum(weights) for atom, weights in grouped.items()}
        total = math.fsum(totals.values())
        mass = tuple((atom, weight / total) for atom, weight in totals.items())
        if any(weight <= 0 for _, weight in mass):
            raise FloatingPointError("Positive belief mass underflowed during normalization")
        key = frozenset(mass)
        object.__setattr__(self, "mass", mass)
        object.__setattr__(self, "_key", key)
        object.__setattr__(self, "_hash", hash(key))

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return isinstance(other, FiniteBelief) and self._key == other._key

    def __reduce__(self):
        # Cached hashes include frame/model identities, which change on unpickle.
        # Reconstruct from values so all derived keys and hashes are recomputed.
        return type(self), (self.mass,)
