"""Canonical, process-independent identity for interactive states and beliefs.

Why this module exists
----------------------
``FiniteBelief`` equality ignores the insertion order of its atoms, and frames
compare physics by object identity. Both are right for in-process caching, but
neither gives an *ordering* or a *digest* that is the same for two equal beliefs
built along different update paths, or in different processes. Every consumer
that must behave identically on equal beliefs therefore needs one canonical
encoding:

- search seeding (equal beliefs must receive equal search randomness);
- the order in which belief atoms are presented to ``random.choices``;
- deterministic bounded resampling in the interactive filter.

Before this module existed, the solver bank had such an encoding but the filter
did not, so bounded resampling depended on insertion order (two equal beliefs
could compress to different posteriors). Keeping the encoding here, next to the
belief types, lets every layer share it without importing the solver layer.

Encoding rules
--------------
- Values are encoded structurally by ``stable_value``: never ``repr``, object
  ids, or Python's randomized ``hash``. Floats are encoded exactly via ``hex``.
- Physics is identified by its class and its public attribute values, so two
  separately constructed but identical models encode identically.
- A nested belief is encoded by its digest, computed recursively and memoized.
  The memo is keyed by ``FiniteBelief`` equality, which is exact (no rounding).
"""

import hashlib
import json
import math
from dataclasses import fields, is_dataclass
from functools import lru_cache


def stable_value(value):
    """Encode plain data canonically; never use repr, object IDs or randomized hashes.

    Supported: None, str, bool, int, finite float (as exact hex), dataclasses,
    dicts, sets/frozensets and sequences, recursively. Anything else raises,
    because a silently lossy encoding would merge distinct models.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Nonfinite value cannot identify a model")
        return ["float", value.hex()]
    if is_dataclass(value):
        return [
            type(value).__module__,
            type(value).__qualname__,
            [(f.name, stable_value(getattr(value, f.name))) for f in fields(value)],
        ]
    if isinstance(value, dict):
        return [
            "dict",
            sorted([(stable_value(k), stable_value(v)) for k, v in value.items()], key=json.dumps),
        ]
    if isinstance(value, (set, frozenset)):
        return ["set", sorted([stable_value(v) for v in value], key=json.dumps)]
    if isinstance(value, (list, tuple)):
        return ["sequence", [stable_value(v) for v in value]]
    raise TypeError(f"No canonical model encoding for {type(value).__qualname__}")


def digest(value):
    """SHA-256 of the compact canonical JSON of an already stable value."""
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def physics_identity(physics):
    """Class plus public attribute values; equal configurations encode equally."""
    return [type(physics).__module__, type(physics).__qualname__, stable_value(vars(physics))]


def state_key(state):
    """Canonical text key of a physical state, used for ordering and strata."""
    return json.dumps(stable_value(state), sort_keys=True)


def atom_row(atom, mass):
    """Canonical encoding of one weighted interactive-state atom.

    The opponent's private belief enters through its digest, so the row has
    bounded size regardless of nesting depth.
    """
    opponent = atom.opponent
    return [
        stable_value(atom.state),
        mass.hex(),
        stable_value(opponent.frame.agent_id),
        opponent.frame.level,
        *physics_identity(opponent.frame.pomdp_model),
        belief_digest(opponent.belief) if opponent.belief is not None else None,
    ]


def atom_sort_key(row):
    """Sort key for an (atom, mass) pair, consistent with ``belief_digest``."""
    return json.dumps(atom_row(*row))


@lru_cache(maxsize=65536)
def belief_digest(belief):
    """Order-independent digest of a FiniteBelief, recursive over nested beliefs."""
    return digest(sorted([atom_row(atom, mass) for atom, mass in belief.mass], key=json.dumps))


def canonical_mass(belief):
    """The belief's (atom, mass) pairs in canonical order; masses are untouched."""
    return tuple(sorted(belief.mass, key=atom_sort_key))
