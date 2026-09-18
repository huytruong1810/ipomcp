"""One greedy policy contract and reproducible, isolated search randomness.

MCTS and sampled lookahead approximate Q; action execution and opponent prediction
both distribute mass uniformly over maximizing estimates. A bounded solve is not
an optimal policy. In particular, different real/model budgets remain an explicit
approximation. Nested solves restore the caller's RNG stream even on failure.
"""

import hashlib
import json
import math
import random
from contextlib import contextmanager
from dataclasses import fields, is_dataclass


def stable_value(value):
    """Canonical data encoding; never use repr, object IDs or randomized hashes."""
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
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def greedy_policy(values):
    if not values or any(not math.isfinite(q) for q in values.values()):
        raise ValueError("A policy requires finite action values")
    best = max(values.values())
    actions = [a for a, q in values.items() if q == best]
    return {a: 1.0 / len(actions) for a in actions}


def sample_policy(policy):
    return random.choices(list(policy), weights=list(policy.values()), k=1)[0]


@contextmanager
def search_randomness(seed):
    """Process-local serial searches only; simultaneous threads are not supported."""
    previous = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(previous)
