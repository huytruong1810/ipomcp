"""Alpha-vector representations and exact pruning operators.

Exact upper-envelope pruning for alpha vectors over a two-state belief, with the
interval of beliefs on which each surviving vector is optimal. Used by the L1
Tiger value iteration, the independent exact POMDP reference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class AlphaVector2D:
    """An alpha vector over a binary state space S = {s_0, s_1}.

    Under belief b(s_0) = p, b(s_1) = 1 - p:
        V(p) = p * v0 + (1 - p) * v1 = v1 + p * (v0 - v1)
    """

    v0: float
    v1: float
    action: Optional[Any] = None

    def value(self, p: float) -> float:
        """Evaluate the alpha vector at belief p = P(s_0)."""
        return p * self.v0 + (1.0 - p) * self.v1

    @property
    def slope(self) -> float:
        """Slope m in y(p) = m * p + c."""
        return self.v0 - self.v1

    @property
    def intercept(self) -> float:
        """Intercept c at p = 0."""
        return self.v1

    def __repr__(self) -> str:
        act_str = f"[{self.action}] " if self.action is not None else ""
        return f"Alpha2D({act_str}v0={self.v0:.4f}, v1={self.v1:.4f})"


def _intersect_2d(v1: AlphaVector2D, v2: AlphaVector2D) -> float:
    """Find the intersection belief p where v1.value(p) == v2.value(p).

    v1.intercept + p * v1.slope = v2.intercept + p * v2.slope
    p * (v1.slope - v2.slope) = v2.intercept - v1.intercept
    """
    denom = v1.slope - v2.slope
    if abs(denom) < 1e-12:
        return -float("inf") if v1.intercept > v2.intercept else float("inf")
    return (v2.intercept - v1.intercept) / denom


def prune_2d_with_intervals(
    vectors: Sequence[AlphaVector2D],
    tol: float = 1e-9,
) -> List[Tuple[AlphaVector2D, float, float]]:
    """Compute the exact upper envelope of 2D alpha vectors over p in [0, 1].

    Returns a list of tuples: (alpha_vector, left_p, right_p) defining the
    contiguous interval of optimality for each non-dominated vector.
    Runs in O(K log K) time using the dual lower convex hull sweep algorithm.
    """
    if not vectors:
        return []

    # 1. Deduplicate identical vectors within floating tolerance
    unique: dict[Tuple[float, float], AlphaVector2D] = {}
    for v in vectors:
        key = (round(v.v0, 9), round(v.v1, 9))
        if key not in unique or (unique[key].action is None and v.action is not None):
            unique[key] = v

    vecs = list(unique.values())
    if len(vecs) <= 1:
        return [(vecs[0], 0.0, 1.0)]

    # 2. Sort by slope ascending, breaking ties by intercept descending
    vecs.sort(key=lambda v: (v.slope, v.intercept))

    # 3. Filter parallel lines: keep only the highest intercept for identical slopes
    filtered: List[AlphaVector2D] = []
    for v in vecs:
        if filtered and math.isclose(filtered[-1].slope, v.slope, rel_tol=0.0, abs_tol=tol):
            if v.intercept > filtered[-1].intercept:
                filtered[-1] = v
        else:
            filtered.append(v)
    vecs = filtered

    if len(vecs) <= 1:
        return [(vecs[0], 0.0, 1.0)]

    if len(vecs) == 2:
        v0, v1 = vecs[0], vecs[1]
        if v0.v0 >= v1.v0 - tol and v0.v1 >= v1.v1 - tol:
            return [(v0, 0.0, 1.0)]
        if v1.v0 >= v0.v0 - tol and v1.v1 >= v0.v1 - tol:
            return [(v1, 0.0, 1.0)]
        x_cross = _intersect_2d(v0, v1)
        if x_cross <= tol:
            return [(v1, 0.0, 1.0)]
        if x_cross >= 1.0 - tol:
            return [(v0, 0.0, 1.0)]
        return [(v0, 0.0, x_cross), (v1, x_cross, 1.0)]

    # 4. Monotone chain / upper envelope maintenance
    stack: List[AlphaVector2D] = []
    for v in vecs:
        while len(stack) >= 2:
            x_prev = _intersect_2d(stack[-2], stack[-1])
            x_curr = _intersect_2d(stack[-1], v)
            if x_curr <= x_prev:
                stack.pop()
            else:
                break
        stack.append(v)

    # 5. Clip intervals to [0, 1] and drop vectors with empty intervals
    intervals: List[Tuple[AlphaVector2D, float, float]] = []
    for i, v in enumerate(stack):
        left_x = 0.0 if i == 0 else max(0.0, _intersect_2d(stack[i - 1], v))
        right_x = 1.0 if i == len(stack) - 1 else min(1.0, _intersect_2d(v, stack[i + 1]))
        if right_x >= left_x - tol and right_x >= 0.0 and left_x <= 1.0:
            intervals.append((v, left_x, right_x))

    return intervals


def prune_2d(vectors: Sequence[AlphaVector2D], tol: float = 1e-9) -> List[AlphaVector2D]:
    """Prune dominated 2D alpha vectors, returning only the non-dominated set."""
    return [v for v, _, _ in prune_2d_with_intervals(vectors, tol=tol)]
