"""Alpha-vector representations and exact pruning operators.

Provides:
1. Exact 2D upper convex hull pruning in O(K log K) time with analytical
   interval extraction (for 2-state POMDPs / Level 1 I-POMDP against L0).
2. General D-dimensional alpha-vector representations with Linear Programming
   (LP) dominance pruning via scipy.optimize.linprog.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog


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
        if filtered and math.isclose(filtered[-1].slope, v.slope, abs_tol=tol):
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


@dataclass(frozen=True)
class AlphaVectorND:
    """An alpha vector over a general D-dimensional discrete state space."""

    values: Tuple[float, ...]
    action: Optional[Any] = None

    def value(self, belief: Sequence[float]) -> float:
        """Evaluate the alpha vector under a belief distribution."""
        return float(np.dot(self.values, belief))

    @property
    def dim(self) -> int:
        return len(self.values)

    def __repr__(self) -> str:
        act_str = f"[{self.action}] " if self.action is not None else ""
        vals_str = ", ".join(f"{x:.3f}" for x in self.values[:4])
        if len(self.values) > 4:
            vals_str += ", ..."
        return f"AlphaND({act_str}[{vals_str}])"


def prune_nd(
    vectors: Sequence[AlphaVectorND],
    tol: float = 1e-7,
) -> List[AlphaVectorND]:
    """Prune dominated multi-dimensional alpha vectors using Linear Programming.

    For each vector v in Gamma, we solve the LP:
        max delta
        s.t.
            <v - w, b> >= delta,  for all w in Gamma \\ {v}
            sum(b) == 1
            b >= 0
    If delta* > tol, vector v is non-dominated at the witness belief point b*.
    """
    if not vectors:
        return []
    if len(vectors) == 1:
        return [vectors[0]]

    # Deduplicate
    unique: List[AlphaVectorND] = []
    seen: set[Tuple[float, ...]] = set()
    for v in vectors:
        rounded = tuple(round(x, 7) for x in v.values)
        if rounded not in seen:
            seen.add(rounded)
            unique.append(v)

    if len(unique) <= 1:
        return unique

    dim = unique[0].dim
    non_dominated: List[AlphaVectorND] = []

    # Variables for LP: x = [b_0, b_1, ..., b_{dim-1}, delta]
    # Objective: maximize delta <=> minimize -delta
    c = np.zeros(dim + 1)
    c[-1] = -1.0

    # Bounds: b_i >= 0, delta unbounded (or delta >= -inf)
    bounds = [(0.0, 1.0) for _ in range(dim)] + [(None, None)]

    # Equality constraint: sum(b_i) == 1
    A_eq = np.zeros((1, dim + 1))
    A_eq[0, :dim] = 1.0
    b_eq = np.array([1.0])

    for i, v in enumerate(unique):
        # Build other vectors
        others = [w for j, w in enumerate(unique) if j != i]
        n_others = len(others)

        # Inequality constraints: <v - w, b> >= delta
        # <=> delta - <v - w, b> <= 0
        # <=> sum_k (w_k - v_k) * b_k + delta <= 0
        A_ub = np.zeros((n_others, dim + 1))
        b_ub = np.zeros(n_others)

        for row, w in enumerate(others):
            diff = np.array(w.values) - np.array(v.values)
            A_ub[row, :dim] = diff
            A_ub[row, -1] = 1.0

        res = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )

        if res.success and -res.fun > tol:
            non_dominated.append(v)

    return non_dominated
