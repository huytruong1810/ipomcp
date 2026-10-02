"""
distribution.py — Probability-distribution abstractions for belief representation.

Provides discrete probability distribution interfaces and implementations:
- `ParticleDistribution`: Weighted and unweighted particle sets for Interactive Particle
  Filtering (IPF) and Monte Carlo sampling with Stochastic Universal Sampling (SUS).
- `DictDistribution`: Explicit categorical distribution over discrete supports.
"""

import abc
import bisect
import math
import random
from typing import Dict, Generic, Iterable, List, Optional, Tuple, TypeVar

T = TypeVar("T")


class Distribution(abc.ABC, Generic[T]):
    """Abstract base class for discrete probability distributions."""

    @abc.abstractmethod
    def sample(self) -> T:
        """Draw a single sample from the distribution."""
        pass

    @abc.abstractmethod
    def __getitem__(self, item: T) -> float:
        """Evaluate the probability mass of a specific item."""
        pass

    @abc.abstractmethod
    def get_support(self) -> Iterable[T]:
        """Return all items with non-zero probability."""
        pass


class ParticleDistribution(Distribution[T]):
    """
    A distribution represented by a list of discrete particles and their weights.
    Primarily used for Interactive Particle Filtering (IPF).
    """

    def __init__(self, particles: List[T], weights: List[float] = None):
        self._particles = list(particles)
        if weights is None:
            n = len(particles)
            self._weights = [1.0 / n] * n if n > 0 else []
        else:
            if len(particles) != len(weights):
                raise ValueError("Particles and weights must be the same length.")
            self._weights = list(weights)

        if any(not math.isfinite(w) or w < 0 for w in self._weights):
            raise ValueError("Weights must be finite and nonnegative.")
        if self._weights and (not math.isfinite(sum(self._weights)) or sum(self._weights) <= 0):
            raise ValueError("Weights must have finite positive total mass.")
        self._cdf = self._compute_cdf(self._weights)

        # Lazy O(1) Lookup Cache
        self._lookup_cache: Optional[Dict[T, float]] = None

    def _compute_cdf(self, weights: List[float]) -> List[float]:
        cdf = []
        cumsum = 0.0
        for w in weights:
            cumsum += w
            cdf.append(cumsum)
        return cdf

    def sample(self) -> T:
        """Draw a single particle in O(log N) using binary search over the CDF."""
        if not self._particles:
            raise ValueError("Cannot sample from an empty distribution.")

        r = random.random() * self._cdf[-1]
        idx = bisect.bisect_right(self._cdf, r)
        idx = min(idx, len(self._particles) - 1)
        return self._particles[idx]

    def resample(self, n_samples: int) -> List[T]:
        """
        Stochastic Universal Sampling (Systematic Resampling).
        O(N + n_samples) systematic resampling; variance depends on particle ordering.
        """
        if not isinstance(n_samples, int) or n_samples < 0:
            raise ValueError("n_samples must be a nonnegative integer.")
        if n_samples == 0:
            return []
        if not self._particles:
            raise ValueError("Cannot resample an empty distribution.")

        step = self._cdf[-1] / n_samples
        r = random.random() * step

        resampled = []
        idx = 0
        max_idx = len(self._particles) - 1
        for _ in range(n_samples):
            while idx < max_idx and r >= self._cdf[idx]:
                idx += 1
            resampled.append(self._particles[idx])
            r += step

        return resampled

    def __getitem__(self, item: T) -> float:
        """
        Evaluates the exact probability mass of item in O(1) time via lazy lookup caching.
        """
        if self._lookup_cache is None:
            self._lookup_cache = {}
            for p, w in zip(self._particles, self._weights):
                self._lookup_cache[p] = self._lookup_cache.get(p, 0.0) + w

        return self._lookup_cache.get(item, 0.0) / self._cdf[-1] if self._cdf else 0.0

    def get_support(self) -> Iterable[T]:
        """Returns all unique items with non-zero probability mass."""
        if self._lookup_cache is None:
            self._lookup_cache = {}
            for p, w in zip(self._particles, self._weights):
                self._lookup_cache[p] = self._lookup_cache.get(p, 0.0) + w

        return [item for item, prob in self._lookup_cache.items() if prob > 0.0]

    def values(self) -> Tuple[List[T], List[float]]:
        return list(self._particles), list(self._weights)

    def normalize(self) -> None:
        """Normalizes the particle weights to sum to 1.0."""
        if not self._weights:
            return

        total = self._cdf[-1]

        self._weights = [w / total for w in self._weights]

        self._cdf = self._compute_cdf(self._weights)

        # Invalidate the lookup cache since probabilities have changed
        self._lookup_cache = None


class DictDistribution(Distribution[T]):
    """
    A distribution represented natively by a categorical mapping from items to probabilities.
    Copies caller data and rejects invalid probability mass. An empty support cannot be sampled.
    """

    def __init__(self, probabilities: Dict[T, float]):
        # Defensive shallow copy to prevent mutating caller's data
        self._probs: Dict[T, float] = dict(probabilities)
        self._items: List[T] = list(self._probs.keys())
        raw_weights: List[float] = [float(self._probs[k]) for k in self._items]

        if any(not math.isfinite(w) or w < 0 for w in raw_weights):
            raise ValueError("Weights must be finite and nonnegative.")
        total = sum(raw_weights)
        if raw_weights and (not math.isfinite(total) or total <= 0):
            raise ValueError("Weights must have finite positive total mass.")
        self._weights = [w / total for w in raw_weights]
        self._probs = dict(zip(self._items, self._weights))

    def sample(self) -> T:
        if not self._items:
            raise ValueError("Cannot sample from an empty distribution.")
        return random.choices(self._items, weights=self._weights, k=1)[0]

    def __getitem__(self, item: T) -> float:
        return self._probs.get(item, 0.0)

    def get_support(self) -> Iterable[T]:
        return [item for item, p in self._probs.items() if p > 0.0]
