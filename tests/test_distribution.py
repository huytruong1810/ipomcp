import pytest
import math
from core.distribution import ParticleDistribution, DictDistribution


def test_particle_distribution_uniform():
    particles = ["TL", "TR"]
    dist = ParticleDistribution(particles)
    assert dist["TL"] == pytest.approx(0.5)
    assert dist["TR"] == pytest.approx(0.5)
    assert dist["OTHER"] == 0.0
    
    samples = [dist.sample() for _ in range(500)]
    assert "TL" in samples and "TR" in samples


def test_particle_distribution_weighted():
    particles = ["A", "B", "C"]
    weights = [0.1, 0.7, 0.2]
    dist = ParticleDistribution(particles, weights)
    
    assert dist["A"] == pytest.approx(0.1)
    assert dist["B"] == pytest.approx(0.7)
    assert dist["C"] == pytest.approx(0.2)
    
    # Test lazy cache invalidation on normalize
    dist.normalize()
    assert dist["B"] == pytest.approx(0.7)


def test_particle_distribution_resample():
    particles = ["A", "B"]
    weights = [0.8, 0.2]
    dist = ParticleDistribution(particles, weights)
    
    resampled = dist.resample(1000)
    assert len(resampled) == 1000
    count_a = resampled.count("A")
    # SUS (Systematic Resampling) has ultra-low variance
    assert 750 <= count_a <= 850


def test_particle_distribution_zero_weights_fallback():
    particles = ["A", "B"]
    weights = [0.0, 0.0]
    dist = ParticleDistribution(particles, weights)
    dist.normalize()
    assert dist["A"] == pytest.approx(0.5)
    assert dist["B"] == pytest.approx(0.5)


def test_dict_distribution():
    probs = {"N": 0.25, "S": 0.25, "E": 0.25, "W": 0.25}
    dist = DictDistribution(probs)
    
    assert dist["N"] == 0.25
    assert dist["UNKNOWN"] == 0.0
    sample = dist.sample()
    assert sample in probs
    assert set(dist.get_support()) == {"N", "S", "E", "W"}


def test_particle_distribution_unique_support():
    particles = ["A", "A", "B", "B", "C"]
    dist = ParticleDistribution(particles)
    support = list(dist.get_support())
    assert len(support) == 3
    assert set(support) == {"A", "B", "C"}


def test_particle_distribution_resample_boundary_stability():
    # Test boundary condition with large particle count and numerical drift
    n = 2000
    particles = [f"p_{i}" for i in range(n)]
    weights = [1.0 / n] * n
    dist = ParticleDistribution(particles, weights)
    # Must not raise IndexError
    resampled = dist.resample(n)
    assert len(resampled) == n

