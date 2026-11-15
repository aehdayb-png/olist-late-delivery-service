import numpy as np

from scripts.drift_report import psi


def test_psi_is_near_zero_for_identical_samples():
    sample = np.random.default_rng(0).normal(size=1000)
    assert psi(sample, sample) < 0.01


def test_psi_is_large_for_shifted_samples():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(0, 1, 1000), rng.normal(3, 1, 1000)) > 0.25
