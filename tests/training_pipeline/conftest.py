"""
Shared fixtures for training_pipeline tests.

Env vars are seeded at import time so src.config doesn't KeyError when
test modules import src.training_pipeline.*.
"""
import os

os.environ.setdefault("NASA_API_KEY", "test-nasa-key")
os.environ.setdefault("HOPSWORKS_API_KEY", "test-hopsworks-key")
os.environ.setdefault("HOPSWORKS_PROJECT_NAME", "test-project")
os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
os.environ.setdefault("MLFLOW_TRACKING_USERNAME", "test-user")
os.environ.setdefault("MLFLOW_TRACKING_PASSWORD", "test-pw")

import numpy as np
import pandas as pd
import pytest


# ─── Sample data builders ─────────────────────────────────────────

def _make_features(n_rows: int = 50, seed: int = 0) -> pd.DataFrame:
    """A small feature DataFrame matching get_feature_columns() schema."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        # Raw
        "est_diameter_min_km":   rng.uniform(0.01, 1.0, n_rows),
        "est_diameter_max_km":   rng.uniform(1.0, 5.0, n_rows),
        "absolute_magnitude_h":  rng.uniform(15, 28, n_rows),
        "relative_velocity_kmh": rng.uniform(1e3, 1e5, n_rows),
        "miss_distance_km":     rng.uniform(1e5, 1e8, n_rows),
        "is_sentry_object":      rng.integers(0, 2, n_rows),
        # Orbital
        "moid_au":             rng.uniform(0.0, 0.5, n_rows),
        "perihelion_dist_au":  rng.uniform(0.5, 1.5, n_rows),
        "aphelion_dist_au":    rng.uniform(1.5, 4.0, n_rows),
        "eccentricity":        rng.uniform(0.0, 0.9, n_rows),
        "semi_major_axis_au":  rng.uniform(1.0, 3.0, n_rows),
        "inclination_deg":     rng.uniform(0.0, 30.0, n_rows),
        "jupiter_tisserand":   rng.uniform(2.0, 6.0, n_rows),
        # Computed
        "est_diameter_mean_km":      rng.uniform(0.5, 3.0, n_rows),
        "diameter_to_distance_ratio": rng.uniform(1e-10, 1e-5, n_rows),
        "kinetic_energy_proxy":       rng.uniform(1e8, 1e14, n_rows),
        "log_miss_distance_km":       rng.uniform(10, 20, n_rows),
        "log_kinetic_energy_proxy":   rng.uniform(15, 30, n_rows),
    })


def _make_labels(n_rows: int = 50, positive_fraction: float = 0.3,
                 seed: int = 0) -> pd.Series:
    """Imbalanced binary labels — mirrors the real PHA class distribution."""
    rng = np.random.default_rng(seed)
    n_positive = max(1, int(n_rows * positive_fraction))
    labels = np.zeros(n_rows, dtype=int)
    labels[:n_positive] = 1
    rng.shuffle(labels)
    return pd.Series(labels, name="is_potentially_hazardous")


@pytest.fixture
def X_features():
    """50-row feature DataFrame."""
    return _make_features(n_rows=50, seed=0)


@pytest.fixture
def y_labels():
    """50-row label series with ~30% positives."""
    return _make_labels(n_rows=50, positive_fraction=0.3, seed=0)


@pytest.fixture
def make_features():
    """Factory for tests that need custom-sized features."""
    return _make_features


@pytest.fixture
def make_labels():
    """Factory for tests that need custom-distribution labels."""
    return _make_labels