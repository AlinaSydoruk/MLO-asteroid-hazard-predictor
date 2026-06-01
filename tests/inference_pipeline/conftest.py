"""Shared fixtures for inference_pipeline tests."""
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


def _make_features_with_identity(n_rows: int = 10, seed: int = 0) -> pd.DataFrame:
    """
    DataFrame matching what AsteroidFeatureView.get_batch_data() returns:
    identity columns + all model features.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        # Identity columns
        "asteroid_id":            [f"id_{i}" for i in range(n_rows)],
        "name":                   [f"A_{i}" for i in range(n_rows)],
        "close_approach_date":    pd.date_range("2024-06-01", periods=n_rows),
        "first_observation_date": pd.date_range("2010-01-01", periods=n_rows, freq="365D"),
        # Raw features
        "est_diameter_min_km":   rng.uniform(0.01, 1.0, n_rows),
        "est_diameter_max_km":   rng.uniform(1.0, 5.0, n_rows),
        "absolute_magnitude_h":  rng.uniform(15, 28, n_rows),
        "relative_velocity_kmh": rng.uniform(1e3, 1e5, n_rows),
        "miss_distance_km":     rng.uniform(1e5, 1e8, n_rows),
        "is_sentry_object":      rng.integers(0, 2, n_rows),
        # Orbital features
        "moid_au":             rng.uniform(0.0, 0.5, n_rows),
        "perihelion_dist_au":  rng.uniform(0.5, 1.5, n_rows),
        "aphelion_dist_au":    rng.uniform(1.5, 4.0, n_rows),
        "eccentricity":        rng.uniform(0.0, 0.9, n_rows),
        "semi_major_axis_au":  rng.uniform(1.0, 3.0, n_rows),
        "inclination_deg":     rng.uniform(0.0, 30.0, n_rows),
        "jupiter_tisserand":   rng.uniform(2.0, 6.0, n_rows),
        # Computed features
        "est_diameter_mean_km":       rng.uniform(0.5, 3.0, n_rows),
        "diameter_to_distance_ratio": rng.uniform(1e-10, 1e-5, n_rows),
        "kinetic_energy_proxy":       rng.uniform(1e8, 1e14, n_rows),
        "log_miss_distance_km":       rng.uniform(10, 20, n_rows),
        "log_kinetic_energy_proxy":   rng.uniform(15, 30, n_rows),
    })


@pytest.fixture
def features_df():
    """Default 10-row DataFrame matching feature view output."""
    return _make_features_with_identity(n_rows=10, seed=0)


@pytest.fixture
def make_features_df():
    """Factory for tests that need custom-shaped feature DataFrames."""
    return _make_features_with_identity