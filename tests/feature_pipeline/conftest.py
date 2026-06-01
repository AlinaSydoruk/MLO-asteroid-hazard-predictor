"""
Shared fixtures and environment setup for feature_pipeline tests.

Env vars MUST be set before `src.config` is imported, since that module
reads them at import time. pytest imports conftest.py first, so the
os.environ.setdefault calls below run before any test module touches
`src.*`.
"""
import os

# ─── Env vars required by src.config ───────────────────────────────
os.environ.setdefault("NASA_API_KEY", "test-nasa-key")
os.environ.setdefault("HOPSWORKS_API_KEY", "test-hopsworks-key")
os.environ.setdefault("HOPSWORKS_PROJECT_NAME", "test-project")
os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
os.environ.setdefault("MLFLOW_TRACKING_USERNAME", "test-user")
os.environ.setdefault("MLFLOW_TRACKING_PASSWORD", "test-pw")

import copy
import pytest


# ─── Sample NASA API payloads ──────────────────────────────────────

def _make_asteroid(
    asteroid_id: str = "3542519",
    name: str = "(2010 PK9)",
    date_str: str = "2024-01-01",
    diameter_min: float = 0.1,
    diameter_max: float = 0.3,
    magnitude: float = 22.5,
    velocity_kmh: str = "50000.0",
    miss_km: str = "1000000.0",
    hazardous: bool = True,
    sentry: bool = False,
    with_orbital: bool = True,
) -> dict:
    """Build a single asteroid dict matching NASA NeoWs response shape."""
    asteroid = {
        "id": asteroid_id,
        "name": name,
        "absolute_magnitude_h": magnitude,
        "is_potentially_hazardous_asteroid": hazardous,
        "is_sentry_object": sentry,
        "estimated_diameter": {
            "kilometers": {
                "estimated_diameter_min": diameter_min,
                "estimated_diameter_max": diameter_max,
            }
        },
        "close_approach_data": [
            {
                "close_approach_date": date_str,
                "relative_velocity": {"kilometers_per_hour": velocity_kmh},
                "miss_distance": {"kilometers": miss_km},
            }
        ],
    }
    if with_orbital:
        asteroid["orbital_data"] = {
            "minimum_orbit_intersection": "0.05",
            "perihelion_distance": "0.9",
            "aphelion_distance": "2.1",
            "eccentricity": "0.4",
            "semi_major_axis": "1.5",
            "inclination": "10.0",
            "jupiter_tisserand_invariant": "3.5",
            "first_observation_date": "2010-05-01",
        }
    return asteroid


@pytest.fixture
def sample_asteroid():
    """A single, well-formed asteroid dict."""
    return _make_asteroid()


@pytest.fixture
def sample_feed():
    """A two-date feed response with three asteroids total."""
    return {
        "near_earth_objects": {
            "2024-01-01": [
                _make_asteroid(asteroid_id="111", name="A1", date_str="2024-01-01"),
                _make_asteroid(
                    asteroid_id="222",
                    name="A2",
                    date_str="2024-01-01",
                    hazardous=False,
                ),
            ],
            "2024-01-02": [
                _make_asteroid(asteroid_id="333", name="A3", date_str="2024-01-02"),
            ],
        }
    }


@pytest.fixture
def sample_browse_page():
    """One page of /neo/browse response."""
    return {
        "page": {"size": 20, "total_pages": 3, "total_elements": 60, "number": 0},
        "near_earth_objects": [
            _make_asteroid(asteroid_id="111", name="A1"),
            _make_asteroid(asteroid_id="222", name="A2", hazardous=False),
        ],
    }


@pytest.fixture
def make_asteroid():
    """Factory exposed to tests that need to customize fields."""
    return _make_asteroid


@pytest.fixture
def deep_copy():
    """Helper so tests don't accidentally mutate shared fixtures."""
    return copy.deepcopy