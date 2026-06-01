"""Unit tests for src.feature_pipeline.feature_computer."""
import numpy as np
import pandas as pd
import pytest

from src.feature_pipeline.feature_computer import compute_features


def _minimal_row(**overrides) -> dict:
    """Build one row matching the parser's output schema."""
    row = {
        "asteroid_id": "111",
        "name": "A1",
        "close_approach_date": "2024-01-01",
        "est_diameter_min_km": 0.1,
        "est_diameter_max_km": 0.3,
        "absolute_magnitude_h": 22.5,
        "relative_velocity_kmh": 50000.0,
        "miss_distance_km": 1_000_000.0,
        "is_sentry_object": 0,
        "is_potentially_hazardous": 1,
        "moid_au": 0.05,
        "perihelion_dist_au": 0.9,
        "aphelion_dist_au": 2.1,
        "eccentricity": 0.4,
        "semi_major_axis_au": 1.5,
        "inclination_deg": 10.0,
        "jupiter_tisserand": 3.5,
        "first_observation_date": "2010-05-01",
    }
    row.update(overrides)
    return row


@pytest.fixture
def base_df():
    return pd.DataFrame([_minimal_row()])


class TestEngineeredColumns:
    def test_mean_diameter_is_average_of_min_and_max(self, base_df):
        out = compute_features(base_df)
        assert out["est_diameter_mean_km"].iloc[0] == pytest.approx(0.2)

    def test_diameter_to_distance_ratio(self, base_df):
        out = compute_features(base_df)
        expected = 0.2 / 1_000_000.0
        assert out["diameter_to_distance_ratio"].iloc[0] == pytest.approx(expected)

    def test_diameter_to_distance_clips_zero_distance(self):
        # A zero miss-distance would otherwise produce inf; verify clip kicks in
        df = pd.DataFrame([_minimal_row(miss_distance_km=0.0)])
        out = compute_features(df)
        assert np.isfinite(out["diameter_to_distance_ratio"].iloc[0])

    def test_kinetic_energy_proxy_formula(self, base_df):
        out = compute_features(base_df)
        expected = (0.2 ** 3) * (50000.0 ** 2)
        assert out["kinetic_energy_proxy"].iloc[0] == pytest.approx(expected)

    def test_log_miss_distance_uses_log1p(self, base_df):
        out = compute_features(base_df)
        assert out["log_miss_distance_km"].iloc[0] == pytest.approx(
            np.log1p(1_000_000.0)
        )

    def test_log_kinetic_energy_uses_log1p(self, base_df):
        out = compute_features(base_df)
        ke = (0.2 ** 3) * (50000.0 ** 2)
        assert out["log_kinetic_energy_proxy"].iloc[0] == pytest.approx(np.log1p(ke))


class TestDropping:
    @pytest.mark.parametrize(
        "missing_col",
        ["est_diameter_min_km", "absolute_magnitude_h", "moid_au"],
    )
    def test_rows_with_missing_critical_values_are_dropped(self, missing_col):
        good = _minimal_row()
        bad = _minimal_row(asteroid_id="222", **{missing_col: None})
        df = pd.DataFrame([good, bad])

        out = compute_features(df)

        assert len(out) == 1
        assert out["asteroid_id"].iloc[0] == "111"

    def test_keeps_rows_when_non_critical_field_is_null(self):
        # eccentricity is engineered downstream but not in the dropna list
        df = pd.DataFrame([_minimal_row(eccentricity=None)])
        out = compute_features(df)
        assert len(out) == 1


class TestTypeCoercion:
    def test_asteroid_id_is_string(self):
        df = pd.DataFrame([_minimal_row(asteroid_id=12345)])
        out = compute_features(df)
        # pandas may use either `object` or StringDtype — what matters is
        # the value got coerced to "12345"
        assert isinstance(out["asteroid_id"].iloc[0], str)
        assert out["asteroid_id"].iloc[0] == "12345"

    def test_close_approach_date_is_datetime(self, base_df):
        out = compute_features(base_df)
        assert pd.api.types.is_datetime64_any_dtype(out["close_approach_date"])

    def test_label_and_sentry_are_int(self, base_df):
        out = compute_features(base_df)
        assert pd.api.types.is_integer_dtype(out["is_potentially_hazardous"])
        assert pd.api.types.is_integer_dtype(out["is_sentry_object"])

    def test_first_observation_date_coerces_invalid_to_nat(self):
        df = pd.DataFrame([_minimal_row(first_observation_date="not-a-date")])
        out = compute_features(df)
        assert pd.isna(out["first_observation_date"].iloc[0])


class TestPurity:
    def test_does_not_mutate_input(self, base_df):
        snapshot = base_df.copy(deep=True)
        compute_features(base_df)
        pd.testing.assert_frame_equal(base_df, snapshot)