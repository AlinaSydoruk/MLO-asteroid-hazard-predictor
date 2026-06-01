"""Unit tests for src.monitoring_pipeline.data_quality_check."""
import numpy as np
import pandas as pd
import pytest

from src.monitoring_pipeline.data_quality_check import DataQualityChecker
from src.config import EXPECTED_MIN_ROWS_PER_DAY


@pytest.fixture
def checker():
    return DataQualityChecker()


def _healthy_df(n_rows: int) -> pd.DataFrame:
    """Clean DataFrame with no nulls or duplicates."""
    return pd.DataFrame({
        "asteroid_id": [f"id_{i}" for i in range(n_rows)],
        "moid_au":     [0.1 * i for i in range(n_rows)],
    })


class TestEmptyOrNone:
    def test_empty_df_is_unhealthy(self, checker):
        result = checker.check(pd.DataFrame())
        assert result["rows"] == 0
        assert result["is_healthy"] is False

    def test_none_input_is_unhealthy(self, checker):
        result = checker.check(None)
        assert result["rows"] == 0
        assert result["is_healthy"] is False


class TestHealthyPath:
    def test_clean_data_is_healthy(self, checker):
        df = _healthy_df(EXPECTED_MIN_ROWS_PER_DAY + 10)
        result = checker.check(df)
        assert result["is_healthy"] is True
        assert result["nulls"] == 0
        assert result["duplicates"] == 0
        assert result["null_ratio"] == 0.0

    def test_row_count_reported_correctly(self, checker):
        df = _healthy_df(EXPECTED_MIN_ROWS_PER_DAY + 50)
        result = checker.check(df)
        assert result["rows"] == EXPECTED_MIN_ROWS_PER_DAY + 50


class TestUnhealthyConditions:
    def test_too_few_rows_is_unhealthy(self, checker):
        # One row — well below the daily minimum
        df = _healthy_df(1)
        result = checker.check(df)
        assert result["is_healthy"] is False

    def test_high_null_ratio_is_unhealthy(self, checker):
        df = _healthy_df(EXPECTED_MIN_ROWS_PER_DAY + 10)
        # Wipe out 50% of one column → null_ratio well above 5%
        df.loc[df.index[: len(df) // 2], "moid_au"] = np.nan
        result = checker.check(df)
        assert result["is_healthy"] is False
        assert result["null_ratio"] > 0.05

    def test_counts_nulls_correctly(self, checker):
        df = _healthy_df(10)
        df.loc[0, "moid_au"] = np.nan
        df.loc[1, "moid_au"] = np.nan
        df.loc[2, "asteroid_id"] = None
        result = checker.check(df)
        assert result["nulls"] == 3

    def test_counts_duplicates_correctly(self, checker):
        df = pd.DataFrame({
            "asteroid_id": ["a", "a", "b", "c"],
            "moid_au":     [0.1, 0.1, 0.2, 0.3],
        })
        result = checker.check(df)
        assert result["duplicates"] == 1


class TestReturnedTypes:
    def test_returned_values_are_native_python(self, checker):
        """MLflow can't serialize numpy ints — verify type coercion."""
        df = _healthy_df(EXPECTED_MIN_ROWS_PER_DAY + 5)
        result = checker.check(df)
        assert type(result["rows"]) is int
        assert type(result["nulls"]) is int
        assert type(result["duplicates"]) is int
        assert type(result["null_ratio"]) is float
        assert type(result["is_healthy"]) is bool