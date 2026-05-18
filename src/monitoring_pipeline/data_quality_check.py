import pandas as pd

from src.utils import get_logger
from src.config import EXPECTED_MIN_ROWS_PER_DAY

log = get_logger(__name__)


class DataQualityChecker:
    """Checks the most recent feature pipeline output for basic health."""

    def check(self, df: pd.DataFrame) -> dict:
        if df is None or len(df) == 0:
            log.warning("Empty dataframe — feature pipeline likely failed.")
            return {"rows": 0, "nulls": 0, "duplicates": 0, "is_healthy": False}

        metrics = {
            "rows": int(len(df)),
            "nulls": int(df.isnull().sum().sum()),
            "duplicates": int(df.duplicated().sum()),
            "null_ratio": float(df.isnull().sum().sum() / df.size),
        }
        metrics["is_healthy"] = (
            metrics["rows"] >= EXPECTED_MIN_ROWS_PER_DAY
            and metrics["null_ratio"] < 0.05
        )
        log.info(f"Data quality: {metrics}")
        return metrics