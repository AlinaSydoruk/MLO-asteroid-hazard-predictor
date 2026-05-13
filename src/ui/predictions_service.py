from datetime import date, timedelta
import pandas as pd

from src.common.hopsworks.connection_manager import HopsworksConnectionManager
from src.inference_pipeline.daily_predictions_repository import create_predictions_repository
from src.utils import get_logger

log = get_logger(__name__)


class PredictionsService:
    """Reads predictions from Hopsworks and prepares them for display."""

    def __init__(self, connection: HopsworksConnectionManager = None):
        self.connection = connection or HopsworksConnectionManager()
        self.repo = create_predictions_repository(connection=self.connection)

    def get_by_range(self, start: date, end: date) -> pd.DataFrame:
        log.info(f"Fetching predictions {start} → {end}")
        df = self.repo.read_by_date(start.isoformat(), end.isoformat())
        return self._format(df)

    def get_today(self) -> pd.DataFrame:
        today = date.today()
        return self.get_by_range(today, today)

    def get_last_n_days(self, n: int) -> pd.DataFrame:
        end = date.today()
        start = end - timedelta(days=n)
        return self.get_by_range(start, end)

    def compute_stats(self, df: pd.DataFrame) -> dict:
        if df.empty:
            return {"total": 0, "hazardous": 0, "safe": 0,
                    "avg_prob": 0.0, "model": "—"}
        return {
            "total": len(df),
            "hazardous": int(df["predicted_hazardous"].sum()),
            "safe": int((df["predicted_hazardous"] == 0).sum()),
            "avg_prob": float(df["hazard_probability"].mean()),
            "model": f"v{df['model_version'].iloc[0]}",
        }

    @staticmethod
    def _format(df: pd.DataFrame) -> pd.DataFrame:
        """Sort, label, and round for display."""
        if df.empty:
            return df
        df = df.sort_values("hazard_probability", ascending=False).copy()
        df["status"] = df["predicted_hazardous"].map(
            {1: "Hazardous", 0: "Safe"}
        )
        df["hazard_probability"] = (df["hazard_probability"] * 100).round(2)
        return df[["name", "close_approach_date", "status",
                   "hazard_probability", "asteroid_id"]].rename(columns={
            "name": "Asteroid",
            "close_approach_date": "Close Approach",
            "status": "Status",
            "hazard_probability": "Hazard %",
            "asteroid_id": "ID",
        })