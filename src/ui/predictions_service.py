from datetime import date, timedelta
import pandas as pd

from src.common.hopsworks.connection_manager import HopsworksConnectionManager
from src.inference_pipeline.daily_predictions_repository import create_predictions_repository
from src.utils import get_logger
from src.config import WEEKLY_RETRAIN_DAY, WEEKLY_RETRAIN_HOUR_UTC

log = get_logger(__name__)
from datetime import datetime, time, timezone

class PredictionsService:
    """Reads predictions from Hopsworks and prepares them for display."""

    def __init__(self, connection: HopsworksConnectionManager = None):
        self.connection = connection or HopsworksConnectionManager()
        self.repo = create_predictions_repository(connection=self.connection)

    def get_by_range(self, start: date, end: date) -> pd.DataFrame:
        log.info(f"Fetching predictions {start} → {end}")
        return self.repo.read_by_date(start.isoformat(), end.isoformat())

    def get_today(self) -> pd.DataFrame:
        today = date.today()
        return self.get_by_range(today, today)

    def get_last_n_days(self, n: int) -> pd.DataFrame:
        end = date.today()
        start = end - timedelta(days=n)
        return self.get_by_range(start, end)

    def get_next_train_time(self) -> datetime:
        """Next scheduled retraining time (UTC)."""
        now = datetime.now(timezone.utc)
        days_ahead = (WEEKLY_RETRAIN_DAY - now.weekday()) % 7
        target = now.replace(
            hour=WEEKLY_RETRAIN_HOUR_UTC, minute=0, second=0, microsecond=0,
        ) + timedelta(days=days_ahead)
        if target <= now:
            target += timedelta(days=7)
        return target

    def get_time_until_next_train(self) -> str:
        """Human-readable countdown e.g. 'in 3d 4h'."""
        delta = self.get_next_train_time() - datetime.now(timezone.utc)
        days = delta.days
        hours = delta.seconds // 3600
        if days > 0:
            return f"in {days}d {hours}h"
        if hours > 0:
            return f"in {hours}h"
        return "soon"

    def get_history_days(self) -> int:
        """How many days of accumulated predictions exist."""
        df = self.repo.read()
        if df.empty:
            return 0
        dates = pd.to_datetime(df["close_approach_date"]).dt.date
        return (dates.max() - dates.min()).days + 1

    def get_all(self) -> pd.DataFrame:
        """Return all predictions regardless of date range."""
        return self.repo.read()

    def compute_stats(self, df: pd.DataFrame) -> dict:
        if df.empty:
            return {
                "total": 0,
                "hazardous": 0,
                "safe": 0,
                "avg_prob": 0.0,
                "model": "—",
                "predicted": 0,
                "next_train": self.get_time_until_next_train(),
                "history_days": self.get_history_days(),
            }
        return {
            "total": len(df),
            "hazardous": int(df["predicted_hazardous"].sum()),
            "safe": int((df["predicted_hazardous"] == 0).sum()),
            "predicted": int((df["source"] == "model_prediction").sum()),
            "avg_prob": float(df["hazard_probability"].mean()),
            "model": f"v{df['model_version'].iloc[0]}",
            "next_train": self.get_time_until_next_train(),
            "history_days": self.get_history_days(),
        }

    # Display formatting
    @staticmethod
    def format_for_display(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.sort_values("hazard_probability", ascending=False).copy()
        df["status"] = df["predicted_hazardous"].map({1: "Hazardous", 0: "Safe"})
        df["hazard_probability"] = (df["hazard_probability"] * 100).round(2)

        if "first_observation_date" in df.columns:
            df["first_observation_date"] = pd.to_datetime(
                df["first_observation_date"]
            ).dt.date

        return df[[
            "name",
            "close_approach_date",
            "first_observation_date",
            "status",
            "hazard_probability",
            "source",
            "asteroid_id",
        ]].rename(columns={
            "name": "Asteroid",
            "close_approach_date": "Close Approach",
            "first_observation_date": "Discovered",
            "status": "Status",
            "hazard_probability": "Hazard %",
            "source": "Source",
            "asteroid_id": "ID",
        })