"""
Inference Pipeline.
Loads today's asteroids, predicts hazards, stores results.

Usage:
    python -m src.inference_pipeline.pipeline
"""
from datetime import date
import pandas as pd

from src.common.hopsworks.connection_manager import HopsworksConnectionManager
from src.common.hopsworks.feature_view_repo import FeatureViewRepository
from src.inference_pipeline.predictor import AsteroidPredictor
from src.inference_pipeline.daily_predictions_repository import create_predictions_repository
from src.common.feature_schema import get_identity_columns
from src.utils import get_logger
from datetime import timedelta

log = get_logger(__name__)


class InferencePipeline:
    """
    Orchestrates inference:
        Feature View → Predictor → Prediction Store
    """

    def __init__(self):
        connection = HopsworksConnectionManager()
        self.feature_view = FeatureViewRepository(connection=connection)
        self.predictor = AsteroidPredictor()
        self.prediction_repo = create_predictions_repository(connection=connection)

    def run(self, days_back: int = 5) -> pd.DataFrame:
        """ Run inference for asteroids in the last N days."""
        log.info("Starting INFERENCE pipeline")

        # Get recent asteroids from Feature View
        log.info("Loading recent asteroids from Feature Store...")
        end = date.today()
        start = end - timedelta(days=days_back)
        df = self.feature_view.get_batch_data(
            start_time=start.isoformat(),
            end_time=end.isoformat(),
        )


        if df.empty:
            log.warning("No asteroids found for inference.")
            return df

        log.info(f"Loaded {len(df)} asteroids for prediction")

        # Generate predictions
        log.info("Generating predictions...")
        predictions = self.predictor.predict(df)

        # Save predictions
        log.info("Saving predictions to Hopsworks...")

        # Keep only columns we want to store
        cols_to_save = (get_identity_columns() + ["predicted_hazardous", "hazard_probability", "model_version"])
        predictions_to_save = predictions[
            [c for c in cols_to_save if c in predictions.columns]
        ]

        self.prediction_repo.insert(predictions_to_save)

        log.info("Inference pipeline complete")
        log.info(f"Model: v{self.predictor.model_version}")
        return predictions

# TODO REPLACE TO VARIABLE
def main():
    pipeline = InferencePipeline()
    pipeline.run(days_back=5)


if __name__ == "__main__":
    main()