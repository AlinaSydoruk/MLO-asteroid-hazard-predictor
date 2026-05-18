from datetime import date, timedelta
import argparse
import pandas as pd
from src.common.hopsworks.connection_manager import HopsworksConnectionManager
from src.common.hopsworks.feature_view_repo import FeatureViewRepository
from src.inference_pipeline.predictor import AsteroidPredictor
from src.inference_pipeline.daily_predictions_repository import create_predictions_repository
from src.common.feature_schema import get_identity_columns
from src.utils import get_logger
from src.config import INFERENCE_DAYS_BACK

log = get_logger(__name__)


class InferencePipeline:
    """
    Orchestrates inference:
        Feature View → split → Predictor → Prediction Store
    """

    def __init__(self):
        connection = HopsworksConnectionManager()
        self.feature_view = FeatureViewRepository(connection=connection)
        self.predictor = AsteroidPredictor()
        self.prediction_repo = create_predictions_repository(connection=connection)

    def run(self) -> pd.DataFrame:
        """Run inference for asteroids"""
        log.info("Starting INFERENCE pipeline")

        log.info("Loading recent asteroids from Feature Store...")
        df = self.feature_view.get_batch_data(
            start_time=date.today().isoformat(),
            end_time=(date.today() + timedelta(days=1)).isoformat(),
        )

        if df.empty:
            log.warning("No asteroids found for inference.")
            return df

        log.info(f"Loaded {len(df)} asteroids")

        # Load model
        self.predictor.load_champion()
        training_cutoff = self.predictor.training_cutoff
        model_version = self.predictor.model_version
        log.info(f"Model v{model_version} trained on: {training_cutoff}")

        predictions = self.predictor.predict(df)

        #  was it discovered after the model was trained?
        if training_cutoff and "first_observation_date" in df.columns:
            cutoff = pd.to_datetime(training_cutoff)
            is_new = (df["first_observation_date"] > cutoff).fillna(False)
            predictions["source"] = is_new.map({True: "model_prediction", False: "nasa_catalog"})
        else:
            log.warning("No training_cutoff_date or no first_observation_date — all tagged as nasa_catalog.")
            predictions["source"] = "nasa_catalog"
            is_new = pd.Series(False, index=df.index)

        n_new = int(is_new.sum())
        n_known = len(df) - n_new
        log.info(f"NEW: {n_new} | KNOWN: {n_known}")

        # Save to Hopsworks
        log.info("Saving predictions to Hopsworks...")
        cols_to_save = (
                get_identity_columns()
                + ["predicted_hazardous", "hazard_probability", "model_version", "source"]
        )
        predictions_to_save = predictions[
            [c for c in cols_to_save if c in predictions.columns]
        ]
        self.prediction_repo.insert(predictions_to_save)

        log.info("Inference pipeline complete")
        log.info(f"  Total:     {len(predictions)}")
        log.info(f"  New:       {n_new}")
        log.info(f"  Known:     {n_known}")
        log.info(f"  Hazardous: {int(predictions['predicted_hazardous'].sum())}")
        log.info(f"  Model:     v{model_version}")
        return predictions

    def run_for_range(self, start: date, end: date) -> pd.DataFrame:
        log.info(f"Starting RANGE INFERENCE: {start} → {end}")
        end_exclusive = end + timedelta(days=1)

        log.info(f"Loading asteroids from Feature Store ({start} → {end})...")
        df = self.feature_view.get_batch_data(
            start_time=start.isoformat(),
            end_time=end_exclusive.isoformat(),
        )

        if df.empty:
            log.warning("No asteroids found for inference.")
            return df

        log.info(f"Loaded {len(df)} asteroids")
        self.predictor.load_champion()
        training_cutoff = self.predictor.training_cutoff
        model_version = self.predictor.model_version
        log.info(f"Model v{model_version} trained on: {training_cutoff}")

        predictions = self.predictor.predict(df)

        if training_cutoff and "first_observation_date" in df.columns:
            cutoff = pd.to_datetime(training_cutoff)
            is_new = (df["first_observation_date"] > cutoff).fillna(False)
            predictions["source"] = is_new.map(
                {True: "model_prediction", False: "nasa_catalog"}
            )
        else:
            predictions["source"] = "nasa_catalog"
            is_new = pd.Series(False, index=df.index)

        cols_to_save = (
                get_identity_columns()
                + ["predicted_hazardous", "hazard_probability", "model_version", "source"]
        )
        predictions_to_save = predictions[
            [c for c in cols_to_save if c in predictions.columns]
        ]
        self.prediction_repo.insert(predictions_to_save)

        log.info(f"Range inference complete: {len(predictions)} predictions")
        log.info(f"  Real predictions: {int(is_new.sum())}")
        log.info(f"  Hazardous: {int(predictions['predicted_hazardous'].sum())}")
        return predictions

def main():
    parser = argparse.ArgumentParser(description="Asteroid Inference Pipeline")
    parser.add_argument(
        "--mode", choices=["daily", "range"], default="daily",
    )
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date",   type=str, default=None)
    args = parser.parse_args()

    pipe = InferencePipeline()

    if args.mode == "daily":
        pipe.run()
    else:
        if not args.start_date or not args.end_date:
            raise ValueError("--mode range requires --start-date and --end-date")
        start = date.fromisoformat(args.start_date)
        end   = date.fromisoformat(args.end_date)
        pipe.run_for_range(start, end)


if __name__ == "__main__":
    main()