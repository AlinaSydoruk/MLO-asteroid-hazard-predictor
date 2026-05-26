import argparse
from datetime import date, timedelta

import mlflow
import pandas as pd

from src.common.hopsworks.connection_manager import HopsworksConnectionManager
from src.common.hopsworks.feature_view_repo import (
    FeatureViewRepository,
    DedupFeatureViewRepository,
)
from src.feature_pipeline.main_features_repository import AsteroidFeaturesRepository
from src.inference_pipeline.daily_predictions_repository import (
    create_predictions_repository,
)
from src.training_pipeline.mlflow.connection import MLflowConnectionManager
from src.monitoring_pipeline.data_quality_check import DataQualityChecker
from src.monitoring_pipeline.feature_drift import FeatureDriftDetector
from src.monitoring_pipeline.prediction_drift import PredictionDriftDetector
from src.monitoring_pipeline.performance_tracker import PerformanceTracker
from src.config import (
    MONITORING_EXPERIMENT_NAME,
    MONITORING_LOOKBACK_DAYS,
    PERFORMANCE_LAG_DAYS,
    DRIFT_ALERT_THRESHOLD,
)
from src.utils import get_logger

log = get_logger(__name__)


class MonitoringPipeline:
    """Runs all 4 monitoring levels and logs everything to MLflow."""

    def __inFit__(self):
        connection = HopsworksConnectionManager()
        self.fv_raw = FeatureViewRepository(connection=connection)
        self.fv_dedup = DedupFeatureViewRepository(connection=connection)
        self.pred_repo = create_predictions_repository(connection=connection)
        self.feature_group = AsteroidFeaturesRepository(connection=connection)
        self.quality = DataQualityChecker()
        self.feature_drift = FeatureDriftDetector()
        self.pred_drift = PredictionDriftDetector()
        self.perf = PerformanceTracker()

        MLflowConnectionManager().connect()
        mlflow.set_experiment(MONITORING_EXPERIMENT_NAME)

    def run(self) -> dict:
        log.info("Starting MONITORING pipeline")
        today = date.today()
        start = today - timedelta(days=MONITORING_LOOKBACK_DAYS)

        with mlflow.start_run(run_name=f"monitoring_{today.isoformat()}"):
            mlflow.log_param("window_start", start.isoformat())
            mlflow.log_param("window_end", today.isoformat())

            #  Data Quality
            current_features = self.fv_raw.get_batch_data(
                start_time=start.isoformat(),
                end_time=today.isoformat(),
            )
            dq = self.quality.check(current_features)
            mlflow.log_metrics({f"dq_{k}": v for k, v in dq.items()
                                if isinstance(v, (int, float, bool))})

            #  Feature Drift
            reference = self.fv_dedup.get_batch_data(
                start_time=(start - timedelta(days=90)).isoformat(),
                end_time=start.isoformat(),
            )
            drift = self.feature_drift.detect(reference, current_features)
            mlflow.log_metrics({
                "drifted_features": drift["drifted_features"],
                "share_drifted":    drift["share_drifted"],
                "dataset_drift":    int(drift["dataset_drift"]),
            })
            if drift.get("report_path"):
                mlflow.log_artifact(drift["report_path"])

            #  Prediction Drift
            preds_fg = self.pred_repo.get_or_create()
            recent_preds = preds_fg.read(
                read_options={"use_hive": True}
            ).query(f"close_approach_date >= '{start.isoformat()}'")
            pred_metrics = self.pred_drift.detect(recent_preds)
            mlflow.log_metrics(pred_metrics)

            #  Lagged Performance
            lag_start = today - timedelta(days=PERFORMANCE_LAG_DAYS + 7)
            lag_end = today - timedelta(days=PERFORMANCE_LAG_DAYS)

            raw = preds_fg.read(read_options={"use_hive": True})
            raw["close_approach_date"] = pd.to_datetime(raw["close_approach_date"])
            old_preds = raw.query("@lag_start <= close_approach_date <= @lag_end")

            current_labels = self.feature_group.read_by_date(
                start_date=lag_start.isoformat(),
                end_date=lag_end.isoformat(),
            )
            if old_preds.empty:
                log.warning(f"No predictions found in lag window {lag_start} → {lag_end}, skipping performance eval.")
            else:
                perf = self.perf.evaluate(old_preds, current_labels)
                mlflow.log_metrics({f"perf_{k}": v for k, v in perf.items()
                                    if isinstance(v, (int, float)) and v is not None})

            # Alerting
            if drift["share_drifted"] >= DRIFT_ALERT_THRESHOLD:
                log.warning(f"DRIFT ALERT: {drift['share_drifted']:.0%} of features drifted")
                mlflow.set_tag("alert", "drift")

            log.info("Monitoring pipeline complete.")
            return {"quality": dq, "drift": drift,
                    "predictions": pred_metrics, "performance": perf}


def main():
    parser = argparse.ArgumentParser(description="Asteroid Monitoring Pipeline")
    parser.parse_args()
    MonitoringPipeline().run()


if __name__ == "__main__":
    main()