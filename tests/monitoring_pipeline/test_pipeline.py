"""Unit tests for src.monitoring_pipeline.pipeline.MonitoringPipeline."""
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


PIPELINE_MOD = "src.monitoring_pipeline.pipeline"


@pytest.fixture
def pipeline():
    """A MonitoringPipeline with every collaborator and mlflow mocked."""
    import src.monitoring_pipeline.pipeline  # noqa: F401

    with patch(f"{PIPELINE_MOD}.HopsworksConnectionManager"), \
         patch(f"{PIPELINE_MOD}.MLflowConnectionManager"), \
         patch(f"{PIPELINE_MOD}.AsteroidFeatureView"), \
         patch(f"{PIPELINE_MOD}.AsteroidDedupFeatureView"), \
         patch(f"{PIPELINE_MOD}.AsteroidMonitoringFeatureView"), \
         patch(f"{PIPELINE_MOD}.AsteroidPredictionsRepository"), \
         patch(f"{PIPELINE_MOD}.DataQualityChecker"), \
         patch(f"{PIPELINE_MOD}.FeatureDriftDetector"), \
         patch(f"{PIPELINE_MOD}.PredictionDriftDetector"), \
         patch(f"{PIPELINE_MOD}.PerformanceTracker"), \
         patch(f"{PIPELINE_MOD}.mlflow") as mock_mlflow:

        # mlflow.start_run is used as a context manager
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(
            return_value=MagicMock()
        )
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        from src.monitoring_pipeline.pipeline import MonitoringPipeline
        p = MonitoringPipeline()
        p._mlflow = mock_mlflow  # expose to tests for assertions
        yield p


def _wire(pipeline, share_drifted=0.1, old_preds_empty=False):
    """Wire mock collaborators so run() can flow end-to-end."""
    from datetime import date, timedelta
    from src.config import PERFORMANCE_LAG_DAYS

    features_df = pd.DataFrame({"moid_au": [0.1] * 50})
    pipeline.fv_raw.get_batch_data.return_value = features_df
    pipeline.fv_dedup.get_batch_data.return_value = features_df
    pipeline.monitoring_view.get_batch_data.return_value = features_df

    pipeline.quality.check.return_value = {
        "rows": 50, "nulls": 0, "duplicates": 0,
        "null_ratio": 0.0, "is_healthy": True,
    }
    pipeline.feature_drift.detect.return_value = {
        "drifted_features": 1,
        "share_drifted": share_drifted,
        "dataset_drift": share_drifted >= 0.5,
        "report_path": "drift_report.html",
        "per_feature": {},
    }
    pipeline.pred_drift.detect.return_value = {
        "n_predictions": 100, "mean_proba": 0.3, "pct_hazardous": 0.1,
    }
    pipeline.perf.evaluate.return_value = {
        "n_evaluated": 20, "f1": 0.8, "precision": 0.9, "recall": 0.7,
    }

    # Predictions feature group. The orchestrator filters by close_approach_date
    # falling inside [today - LAG - 7, today - LAG]. Place rows accordingly.
    today = date.today()
    in_lag_window = today - timedelta(days=PERFORMANCE_LAG_DAYS + 3)
    future_date = today + timedelta(days=365)

    pred_fg = MagicMock()
    if old_preds_empty:
        # All predictions far in the future → none in the lagged window
        pred_fg.read.return_value = pd.DataFrame({
            "close_approach_date": pd.to_datetime([future_date] * 5),
            "predicted_hazardous": [0] * 5,
        })
    else:
        pred_fg.read.return_value = pd.DataFrame({
            "close_approach_date": pd.to_datetime([in_lag_window] * 5),
            "predicted_hazardous": [0] * 5,
        })
    pipeline.pred_repo.get_or_create.return_value = pred_fg


class TestOrchestration:
    def test_all_four_detectors_invoked(self, pipeline):
        _wire(pipeline)
        result = pipeline.run()

        pipeline.quality.check.assert_called_once()
        pipeline.feature_drift.detect.assert_called_once()
        pipeline.pred_drift.detect.assert_called_once()
        pipeline.perf.evaluate.assert_called_once()
        # Each detector's output shows up in the returned dict
        assert "quality" in result
        assert "drift" in result
        assert "predictions" in result
        assert "performance" in result

    def test_mlflow_run_started_and_metrics_logged(self, pipeline):
        _wire(pipeline)
        pipeline.run()

        pipeline._mlflow.start_run.assert_called_once()
        # At least one log_metrics call per detector → many total
        assert pipeline._mlflow.log_metrics.call_count >= 3
        # Window dates logged as params
        assert pipeline._mlflow.log_param.call_count >= 2


class TestAlerting:
    def test_alert_tag_set_when_drift_exceeds_threshold(self, pipeline):
        # 60% of features drifted — above the typical alert threshold
        _wire(pipeline, share_drifted=0.6)
        pipeline.run()

        # set_tag called with ("alert", "drift")
        tag_calls = [c for c in pipeline._mlflow.set_tag.call_args_list]
        tags_set = [c.args for c in tag_calls]
        assert ("alert", "drift") in tags_set

    def test_no_alert_when_drift_below_threshold(self, pipeline):
        _wire(pipeline, share_drifted=0.05)
        pipeline.run()

        tag_calls = [c.args for c in pipeline._mlflow.set_tag.call_args_list]
        assert ("alert", "drift") not in tag_calls


class TestPerformanceSkipping:
    def test_skips_performance_eval_when_no_old_predictions(self, pipeline):
        _wire(pipeline, old_preds_empty=True)
        result = pipeline.run()

        # Performance tracker not invoked when no predictions in the lag window
        pipeline.perf.evaluate.assert_not_called()
        assert result["performance"] == {}

    def test_invokes_performance_eval_with_old_predictions(self, pipeline):
        _wire(pipeline, old_preds_empty=False)
        pipeline.run()

        pipeline.perf.evaluate.assert_called_once()