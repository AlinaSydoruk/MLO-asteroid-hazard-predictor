"""Unit tests for src.inference_pipeline.pipeline.InferencePipeline."""
from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest


PIPELINE_MOD = "src.inference_pipeline.pipeline"


@pytest.fixture
def pipeline():
    """InferencePipeline with all external dependencies mocked."""
    import src.inference_pipeline.pipeline  # noqa: F401

    with patch(f"{PIPELINE_MOD}.HopsworksConnectionManager"), \
         patch(f"{PIPELINE_MOD}.AsteroidFeatureView"), \
         patch(f"{PIPELINE_MOD}.AsteroidPredictor"), \
         patch(f"{PIPELINE_MOD}.AsteroidPredictionsRepository"):

        from src.inference_pipeline.pipeline import InferencePipeline
        p = InferencePipeline()
        yield p


def _predicted(features_df: pd.DataFrame) -> pd.DataFrame:
    """Mimic what AsteroidPredictor.predict() returns: input + 3 new cols."""
    out = features_df.copy()
    out["predicted_hazardous"] = [1, 0] * (len(features_df) // 2) + \
                                 [0] * (len(features_df) % 2)
    out["hazard_probability"] = 0.5
    out["model_version"] = "7"
    return out


def _wire(pipeline, batch_df, training_cutoff="2020-01-01", model_version="7"):
    """Hook up the mocked collaborators."""
    pipeline.feature_view.get_batch_data.return_value = batch_df
    pipeline.predictor.training_cutoff = training_cutoff
    pipeline.predictor.model_version = model_version
    pipeline.predictor.predict.side_effect = _predicted


class TestRunHappyPath:
    def test_calls_each_stage(self, pipeline, features_df):
        _wire(pipeline, features_df)

        result = pipeline.run()

        pipeline.feature_view.get_batch_data.assert_called_once()
        pipeline.predictor.load_champion.assert_called_once()
        pipeline.predictor.predict.assert_called_once()
        pipeline.prediction_repo.insert.assert_called_once()
        assert len(result) == len(features_df)

    def test_returns_predictions_with_source_column(self, pipeline, features_df):
        _wire(pipeline, features_df)
        result = pipeline.run()
        assert "source" in result.columns

    def test_fetches_today_to_tomorrow_window(self, pipeline, features_df):
        _wire(pipeline, features_df)
        pipeline.run()

        kwargs = pipeline.feature_view.get_batch_data.call_args.kwargs
        start = date.fromisoformat(kwargs["start_time"])
        end = date.fromisoformat(kwargs["end_time"])
        # End is exclusive (today + 1) so the window covers today only
        assert (end - start).days == 1


class TestEmptyData:
    def test_returns_early_when_no_asteroids(self, pipeline):
        pipeline.feature_view.get_batch_data.return_value = pd.DataFrame()

        result = pipeline.run()

        assert result.empty
        # No predictions, no inserts when there's nothing to predict on
        pipeline.predictor.predict.assert_not_called()
        pipeline.prediction_repo.insert.assert_not_called()


class TestSourceTagging:
    """
    Asteroids first observed AFTER the training cutoff are 'model_prediction'
    (the model has never seen them). Those observed before are 'nasa_catalog'.
    """

    def test_new_asteroids_tagged_model_prediction(
        self, pipeline, make_features_df
    ):
        df = make_features_df(n_rows=4)
        # Training cutoff before all first_observation_dates → all NEW
        df["first_observation_date"] = pd.to_datetime("2024-01-01")
        _wire(pipeline, df, training_cutoff="2020-01-01")

        result = pipeline.run()

        assert (result["source"] == "model_prediction").all()

    def test_known_asteroids_tagged_nasa_catalog(
        self, pipeline, make_features_df
    ):
        df = make_features_df(n_rows=4)
        # All asteroids observed BEFORE the training cutoff → all KNOWN
        df["first_observation_date"] = pd.to_datetime("2015-01-01")
        _wire(pipeline, df, training_cutoff="2020-01-01")

        result = pipeline.run()

        assert (result["source"] == "nasa_catalog").all()

    def test_mixed_known_and_new_tagged_correctly(
        self, pipeline, make_features_df
    ):
        df = make_features_df(n_rows=4)
        df["first_observation_date"] = pd.to_datetime(
            ["2015-01-01", "2024-01-01", "2015-01-01", "2024-01-01"]
        )
        _wire(pipeline, df, training_cutoff="2020-01-01")

        result = pipeline.run()

        # Row order preserved → indices 1, 3 are new, 0, 2 are known
        assert list(result["source"]) == [
            "nasa_catalog", "model_prediction",
            "nasa_catalog", "model_prediction",
        ]

    def test_missing_training_cutoff_defaults_to_nasa_catalog(
        self, pipeline, features_df
    ):
        """If the model has no recorded cutoff, we can't tell — assume known."""
        _wire(pipeline, features_df, training_cutoff=None)

        result = pipeline.run()

        assert (result["source"] == "nasa_catalog").all()


class TestPersistence:
    def test_saves_only_identity_and_prediction_columns(
        self, pipeline, features_df
    ):
        _wire(pipeline, features_df)
        pipeline.run()

        saved_df = pipeline.prediction_repo.insert.call_args.args[0]

        # Should include identity columns
        assert "asteroid_id" in saved_df.columns
        # Should include prediction columns
        assert "predicted_hazardous" in saved_df.columns
        assert "hazard_probability" in saved_df.columns
        assert "model_version" in saved_df.columns
        assert "source" in saved_df.columns
        # Should NOT include raw model features — those bloat the predictions
        # store and are already available in the feature group
        assert "moid_au" not in saved_df.columns
        assert "kinetic_energy_proxy" not in saved_df.columns


class TestRunForRange:
    def test_fetches_explicit_date_range(self, pipeline, features_df):
        _wire(pipeline, features_df)

        pipeline.run_for_range(
            start=date(2024, 6, 1),
            end=date(2024, 6, 7),
        )

        kwargs = pipeline.feature_view.get_batch_data.call_args.kwargs
        assert kwargs["start_time"] == "2024-06-01"
        # end is exclusive — June 7 inclusive means June 8 as the boundary
        assert kwargs["end_time"] == "2024-06-08"

    def test_returns_empty_when_no_data_in_range(self, pipeline):
        pipeline.feature_view.get_batch_data.return_value = pd.DataFrame()

        result = pipeline.run_for_range(
            start=date(2024, 6, 1),
            end=date(2024, 6, 7),
        )

        assert result.empty
        pipeline.predictor.predict.assert_not_called()

    def test_range_inference_persists_predictions(self, pipeline, features_df):
        _wire(pipeline, features_df)

        pipeline.run_for_range(
            start=date(2024, 6, 1),
            end=date(2024, 6, 7),
        )

        pipeline.prediction_repo.insert.assert_called_once()