"""Unit tests for src.inference_pipeline.predictor."""
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.inference_pipeline.predictor import AsteroidPredictor
from src.common.features.schema import (
    get_feature_columns,
    get_prediction_column,
    get_probability_column,
)


@pytest.fixture
def mock_connection():
    """An MLflowConnectionManager with a mocked .client."""
    conn = MagicMock()
    client = MagicMock()
    conn.client = client

    # Default behaviour: champion model exists with cutoff in run params
    version = MagicMock()
    version.version = "7"
    version.run_id = "RUN_ABC"
    client.get_model_version_by_alias.return_value = version

    run = MagicMock()
    run.data.params = {"training_cutoff_date": "2024-01-15"}
    client.get_run.return_value = run

    return conn


@pytest.fixture
def predictor(mock_connection):
    """A predictor with mocked MLflow connection — no real model load."""
    return AsteroidPredictor(connection=mock_connection)


class TestLoadChampion:
    def test_populates_model_and_metadata(self, predictor):
        with patch(
            "mlflow.xgboost.load_model"
        ) as mock_load:
            mock_load.return_value = "MODEL_OBJECT"
            predictor.load_champion()

        assert predictor._model == "MODEL_OBJECT"
        assert predictor._model_version == "7"
        assert predictor._training_cutoff == "2024-01-15"

    def test_uses_alias_uri(self, predictor):
        with patch(
            "mlflow.xgboost.load_model"
        ) as mock_load:
            mock_load.return_value = "MODEL"
            predictor.load_champion()

        called_uri = mock_load.call_args.args[0]
        # Format: models:/{name}@{alias}
        assert called_uri.startswith("models:/")
        assert "@" in called_uri

    def test_is_idempotent(self, predictor):
        """Calling load_champion twice should only fetch once."""
        with patch(
            "mlflow.xgboost.load_model"
        ) as mock_load:
            mock_load.return_value = "MODEL"
            predictor.load_champion()
            predictor.load_champion()
            predictor.load_champion()

        assert mock_load.call_count == 1

    def test_training_cutoff_missing_is_none(self, predictor, mock_connection):
        # Run has no training_cutoff_date param
        mock_connection.client.get_run.return_value.data.params = {}

        with patch(
            "mlflow.xgboost.load_model"
        ):
            predictor.load_champion()

        assert predictor._training_cutoff is None


class TestPredict:
    def test_predict_returns_input_plus_prediction_columns(
        self, predictor, features_df
    ):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1] * len(features_df))
        proba = np.array([[0.2, 0.8]] * len(features_df))
        mock_model.predict_proba.return_value = proba

        with patch(
            "mlflow.xgboost.load_model",
            return_value=mock_model,
        ):
            result = predictor.predict(features_df)

        # Original columns preserved
        for col in features_df.columns:
            assert col in result.columns
        # New prediction columns added
        assert get_prediction_column() in result.columns
        assert get_probability_column() in result.columns
        assert "model_version" in result.columns

    def test_predict_filters_to_feature_columns_only(self, predictor, features_df):
        captured = {}
        mock_model = MagicMock()

        def fake_predict(X):
            captured["X"] = X.copy()
            return np.zeros(len(X), dtype=int)

        mock_model.predict.side_effect = fake_predict
        mock_model.predict_proba.return_value = np.zeros((len(features_df), 2))

        with patch(
            "mlflow.xgboost.load_model",
            return_value=mock_model,
        ):
            predictor.predict(features_df)

        # The matrix passed to model.predict must NOT contain identity columns
        assert set(captured["X"].columns) == set(get_feature_columns())
        assert "asteroid_id" not in captured["X"].columns
        assert "name" not in captured["X"].columns

    def test_predictions_are_int_probabilities_are_float(
        self, predictor, features_df
    ):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1, 0, 1] + [0] * 7)
        mock_model.predict_proba.return_value = np.array(
            [[0.2, 0.8], [0.7, 0.3], [0.1, 0.9]] + [[0.6, 0.4]] * 7
        )

        with patch(
            "mlflow.xgboost.load_model",
            return_value=mock_model,
        ):
            result = predictor.predict(features_df)

        assert pd.api.types.is_integer_dtype(result[get_prediction_column()])
        assert pd.api.types.is_float_dtype(result[get_probability_column()])

    def test_probability_is_class_one_column(self, predictor, features_df):
        """We expose P(hazardous), which is column 1 of predict_proba output."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.zeros(len(features_df), dtype=int)
        # P(class 0) = 0.7, P(class 1) = 0.3 — we want the 0.3
        mock_model.predict_proba.return_value = np.array(
            [[0.7, 0.3]] * len(features_df)
        )

        with patch(
            "mlflow.xgboost.load_model",
            return_value=mock_model,
        ):
            result = predictor.predict(features_df)

        assert np.allclose(result[get_probability_column()], 0.3)

    def test_predict_auto_loads_champion(self, predictor, features_df):
        """Calling predict without prior load_champion should still work."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.zeros(len(features_df), dtype=int)
        mock_model.predict_proba.return_value = np.zeros((len(features_df), 2))

        with patch(
            "mlflow.xgboost.load_model",
            return_value=mock_model,
        ):
            assert predictor._model is None
            predictor.predict(features_df)
            assert predictor._model is mock_model


class TestProperties:
    def test_model_version_before_load(self, predictor):
        # Pre-load — returns sentinel, not None
        assert predictor.model_version == "not loaded"

    def test_training_cutoff_triggers_load(self, predictor):
        with patch(
            "mlflow.xgboost.load_model",
            return_value="MODEL",
        ):
            assert predictor._model is None
            cutoff = predictor.training_cutoff
            # Accessing the property auto-loads the champion
            assert predictor._model == "MODEL"
            assert cutoff == "2024-01-15"