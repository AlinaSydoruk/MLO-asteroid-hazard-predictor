"""Unit tests for src.training_pipeline.trainer."""
import math
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.training_pipeline.trainer import ModelTrainer
from src.config import XGBOOST_PARAMS


class TestComputeClassWeight:
    def test_returns_sqrt_of_ratio(self):
        trainer = ModelTrainer()
        # 90 negatives, 10 positives → ratio = 9 → sqrt(9) = 3
        y = pd.Series([0] * 90 + [1] * 10)
        assert trainer._compute_class_weight(y) == pytest.approx(math.sqrt(9))

    def test_balanced_classes_yield_weight_of_one(self):
        trainer = ModelTrainer()
        y = pd.Series([0, 1] * 50)
        assert trainer._compute_class_weight(y) == pytest.approx(1.0)

    def test_raises_when_no_positives(self):
        trainer = ModelTrainer()
        y = pd.Series([0] * 50)
        with pytest.raises(ValueError, match="no positive"):
            trainer._compute_class_weight(y)

    def test_raises_when_no_negatives(self):
        trainer = ModelTrainer()
        y = pd.Series([1] * 50)
        with pytest.raises(ValueError, match="no negative"):
            trainer._compute_class_weight(y)


class TestParams:
    def test_uses_config_defaults(self):
        trainer = ModelTrainer()
        assert trainer.params == XGBOOST_PARAMS

    def test_accepts_override_params(self):
        custom = {"max_depth": 99, "learning_rate": 0.01}
        trainer = ModelTrainer(params=custom)
        assert trainer.params == custom

    def test_params_copy_isolates_from_config(self):
        # Mutating trainer.params must NOT change the global config
        trainer = ModelTrainer()
        original = XGBOOST_PARAMS.copy()
        trainer.params["max_depth"] = 999
        assert XGBOOST_PARAMS == original


class TestTrain:
    def test_fits_xgb_classifier_with_scale_pos_weight(
        self, X_features, y_labels
    ):
        with patch(
            "src.training_pipeline.trainer.xgb.XGBClassifier"
        ) as MockXGB:
            mock_model = MagicMock()
            MockXGB.return_value = mock_model

            trainer = ModelTrainer()
            result = trainer.train(X_features, y_labels)

            # XGBClassifier constructed with scale_pos_weight injected
            ctor_kwargs = MockXGB.call_args.kwargs
            assert "scale_pos_weight" in ctor_kwargs
            assert ctor_kwargs["scale_pos_weight"] > 0
            # fit() called and model returned
            mock_model.fit.assert_called_once()
            assert result is mock_model

    def test_passes_eval_set_when_test_data_given(
        self, X_features, y_labels, make_features, make_labels
    ):
        X_test = make_features(n_rows=20, seed=99)
        y_test = make_labels(n_rows=20, seed=99)

        with patch("src.training_pipeline.trainer.xgb.XGBClassifier") as MockXGB:
            mock_model = MagicMock()
            MockXGB.return_value = mock_model

            trainer = ModelTrainer()
            trainer.train(X_features, y_labels, X_test=X_test, y_test=y_test)

            eval_set = mock_model.fit.call_args.kwargs["eval_set"]
            assert eval_set is not None
            assert eval_set == [(X_test, y_test)]

    def test_no_eval_set_when_test_data_absent(self, X_features, y_labels):
        with patch("src.training_pipeline.trainer.xgb.XGBClassifier") as MockXGB:
            mock_model = MagicMock()
            MockXGB.return_value = mock_model

            trainer = ModelTrainer()
            trainer.train(X_features, y_labels)

            assert mock_model.fit.call_args.kwargs["eval_set"] is None

    def test_stores_model_on_instance(self, X_features, y_labels):
        with patch("src.training_pipeline.trainer.xgb.XGBClassifier") as MockXGB:
            mock_model = MagicMock()
            MockXGB.return_value = mock_model

            trainer = ModelTrainer()
            assert trainer.model is None
            trainer.train(X_features, y_labels)
            assert trainer.model is mock_model


class TestFeatureImportance:
    def test_raises_when_not_trained(self):
        trainer = ModelTrainer()
        with pytest.raises(RuntimeError, match="not trained"):
            trainer.get_feature_importance()

    def test_returns_booster_scores_after_training(self):
        trainer = ModelTrainer()
        mock_booster = MagicMock()
        mock_booster.get_score.return_value = {"f0": 1.5, "f1": 0.8}
        trainer.model = MagicMock()
        trainer.model.get_booster.return_value = mock_booster

        result = trainer.get_feature_importance(importance_type="gain")

        mock_booster.get_score.assert_called_once_with(importance_type="gain")
        assert result == {"f0": 1.5, "f1": 0.8}


class TestEndToEndTraining:
    """One real XGBoost run on tiny data — catches API drift."""

    def test_trains_on_real_data(self, X_features, y_labels):
        # Tiny n_estimators keeps this fast
        trainer = ModelTrainer(params={"n_estimators": 5, "max_depth": 2,
                                       "verbosity": 0})
        model = trainer.train(X_features, y_labels)
        preds = model.predict(X_features)
        assert len(preds) == len(X_features)
        assert set(preds).issubset({0, 1})