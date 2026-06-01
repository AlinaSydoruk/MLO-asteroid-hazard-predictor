"""Unit tests for src.training_pipeline.pipeline.TrainingPipeline."""
from unittest.mock import patch

import pandas as pd
import pytest


PIPELINE_MOD = "src.training_pipeline.pipeline"


@pytest.fixture
def pipeline():
    """A TrainingPipeline with every collaborator mocked."""
    import src.training_pipeline.pipeline  # noqa: F401 — let patch() resolve attrs

    with patch(f"{PIPELINE_MOD}.MLflowConnectionManager"), \
         patch(f"{PIPELINE_MOD}.TrainingDataLoader"), \
         patch(f"{PIPELINE_MOD}.ModelTrainer"), \
         patch(f"{PIPELINE_MOD}.ModelEvaluator"), \
         patch(f"{PIPELINE_MOD}.ModelRegistryRepository"):

        from src.training_pipeline.pipeline import TrainingPipeline
        p = TrainingPipeline(auto_promote=True, promotion_metric="f1")
        yield p


@pytest.fixture
def fake_loader_output(X_features, y_labels):
    """What data_loader.load() returns: (Xtr, Xv, Xt, ytr, yv, yt, iso)."""
    third = len(X_features) // 3
    return (
        X_features.iloc[:third],
        X_features.iloc[third:2 * third],
        X_features.iloc[2 * third:],
        y_labels.iloc[:third],
        y_labels.iloc[third:2 * third],
        y_labels.iloc[2 * third:],
        "2024-06-15",
    )


def _wire_pipeline(pipeline, loader_output, metrics):
    """Hook up mock collaborators so run() can flow end-to-end."""
    pipeline.data_loader.load.return_value = loader_output
    pipeline.trainer.train.return_value = "TRAINED_MODEL"
    pipeline.trainer.params = {"max_depth": 3}
    pipeline.trainer.get_feature_importance.return_value = {"f0": 1.0}
    pipeline.evaluator.evaluate.return_value = metrics
    pipeline.registry.log_run.return_value = "RUN_ID_123"


class TestRunOrchestration:
    def test_calls_each_stage_in_order(self, pipeline, fake_loader_output):
        metrics = {"f1": 0.8, "roc_auc": 0.9}
        _wire_pipeline(pipeline, fake_loader_output, metrics)
        pipeline.registry.should_promote.return_value = False

        result = pipeline.run(training_cutoff=pd.Timestamp("2024-06-15"))

        pipeline.data_loader.load.assert_called_once()
        pipeline.trainer.train.assert_called_once()
        pipeline.evaluator.evaluate.assert_called_once()
        pipeline.registry.log_run.assert_called_once()
        assert result == metrics

    def test_passes_explicit_cutoff_to_loader(self, pipeline, fake_loader_output):
        _wire_pipeline(pipeline, fake_loader_output, {"f1": 0.5})
        pipeline.registry.should_promote.return_value = False

        cutoff = pd.Timestamp("2024-06-15")
        pipeline.run(training_cutoff=cutoff)

        pipeline.data_loader.load.assert_called_once_with(training_cutoff=cutoff)

    def test_defaults_cutoff_when_none(self, pipeline, fake_loader_output):
        _wire_pipeline(pipeline, fake_loader_output, {"f1": 0.5})
        pipeline.registry.should_promote.return_value = False

        pipeline.run(training_cutoff=None)

        # Loader was called with *some* Timestamp — pipeline computed default
        cutoff_arg = pipeline.data_loader.load.call_args.kwargs["training_cutoff"]
        assert isinstance(cutoff_arg, pd.Timestamp)

    def test_log_run_receives_training_artifacts(self, pipeline, fake_loader_output):
        metrics = {"f1": 0.7}
        _wire_pipeline(pipeline, fake_loader_output, metrics)
        pipeline.registry.should_promote.return_value = False

        pipeline.run(training_cutoff=pd.Timestamp("2024-06-15"))

        kwargs = pipeline.registry.log_run.call_args.kwargs
        assert kwargs["model"] == "TRAINED_MODEL"
        assert kwargs["metrics"] == metrics
        assert kwargs["feature_importance"] == {"f0": 1.0}
        # cutoff date must be threaded into the run params
        assert kwargs["params"]["training_cutoff_date"] == "2024-06-15"


class TestPromotionLogic:
    def test_promotes_when_should_promote_true(self, pipeline, fake_loader_output):
        _wire_pipeline(pipeline, fake_loader_output, {"f1": 0.9})
        pipeline.registry.should_promote.return_value = True

        pipeline.run(training_cutoff=pd.Timestamp("2024-06-15"))

        pipeline.registry.promote_to_champion.assert_called_once()

    def test_does_not_promote_when_should_promote_false(
        self, pipeline, fake_loader_output
    ):
        _wire_pipeline(pipeline, fake_loader_output, {"f1": 0.3})
        pipeline.registry.should_promote.return_value = False

        pipeline.run(training_cutoff=pd.Timestamp("2024-06-15"))

        pipeline.registry.promote_to_champion.assert_not_called()

    def test_does_not_promote_when_auto_promote_disabled(
        self, fake_loader_output
    ):
        import src.training_pipeline.pipeline  # noqa: F401
        with patch(f"{PIPELINE_MOD}.MLflowConnectionManager"), \
             patch(f"{PIPELINE_MOD}.TrainingDataLoader"), \
             patch(f"{PIPELINE_MOD}.ModelTrainer"), \
             patch(f"{PIPELINE_MOD}.ModelEvaluator"), \
             patch(f"{PIPELINE_MOD}.ModelRegistryRepository"):

            from src.training_pipeline.pipeline import TrainingPipeline
            p = TrainingPipeline(auto_promote=False, promotion_metric="f1")
            _wire_pipeline(p, fake_loader_output, {"f1": 0.99})
            # Even a glowing score shouldn't promote when auto_promote=False
            p.registry.should_promote.return_value = True

            p.run(training_cutoff=pd.Timestamp("2024-06-15"))

            p.registry.promote_to_champion.assert_not_called()

    def test_force_promote_bypasses_should_promote(self, fake_loader_output):
        import src.training_pipeline.pipeline  # noqa: F401
        with patch(f"{PIPELINE_MOD}.MLflowConnectionManager"), \
             patch(f"{PIPELINE_MOD}.TrainingDataLoader"), \
             patch(f"{PIPELINE_MOD}.ModelTrainer"), \
             patch(f"{PIPELINE_MOD}.ModelEvaluator"), \
             patch(f"{PIPELINE_MOD}.ModelRegistryRepository"):

            from src.training_pipeline.pipeline import TrainingPipeline
            p = TrainingPipeline(
                auto_promote=False,
                promotion_metric="f1",
                force_promote=True,
            )
            _wire_pipeline(p, fake_loader_output, {"f1": 0.1})
            # Should_promote would say no, but force_promote overrides it
            p.registry.should_promote.return_value = False

            p.run(training_cutoff=pd.Timestamp("2024-06-15"))

            p.registry.promote_to_champion.assert_called_once()

    def test_should_promote_called_with_chosen_metric(
        self, pipeline, fake_loader_output
    ):
        metrics = {"f1": 0.7, "roc_auc": 0.9}
        _wire_pipeline(pipeline, fake_loader_output, metrics)
        pipeline.registry.should_promote.return_value = False

        pipeline.run(training_cutoff=pd.Timestamp("2024-06-15"))

        pipeline.registry.should_promote.assert_called_once_with(
            new_metrics=metrics,
            metric_name="f1",
        )