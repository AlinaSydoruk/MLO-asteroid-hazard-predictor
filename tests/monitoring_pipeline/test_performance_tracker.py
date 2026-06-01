"""Unit tests for src.monitoring_pipeline.performance_tracker."""
import pandas as pd
import pytest

from src.monitoring_pipeline.performance_tracker import PerformanceTracker


@pytest.fixture
def tracker():
    return PerformanceTracker()


def _preds(asteroid_ids, dates, preds, probas=None):
    """Build an old-predictions DataFrame."""
    df = pd.DataFrame({
        "asteroid_id":          asteroid_ids,
        "close_approach_date":  pd.to_datetime(dates),
        "predicted_hazardous":  preds,
    })
    if probas is not None:
        df["hazard_probability"] = probas
    return df


def _labels(asteroid_ids, dates, labels):
    """Build a current-labels DataFrame (from the feature view)."""
    return pd.DataFrame({
        "asteroid_id":              asteroid_ids,
        "close_approach_date":      pd.to_datetime(dates),
        "is_potentially_hazardous": labels,
    })


class TestEmptyInputs:
    def test_empty_predictions_returns_none_metrics(self, tracker):
        result = tracker.evaluate(
            old_predictions=pd.DataFrame(),
            current_labels=_labels(["a"], ["2024-01-01"], [1]),
        )
        assert result["n_evaluated"] == 0
        assert result["f1"] is None

    def test_empty_labels_returns_none_metrics(self, tracker):
        result = tracker.evaluate(
            old_predictions=_preds(["a"], ["2024-01-01"], [1]),
            current_labels=pd.DataFrame(),
        )
        assert result["n_evaluated"] == 0


class TestJoinAndMetrics:
    def test_perfect_match_yields_perfect_metrics(self, tracker):
        preds = _preds(
            ["a", "b", "c", "d"],
            ["2024-01-01"] * 4,
            [1, 0, 1, 0],
            probas=[0.9, 0.1, 0.8, 0.2],
        )
        labels = _labels(
            ["a", "b", "c", "d"],
            ["2024-01-01"] * 4,
            [1, 0, 1, 0],  # same as preds
        )
        result = tracker.evaluate(preds, labels)

        assert result["n_evaluated"] == 4
        assert result["f1"] == pytest.approx(1.0)
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(1.0)
        assert result["auc"] == pytest.approx(1.0)

    def test_zero_recall_with_all_misses(self, tracker):
        # All positives predicted as negative
        preds = _preds(["a", "b"], ["2024-01-01"] * 2, [0, 0])
        labels = _labels(["a", "b"], ["2024-01-01"] * 2, [1, 1])
        result = tracker.evaluate(preds, labels)

        assert result["recall"] == pytest.approx(0.0)
        assert result["f1"] == pytest.approx(0.0)

    def test_partial_overlap_evaluates_only_intersection(self, tracker):
        preds = _preds(
            ["a", "b", "c"],
            ["2024-01-01"] * 3,
            [1, 0, 1],
        )
        labels = _labels(
            ["a", "b"],  # 'c' has no label yet
            ["2024-01-01"] * 2,
            [1, 0],
        )
        result = tracker.evaluate(preds, labels)
        assert result["n_evaluated"] == 2

    def test_zero_overlap_returns_none(self, tracker):
        preds = _preds(["x", "y"], ["2024-01-01"] * 2, [1, 0])
        labels = _labels(["a", "b"], ["2024-01-01"] * 2, [1, 0])
        result = tracker.evaluate(preds, labels)
        assert result["n_evaluated"] == 0
        assert result["f1"] is None


class TestAUC:
    def test_auc_computed_when_probabilities_present(self, tracker):
        preds = _preds(
            ["a", "b", "c", "d"],
            ["2024-01-01"] * 4,
            [1, 0, 1, 0],
            probas=[0.9, 0.1, 0.8, 0.2],
        )
        labels = _labels(["a", "b", "c", "d"], ["2024-01-01"] * 4, [1, 0, 1, 0])
        result = tracker.evaluate(preds, labels)
        assert "auc" in result
        assert result["auc"] == pytest.approx(1.0)

    def test_auc_skipped_when_only_one_class_in_labels(self, tracker):
        # All positives — roc_auc would raise; tracker should skip it
        preds = _preds(["a", "b"], ["2024-01-01"] * 2, [1, 1],
                       probas=[0.9, 0.8])
        labels = _labels(["a", "b"], ["2024-01-01"] * 2, [1, 1])
        result = tracker.evaluate(preds, labels)
        # f1/precision/recall computed, but no AUC
        assert "auc" not in result

    def test_auc_skipped_when_no_probability_column(self, tracker):
        preds = _preds(["a", "b"], ["2024-01-01"] * 2, [1, 0])
        labels = _labels(["a", "b"], ["2024-01-01"] * 2, [1, 0])
        result = tracker.evaluate(preds, labels)
        assert "auc" not in result


class TestCustomKeys:
    def test_accepts_explicit_join_keys(self, tracker):
        # Use a single join key instead of the schema default
        preds = pd.DataFrame({
            "id":                  ["a", "b"],
            "predicted_hazardous": [1, 0],
        })
        labels = pd.DataFrame({
            "id":                       ["a", "b"],
            "is_potentially_hazardous": [1, 0],
        })
        result = tracker.evaluate(preds, labels, join_keys=["id"])
        assert result["n_evaluated"] == 2
        assert result["f1"] == pytest.approx(1.0)