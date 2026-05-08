import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
)
from src.utils import get_logger

log = get_logger(__name__)


class ModelEvaluator:
    """
    Evaluates a trained model.
    Returns metrics.
    """

    def evaluate(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> dict:
        """
        Compute classification metrics.

        Args:
            model: trained classifier with predict + predict_proba
            X:     feature DataFrame
            y:     true labels
        Returns:
            dict of metrics
        """
        preds = model.predict(X)
        proba = model.predict_proba(X)[:, 1]

        cm = confusion_matrix(y, preds)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        metrics = {
            "accuracy":  accuracy_score(y, preds),
            "f1":        f1_score(y, preds, zero_division=0),
            "roc_auc":   roc_auc_score(y, proba),
            "precision": precision_score(y, preds, zero_division=0),
            "recall":    recall_score(y, preds, zero_division=0),
            "true_positives":  int(tp),
            "false_positives": int(fp),
            "true_negatives":  int(tn),
            "false_negatives": int(fn),
        }

        self._log_metrics(metrics)
        return metrics

    def _log_metrics(self, metrics: dict) -> None:
        """Pretty-print metrics."""
        log.info("─" * 40)
        log.info("Evaluation Metrics:")
        log.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        log.info(f"  F1:        {metrics['f1']:.4f}")
        log.info(f"  Precision: {metrics['precision']:.4f}")
        log.info(f"  Recall:    {metrics['recall']:.4f}")
        log.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        log.info(
            f"  Confusion: TP={metrics['true_positives']} "
            f"FP={metrics['false_positives']} "
            f"TN={metrics['true_negatives']} "
            f"FN={metrics['false_negatives']}"
        )
        log.info("─" * 40)