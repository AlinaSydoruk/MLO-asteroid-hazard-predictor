import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from src.utils import get_logger

log = get_logger(__name__)


class PerformanceTracker:
    """Joins old predictions with up-to-date labels and computes real metrics."""

    def evaluate(
        self,
        old_predictions: pd.DataFrame,
        current_labels: pd.DataFrame,
        join_key: str = "asteroid_id",
        label_col: str = "is_potentially_hazardous",
        pred_col: str = "pred_label",
        proba_col: str = "pred_proba",
    ) -> dict:
        if len(old_predictions) == 0 or len(current_labels) == 0:
            return {"f1": None, "precision": None, "recall": None, "auc": None,
                    "n_evaluated": 0}

        merged = old_predictions.merge(
            current_labels[[join_key, label_col]],
            on=join_key, how="inner",
        )
        if len(merged) == 0:
            log.warning("No overlap between old predictions and current labels.")
            return {"f1": None, "n_evaluated": 0}

        y_true = merged[label_col].astype(int)
        y_pred = merged[pred_col].astype(int)
        out = {
            "n_evaluated": int(len(merged)),
            "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        }
        if proba_col in merged.columns and y_true.nunique() > 1:
            out["auc"] = float(roc_auc_score(y_true, merged[proba_col]))
        log.info(f"Lagged performance: {out}")
        return out