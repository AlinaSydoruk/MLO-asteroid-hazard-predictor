import pandas as pd

from src.utils import get_logger

log = get_logger(__name__)


class PredictionDriftDetector:
    """Tracks the distribution of recent predictions."""

    def detect(self, predictions: pd.DataFrame) -> dict:
        if predictions is None or len(predictions) == 0:
            return {"n_predictions": 0, "mean_proba": 0.0, "pct_hazardous": 0.0}

        proba_col = next((c for c in ["pred_proba", "hazard_probability", "score"]
                          if c in predictions.columns), None)
        label_col = next((c for c in ["pred_label", "prediction", "hazardous"]
                          if c in predictions.columns), None)

        out = {
            "n_predictions": int(len(predictions)),
            "mean_proba": float(predictions[proba_col].mean()) if proba_col else 0.0,
            "std_proba":  float(predictions[proba_col].std())  if proba_col else 0.0,
            "pct_hazardous": float((predictions[label_col] == 1).mean())
                             if label_col else 0.0,
        }
        log.info(f"Prediction drift: {out}")
        return out