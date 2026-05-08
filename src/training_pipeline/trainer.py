import math
import pandas as pd
import xgboost as xgb
from src.utils import get_logger

log = get_logger(__name__)

from src.config import XGBOOST_PARAMS


class ModelTrainer:
    """
    Trains XGBoost classifier for asteroid hazard prediction.
    Returns trained model.
    """
    def __init__(
            self,
            params: dict | None = None,
    ):
        self.params = params or XGBOOST_PARAMS.copy()
        self.model: xgb.XGBClassifier | None = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_eval: pd.DataFrame = None,
        y_eval: pd.Series = None,
    ) -> xgb.XGBClassifier:
        """
        Train XGBoost on given data.

        Args:
            X_train, y_train: training data
            X_eval, y_eval:   optional eval set for monitoring during training
        Returns:
            trained XGBoost classifier
        """
        # Handle class imbalance — PHAs are rare (~5-10%)
        scale_pos_weight = self._compute_class_weight(y_train)
        log.info(f"Class weight (negative/positive): {scale_pos_weight:.2f}")

        # Build model
        self.model = xgb.XGBClassifier(
            **self.params,
            scale_pos_weight=scale_pos_weight, # Mistakes on hazardous asteroids are more important.
        )

        # Fit
        if X_eval is not None and y_eval is not None:
            eval_set = [(X_eval, y_eval)]
        else:
            eval_set = None
        log.info("Training XGBoost...")
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False,
        )
        log.info("Training complete.")

        return self.model

    def _compute_class_weight(self, y: pd.Series) -> float:
        y = y.squeeze()
        negative_count = int((y == 0).sum())
        positive_count = int((y == 1).sum())

        if positive_count == 0:
            raise ValueError("Training labels contain no positive examples.")
        if negative_count == 0:
            raise ValueError("Training labels contain no negative examples.")

        ratio = negative_count / positive_count
        weight = math.sqrt(ratio)  # ← softer than full ratio
        log.info(f"Class ratio: {ratio:.2f} → scale_pos_weight: {weight:.2f}")
        return weight

    def get_feature_importance(self, importance_type: str = "gain") -> dict:
        """Return feature importance after training."""
        if self.model is None:
            raise RuntimeError("Model not trained yet.")

        return self.model.get_booster().get_score(
            importance_type=importance_type
        )