import pandas as pd

from src.config import (
    VALIDATION_SIZE,
    TEST_SIZE,
)
from src.common.hopsworks.feature_view_repo import DedupFeatureViewRepository
from src.common.feature_schema import get_feature_columns
from src.utils import get_logger

log = get_logger(__name__)


class TrainingDataLoader:
    """Loads training data via the dedup feature view."""

    def __init__(
        self,
        feature_view_repo: DedupFeatureViewRepository | None = None,
        val_fraction: float = VALIDATION_SIZE,
        test_fraction: float = TEST_SIZE,
    ):
        self.feature_view_repo = feature_view_repo or DedupFeatureViewRepository()
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction

    def load(self, training_cutoff: pd.Timestamp) -> tuple:
        X_train, X_val, X_test, y_train, y_val, y_test = self.feature_view_repo.get_training_splits(
            training_cutoff=training_cutoff,
            validation_size=self.val_fraction,
            test_size=self.test_fraction,
        )

        # Keep only model features (drop identity columns like asteroid_id, name, dates).
        feature_cols = get_feature_columns()
        X_train = X_train[feature_cols]
        X_val   = X_val[feature_cols]
        X_test  = X_test[feature_cols]

        log.info(
            f"Split @ cutoff={training_cutoff.date()}:  "
            f"train={len(X_train)}, val={len(X_val)}, test={len(X_test)}  "
            f"(asteroids first observed AFTER {training_cutoff.date()} are reserved for inference)"
        )

        return (
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            training_cutoff.date().isoformat(),
        )