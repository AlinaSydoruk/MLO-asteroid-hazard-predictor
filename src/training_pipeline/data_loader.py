import pandas as pd

from src.config import TEST_SIZE
from src.common.hopsworks.feature_view_repo import FeatureViewRepository
from src.common.feature_schema import get_feature_columns
from src.utils import get_logger


log = get_logger(__name__)


class TrainingDataLoader:
    """
    Loads training data from Hopsworks Feature View.
    Handles train/test split and feature column selection.
    """

    def __init__(
        self,
        feature_view_repo: FeatureViewRepository | None = None,
        test_size: float = TEST_SIZE,
    ):
        self.feature_view_repo = feature_view_repo or FeatureViewRepository()
        self.test_size = test_size

    def load(self) -> tuple:
        """ Returns: X_train, X_val, X_test, y_train, y_val, y_test """
        log.info("Loading training data...")
        X_train, X_val, X_test, y_train, y_val, y_test = (
            self.feature_view_repo.get_training_data()
        )

        feature_cols = get_feature_columns()
        X_train = X_train[feature_cols]
        X_val = X_val[feature_cols]
        X_test = X_test[feature_cols]

        log.info("Train: %s rows | Val: %s rows | Test: %s rows",
                 len(X_train), len(X_val), len(X_test))
        log.info("Features: %s", X_train.shape[1])

        return X_train, X_val, X_test, y_train, y_val, y_test