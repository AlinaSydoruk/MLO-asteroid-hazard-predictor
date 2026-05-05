"""
Feature View Repository.
Single responsibility: manage the asteroid_pha_view Feature View.
Used by training pipeline and inference pipeline.
"""
import pandas as pd
from datetime import datetime

from src.config import (
    FEATURE_VIEW_NAME,
    FEATURE_VIEW_VERSION,
)
from src.feature_pipeline.hopsworks.connection_manager import HopsworksConnectionManager
from src.feature_pipeline.hopsworks.feature_group_repo import FeatureGroupRepository
from src.utils import get_logger

log = get_logger(__name__)


class FeatureViewRepository:
    """
    Manages all Feature View operations.

    """

    def __init__(
        self,
        name: str = FEATURE_VIEW_NAME,
        version: int = FEATURE_VIEW_VERSION,
        connection: HopsworksConnectionManager = None,
        feature_group_repo: FeatureGroupRepository = None,
    ):
        self.name = name
        self.version = version
        self.connection = connection or HopsworksConnectionManager()
        self.feature_group_repo = (
            feature_group_repo or
            FeatureGroupRepository(connection=self.connection)  # share same connection
        )
        self._fv = None

    def get_or_create(self):
        """
        Get or create the Feature View.
        Caches the result — no repeated API calls.
        """
        if self._fv is not None:
            return self._fv

        log.info(f"Getting or creating feature view: {self.name} v{self.version}")
        fg = self.feature_group_repo.get_or_create()

        # Select all columns except identity columns
        # (asteroid_id, name are not useful features for the model)
        query = fg.select_except([
            "asteroid_id",
            "name",
            "close_approach_date",
        ])

        self._fv = self.connection.feature_store.get_or_create_feature_view(
            name=self.name,
            version=self.version,
            query=query,
            labels=["is_potentially_hazardous"],
        )
        log.info(f"Feature view ready: {self.name}")
        return self._fv

    def get_training_data(
        self,
        test_size: float = 0.2,
        training_dataset_version: int = 1,
    ) -> tuple:
        """
        Get train/test split for model training.

        Args:
            test_size: fraction for test set (default 0.2 = 20%)
            training_dataset_version: version to create in Hopsworks
        Returns:
            (X_train, X_test, y_train, y_test) tuple
        """
        fv = self.get_or_create()
        log.info(f"Getting training data (test_size={test_size})...")

        X_train, X_test, y_train, y_test = fv.train_test_split(
            test_size=test_size,
            description=f"Training dataset created {datetime.now().date()}",
        )
        log.info(
            f"Training set: {len(X_train)} rows | "
            f"Test set: {len(X_test)} rows"
        )
        return X_train, X_test, y_train, y_test


    def get_batch_data(
        self,
        start_time: str = None,
        end_time: str = None,
    ) -> pd.DataFrame:
        """
        Get batch of features for inference.

        Args:
            start_time: start of window "YYYY-MM-DD" (optional)
            end_time:   end of window "YYYY-MM-DD" (optional)
        Returns:
            DataFrame with features for prediction
        """
        fv = self.get_or_create()
        log.info(f"Getting batch data ({start_time} → {end_time})...")

        df = fv.get_batch_data(
            start_time=start_time,
            end_time=end_time,
        )
        log.info(f"Batch data: {len(df)} rows")
        return df