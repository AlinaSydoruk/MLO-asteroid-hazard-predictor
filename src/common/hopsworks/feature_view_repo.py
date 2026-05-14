import pandas as pd
from datetime import datetime

from src.config import (
    FEATURE_VIEW_NAME,
    FEATURE_VIEW_VERSION,
)
from src.config import VALIDATION_SIZE, TEST_SIZE
from src.common.hopsworks.connection_manager import HopsworksConnectionManager
from src.common.hopsworks.feature_group_repo import FeatureGroupRepository
from src.common.feature_schema import get_identity_columns
from src.utils import get_logger
from src.feature_pipeline.main_features_repository import create_features_repository
log = get_logger(__name__)


class FeatureViewRepository:
    """ Manages all Feature View operations. """

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
        self.feature_group_repo = feature_group_repo or create_features_repository(connection=connection)
        self._fv = None

    def get_or_create(self):
        """ Get or create the Feature View. """
        if self._fv is not None:
            return self._fv

        log.info(f"Getting or creating feature view: {self.name} v{self.version}")
        fg = self.feature_group_repo.get_or_create()


        self._fv = self.connection.feature_store.get_or_create_feature_view(
            name=self.name,
            version=self.version,
            query=fg.select_all(), # select all columns
            labels=["is_potentially_hazardous"],
        )
        log.info(f"Feature view ready: {self.name}")
        return self._fv

    def get_training_data( self, validation_size: float = VALIDATION_SIZE, test_size: float = TEST_SIZE,) -> tuple:
        """
        Returns 6 splits: X_train, X_val, X_test, y_train, y_val, y_test
        Train = 60%, Validation = 20%, Test = 20%
        """
        fv = self.get_or_create()
        log.info("Creating new training dataset snapshot (train/val/test)...")
        return fv.train_validation_test_split(
            validation_size=validation_size,
            test_size=test_size,
            description=f"Training dataset created {datetime.now().date()}",
        )


    def get_batch_data(
        self,
        start_time: str = None,
        end_time: str = None,
    ) -> pd.DataFrame:
        """ Get batch of features for inference."""

        fv = self.get_or_create()
        log.info(f"Getting batch data ({start_time} → {end_time})...")

        df = fv.get_batch_data(
            start_time=start_time,
            end_time=end_time,
        )
        log.info(f"Batch data: {len(df)} rows | columns: {list(df.columns)}")
        log.info(f"Batch data: {len(df)} rows")
        return df


    def get_full_dataset(self) -> pd.DataFrame:
        """Return all data from the feature view for training."""
        fv = self.get_or_create()
        log.info("Loading full dataset for training...")
        df = fv.get_batch_data()
        log.info(f"Loaded {len(df)} rows")
        return df