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
        self.feature_group_repo = (
            feature_group_repo or
            FeatureGroupRepository(connection=self.connection)  # share same connection
        )
        self._fv = None

    def get_or_create(self):
        """ Get or create the Feature View. """
        if self._fv is not None:
            return self._fv

        log.info(f"Getting or creating feature view: {self.name} v{self.version}")
        fg = self.feature_group_repo.get_or_create()

        # Select all columns except identity columns
        # (asteroid_id, name are not useful features for the model)
        query = fg.select_except(get_identity_columns())

        self._fv = self.connection.feature_store.get_or_create_feature_view(
            name=self.name,
            version=self.version,
            query=query,
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