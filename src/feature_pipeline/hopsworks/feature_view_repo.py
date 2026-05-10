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

    def get_training_data(self, test_size: float = 0.2) -> tuple:
        fv = self.get_or_create()
        log.info("Creating new training dataset snapshot...")
        return fv.train_test_split(
            test_size=test_size,
            description=f"Training dataset created {datetime.now().date()}",
        )


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