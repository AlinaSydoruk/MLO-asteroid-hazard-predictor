import pandas as pd

from src.config import (
    FEATURE_VIEW_NAME,
    FEATURE_VIEW_VERSION,
    FEATURE_VIEW_DEDUP_NAME,
    FEATURE_VIEW_DEDUP_VERSION,
)
from src.common.hopsworks.connection_manager import HopsworksConnectionManager
from src.common.hopsworks.feature_group_repo import FeatureGroupRepository
from src.feature_pipeline.main_features_repository import (
    AsteroidFeaturesRepository,
    AsteroidFeaturesDedupRepository,
)
from src.utils import get_logger

log = get_logger(__name__)


class FeatureViewRepository:
    """
    Feature view bound to the raw asteroid_features FG.
    Used by the inference pipeline (label column is excluded by Hopsworks).
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
        self.feature_group_repo = feature_group_repo or AsteroidFeaturesRepository(connection=connection)
        self._fv = None

    def get_or_create(self):
        if self._fv is not None:
            return self._fv

        log.info(f"Getting or creating feature view: {self.name} v{self.version}")
        fg = self.feature_group_repo.get_or_create()
        self._fv = self.connection.feature_store.get_or_create_feature_view(
            name=self.name,
            version=self.version,
            query=fg.select_all(),
            labels=["is_potentially_hazardous"],
        )
        log.info(f"Feature view ready: {self.name}")
        return self._fv

    def get_batch_data(
            self,
            start_time: str = None,
            end_time: str = None,
    ) -> pd.DataFrame:
        """Get a batch of features for inference (label is excluded)."""
        fv = self.get_or_create()
        log.info(f"Getting batch data ({start_time} → {end_time})...")
        df = fv.get_batch_data(start_time=start_time, end_time=end_time)
        log.info(f"Batch data: {len(df)} rows")
        return df


class DedupFeatureViewRepository(FeatureViewRepository):
    """
    Feature view bound to the asteroid_features_dedup FG.
    Used by the training pipeline. Exposes a training split that filters by
    cutoff and randomly splits via Hopsworks `train_validation_test_split`.
    """

    def __init__(
            self,
            connection: HopsworksConnectionManager = None,
            feature_group_repo: FeatureGroupRepository = None,
    ):
        super().__init__(
            name=FEATURE_VIEW_DEDUP_NAME,
            version=FEATURE_VIEW_DEDUP_VERSION,
            connection=connection,
            feature_group_repo=feature_group_repo or AsteroidFeaturesDedupRepository(connection=connection),
        )

    def get_training_splits(
            self,
            training_cutoff: pd.Timestamp,
            validation_size: float,
            test_size: float,
    ) -> tuple:
        """
        Returns: X_train, X_val, X_test, y_train, y_val, y_test

        Filters out asteroids first observed after `training_cutoff` (reserved
        for inference), then randomly splits the remainder into train/val/test.
        Filter is pushed down to Hopsworks via `extra_filter`.
        """
        fv = self.get_or_create()
        fg = self.feature_group_repo.get_or_create()

        cutoff_str = training_cutoff.strftime("%Y-%m-%d")
        log.info(f"Training split @ cutoff={cutoff_str} (val={validation_size}, test={test_size})")

        return fv.train_validation_test_split(
            validation_size=validation_size,
            test_size=test_size,
            extra_filter=fg.first_observation_date <= cutoff_str,
            description=f"training @ cutoff {cutoff_str}",
        )