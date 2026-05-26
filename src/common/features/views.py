from src.common.hopsworks.feature_view_repo import FeatureViewRepository
from src.common.hopsworks.connection_manager import HopsworksConnectionManager
from src.common.features.repositories import (
    AsteroidFeaturesRepository,
    AsteroidFeaturesDedupRepository,
)
from src.utils import get_logger
from src.common.features.schema import get_label_column
import pandas as pd
from src.config import (
    FEATURE_VIEW_NAME, FEATURE_VIEW_VERSION,
    FEATURE_VIEW_DEDUP_NAME, FEATURE_VIEW_DEDUP_VERSION,
    FEATURE_VIEW_MONITORING_NAME
)
log = get_logger(__name__)



class AsteroidFeatureView(FeatureViewRepository):
    """Inference FV"""
    def __init__(self, connection: HopsworksConnectionManager = None):
        super().__init__(
            name=FEATURE_VIEW_NAME,
            version=FEATURE_VIEW_VERSION,
            feature_group_repo=AsteroidFeaturesRepository(connection=connection),
            labels=[get_label_column()],
            connection=connection,
        )




class AsteroidDedupFeatureView(FeatureViewRepository):
    """Training FV"""
    def __init__(self, connection: HopsworksConnectionManager = None):
        super().__init__(
            name=FEATURE_VIEW_DEDUP_NAME,
            version=FEATURE_VIEW_DEDUP_VERSION,
            feature_group_repo=AsteroidFeaturesDedupRepository(connection=connection),
            labels=[get_label_column()],
            connection=connection,
        )
    def get_training_splits(
            self,
            training_cutoff: pd.Timestamp,
            validation_size: float,
            test_size: float,
    ) -> tuple:
        """
        Returns X_train, X_val, X_test, y_train, y_val, y_test.
        Filters out asteroids first observed after training_cutoff
        (those are reserved for inference), then splits the rest.
        """
        fv = self.get_or_create()
        fg = self.feature_group_repo.get_or_create()

        cutoff_str = training_cutoff.strftime("%Y-%m-%d")
        log.info(f"Training split @ cutoff={cutoff_str} "
                 f"(val={validation_size}, test={test_size})")

        return fv.train_validation_test_split(
            validation_size=validation_size,
            test_size=test_size,
            extra_filter=fg.first_observation_date <= cutoff_str,
            description=f"training @ cutoff {cutoff_str}",
        )


class AsteroidMonitoringFeatureView(FeatureViewRepository):
    """Monitoring FV"""
    def __init__(self, connection: HopsworksConnectionManager = None):
        super().__init__(
            name=FEATURE_VIEW_MONITORING_NAME,
            version=FEATURE_VIEW_VERSION,
            feature_group_repo=AsteroidFeaturesRepository(connection=connection),
            labels=[],
            connection=connection,
        )