import pandas as pd

from src.common.hopsworks.connection_manager import HopsworksConnectionManager
from src.common.hopsworks.feature_group_repo import FeatureGroupRepository
from src.utils import get_logger

log = get_logger(__name__)


class FeatureViewRepository:
    """Generic wrapper for any Hopsworks Feature View."""

    def __init__(
        self,
        name: str,
        version: int,
        feature_group_repo: FeatureGroupRepository,
        labels: list[str] | None = None,
        connection: HopsworksConnectionManager | None = None,
    ):
        self.name = name
        self.version = version
        self.feature_group_repo = feature_group_repo
        self.labels = labels or []
        self.connection = connection or HopsworksConnectionManager()
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
            labels=self.labels,
        )
        log.info(f"Feature view ready: {self.name}")
        return self._fv

    def get_batch_data(self, start_time: str = None, end_time: str = None) -> pd.DataFrame:
        fv = self.get_or_create()
        log.info(f"Getting batch data ({start_time} → {end_time})...")
        df = fv.get_batch_data(start_time=start_time, end_time=end_time)
        log.info(f"Batch data: {len(df)} rows")
        return df