import pandas as pd

from src.config import (
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
)
from src.common.hopsworks.connection_manager import HopsworksConnectionManager
from src.utils import get_logger

log = get_logger(__name__)


class FeatureGroupRepository:
    """
    Manages all Feature Group operations.
    read/write the asteroid_features Feature Group.
    """

    def __init__(
        self,
        name: str = FEATURE_GROUP_NAME,
        version: int = FEATURE_GROUP_VERSION,
        connection: HopsworksConnectionManager = None,
    ):
        self.name = name
        self.version = version
        self.connection = connection or HopsworksConnectionManager()
        self._fg = None

    def get_or_create(self):
        if self._fg is not None:
            return self._fg

        log.info(f"Getting or creating feature group: {self.name} v{self.version}")
        self._fg = self.connection.feature_store.get_or_create_feature_group(
            name=self.name,
            version=self.version,
            primary_key=["asteroid_id", "close_approach_date"],
            event_time="close_approach_date",
            description="NEO asteroid features and PHA label",
        )
        log.info(f"Feature group ready: {self.name}")
        return self._fg

    def insert(self, df: pd.DataFrame) -> bool:
        if df.empty:
            log.warning("Empty DataFrame, nothing to insert.")
            return False

        fg = self.get_or_create()
        log.info(f"Inserting {len(df)} rows into {self.name}...")

        try:
            job, execution = fg.insert(df, write_options={"wait_for_job": True})

            if execution is not None and execution.final_status == "FAILED":
                log.error("Materialization job FAILED. Data not written to offline store.")
                raise RuntimeError(f"Hopsworks materialization failed: {execution.final_status}")

            log.info("Insert complete.")
            return True

        except Exception as e:
            log.error(f"Insert FAILED: {e}")
            raise

    def read(self) -> pd.DataFrame:
        """Read all data from the Feature Group."""
        log.info(f"Reading from feature group: {self.name}")
        fg = self.get_or_create()
        return fg.read()

    def read_by_date(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Read features for a specific date range.

        Args:
            start_date: "YYYY-MM-DD"
            end_date:   "YYYY-MM-DD"
        Returns:
            filtered DataFrame
        """
        df = self.read()
        mask = (
            (df["close_approach_date"] >= start_date) &
            (df["close_approach_date"] <= end_date)
        )
        return df[mask]
