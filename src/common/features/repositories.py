from src.config import (
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
    FEATURE_GROUP_DEDUP_NAME,
    FEATURE_GROUP_DEDUP_VERSION,
)
from src.common.hopsworks.connection_manager import HopsworksConnectionManager
from src.common.hopsworks.feature_group_repo import FeatureGroupRepository
from src.utils import get_logger
import pandas as pd
from src.config import PREDICTIONS_GROUP_NAME, PREDICTIONS_GROUP_VERSION
log = get_logger(__name__)


class AsteroidFeaturesRepository(FeatureGroupRepository):
    """Raw feature group: one row per (asteroid_id, close_approach_date)."""

    def __init__(self, connection: HopsworksConnectionManager = None):
        super().__init__(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION,
            primary_key=["asteroid_id", "close_approach_date"],
            event_time="close_approach_date",
            description="NEO asteroid features and PHA label",
            connection=connection,
        )


class AsteroidFeaturesDedupRepository(FeatureGroupRepository):
    """
    Deduplicated feature group: one row per asteroid_id.

    Designed for training, where each asteroid must appear in exactly one split (preventing leakage
    from the same asteroid showing up in both train and test through different close approaches).
    """

    def __init__(
        self,
        connection: HopsworksConnectionManager = None,
    ):
        super().__init__(
            name=FEATURE_GROUP_DEDUP_NAME,
            version=FEATURE_GROUP_DEDUP_VERSION,
            primary_key=["asteroid_id"],
            event_time="close_approach_date",
            description="One row per asteroid (most recent close approach) — training source",
            connection=connection,
        )
    def dedup_and_insert(self, df: pd.DataFrame) -> None:

        if df.empty:
            log.warning("Empty DataFrame, nothing to insert.")
            return

        dedup_df = (
            df.sort_values(["asteroid_id", "close_approach_date"]).drop_duplicates(subset=["asteroid_id"], keep="last")
        )
        log.info(
            f"Dedup: {len(df)} raw rows → {len(dedup_df)} unique asteroids"
        )
        self.insert(dedup_df)



class AsteroidPredictionsRepository(FeatureGroupRepository):
    """Feature group storing daily hazard predictions from the inference pipeline."""

    def __init__(self, connection: HopsworksConnectionManager = None):
        super().__init__(
            name=PREDICTIONS_GROUP_NAME,
            version=PREDICTIONS_GROUP_VERSION,
            primary_key=["asteroid_id", "close_approach_date"],
            event_time="close_approach_date",
            description="Daily hazard predictions for asteroids",
            connection=connection,
        )


