from src.config import (
    PREDICTIONS_GROUP_NAME,
    PREDICTIONS_GROUP_VERSION,
)
from src.common.hopsworks.connection_manager import HopsworksConnectionManager
from src.common.hopsworks.feature_group_repo import FeatureGroupRepository


def create_predictions_repository(
    connection: HopsworksConnectionManager = None,
) -> FeatureGroupRepository:
    """
    Repository for the asteroid_predictions Feature Group.
    Contains daily hazard predictions from the inference pipeline.
    """
    return FeatureGroupRepository(
        name=PREDICTIONS_GROUP_NAME,
        version=PREDICTIONS_GROUP_VERSION,
        primary_key=["asteroid_id", "close_approach_date"],
        event_time="close_approach_date",
        description="Daily hazard predictions for asteroids",
        connection=connection,
    )