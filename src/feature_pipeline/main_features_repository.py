from src.config import (
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
)
from src.common.hopsworks.connection_manager import HopsworksConnectionManager
from src.common.hopsworks.feature_group_repo import FeatureGroupRepository

def create_features_repository(
    connection: HopsworksConnectionManager = None,
) -> FeatureGroupRepository:
    """
    Repository for the asteroid_features Feature Group.
    Contains raw + engineered features used by training and inference.
    """
    return FeatureGroupRepository(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        primary_key=["asteroid_id", "close_approach_date"],
        event_time="close_approach_date",
        description="NEO asteroid features and PHA label",
        connection=connection,
    )