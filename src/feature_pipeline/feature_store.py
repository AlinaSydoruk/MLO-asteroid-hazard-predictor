"""
Hopsworks Feature Store wrapper.
Encapsulates all interaction with Hopsworks Feature Store.
"""

import hopsworks
import pandas as pd

from src.config import (
    HOPSWORKS_API_KEY,
    HOPSWORKS_PROJECT_NAME,
    HOPSWORKS_HOST,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
    FEATURE_VIEW_NAME,
    FEATURE_VIEW_VERSION,
)


class FeatureStore:
    """Wraps Hopsworks Feature Store operations."""

    def __init__(self):
        self.project = None
        self.fs = None

    def connect(self) -> None:
        """Connect to Hopsworks. Reuses connection if already open."""
        if self.project is not None:
            return
        self.project = hopsworks.login(
            host=HOPSWORKS_HOST,
            api_key_value=HOPSWORKS_API_KEY,
            project=HOPSWORKS_PROJECT_NAME,
        )
        self.fs = self.project.get_feature_store()
        print(f"Connected to Hopsworks project: {self.project.name}")



    def get_or_create_feature_group(
        self,
        name: str = FEATURE_GROUP_NAME,
        version: int = FEATURE_GROUP_VERSION,
        description: str = "NEO asteroid features and PHA label",
    ):
        """Get existing feature group or create a new one."""
        self.connect()
        return self.fs.get_or_create_feature_group(
            name=name,
            version=version,
            primary_key=["asteroid_id", "close_approach_date"],
            event_time="close_approach_date",
            description=description,
        )

    def insert(self, df: pd.DataFrame) -> None:
        """Insert features DataFrame into the asteroid feature group."""
        if df.empty:
            print("Empty DataFrame — nothing to insert.")
            return

        fg = self.get_or_create_feature_group()
        print(f"Inserting {len(df)} rows into {FEATURE_GROUP_NAME}...")
        fg.insert(df, write_options={"wait_for_job": True})
        print("Insert complete.")



    def get_or_create_feature_view(
        self,
        name: str = FEATURE_VIEW_NAME,
        version: int = FEATURE_VIEW_VERSION,
    ):
        """Get or create the Feature View used by training/inference."""
        self.connect()
        fg = self.get_or_create_feature_group()
        query = fg.select_except(["asteroid_id", "name", "close_approach_date"])
        return self.fs.get_or_create_feature_view(
            name=name,
            version=version,
            query=query,
            labels=["is_potentially_hazardous"],
        )