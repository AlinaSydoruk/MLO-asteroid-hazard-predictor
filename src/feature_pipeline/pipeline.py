"""Feature Pipeline orchestrator."""
import argparse
import pandas as pd

from src.feature_pipeline.nasa_client import NASAClient
from src.feature_pipeline.asteroid_fetcher import AsteroidFetcher
from src.feature_pipeline.asteroid_parser import AsteroidParser
from src.feature_pipeline.feature_computer import compute_features
from src.feature_pipeline.hopsworks.connection_manager import HopsworksConnectionManager
from src.feature_pipeline.hopsworks.feature_group_repo import FeatureGroupRepository
from src.feature_pipeline.hopsworks.feature_view_repo import FeatureViewRepository
from src.utils import get_logger

log = get_logger(__name__)


class FeaturePipeline:
    """Orchestrates the entire feature pipeline."""

    def __init__(self):
        connection = HopsworksConnectionManager()
        self.fetcher = AsteroidFetcher(client=NASAClient())
        self.parser = AsteroidParser()
        self.feature_group = FeatureGroupRepository(connection=connection)
        self.feature_view = FeatureViewRepository(
            connection=connection,
            feature_group_repo=self.feature_group,   # share same fg instance
        )

    def run_incremental(self) -> pd.DataFrame:
        """Fetch today's asteroids, engineer features, store."""
        log.info("Running INCREMENTAL feature pipeline...")

        raw = self.fetcher.get_today_feed()
        total_raw = sum(len(v) for v in raw["near_earth_objects"].values())
        log.info(f"NASA API returned {total_raw} raw asteroids")

        df = self.parser.parse_feed(raw)
        log.info(f"Parsed {len(df)} records | "
                 f"Hazardous: {df['is_potentially_hazardous'].sum()}")

        df = compute_features(df)

        success = self.feature_group.insert(df)
        if not success:
            log.error("Pipeline failed, insert unsuccessful.")
            raise RuntimeError("Feature group insert failed")

        log.info(" Incremental run complete.")
        return df

    def run_weekly(self, start_date: str = None) -> pd.DataFrame:
        """Fetch last 7 days of asteroids."""
        log.info(f"Running WEEKLY feature pipeline (start={start_date})...")
        raw = self.fetcher.get_week_feed(start_date=start_date)
        df = self.parser.parse_feed(raw)
        df = compute_features(df)
        self.feature_group.insert(df)
        log.info("Weekly run complete.")
        return df

    def run_backfill(self, max_pages: int = None) -> pd.DataFrame:
        """Fetch ALL historical asteroids and run once at start."""
        log.info(f"Running BACKFILL (max_pages={max_pages})...")
        pages = self.fetcher.get_all_pages(max_pages=max_pages)
        df = self.parser.parse_all_browse_pages(pages)
        log.info(f"Parsed {len(df)} unique asteroids")
        df = compute_features(df)
        self.feature_group.insert(df)
        log.info(" Backfill complete.")
        return df

    def create_feature_view(self) -> None:
        """Create the Feature View for training/inference pipelines."""
        log.info("Creating feature view...")
        self.feature_view.get_or_create()
        log.info("Feature view created.")


def main():
    parser = argparse.ArgumentParser(description="Asteroid Feature Pipeline")
    parser.add_argument(
        "--mode",
        choices=["incremental", "weekly", "backfill", "create-feature-view"],
        default="incremental",
    )
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--start-date", type=str, default=None)
    args = parser.parse_args()

    pipeline = FeaturePipeline()

    if args.mode == "incremental":
        pipeline.run_incremental()
    elif args.mode == "weekly":
        pipeline.run_weekly(start_date=args.start_date)
    elif args.mode == "backfill":
        pipeline.run_backfill(max_pages=args.max_pages)
    elif args.mode == "create-feature-view":
        pipeline.create_feature_view()


if __name__ == "__main__":
    main()