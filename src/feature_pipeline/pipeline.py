"""Feature Pipeline orchestrator."""
import argparse
import pandas as pd
from src.config import FEATURE_VIEW_NAME, FEATURE_VIEW_VERSION
from src.feature_pipeline.nasa_client import NASAClient
from src.feature_pipeline.asteroid_fetcher import AsteroidFetcher
from src.feature_pipeline.asteroid_parser import AsteroidParser
from src.feature_pipeline.feature_computer import compute_features
from src.common.hopsworks.connection_manager import HopsworksConnectionManager
from src.common.features.repositories import (
    AsteroidFeaturesRepository,
    AsteroidFeaturesDedupRepository,
)
from src.common.hopsworks.feature_view_repo import FeatureViewRepository
from src.utils import get_logger

log = get_logger(__name__)


class FeaturePipeline:
    """Orchestrates the entire feature pipeline."""

    def __init__(self):
        connection = HopsworksConnectionManager()
        self.fetcher = AsteroidFetcher(client=NASAClient())
        self.parser = AsteroidParser()
        self.feature_group = AsteroidFeaturesRepository(connection=connection)
        self.feature_group_dedup = AsteroidFeaturesDedupRepository(
            connection=connection,
        )
        self.feature_view = FeatureViewRepository(
            name=FEATURE_VIEW_NAME,
            version=FEATURE_VIEW_VERSION,
            connection=connection,
            feature_group_repo=self.feature_group,
        )

    def run_incremental(self) -> pd.DataFrame:
        """Fetch today's asteroids, engineer features, store."""
        log.info("Running INCREMENTAL feature pipeline...")

        feed_response = self.fetcher.get_today_feed()

        asteroid_count = sum(len(v) for v in feed_response["near_earth_objects"].values())
        log.info(f"NASA API returned {asteroid_count} asteroids")

        asteroid_ids = self._extract_asteroid_ids(feed_response)
        orbital_data_by_id = self.fetcher.get_orbital_data(asteroid_ids)
        enriched_feed = self._attach_orbital_data(feed_response, orbital_data_by_id)

        df = self.parser.parse_feed(enriched_feed)
        log.info(f"Parsed {len(df)} records | "
                 f"Hazardous: {df['is_potentially_hazardous'].sum()}")

        df = compute_features(df)

        success = self.feature_group.insert(df)
        if not success:
            log.error("Pipeline failed, insert unsuccessful.")
            raise RuntimeError("Feature group insert failed")

        self.feature_group_dedup.dedup_and_insert(df)

        log.info("Incremental run complete.")

        return df

    def run_weekly(self, start_date: str = None) -> pd.DataFrame:
        """Fetch last 7 days of asteroids."""
        log.info(f"Running WEEKLY feature pipeline (start={start_date})...")

        feed_response = self.fetcher.get_week_feed(start_date=start_date)

        asteroid_count = sum(len(v) for v in feed_response["near_earth_objects"].values())
        log.info(f"NASA API returned {asteroid_count} asteroids")

        asteroid_ids = self._extract_asteroid_ids(feed_response)
        orbital_data_by_id = self.fetcher.get_orbital_data(asteroid_ids)
        enriched_feed = self._attach_orbital_data(feed_response, orbital_data_by_id)

        df = self.parser.parse_feed(enriched_feed)
        df = compute_features(df)
        self.feature_group.insert(df)
        self.feature_group_dedup.dedup_and_insert(df)
        log.info("Weekly run complete.")
        return df

    def run_backfill(self, max_pages: int = None, start_page: int = 0) -> pd.DataFrame:
        """Fetch ALL historical asteroids and run once at start."""
        log.info(f"Running BACKFILL (max_pages={max_pages})...")
        pages = self.fetcher.get_all_pages(max_pages=max_pages, start_page=start_page)
        df = self.parser.parse_all_browse_pages(pages)
        log.info(f"Parsed {len(df)} unique asteroids")
        df = compute_features(df)
        self.feature_group.insert(df)
        self.feature_group_dedup.dedup_and_insert(df)
        log.info("Backfill complete.")
        return df

    def create_feature_view(self) -> None:
        """Create the Feature View."""
        log.info("Creating feature view...")
        self.feature_view.get_or_create()
        log.info("Feature view created.")


    def _attach_orbital_data(
            self,
            raw_feed: dict,
            orbital_data: dict[str, dict],
    ) -> dict:
        for date_str, asteroids in raw_feed["near_earth_objects"].items():
            for ast in asteroids:
                asteroid_id = ast["id"]
                ast["orbital_data"] = orbital_data.get(asteroid_id, {})
        return raw_feed

    def _extract_asteroid_ids(self, raw_feed: dict) -> list[str]:
        asteroid_ids = []
        for asteroids in raw_feed["near_earth_objects"].values():
            for ast in asteroids:
                asteroid_ids.append(ast["id"])
        return asteroid_ids

def main():
    parser = argparse.ArgumentParser(description="Asteroid Feature Pipeline")
    parser.add_argument(
        "--mode",
        choices=["incremental", "weekly", "backfill", "create-feature-view"],
        default="incremental",
    )
    parser.add_argument("--start-page", type=int, default=0)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--start-date", type=str, default=None)
    args = parser.parse_args()

    pipeline = FeaturePipeline()

    if args.mode == "incremental":
        pipeline.run_incremental()
    elif args.mode == "weekly":
        pipeline.run_weekly(start_date=args.start_date)
    elif args.mode == "backfill":
        pipeline.run_backfill(max_pages=args.max_pages, start_page=args.start_page)
    elif args.mode == "create-feature-view":
        pipeline.create_feature_view()


if __name__ == "__main__":
    main()