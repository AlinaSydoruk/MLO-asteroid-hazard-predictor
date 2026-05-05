"""
Feature Pipeline — main entry point.
Composes all classes to fetch, parse, engineer, and store features.

Usage:
    python -m src.feature_pipeline --mode incremental
    python -m src.feature_pipeline --mode weekly
    python -m src.feature_pipeline --mode backfill --max-pages 5
"""
import argparse
import pandas as pd

from src.feature_pipeline.asteroid_fetcher import AsteroidFetcher
from src.feature_pipeline.asteroid_parser import AsteroidParser
from src.feature_pipeline.feature_computer import compute_features
from src.feature_pipeline.feature_store import FeatureStore
from src.feature_pipeline.nasa_client import NASAClient
from src.utils import get_logger

log = get_logger(__name__)


class FeaturePipeline:
    """Orchestrates the entire feature pipeline."""

    def __init__(self):
        self.fetcher = AsteroidFetcher(client=NASAClient())
        self.parser = AsteroidParser()
        self.store = FeatureStore()

    def run_incremental(self) -> pd.DataFrame:
        log.info("Running INCREMENTAL feature pipeline...")

        raw = self.fetcher.get_today_feed()
        total_raw = sum(len(v) for v in raw["near_earth_objects"].values())
        log.info(f"NASA API returned {total_raw} raw asteroids")

        df = self.parser.parse_feed(raw)
        log.info(f"Parsed {len(df)} asteroid records")
        log.info(f"Hazardous: {df['is_potentially_hazardous'].sum()} | "
                 f"Safe: {(df['is_potentially_hazardous'] == 0).sum()}")
        log.info(f"Date range: {df['close_approach_date'].min()} → "
                 f"{df['close_approach_date'].max()}")

        df = compute_features(df)
        log.info(f"Engineered features for {len(df)} rows")
        log.info(f"Columns: {list(df.columns)}")

        self.store.insert(df)
        log.info("Incremental run complete.")
        return df

    def run_weekly(self, start_date: str = None) -> pd.DataFrame:
        """Fetch last 7 days of asteroids."""
        log.info(f"Running WEEKLY feature pipeline (start={start_date})...")
        raw = self.fetcher.get_week_feed(start_date=start_date)
        df = self.parser.parse_feed(raw)
        df = compute_features(df)
        self.store.insert(df)
        log.info("Weekly run complete.")
        return df

    def run_backfill(self, max_pages: int = None) -> pd.DataFrame:
        """Fetch ALL historical asteroids — run once at start."""
        log.info(f"Running BACKFILL (max_pages={max_pages})...")
        pages = self.fetcher.get_all_pages(max_pages=max_pages)
        df = self.parser.parse_all_browse_pages(pages)
        log.info(f"Parsed {len(df)} unique asteroids")

        df = compute_features(df)
        self.store.insert(df)
        log.info("Backfill complete.")
        return df


def main():
    parser = argparse.ArgumentParser(description="Asteroid Feature Pipeline")
    parser.add_argument(
        "--mode",
        choices=["incremental", "weekly", "backfill"],
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


if __name__ == "__main__":
    main()