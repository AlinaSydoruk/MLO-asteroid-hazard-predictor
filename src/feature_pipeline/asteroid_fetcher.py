import time
from datetime import date, timedelta

from src.feature_pipeline.nasa_client import NASAClient
from src.config import (
    NASA_FEED_ENDPOINT,
    NASA_BROWSE_ENDPOINT,
    NASA_MAX_FEED_DAYS,
    NASA_BROWSE_PAGE_SIZE,
    NASA_POLITE_DELAY_SECONDS,
)
from src.utils import get_logger

log = get_logger(__name__)


class AsteroidFetcher:
    """Fetches asteroid data from NASA NeoWs API."""

    def __init__(self, client: NASAClient = None):
        self.client = client


    def get_feed(self, start_date: str, end_date: str) -> dict:
        """Fetch asteroids by close approach date (max 7-day window)."""
        return self.client.get(
            NASA_FEED_ENDPOINT,
            params={"start_date": start_date, "end_date": end_date},
        )

    def get_today_feed(self) -> dict:
        """Fetch today's asteroid close approaches."""
        today = date.today().isoformat()
        return self.get_feed(start_date=today, end_date=today)

    def get_week_feed(self, start_date: str = None) -> dict:
        """Fetch a 7-day window of asteroid close approaches."""
        if start_date is None:
            start_date = (
                date.today() - timedelta(days=NASA_MAX_FEED_DAYS)
            ).isoformat()
        end_date = (
            date.fromisoformat(start_date)
            + timedelta(days=NASA_MAX_FEED_DAYS - 1)
        ).isoformat()
        return self.get_feed(start_date=start_date, end_date=end_date)


    def get_browse(self, page: int = 0) -> dict:
        """Fetch one page of the full asteroid dataset."""
        return self.client.get(
            NASA_BROWSE_ENDPOINT,
            params={"page": page, "size": NASA_BROWSE_PAGE_SIZE},
        )

    def get_all_pages(self, max_pages: int = None) -> list[dict]:
        """Fetch ALL pages of /browse endpoint. Used for backfill."""
        pages = []
        page = 0

        while True:
            log.info(f"Fetching browse page {page}...")
            raw = self.get_browse(page=page)
            pages.append(raw)

            total_pages = raw["page"]["total_pages"]
            log.info(f"  Page {page + 1}/{total_pages}")

            if max_pages and page >= max_pages - 1:
                log.info(f"Stopped at max_pages={max_pages}")
                break
            if page >= total_pages - 1:
                log.info(f"All {total_pages} pages fetched.")
                break

            page += 1
            time.sleep(NASA_POLITE_DELAY_SECONDS)

        return pages


    def get_asteroid(self, asteroid_id: str) -> dict:
        """Look up one specific asteroid by NASA JPL ID."""
        return self.client.get(f"/neo/{asteroid_id}")