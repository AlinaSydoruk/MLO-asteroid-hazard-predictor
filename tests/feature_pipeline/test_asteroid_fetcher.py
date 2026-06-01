"""Unit tests for src.feature_pipeline.asteroid_fetcher."""
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from src.feature_pipeline.asteroid_fetcher import AsteroidFetcher
from src.config import (
    NASA_FEED_ENDPOINT,
    NASA_BROWSE_ENDPOINT,
    NASA_BROWSE_PAGE_SIZE,
    NASA_MAX_FEED_DAYS,
)


@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def fetcher(mock_client):
    # Patch sleep globally for this fetcher's tests — no real waits
    with patch("src.feature_pipeline.asteroid_fetcher.time.sleep"):
        yield AsteroidFetcher(client=mock_client)


class TestGetFeed:
    def test_get_feed_passes_date_range(self, fetcher, mock_client):
        fetcher.get_feed("2024-01-01", "2024-01-07")
        mock_client.get.assert_called_once_with(
            NASA_FEED_ENDPOINT,
            params={"start_date": "2024-01-01", "end_date": "2024-01-07"},
        )

    def test_get_today_feed_uses_today(self, fetcher, mock_client):
        with patch("src.feature_pipeline.asteroid_fetcher.date") as mock_date:
            mock_date.today.return_value = date(2024, 6, 15)
            mock_date.fromisoformat = date.fromisoformat  # keep real
            fetcher.get_today_feed()
        mock_client.get.assert_called_once_with(
            NASA_FEED_ENDPOINT,
            params={"start_date": "2024-06-15", "end_date": "2024-06-15"},
        )

    def test_get_week_feed_with_explicit_start_date(self, fetcher, mock_client):
        fetcher.get_week_feed(start_date="2024-01-01")
        expected_end = "2024-01-07"  # NASA_MAX_FEED_DAYS=7 → window length 7
        assert NASA_MAX_FEED_DAYS == 7  # guard against config drift
        mock_client.get.assert_called_once_with(
            NASA_FEED_ENDPOINT,
            params={"start_date": "2024-01-01", "end_date": expected_end},
        )

    def test_get_week_feed_defaults_to_last_seven_days(self, fetcher, mock_client):
        with patch("src.feature_pipeline.asteroid_fetcher.date") as mock_date:
            mock_date.today.return_value = date(2024, 6, 15)
            mock_date.fromisoformat = date.fromisoformat
            fetcher.get_week_feed()
        called_params = mock_client.get.call_args.kwargs["params"]
        # Should start 7 days before today and span 7 days
        assert called_params["start_date"] == "2024-06-08"
        assert called_params["end_date"] == "2024-06-14"


class TestGetBrowse:
    def test_get_browse_uses_page_and_size(self, fetcher, mock_client):
        fetcher.get_browse(page=3)
        mock_client.get.assert_called_once_with(
            NASA_BROWSE_ENDPOINT,
            params={"page": 3, "size": NASA_BROWSE_PAGE_SIZE},
        )

    def test_get_all_pages_stops_at_max_pages(self, fetcher, mock_client):
        mock_client.get.return_value = {"page": {"total_pages": 100}}
        pages = fetcher.get_all_pages(max_pages=3)
        assert len(pages) == 3
        assert mock_client.get.call_count == 3

    def test_get_all_pages_stops_at_last_page(self, fetcher, mock_client):
        mock_client.get.return_value = {"page": {"total_pages": 2}}
        pages = fetcher.get_all_pages()
        assert len(pages) == 2

    def test_get_all_pages_respects_start_page(self, fetcher, mock_client):
        mock_client.get.return_value = {"page": {"total_pages": 100}}
        fetcher.get_all_pages(max_pages=2, start_page=50)
        called_pages = [c.kwargs["params"]["page"] for c in mock_client.get.call_args_list]
        assert called_pages == [50, 51]


class TestGetOrbitalData:
    def test_fetches_orbital_data_per_id(self, fetcher, mock_client):
        mock_client.get.side_effect = [
            {"orbital_data": {"eccentricity": "0.1"}},
            {"orbital_data": {"eccentricity": "0.2"}},
        ]
        result = fetcher.get_orbital_data(["111", "222"])
        assert result == {
            "111": {"eccentricity": "0.1"},
            "222": {"eccentricity": "0.2"},
        }

    def test_missing_orbital_data_returns_empty_dict(self, fetcher, mock_client):
        mock_client.get.return_value = {}  # no 'orbital_data' key
        result = fetcher.get_orbital_data(["111"])
        assert result == {"111": {}}

    def test_failed_lookup_is_swallowed_and_returns_empty(self, fetcher, mock_client):
        mock_client.get.side_effect = [
            RuntimeError("404"),
            {"orbital_data": {"eccentricity": "0.2"}},
        ]
        result = fetcher.get_orbital_data(["bad", "222"])
        assert result["bad"] == {}
        assert result["222"] == {"eccentricity": "0.2"}

    def test_empty_id_list_returns_empty_dict(self, fetcher, mock_client):
        assert fetcher.get_orbital_data([]) == {}
        mock_client.get.assert_not_called()