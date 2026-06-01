"""Unit tests for src.feature_pipeline.pipeline.FeaturePipeline."""
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# Where the pipeline module imports its collaborators from — we patch
# *these* names, not the original definition modules.
PIPELINE_MOD = "src.feature_pipeline.pipeline"


@pytest.fixture
def pipeline():
    """Construct a FeaturePipeline with every external dep mocked."""
    # Import the module up-front so patch() can resolve the attribute path
    import src.feature_pipeline.pipeline  # noqa: F401

    with patch(f"{PIPELINE_MOD}.HopsworksConnectionManager"), \
         patch(f"{PIPELINE_MOD}.NASAClient"), \
         patch(f"{PIPELINE_MOD}.AsteroidFetcher") as MockFetcher, \
         patch(f"{PIPELINE_MOD}.AsteroidParser") as MockParser, \
         patch(f"{PIPELINE_MOD}.AsteroidFeaturesRepository") as MockFG, \
         patch(f"{PIPELINE_MOD}.AsteroidFeaturesDedupRepository") as MockFGDedup, \
         patch(f"{PIPELINE_MOD}.FeatureViewRepository") as MockFV:

        from src.feature_pipeline.pipeline import FeaturePipeline
        p = FeaturePipeline()

        # Default behaviour: insert succeeds
        p.feature_group.insert.return_value = True
        yield p


@pytest.fixture
def feed_payload():
    """Minimal raw feed returned by fetcher.get_today_feed()."""
    return {
        "near_earth_objects": {
            "2024-01-01": [
                {"id": "111", "name": "A1"},
                {"id": "222", "name": "A2"},
            ],
            "2024-01-02": [
                {"id": "333", "name": "A3"},
            ],
        }
    }


@pytest.fixture
def parsed_df():
    """Stand-in for parser.parse_feed output."""
    return pd.DataFrame(
        {
            "asteroid_id": ["111", "222", "333"],
            "is_potentially_hazardous": [1, 0, 1],
            "close_approach_date": ["2024-01-01"] * 3,
        }
    )


class TestExtractAsteroidIds:
    def test_flattens_ids_across_dates(self, pipeline, feed_payload):
        ids = pipeline._extract_asteroid_ids(feed_payload)
        assert sorted(ids) == ["111", "222", "333"]

    def test_returns_empty_list_for_empty_feed(self, pipeline):
        ids = pipeline._extract_asteroid_ids({"near_earth_objects": {}})
        assert ids == []


class TestAttachOrbitalData:
    def test_attaches_orbital_data_keyed_by_id(self, pipeline, feed_payload):
        orbital = {"111": {"eccentricity": "0.1"}, "222": {"eccentricity": "0.2"}}
        enriched = pipeline._attach_orbital_data(feed_payload, orbital)

        a1 = enriched["near_earth_objects"]["2024-01-01"][0]
        a2 = enriched["near_earth_objects"]["2024-01-01"][1]
        assert a1["orbital_data"] == {"eccentricity": "0.1"}
        assert a2["orbital_data"] == {"eccentricity": "0.2"}

    def test_missing_orbital_yields_empty_dict(self, pipeline, feed_payload):
        # Asteroid "333" has no entry in the orbital map
        orbital = {"111": {"eccentricity": "0.1"}}
        enriched = pipeline._attach_orbital_data(feed_payload, orbital)
        a3 = enriched["near_earth_objects"]["2024-01-02"][0]
        assert a3["orbital_data"] == {}


class TestRunIncremental:
    def test_happy_path_calls_each_stage_in_order(
        self, pipeline, feed_payload, parsed_df
    ):
        pipeline.fetcher.get_today_feed.return_value = feed_payload
        pipeline.fetcher.get_orbital_data.return_value = {
            "111": {}, "222": {}, "333": {},
        }
        pipeline.parser.parse_feed.return_value = parsed_df

        with patch(f"{PIPELINE_MOD}.compute_features", return_value=parsed_df) as mock_compute:
            result = pipeline.run_incremental()

        pipeline.fetcher.get_today_feed.assert_called_once()
        pipeline.fetcher.get_orbital_data.assert_called_once()
        pipeline.parser.parse_feed.assert_called_once()
        mock_compute.assert_called_once()
        pipeline.feature_group.insert.assert_called_once()
        pipeline.feature_group_dedup.dedup_and_insert.assert_called_once()
        pd.testing.assert_frame_equal(result, parsed_df)

    def test_raises_when_insert_fails(self, pipeline, feed_payload, parsed_df):
        pipeline.fetcher.get_today_feed.return_value = feed_payload
        pipeline.fetcher.get_orbital_data.return_value = {}
        pipeline.parser.parse_feed.return_value = parsed_df
        pipeline.feature_group.insert.return_value = False

        with patch(f"{PIPELINE_MOD}.compute_features", return_value=parsed_df):
            with pytest.raises(RuntimeError, match="insert failed"):
                pipeline.run_incremental()

        # Dedup must NOT run after a failed insert
        pipeline.feature_group_dedup.dedup_and_insert.assert_not_called()

    def test_passes_extracted_ids_to_orbital_lookup(
        self, pipeline, feed_payload, parsed_df
    ):
        pipeline.fetcher.get_today_feed.return_value = feed_payload
        pipeline.fetcher.get_orbital_data.return_value = {}
        pipeline.parser.parse_feed.return_value = parsed_df

        with patch(f"{PIPELINE_MOD}.compute_features", return_value=parsed_df):
            pipeline.run_incremental()

        ids_arg = pipeline.fetcher.get_orbital_data.call_args.args[0]
        assert sorted(ids_arg) == ["111", "222", "333"]


class TestRunWeekly:
    def test_passes_start_date_to_fetcher(self, pipeline, feed_payload, parsed_df):
        pipeline.fetcher.get_week_feed.return_value = feed_payload
        pipeline.fetcher.get_orbital_data.return_value = {}
        pipeline.parser.parse_feed.return_value = parsed_df

        with patch(f"{PIPELINE_MOD}.compute_features", return_value=parsed_df):
            pipeline.run_weekly(start_date="2024-01-01")

        pipeline.fetcher.get_week_feed.assert_called_once_with(start_date="2024-01-01")
        pipeline.feature_group.insert.assert_called_once()
        pipeline.feature_group_dedup.dedup_and_insert.assert_called_once()

    def test_defaults_start_date_to_none(self, pipeline, feed_payload, parsed_df):
        pipeline.fetcher.get_week_feed.return_value = feed_payload
        pipeline.fetcher.get_orbital_data.return_value = {}
        pipeline.parser.parse_feed.return_value = parsed_df

        with patch(f"{PIPELINE_MOD}.compute_features", return_value=parsed_df):
            pipeline.run_weekly()

        pipeline.fetcher.get_week_feed.assert_called_once_with(start_date=None)


class TestRunBackfill:
    def test_uses_browse_endpoint(self, pipeline, parsed_df):
        fake_pages = [{"page": {}, "near_earth_objects": []}] * 3
        pipeline.fetcher.get_all_pages.return_value = fake_pages
        pipeline.parser.parse_all_browse_pages.return_value = parsed_df

        with patch(f"{PIPELINE_MOD}.compute_features", return_value=parsed_df):
            pipeline.run_backfill(max_pages=3, start_page=10)

        pipeline.fetcher.get_all_pages.assert_called_once_with(
            max_pages=3, start_page=10
        )
        pipeline.parser.parse_all_browse_pages.assert_called_once_with(fake_pages)
        pipeline.feature_group.insert.assert_called_once()
        pipeline.feature_group_dedup.dedup_and_insert.assert_called_once()


class TestCreateFeatureView:
    def test_delegates_to_feature_view_repo(self, pipeline):
        pipeline.create_feature_view()
        pipeline.feature_view.get_or_create.assert_called_once()