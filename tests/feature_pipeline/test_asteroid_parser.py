"""Unit tests for src.feature_pipeline.asteroid_parser."""
import pandas as pd
import pytest

from src.feature_pipeline.asteroid_parser import (
    AsteroidParser,
    ORBITAL_API_FIELDS,
    _to_float,
)


@pytest.fixture
def parser():
    return AsteroidParser()


class TestToFloat:
    @pytest.mark.parametrize(
        "value, expected",
        [
            ("1.5", 1.5),
            (2, 2.0),
            (0, 0.0),
            (None, None),
            ("not-a-number", None),
            ([], None),
        ],
    )
    def test_to_float(self, value, expected):
        assert _to_float(value) == expected


class TestParseFeed:
    def test_extracts_one_row_per_asteroid(self, parser, sample_feed):
        df = parser.parse_feed(sample_feed)
        assert len(df) == 3
        assert set(df["asteroid_id"]) == {"111", "222", "333"}

    def test_extracts_all_expected_columns(self, parser, sample_feed):
        df = parser.parse_feed(sample_feed)
        required = {
            "asteroid_id",
            "name",
            "close_approach_date",
            "est_diameter_min_km",
            "est_diameter_max_km",
            "absolute_magnitude_h",
            "relative_velocity_kmh",
            "miss_distance_km",
            "is_sentry_object",
            "is_potentially_hazardous",
            "first_observation_date",
        }
        assert required.issubset(df.columns)
        # All orbital cols are present
        assert set(ORBITAL_API_FIELDS.keys()).issubset(df.columns)

    def test_hazardous_label_is_int(self, parser, sample_feed):
        df = parser.parse_feed(sample_feed)
        # Hazardous=True asteroids should be 1, False should be 0
        labels_by_id = dict(zip(df["asteroid_id"], df["is_potentially_hazardous"]))
        assert labels_by_id["111"] == 1
        assert labels_by_id["222"] == 0

    def test_numeric_strings_coerced_to_float(self, parser, sample_feed):
        df = parser.parse_feed(sample_feed)
        assert df["relative_velocity_kmh"].dtype == float
        assert df["miss_distance_km"].dtype == float

    def test_orbital_data_mapped_via_field_dict(self, parser, sample_feed):
        df = parser.parse_feed(sample_feed)
        row = df[df["asteroid_id"] == "111"].iloc[0]
        assert row["moid_au"] == pytest.approx(0.05)
        assert row["eccentricity"] == pytest.approx(0.4)
        assert row["jupiter_tisserand"] == pytest.approx(3.5)

    def test_skips_asteroid_without_close_approach_data(self, parser, make_asteroid):
        bad = make_asteroid(asteroid_id="999")
        bad["close_approach_data"] = []
        feed = {"near_earth_objects": {"2024-01-01": [bad]}}
        df = parser.parse_feed(feed)
        assert df.empty

    def test_returns_empty_df_when_feed_has_no_asteroids(self, parser):
        df = parser.parse_feed({"near_earth_objects": {}})
        assert df.empty


class TestParseBrowse:
    def test_parses_browse_page(self, parser, sample_browse_page):
        df = parser.parse_browse(sample_browse_page)
        assert len(df) == 2
        assert set(df["asteroid_id"]) == {"111", "222"}

    def test_handles_empty_browse_page(self, parser):
        df = parser.parse_browse({"near_earth_objects": []})
        assert df.empty


class TestParseAllBrowsePages:
    def test_concatenates_and_dedupes_across_pages(
        self, parser, make_asteroid, sample_browse_page
    ):
        # Second page repeats asteroid "111" — must be deduped
        page2 = {
            "page": sample_browse_page["page"],
            "near_earth_objects": [
                make_asteroid(asteroid_id="111", name="A1"),
                make_asteroid(asteroid_id="444", name="A4"),
            ],
        }
        df = parser.parse_all_browse_pages([sample_browse_page, page2])
        assert set(df["asteroid_id"]) == {"111", "222", "444"}

    def test_returns_empty_df_when_no_pages(self, parser):
        df = parser.parse_all_browse_pages([])
        assert df.empty

    def test_returns_empty_df_when_all_pages_empty(self, parser):
        empty = {"page": {}, "near_earth_objects": []}
        df = parser.parse_all_browse_pages([empty, empty])
        assert df.empty


class TestExtractRecordResilience:
    def test_returns_none_when_required_key_missing(self, parser):
        # No 'estimated_diameter' → KeyError caught, None returned
        broken = {
            "id": "x",
            "name": "x",
            "close_approach_data": [
                {
                    "close_approach_date": "2024-01-01",
                    "relative_velocity": {"kilometers_per_hour": "1"},
                    "miss_distance": {"kilometers": "1"},
                }
            ],
        }
        feed = {"near_earth_objects": {"2024-01-01": [broken]}}
        df = parser.parse_feed(feed)
        assert df.empty

    def test_orbital_data_absent_yields_none_floats(self, parser, make_asteroid):
        ast = make_asteroid(asteroid_id="555", with_orbital=False)
        feed = {"near_earth_objects": {"2024-01-01": [ast]}}
        df = parser.parse_feed(feed)
        row = df.iloc[0]
        for col in ORBITAL_API_FIELDS.keys():
            assert pd.isna(row[col])