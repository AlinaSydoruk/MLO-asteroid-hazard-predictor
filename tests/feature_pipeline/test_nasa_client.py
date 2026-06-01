"""Unit tests for src.feature_pipeline.nasa_client."""
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.feature_pipeline.nasa_client import NASAClient


def _mock_response(status_code: int, json_payload=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_payload or {}
    if status_code >= 400:
        http_err = requests.HTTPError(response=resp)
        resp.raise_for_status.side_effect = http_err
    else:
        resp.raise_for_status.return_value = None
    return resp


@pytest.fixture
def client():
    # max_retries small, delays zero — fast tests
    c = NASAClient(api_key="KEY", max_retries=3, retry_delay=0)
    return c


@pytest.fixture(autouse=True)
def no_sleep():
    """Stop tests from actually sleeping during retries."""
    with patch("src.feature_pipeline.nasa_client.time.sleep"):
        yield


class TestGetSuccess:
    def test_returns_parsed_json(self, client):
        response = _mock_response(200, {"hello": "world"})
        with patch.object(client.session, "get", return_value=response) as mock_get:
            result = client.get("/feed", params={"start_date": "2024-01-01"})
        assert result == {"hello": "world"}
        # Verify api_key gets injected
        called_params = mock_get.call_args.kwargs["params"]
        assert called_params["api_key"] == "KEY"
        assert called_params["start_date"] == "2024-01-01"

    def test_builds_full_url(self, client):
        response = _mock_response(200)
        with patch.object(client.session, "get", return_value=response) as mock_get:
            client.get("/neo/browse")
        url_arg = mock_get.call_args.args[0]
        assert url_arg.endswith("/neo/browse")

    def test_works_without_explicit_params(self, client):
        response = _mock_response(200, {"ok": True})
        with patch.object(client.session, "get", return_value=response) as mock_get:
            client.get("/neo/browse")
        # api_key still added even when params=None
        assert mock_get.call_args.kwargs["params"] == {"api_key": "KEY"}


class TestGetErrors:
    def test_4xx_not_429_raises_immediately(self, client):
        response = _mock_response(404)
        with patch.object(client.session, "get", return_value=response) as mock_get:
            with pytest.raises(requests.HTTPError):
                client.get("/neo/missing")
        # No retries for 4xx
        assert mock_get.call_count == 1

    def test_429_retries_then_succeeds(self, client):
        bad = _mock_response(429)
        good = _mock_response(200, {"ok": True})
        with patch.object(client.session, "get", side_effect=[bad, bad, good]) as mock_get:
            result = client.get("/feed")
        assert result == {"ok": True}
        assert mock_get.call_count == 3

    def test_5xx_retries_then_raises_after_max(self, client):
        response = _mock_response(500)
        with patch.object(client.session, "get", return_value=response) as mock_get:
            with pytest.raises(requests.HTTPError):
                client.get("/feed")
        assert mock_get.call_count == client.max_retries

    def test_5xx_retries_then_succeeds(self, client):
        bad = _mock_response(503)
        good = _mock_response(200, {"recovered": True})
        with patch.object(client.session, "get", side_effect=[bad, good]) as mock_get:
            result = client.get("/feed")
        assert result == {"recovered": True}
        assert mock_get.call_count == 2

    def test_429_failures_exhaust_retries(self, client):
        # All 429s — must eventually exhaust and raise
        response = _mock_response(429)
        with patch.object(client.session, "get", return_value=response) as mock_get:
            with pytest.raises(requests.HTTPError):
                client.get("/feed")
        assert mock_get.call_count == client.max_retries