import time
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env.local")

from src.config import (
    NASA_API_KEY,
    NASA_BASE_URL,
    NASA_MAX_RETRIES,
    NASA_TIMEOUT_SECONDS,
    NASA_RETRY_DELAY_SECONDS,
)


class NASAClient:
    """
    HTTP client for NASA API.
    """

    def __init__(
        self,
        api_key: str = NASA_API_KEY,
        base_url: str = NASA_BASE_URL,
        max_retries: int = NASA_MAX_RETRIES,
        retry_delay: float = NASA_RETRY_DELAY_SECONDS,
        timeout: int = NASA_TIMEOUT_SECONDS,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.session = requests.Session()


    def get(self, endpoint: str, params: dict = None) -> dict:
        """
        Send a GET request to NASA API.

        Args:
            endpoint: API path e.g. "/feed" or "/neo/browse"
            params:   query parameters (api_key added automatically)
        Returns:
            parsed JSON response as dict
        Raises:
            requests.HTTPError: if all retries fail
        """
        url = f"{self.base_url}{endpoint}"
        params = params or {}
        params["api_key"] = self.api_key

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                return response.json()

            except requests.HTTPError as e:
                status = e.response.status_code if e.response else None

                # 429 = rate limited — wait longer
                if status == 429:
                    wait = self.retry_delay * attempt * 2
                    print(f"Rate limited. Waiting {wait}s before retry {attempt}/{self.max_retries}")
                    time.sleep(wait)

                # 4xx errors (except 429) — don't retry
                elif status and 400 <= status < 500:
                    raise


                # 5xx or network errors — retry
                else:
                    if attempt < self.max_retries:
                        print(f"Request failed (attempt {attempt}/{self.max_retries}): {e}")
                        time.sleep(self.retry_delay * attempt)
                    else:
                        raise

        raise requests.HTTPError(f"All {self.max_retries} retries failed for {url}")