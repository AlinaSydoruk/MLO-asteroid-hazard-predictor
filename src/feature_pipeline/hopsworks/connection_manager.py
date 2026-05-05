import hopsworks

from src.config import (
    HOPSWORKS_API_KEY,
    HOPSWORKS_PROJECT_NAME,
    HOPSWORKS_HOST,
)
from src.utils import get_logger

log = get_logger(__name__)


class HopsworksConnectionManager:
    """
    Manages the Hopsworks connection.
    Connects lazily (only when first needed)
    Reuses existing connection (no repeated logins)

    """

    def __init__(
        self,
        api_key: str = HOPSWORKS_API_KEY,
        project_name: str = HOPSWORKS_PROJECT_NAME,
        host: str = HOPSWORKS_HOST,
    ):
        self.api_key = api_key
        self.project_name = project_name
        self.host = host
        self._project = None
        self._fs = None


    def connect(self) -> None:
        """Connect to Hopsworks. Does nothing if already connected."""
        if self._project is not None:
            return

        log.info(f"Connecting to Hopsworks project: {self.project_name}")
        self._project = hopsworks.login(
            host=self.host,
            api_key_value=self.api_key,
            project=self.project_name,
        )
        self._fs = self._project.get_feature_store()
        log.info(f"Connected to Hopsworks: {self._project.name}")

    def reconnect(self) -> None:
        """Force a fresh connection"""
        log.info("Reconnecting to Hopsworks...")
        self._project = None
        self._fs = None
        self.connect()

    def is_connected(self) -> bool:
        """Check if connection is active."""
        return self._project is not None

    # ─── Properties ─────────────────────────

    @property
    def project(self):
        """Get the Hopsworks project"""
        self.connect()
        return self._project

    @property
    def feature_store(self):
        """Get the Feature Store"""
        self.connect()
        return self._fs