# connection.py
import os
import mlflow
from mlflow.tracking import MlflowClient

from src.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_TRACKING_USERNAME,
    MLFLOW_TRACKING_PASSWORD,
    EXPERIMENT_NAME,
)
from src.utils import get_logger

log = get_logger(__name__)


class MLflowConnectionManager:
    def __init__(
        self,
        tracking_uri: str = MLFLOW_TRACKING_URI,
        username: str = MLFLOW_TRACKING_USERNAME,
        password: str = MLFLOW_TRACKING_PASSWORD,
        experiment_name: str = EXPERIMENT_NAME,
    ):
        self.tracking_uri = tracking_uri
        self.username = username
        self.password = password
        self.experiment_name = experiment_name
        self._configured = False
        self._client: MlflowClient | None = None

    def connect(self) -> None:
        if self._configured:
            return
        log.info(f"Configuring MLflow: {self.tracking_uri}")
        os.environ["MLFLOW_TRACKING_USERNAME"] = self.username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = self.password
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self._configured = True

    @property
    def client(self) -> MlflowClient:
        self.connect()
        if self._client is None:
            self._client = MlflowClient()
        return self._client