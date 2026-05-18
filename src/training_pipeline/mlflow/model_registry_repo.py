"""
Model Registry Repository.
Single responsibility: manage models in MLflow.
Mirrors FeatureGroupRepository pattern.
"""
import mlflow
import mlflow.xgboost
import pandas as pd
import xgboost as xgb
from mlflow.models import infer_signature

from src.config import MODEL_NAME, MODEL_ALIAS
from src.training_pipeline.mlflow.connection import MLflowConnectionManager
from src.utils import get_logger

log = get_logger(__name__)


class ModelRegistryRepository:
    """
    All MLflow Model Registry operations in one place.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        alias: str = MODEL_ALIAS,
        connection: MLflowConnectionManager = None,
    ):
        self.model_name = model_name
        self.alias = alias
        self.connection = connection or MLflowConnectionManager()

    def log_run(
        self,
        model: xgb.XGBClassifier,
        metrics: dict,
        params: dict,
        feature_importance: dict,
        X_sample: pd.DataFrame,
    ) -> str:
        """ Log a complete training run to MLflow. """
        self.connection.connect()

        with mlflow.start_run() as run:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.log_dict(feature_importance, "feature_importance.json")

            signature = infer_signature(X_sample, model.predict(X_sample))
            mlflow.xgboost.log_model(
                model,
                name="model",
                signature=signature,
                registered_model_name=self.model_name,
            )

            run_id = run.info.run_id
            log.info(f" Run logged: {run_id}")
            return run_id



    def promote_to_champion(self, version: str = None) -> str:
        """
        Promote a version to champion alias.
        If version is None, uses the latest version.
        """
        client = self.connection.client
        if version is None:
            versions = client.get_latest_versions(self.model_name)
            if not versions:
                raise RuntimeError(
                    f"No versions found for model: {self.model_name}"
                )
            version = versions[-1].version

        log.info(
            f"Promoting {self.model_name} v{version} → '{self.alias}'"
        )
        client.set_registered_model_alias(
            self.model_name, self.alias, version
        )
        log.info(f" v{version} is now the {self.alias}")
        return version

    def get_champion_version(self) -> str:
        """Get the current champion version."""
        client = self.connection.client
        try:
            v = client.get_model_version_by_alias(
                self.model_name, self.alias
            )
            return v.version
        except Exception:
            return "none"

    def get_champion_metrics(self) -> dict:
        """Get metrics of the current champion model."""
        client = self.connection.client
        try:
            v = client.get_model_version_by_alias(
                self.model_name, self.alias
            )
            run = client.get_run(v.run_id)
            return run.data.metrics
        except Exception:
            return {}

    def should_promote(self, new_metrics: dict, metric_name: str = "f1") -> bool:
        """
        Decide if new model should replace current champion.
        """
        champion_metrics = self.get_champion_metrics()

        if not champion_metrics:
            log.info("No current champion, auto-promoting new model.")
            return True

        new_score = new_metrics.get(metric_name, 0)
        champion_score = champion_metrics.get(metric_name, 0)

        log.info(
            f"Comparing {metric_name}: "
            f"champion={champion_score:.4f} | new={new_score:.4f}"
        )

        return new_score >= champion_score