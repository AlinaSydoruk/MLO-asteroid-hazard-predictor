"""
Asteroid Predictor.
Loads the champion model from MLflow and generates predictions.
"""
import pandas as pd
import mlflow

from src.config import MODEL_NAME, MODEL_ALIAS
from src.common.mlflow.connection import MLflowConnectionManager
from src.common.features.schema import get_feature_columns
from src.utils import get_logger
from src.common.features.schema import get_feature_columns, get_probability_column, get_prediction_column


log = get_logger(__name__)


class AsteroidPredictor:
    """
    Loads champion model from MLflow and generates predictions.
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
        self._model = None
        self._training_cutoff = None
        self._model_version = None

    def load_champion(self) -> None:
        """Load the current champion model from MLflow."""
        if self._model is not None:
            return

        client = self.connection.client
        model_uri = f"models:/{self.model_name}@{self.alias}"
        log.info(f"Loading champion model: {model_uri}")

        self._model = mlflow.xgboost.load_model(model_uri)

        # Track version for logging
        version = client.get_model_version_by_alias(
            self.model_name, self.alias
        )
        self._model_version = version.version
        log.info(f"Champion model v{self._model_version} loaded.")

        run = client.get_run(version.run_id)
        self._training_cutoff = run.data.params.get("training_cutoff_date")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Generate predictions for asteroid features. """
        self.load_champion()

        # Select only feature columns the model was trained on
        feature_cols = get_feature_columns()
        X = df[feature_cols]

        log.info(f"Generating predictions for {len(df)} asteroids...")
        predictions = self._model.predict(X)
        probabilities = self._model.predict_proba(X)[:, 1]

        # Build result DataFrame
        result = df.copy()
        result[get_prediction_column()] = predictions.astype(int)
        result[get_probability_column()] = probabilities
        result["model_version"] = self._model_version

        return result

    @property
    def model_version(self) -> str:
        """Get the loaded champion model version."""
        return self._model_version or "not loaded"

    @property
    def training_cutoff(self) -> str | None:
        self.load_champion()
        return self._training_cutoff