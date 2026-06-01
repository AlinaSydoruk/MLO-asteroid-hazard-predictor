"""
Global configuration
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# ─────────────────────────────────────────
# Load secrets from .env.local or .env
# ─────────────────────────────────────────
_root = Path(__file__).parent.parent
_env = _root / ".env.local"
if not _env.exists():
    _env = _root / ".env"
load_dotenv(dotenv_path=_env)

# ─────────────────────────────────────────
# NASA
# ─────────────────────────────────────────
NASA_API_KEY = os.environ["NASA_API_KEY"]
NASA_BASE_URL = "https://api.nasa.gov/neo/rest/v1"
NASA_FEED_ENDPOINT = "/feed"
NASA_BROWSE_ENDPOINT = "/neo/browse"
NASA_NEO_ENDPOINT = "/neo"
NASA_MAX_FEED_DAYS = 7
NASA_BROWSE_PAGE_SIZE = 20
NASA_MAX_RETRIES = 3
NASA_TIMEOUT_SECONDS = 30
NASA_RETRY_DELAY_SECONDS = 2.0
NASA_POLITE_DELAY_SECONDS = 0.5

# ─────────────────────────────────────────
# Hopsworks
# ─────────────────────────────────────────
HOPSWORKS_API_KEY = os.environ["HOPSWORKS_API_KEY"]
HOPSWORKS_PROJECT_NAME = os.environ["HOPSWORKS_PROJECT_NAME"]
HOPSWORKS_HOST = "eu-west.cloud.hopsworks.ai"

FEATURE_GROUP_NAME = "asteroid_features"
FEATURE_GROUP_DEDUP_NAME = "asteroid_features_dedup"
PREDICTIONS_GROUP_NAME = "asteroid_predictions"

FEATURE_GROUP_VERSION = 2
FEATURE_GROUP_DEDUP_VERSION = 1
PREDICTIONS_GROUP_VERSION = 1

FEATURE_VIEW_NAME ="asteroid_features"
FEATURE_VIEW_INFERENCE_NAME = "asteroid_inference"
FEATURE_VIEW_DEDUP_NAME = "asteroid_training_dedup"
FEATURE_VIEW_MONITORING_NAME ="asteroid_features_monitoring"

FEATURE_VIEW_VERSION = 1
FEATURE_VIEW_DEDUP_VERSION =1
FEATURE_VIEW_MONITORING_VERSION =1

# ─────────────────────────────────────────
# MLflow / DagsHub
# ─────────────────────────────────────────
MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
MLFLOW_TRACKING_USERNAME = os.environ["MLFLOW_TRACKING_USERNAME"]
MLFLOW_TRACKING_PASSWORD = os.environ["MLFLOW_TRACKING_PASSWORD"]
EXPERIMENT_NAME = "asteroid_pha_classifier"
MODEL_NAME = "asteroid_pha_classifier"
MODEL_ALIAS = "champion"


# ─────────────────────────────────────────
# XGBoost model parameters
# ─────────────────────────────────────────

XGBOOST_PARAMS = {
    "max_depth": 3,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "eval_metric": "auc",
    "random_state": 42,
    "verbosity": 0,
}

FEATURE_IMPORTANCE_TYPE = "gain"
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
RANDOM_SEED     = 42
AUTO_PROMOTE_MODEL = True
PROMOTION_METRIC = "f1"

TRAINING_CUTOFF_DAYS_BACK = 7

WEEKLY_RETRAIN_DAY = 6           # 0=Mon, 6=Sun
WEEKLY_RETRAIN_HOUR_UTC = 6      # Sunday 06:00 UTC

UI_DEFAULT_LOOKBACK_DAYS = 7

# ─────────────────────────────────────────
# Monitoring
# ─────────────────────────────────────────

MONITORING_EXPERIMENT_NAME = "asteroid_monitoring"
MONITORING_LOOKBACK_DAYS = 7         # window of predictions to evaluate
DRIFT_ALERT_THRESHOLD = 0.5          # share of drifted features that triggers alert
PERFORMANCE_LAG_DAYS = 30            # how far back to check predictions for ground truth

P_VALUE_THRESHOLD = 0.05
EXPECTED_MIN_ROWS_PER_DAY =5


