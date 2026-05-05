import os
import traceback

import requests
import hopsworks
import mlflow
from dotenv import load_dotenv
from pathlib import Path

# Load from .env.local (not committed) or .env as fallback
env_path = Path(__file__).parent.parent / ".env.local"
if not env_path.exists():
    env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

print("Testing connections...\n")

# 1. NASA API
r = requests.get(
    "https://api.nasa.gov/neo/rest/v1/feed",
    params={
        "start_date": "2026-05-01",
        "end_date": "2026-05-02",
        "api_key": os.environ["NASA_API_KEY"],
    },
)
if r.ok:
    count = sum(len(v) for v in r.json()["near_earth_objects"].values())
    print(f"NASA API — OK ({count} asteroids returned)")
else:
    print(f"NASA API — FAIL: {r.status_code} {r.text}")

# 2. Hopsworks
try:
    project = hopsworks.login(
        host="eu-west.cloud.hopsworks.ai",
        api_key_value=os.environ["HOPSWORKS_API_KEY"],
        project=os.environ["HOPSWORKS_PROJECT_NAME"],
    )
    print(f" Hopsworks — OK (project: {project.name})")
except Exception as e:
    traceback.print_exc()

# 3. MLflow / DagsHub
try:
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ["MLFLOW_TRACKING_USERNAME"]
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ["MLFLOW_TRACKING_PASSWORD"]
    mlflow.set_experiment("smoke_test")
    with mlflow.start_run():
        mlflow.log_metric("hello", 1.0)
    print("MLflow / DagsHub — OK (run logged)")
except Exception as e:
    print(f"MLflow / DagsHub — FAIL: {e}")