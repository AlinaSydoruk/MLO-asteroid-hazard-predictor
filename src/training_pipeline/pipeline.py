from src.training_pipeline.data_loader import TrainingDataLoader
from src.training_pipeline.trainer import ModelTrainer
from src.training_pipeline.evaluator import ModelEvaluator
from src.common.mlflow.connection import MLflowConnectionManager
from src.common.mlflow.model_registry_repo import ModelRegistryRepository
from src.utils import get_logger
from src.config import PROMOTION_METRIC
from src.config import TRAINING_CUTOFF_DAYS_BACK
import pandas as pd
from datetime import date, timedelta
log = get_logger(__name__)

class TrainingPipeline:
    """
    Orchestrates the full training pipeline.
    """

    def __init__(
        self,
        auto_promote: bool = True,
        promotion_metric: str = "f1",
        force_promote: bool = False,
    ):
        mlflow_conn = MLflowConnectionManager()

        self.data_loader = TrainingDataLoader()
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()
        self.registry = ModelRegistryRepository(connection=mlflow_conn)

        self.auto_promote = auto_promote
        self.promotion_metric = promotion_metric
        self.force_promote = force_promote

    def run(self, training_cutoff: pd.Timestamp | None = None) -> dict:
        """Run the full training pipeline. Returns final metrics."""
        log.info("Starting TRAINING pipeline")

        if training_cutoff is None:
            training_cutoff = pd.Timestamp(
                date.today() - timedelta(days=TRAINING_CUTOFF_DAYS_BACK)
            )
        log.info(f"Training cutoff: {training_cutoff.date()}")

        log.info("Loading data from Feature Store...")
        X_train, X_val, X_test, y_train, y_val, y_test, cutoff_iso = \
            self.data_loader.load(training_cutoff=training_cutoff)

        log.info("Training XGBoost...")
        model = self.trainer.train(X_train, y_train, X_val, y_val)

        log.info("Evaluating on test set...")
        metrics = self.evaluator.evaluate(model, X_test, y_test)

        log.info("Logging to MLflow and registering model...")
        run_params = {
            **self.trainer.params,
            "training_cutoff_date": cutoff_iso,
        }
        run_id = self.registry.log_run(
            model=model,
            metrics=metrics,
            params=run_params,
            feature_importance=self.trainer.get_feature_importance(),
            X_sample=X_train.head(5),
        )

        #  Promote if better
        log.info("Champion promotion check...")
        if self.force_promote or (
                self.auto_promote
                and self.registry.should_promote(
            new_metrics=metrics,
            metric_name=self.promotion_metric,
        )
        ):
            self.registry.promote_to_champion()
            log.info("New champion!")
        else:
            log.info("Current champion retained.")

        log.info("Training pipeline complete")
        log.info(f"Run ID:  {run_id}")
        log.info(f"Cutoff:  {cutoff_iso}")
        return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", default=None, help="YYYY-MM-DD")
    parser.add_argument("--force-promote", action="store_true")
    args = parser.parse_args()

    cutoff = pd.Timestamp(args.cutoff) if args.cutoff else None

    pipeline = TrainingPipeline(auto_promote=True, promotion_metric=PROMOTION_METRIC, force_promote=args.force_promote)
    pipeline.run(training_cutoff=cutoff)

if __name__ == "__main__":
    main()