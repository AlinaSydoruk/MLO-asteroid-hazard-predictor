from src.training_pipeline.data_loader import TrainingDataLoader
from src.training_pipeline.trainer import ModelTrainer
from src.training_pipeline.evaluator import ModelEvaluator
from src.training_pipeline.mlflow.connection import MLflowConnection
from src.training_pipeline.mlflow.model_registry_repo import ModelRegistryRepository
from src.utils import get_logger
from src.config import PROMOTION_METRIC

log = get_logger(__name__)


class TrainingPipeline:
    """
    Orchestrates the full training pipeline.
    """

    def __init__(
        self,
        auto_promote: bool = True,
        promotion_metric: str = "f1",
    ):
        mlflow_conn = MLflowConnection()

        self.data_loader = TrainingDataLoader()
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()
        self.registry = ModelRegistryRepository(connection=mlflow_conn)

        self.auto_promote = auto_promote
        self.promotion_metric = promotion_metric

    def run(self) -> dict:
        """Run the full training pipeline. Returns final metrics."""
        log.info("Starting TRAINING pipeline")
        log.info("Loading data from Feature Store...")

        X_train, X_val, X_test, y_train, y_val, y_test = self.data_loader.load()

        log.info("Training XGBoost...")
        model = self.trainer.train(X_train, y_train, X_val, y_val)

        log.info("Evaluating on test set...")
        metrics = self.evaluator.evaluate(model, X_test, y_test)

        log.info("Logging to MLflow and registering model...")
        run_id = self.registry.log_run(
            model=model,
            metrics=metrics,
            params=self.trainer.params,
            feature_importance=self.trainer.get_feature_importance(),
            X_sample=X_train.head(5),
        )

        #Promote champion
        log.info("Champion promotion check...")
        if self.auto_promote and self.registry.should_promote(
            new_metrics=metrics,
            metric_name=self.promotion_metric,
        ):
            self.registry.promote_to_champion()
            log.info("New champion!")
        else:
            log.info("Current champion retained.")

        log.info("Training pipeline complete")
        log.info(f"Run ID:  {run_id}")

        return metrics


def main():
    pipeline = TrainingPipeline(
        auto_promote=True,
        promotion_metric=PROMOTION_METRIC,
    )
    pipeline.run()


if __name__ == "__main__":
    main()