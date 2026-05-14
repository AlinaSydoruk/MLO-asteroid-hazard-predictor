import pandas as pd

from src.config import (
    VALIDATION_SIZE,
    TEST_SIZE,
    RANDOM_SEED,
)
from src.common.hopsworks.feature_view_repo import FeatureViewRepository
from src.common.feature_schema import get_feature_columns, get_label_column
from src.training_pipeline.temporal_split import temporal_split
from src.utils import get_logger

log = get_logger(__name__)


class TrainingDataLoader:
    """Loads training data and applies an asteroid-level split."""

    def __init__(
        self,
        feature_view_repo: FeatureViewRepository | None = None,
        val_fraction: float = VALIDATION_SIZE,
        test_fraction: float = TEST_SIZE,
        random_seed: int = RANDOM_SEED,
    ):
        self.feature_view_repo = feature_view_repo or FeatureViewRepository()
        self.val_fraction  = val_fraction
        self.test_fraction = test_fraction
        self.random_seed   = random_seed

    def load(self, training_cutoff: pd.Timestamp) -> tuple:
        """
        Returns: X_train, X_val, X_test, y_train, y_val, y_test, cutoff_iso
        """

        df = self.feature_view_repo.get_full_dataset()

        train_df, val_df, test_df = temporal_split(
            df,
            training_cutoff=training_cutoff,
            val_fraction=self.val_fraction,
            test_fraction=self.test_fraction,
            random_seed=self.random_seed,
        )

        feature_cols = get_feature_columns()
        label_col    = get_label_column()

        X_train, y_train = train_df[feature_cols], train_df[label_col]
        X_val,   y_val   = val_df[feature_cols],   val_df[label_col]
        X_test,  y_test  = test_df[feature_cols],  test_df[label_col]

        log.info(
            f"Asteroid-level split @ cutoff={training_cutoff.date()}:\n"
            f"  Train: {len(X_train)} asteroids\n"
            f"  Val:   {len(X_val)} asteroids\n"
            f"  Test:  {len(X_test)} asteroids\n"
            f"  (asteroids first observed AFTER {training_cutoff.date()} are reserved for inference)"
        )

        return (
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            training_cutoff.date().isoformat(),
        )