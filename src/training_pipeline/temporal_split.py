import pandas as pd

def temporal_split(
    df: pd.DataFrame,
    training_cutoff: pd.Timestamp,
    val_fraction: float,
    test_fraction: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split asteroids into train / val / test.

    Anything first observed AFTER `training_cutoff` is excluded entirely, those asteroids are reserved for the inference pipeline as truly
    unseen data.
    Within the asteroids observed ON OR BEFORE `training_cutoff`:
      1. Deduplicate so each asteroid_id appears exactly once
         (using its most recent close approach as the representative row).
      2. Randomly split asteroids into train / val / test.
    """
    required = {"first_observation_date", "asteroid_id", "close_approach_date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    df = df.copy()
    df["first_observation_date"] = pd.to_datetime(df["first_observation_date"])
    df["close_approach_date"]    = pd.to_datetime(df["close_approach_date"])
    df = df.dropna(subset=["first_observation_date"])

    # Keep only asteroids the model is allowed to learn from
    df = df[df["first_observation_date"] <= training_cutoff]

    # Deduplicate to one row per asteroid (most recent close approach)
    df = df.sort_values(["asteroid_id", "close_approach_date"])
    df = df.drop_duplicates(subset=["asteroid_id"], keep="last")

    # Random split into train / val / test
    shuffled = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    n = len(shuffled)
    n_test = int(n * test_fraction)
    n_val  = int(n * val_fraction)

    test_df  = shuffled.iloc[:n_test].copy()
    val_df   = shuffled.iloc[n_test : n_test + n_val].copy()
    train_df = shuffled.iloc[n_test + n_val :].copy()

    return train_df, val_df, test_df