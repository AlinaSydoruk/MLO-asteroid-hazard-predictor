import pandas as pd
import numpy as np


def compute_features(asteroid_df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features from the flat asteroid DataFrame."""
    asteroid_df = asteroid_df.copy()

    # Size estimate
    asteroid_df["est_diameter_mean_km"] = (asteroid_df["est_diameter_min_km"] + asteroid_df["est_diameter_max_km"]) / 2

    # Size / distance ratio — bigger ratio = more dangerous
    asteroid_df["diameter_to_distance_ratio"] = (
            asteroid_df["est_diameter_mean_km"] / asteroid_df["miss_distance_km"].clip(lower=1e-9)
    )

    # Kinetic energy proxy 
    asteroid_df["kinetic_energy_proxy"] = (
            asteroid_df["est_diameter_mean_km"] ** 3 * asteroid_df["relative_velocity_kmh"] ** 2
    )

    # Log transforms — compress huge value ranges
    asteroid_df["log_miss_distance_km"] = np.log1p(asteroid_df["miss_distance_km"])
    asteroid_df["log_kinetic_energy_proxy"] = np.log1p(asteroid_df["kinetic_energy_proxy"])

    # Drop rows with missing critical values
    asteroid_df = asteroid_df.dropna(subset=[
        "est_diameter_min_km",
        "est_diameter_max_km",
        "relative_velocity_kmh",
        "miss_distance_km",
        "absolute_magnitude_h",
    ])

    # Enforce types
    asteroid_df["asteroid_id"] = asteroid_df["asteroid_id"].astype(str)
    asteroid_df["close_approach_date"] = pd.to_datetime(asteroid_df["close_approach_date"])
    asteroid_df["is_potentially_hazardous"] = asteroid_df["is_potentially_hazardous"].astype(int)
    asteroid_df["is_sentry_object"] = asteroid_df["is_sentry_object"].astype(int)

    return asteroid_df


def get_feature_columns() -> list[str]:
    """feature columns"""
    return [
        "est_diameter_min_km",
        "est_diameter_max_km",
        "est_diameter_mean_km",
        "relative_velocity_kmh",
        "relative_velocity_kms",
        "miss_distance_km",
        "miss_distance_lunar",
        "miss_distance_astronomical",
        "absolute_magnitude_h",
        "is_sentry_object",
        "diameter_to_distance_ratio",
        "kinetic_energy_proxy",
        "log_miss_distance_km",
        "log_kinetic_energy_proxy",
    ]


def get_label_column() -> str:
    """Single source of truth for the label column name."""
    return "is_potentially_hazardous"