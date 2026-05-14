import pandas as pd
import numpy as np


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features from the flat asteroid DataFrame."""
    df = df.copy()


    #TODO delete all useless features

    # Size estimate
    df["est_diameter_mean_km"] = (df["est_diameter_min_km"] + df["est_diameter_max_km"]) / 2

    # Size
    df["diameter_to_distance_ratio"] = (
            df["est_diameter_mean_km"] / df["miss_distance_km"].clip(lower=1e-9)
    )

    # Kinetic energy proxy 
    df["kinetic_energy_proxy"] = (
            df["est_diameter_mean_km"] ** 3 * df["relative_velocity_kmh"] ** 2
    )

    # Log transforms — compress huge value ranges
    df["log_miss_distance_km"] = np.log1p(df["miss_distance_km"])
    df["log_kinetic_energy_proxy"] = np.log1p(df["kinetic_energy_proxy"])

    # Drop rows with missing critical values
    df = df.dropna(subset=[
        "est_diameter_min_km",
        "absolute_magnitude_h",
        "moid_au"
    ])

    # Enforce types
    df["asteroid_id"] = df["asteroid_id"].astype(str)
    df["close_approach_date"] = pd.to_datetime(df["close_approach_date"])
    df["is_potentially_hazardous"] = df["is_potentially_hazardous"].astype(int)
    df["is_sentry_object"] = df["is_sentry_object"].astype(int)
    df["first_observation_date"] = pd.to_datetime(df["first_observation_date"], errors="coerce")
    return df

