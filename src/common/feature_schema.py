def get_feature_columns() -> list[str]:

    return [
        "est_diameter_min_km",
        "est_diameter_max_km",
        "est_diameter_mean_km",
        "relative_velocity_kmh",
        #"relative_velocity_kms",
        "miss_distance_km",
        #"miss_distance_lunar",
        #"miss_distance_astronomical",
        "absolute_magnitude_h",
        #"is_sentry_object",
        "diameter_to_distance_ratio",
        "kinetic_energy_proxy",
        "log_miss_distance_km",
        "log_kinetic_energy_proxy",
    ]


def get_label_column() -> str:
    """Target column — what the model predicts."""
    return "is_potentially_hazardous"


def get_identity_columns() -> list[str]:
    """
    Columns that identify a row but are NOT features.
    These are dropped before training/inference.
    """
    return [
        "asteroid_id",
        "name",
        "close_approach_date",
    ]


def get_all_columns() -> list[str]:
    return (
        get_identity_columns() +
        get_feature_columns() +
        [get_label_column()]
    )