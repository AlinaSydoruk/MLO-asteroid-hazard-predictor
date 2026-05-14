def get_raw_feature_columns() -> list[str]:
    """Features extracted directly from /feed or /browse JSON."""
    return [
        "est_diameter_min_km",
        "est_diameter_max_km",
        "absolute_magnitude_h",
        "relative_velocity_kmh",
        "miss_distance_km",
        "is_sentry_object",
    ]


def get_orbital_feature_columns() -> list[str]:
    """Features extracted from the orbital_data block."""
    return [
        "moid_au",
        "perihelion_dist_au",
        "aphelion_dist_au",
        "eccentricity",
        "semi_major_axis_au",
        "inclination_deg",
        "jupiter_tisserand",
    ]


def get_computed_feature_columns() -> list[str]:
    """Features derived in feature_computer.py."""
    return [
        "est_diameter_mean_km",
        "diameter_to_distance_ratio",
        "kinetic_energy_proxy",
        "log_miss_distance_km",
        "log_kinetic_energy_proxy",
    ]


def get_feature_columns() -> list[str]:
    """All features used by the model (raw + orbital + computed)."""
    return (
        get_raw_feature_columns()
        + get_orbital_feature_columns()
        + get_computed_feature_columns()
    )


def get_label_column() -> str:
    return "is_potentially_hazardous"


def get_identity_columns() -> list[str]:
    return [
        "asteroid_id",
        "name",
        "close_approach_date",
        "first_observation_date",
    ]


def get_all_columns() -> list[str]:
    return (
        get_identity_columns()
        + get_feature_columns()
        + [get_label_column()]
    )