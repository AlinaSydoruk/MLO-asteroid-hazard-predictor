import pandas as pd


# Our column name → NASA's field name in orbital_data
ORBITAL_API_FIELDS = {
    "moid_au":            "minimum_orbit_intersection",
    "perihelion_dist_au": "perihelion_distance",
    "aphelion_dist_au":   "aphelion_distance",
    "eccentricity":       "eccentricity",
    "semi_major_axis_au": "semi_major_axis",
    "inclination_deg":    "inclination",
    "jupiter_tisserand":  "jupiter_tisserand_invariant",
}


def _to_float(value):
    """Cast to float, return None on failure or None input."""
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


class AsteroidParser:
    """Parses NASA NeoWs API responses into DataFrames."""

    def parse_feed(self, raw: dict) -> pd.DataFrame:
        records = []
        for date_str, asteroids in raw["near_earth_objects"].items():
            for ast in asteroids:
                record = self._extract_record(ast, date_str)
                if record:
                    records.append(record)
        return pd.DataFrame(records)

    def parse_browse(self, raw: dict) -> pd.DataFrame:
        records = []
        for ast in raw.get("near_earth_objects", []):
            record = self._extract_record(ast)
            if record:
                records.append(record)
        return pd.DataFrame(records)

    def parse_all_browse_pages(self, pages: list[dict]) -> pd.DataFrame:
        dfs = [self.parse_browse(p) for p in pages]
        dfs = [df for df in dfs if not df.empty]
        if not dfs:
            return pd.DataFrame()
        combined = pd.concat(dfs, ignore_index=True)
        return combined.drop_duplicates(subset=["asteroid_id"])

    def parse_lookup(self, raw: dict) -> pd.DataFrame:
        records = []
        for ca in raw.get("close_approach_data", []):
            record = self._extract_record(raw, ca["close_approach_date"])
            if record:
                records.append(record)
        return pd.DataFrame(records)

    def _extract_record(self, ast: dict, date: str = None) -> dict | None:
        """Extract identity + raw features + orbital features + label."""
        try:
            close_approaches = ast.get("close_approach_data", [])
            if not close_approaches:
                return None
            ca = close_approaches[0]
            diameter_km = ast["estimated_diameter"]["kilometers"]

            record = {
                # Identity
                "asteroid_id": ast["id"],
                "name": ast["name"],
                "close_approach_date": ca.get("close_approach_date", date),

                # Raw features
                "est_diameter_min_km":   diameter_km["estimated_diameter_min"],
                "est_diameter_max_km":   diameter_km["estimated_diameter_max"],
                "absolute_magnitude_h":  ast.get("absolute_magnitude_h"),
                "relative_velocity_kmh": float(ca["relative_velocity"]["kilometers_per_hour"]),
                "miss_distance_km":      float(ca["miss_distance"]["kilometers"]),
                "is_sentry_object":      int(ast.get("is_sentry_object", False)),

                # Label
                "is_potentially_hazardous":
                    int(ast.get("is_potentially_hazardous_asteroid", False)),
            }

            # Orbital features
            orbital = ast.get("orbital_data") or {}
            for our_name, api_name in ORBITAL_API_FIELDS.items():
                record[our_name] = _to_float(orbital.get(api_name))

            record["first_observation_date"] = orbital.get("first_observation_date")
            return record

        except (KeyError, TypeError, ValueError):
            return None