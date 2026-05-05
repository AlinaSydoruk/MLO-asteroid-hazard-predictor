"""
NASA API Response Parser.
Transforms raw API responses into flat DataFrames.
"""
import pandas as pd


class AsteroidParser:
    """Parses NASA NeoWs API responses into DataFrames."""

    def parse_feed(self, raw: dict) -> pd.DataFrame:
        """Parse /feed response — keyed by date."""
        records = []
        for date_str, asteroids in raw["near_earth_objects"].items():
            for ast in asteroids:
                record = self._extract_record(ast, date_str)
                if record:
                    records.append(record)
        return pd.DataFrame(records)

    def parse_browse(self, raw: dict) -> pd.DataFrame:
        """Parse /browse response (one page) — flat list."""
        records = []
        for ast in raw.get("near_earth_objects", []):
            record = self._extract_record(ast)
            if record:
                records.append(record)
        return pd.DataFrame(records)

    def parse_all_browse_pages(self, pages: list[dict]) -> pd.DataFrame:
        """Combine ALL browse pages into one DataFrame, deduplicated."""
        dfs = [self.parse_browse(p) for p in pages]
        dfs = [df for df in dfs if not df.empty]
        if not dfs:
            return pd.DataFrame()
        combined = pd.concat(dfs, ignore_index=True)
        return combined.drop_duplicates(subset=["asteroid_id"])

    def parse_lookup(self, raw: dict) -> pd.DataFrame:
        """Parse /neo/{id} — single asteroid, all close approaches."""
        records = []
        for ca in raw.get("close_approach_data", []):
            record = self._extract_record(raw, ca["close_approach_date"])
            if record:
                records.append(record)
        return pd.DataFrame(records)

    def _extract_record(self, ast: dict, date: str = None) -> dict | None:
        """Extract all fields from a single asteroid dict."""
        try:
            close_approaches = ast.get("close_approach_data", [])
            if not close_approaches:
                return None
            ca = close_approaches[0]

            return {
                # Identity
                "asteroid_id": ast["id"],
                "name": ast["name"],
                "close_approach_date": ca.get("close_approach_date", date),
                # Size
                "est_diameter_min_km":
                    ast["estimated_diameter"]["kilometers"]["estimated_diameter_min"],
                "est_diameter_max_km":
                    ast["estimated_diameter"]["kilometers"]["estimated_diameter_max"],
                # Brightness
                "absolute_magnitude_h": ast.get("absolute_magnitude_h"),
                # Velocity
                "relative_velocity_kmh": float(ca["relative_velocity"]["kilometers_per_hour"]),
                "relative_velocity_kms": float(ca["relative_velocity"]["kilometers_per_second"]),
                # Distance
                "miss_distance_km": float(ca["miss_distance"]["kilometers"]),
                "miss_distance_lunar": float(ca["miss_distance"]["lunar"]),
                "miss_distance_astronomical": float(ca["miss_distance"]["astronomical"]),
                # Flags
                "is_sentry_object": int(ast.get("is_sentry_object", False)),
                # Label
                "is_potentially_hazardous": int(ast.get("is_potentially_hazardous_asteroid", False)),
            }
        except (KeyError, TypeError, ValueError):
            return None