import pandas as pd
from scipy.stats import ks_2samp

from src.utils import get_logger
from src.config import P_VALUE_THRESHOLD

log = get_logger(__name__)


class FeatureDriftDetector:
    """Compares current features against a reference sample using KS test."""

    def detect(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        report_path: str = "drift_report.html",
    ) -> dict:
        if len(current) == 0 or len(reference) == 0:
            log.warning("Empty reference or current — skipping drift.")
            return {"drifted_features": 0, "share_drifted": 0.0,
                    "dataset_drift": False, "per_feature": {}}

        drop_cols = {"is_potentially_hazardous", "asteroid_id",
                     "first_observation_date", "close_approach_date"}
        numeric_cols = [
            c for c in reference.select_dtypes(include="number").columns
            if c not in drop_cols and c in current.columns
        ]

        per_feature = {}
        drifted = 0
        for col in numeric_cols:
            ref = reference[col].dropna()
            cur = current[col].dropna()
            if len(ref) < 2 or len(cur) < 2:
                continue
            stat, p_value = ks_2samp(ref, cur)
            is_drifted = p_value < P_VALUE_THRESHOLD
            per_feature[col] = {
                "ks_stat": float(stat),
                "p_value": float(p_value),
                "drifted": bool(is_drifted),
            }
            if is_drifted:
                drifted += 1

        total = len(per_feature) or 1
        share = drifted / total

        # Write a simple HTML report
        self._write_html_report(per_feature, report_path)

        out = {
            "drifted_features": int(drifted),
            "share_drifted": float(share),
            "dataset_drift": bool(share >= 0.5),
            "report_path": report_path,
            "per_feature": per_feature,
        }
        log.info(f"Feature drift: {drifted}/{total} features drifted ({share:.0%})")
        return out

    def _write_html_report(self, per_feature: dict, path: str) -> None:
        rows = "\n".join(
            f"<tr><td>{name}</td><td>{m['ks_stat']:.4f}</td>"
            f"<td>{m['p_value']:.4f}</td>"
            f"<td style='color:{'red' if m['drifted'] else 'green'}'>"
            f"{'DRIFT' if m['drifted'] else 'OK'}</td></tr>"
            for name, m in per_feature.items()
        )
        html = f"""<!DOCTYPE html>
<html><head><title>Feature Drift Report</title>
<style>
  body {{ font-family: Inter, sans-serif; padding: 2rem; background: #0f172a; color: #e2e8f0; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ padding: 0.5rem; border-bottom: 1px solid #334155; text-align: left; }}
  th {{ background: #1e293b; }}
</style></head>
<body>
<h1>Feature Drift Report (KS test)</h1>
<table>
  <tr><th>Feature</th><th>KS Statistic</th><th>p-value</th><th>Status</th></tr>
  {rows}
</table>
</body></html>"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)