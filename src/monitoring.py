import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def log_predictions(features, results, model_version, log_path):
    """Persist individual prediction records as JSON Lines."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat()

    with path.open("a", encoding="utf-8") as file:
        for position in range(len(results)):
            feature_values = features.iloc[position].to_dict()

            record = {
                "timestamp_utc": timestamp,
                "model_version": model_version,
                "features": {
                    key: _json_safe(value) for key, value in feature_values.items()
                },
                "late_probability": float(results.iloc[position]["late_probability"]),
                "predicted_is_late": int(results.iloc[position]["predicted_is_late"]),
            }

            file.write(json.dumps(record) + "\n")


def _json_safe(value):
    """Convert pandas and NumPy values into JSON-compatible values."""
    if value is None or pd.isna(value):
        return None

    if hasattr(value, "item"):
        return value.item()

    if hasattr(value, "isoformat"):
        return value.isoformat()

    return value
