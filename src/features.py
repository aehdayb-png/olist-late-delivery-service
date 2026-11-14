import pandas as pd

from src.config_loader import load_config


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Same logic as Notebook 5: only info known at purchase time, no leakage."""
    cfg = load_config()["features"]
    out = df.copy()

    purchased = pd.to_datetime(out["order_purchase_timestamp"])
    estimated = pd.to_datetime(out["order_estimated_delivery_date"])

    out["purchase_month"] = purchased.dt.month
    out["purchase_weekday"] = purchased.dt.dayofweek
    out["purchase_hour"] = purchased.dt.hour
    out["estimated_days"] = (estimated - purchased).dt.days

    return out[cfg["base_cols"]]
