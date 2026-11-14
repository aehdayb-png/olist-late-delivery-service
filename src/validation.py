from functools import lru_cache

import great_expectations as gx
import great_expectations.expectations as gxe
import pandas as pd

from src.config_loader import load_config
from src.logger import get_logger

logger = get_logger()


@lru_cache(maxsize=1)
def _suite():
    cfg = load_config()["validation"]
    suite = gx.ExpectationSuite(name="order_input_suite")
    for col in cfg["required_cols"]:
        suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column=col))
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeInSet(
            column=cfg["state_col"], value_set=cfg["allowed_states"]
        )
    )
    for col, (low, high) in cfg["ranges"].items():
        suite.add_expectation(
            gxe.ExpectColumnValuesToBeBetween(column=col, min_value=low, max_value=high)
        )
    return suite


@lru_cache(maxsize=1)
def _batch_definition():
    context = gx.get_context(mode="ephemeral")
    source = context.data_sources.add_pandas("orders_source")
    asset = source.add_dataframe_asset(name="orders_asset")
    return asset.add_batch_definition_whole_dataframe("whole")


def validate_orders(df: pd.DataFrame):
    """Return (df, report). report = {"valid": bool, "errors": [str]}."""
    cfg = load_config()["validation"]
    df = df.copy()

    missing = [c for c in cfg["required_cols"] if c not in df.columns]
    if missing:
        errors = [f"missing required columns: {missing}"]
        logger.warning("Validation failed: %s", errors)
        return df, {"valid": False, "errors": errors}

    # optional numeric columns: flag and let the imputer fill them
    for col in cfg["ranges"]:
        if col not in df.columns:
            logger.warning("Optional column '%s' missing; will be imputed", col)
            df[col] = float("nan")

    errors = []
    batch = _batch_definition().get_batch(batch_parameters={"dataframe": df})
    result = batch.validate(_suite())
    for r in result.results:
        if not r.success:
            kwargs = r.expectation_config.kwargs
            errors.append(
                f"{r.expectation_config.type} failed on '{kwargs.get('column')}'"
            )

    purchased = pd.to_datetime(df["order_purchase_timestamp"], errors="coerce")
    estimated = pd.to_datetime(df["order_estimated_delivery_date"], errors="coerce")
    if purchased.isna().any() or estimated.isna().any():
        errors.append("unparseable timestamp")
    elif ((estimated - purchased).dt.days < 0).any():
        errors.append("estimated delivery date is before purchase date")

    if errors:
        logger.warning("Validation failed: %s", errors)
    return df, {"valid": not errors, "errors": errors}
