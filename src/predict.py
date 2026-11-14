import time

import joblib
import pandas as pd

from src.config_loader import load_config, resolve_path
from src.features import build_features
from src.logger import get_logger
from src.preprocessing import Preprocessor
from src.validation import validate_orders


class Predictor:
    """Validates orders and returns late-delivery predictions and risk scores."""

    def __init__(self):
        cfg = load_config()

        self.model = joblib.load(resolve_path(cfg["model"]["path"]))
        self.model_version = cfg["model"]["version"]
        self.preprocessor = Preprocessor()
        self.logger = get_logger()

        if 1 not in self.model.classes_:
            raise ValueError("Model does not contain the late-delivery class (1).")

        self.logger.info(
            "Predictor initialized | model_version=%s",
            self.model_version,
        )

    def predict(self, orders: pd.DataFrame) -> pd.DataFrame:
        start_time = time.perf_counter()

        input_payload = (
            orders.to_dict(orient="records")
            if isinstance(orders, pd.DataFrame)
            else repr(orders)
        )

        try:
            if not isinstance(orders, pd.DataFrame):
                raise TypeError("orders must be a pandas DataFrame.")

            if orders.empty:
                raise ValueError("orders cannot be empty.")

            validated, report = validate_orders(orders)

            if not report["valid"]:
                raise ValueError(
                    "Invalid order input: " + "; ".join(report["errors"])
                )

            features = build_features(validated)
            processed = self.preprocessor.transform(features)

            late_class_index = list(self.model.classes_).index(1)
            probabilities = self.model.predict_proba(processed)[:, late_class_index]
            predictions = self.model.predict(processed)

            results = validated.copy()
            results["late_probability"] = probabilities
            results["late_risk_percent"] = (probabilities * 100).round(2)
            results["predicted_is_late"] = predictions

            latency_ms = (time.perf_counter() - start_time) * 1000

            self.logger.info(
                "prediction_request_success | model_version=%s | "
                "latency_ms=%.2f | input=%s | output=%s",
                self.model_version,
                latency_ms,
                input_payload,
                results.to_dict(orient="records"),
            )

            return results

        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000

            self.logger.exception(
                "prediction_request_failed | model_version=%s | "
                "latency_ms=%.2f | input=%s | error=%s",
                self.model_version,
                latency_ms,
                input_payload,
                str(exc),
            )

            raise
