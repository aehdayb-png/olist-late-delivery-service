import os
import time

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from src.config_loader import PROJECT_ROOT, load_config, resolve_path
from src.features import build_features
from src.logger import get_logger
from src.preprocessing import Preprocessor
from src.validation import validate_orders


class Predictor:
    """Validates orders and returns late-delivery predictions and risk scores."""

    def __init__(self):
        cfg = load_config()
        model_cfg = cfg["model"]

        self.logger = get_logger()
        self.model_version = model_cfg["version"]
        self.model_source = model_cfg.get("source", "local")

        if self.model_source == "mlflow":
            tracking_uri = os.getenv(
                "MLFLOW_TRACKING_URI",
                f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}",
            )

            registry_uri = os.getenv("MLFLOW_REGISTRY_URI", tracking_uri)

            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_registry_uri(registry_uri)

            model_name = model_cfg["registry_name"]
            model_alias = model_cfg["registry_alias"]
            model_uri = f"models:/{model_name}@{model_alias}"

            try:
                self.model = mlflow.sklearn.load_model(model_uri)
                self.logger.info(
                    "Model loaded from MLflow | model_uri=%s",
                    model_uri,
                )

            except Exception as exc:
                if not model_cfg.get("fallback_to_local", False):
                    raise RuntimeError(
                        f"Could not load model from MLflow: {model_uri}"
                    ) from exc

                self.logger.warning(
                    "MLflow model loading failed; using local fallback | "
                    "model_uri=%s | error_type=%s",
                    model_uri,
                    type(exc).__name__,
                )

                self.model = joblib.load(
                    resolve_path(model_cfg["path"])
                )
                self.model_source = "local_fallback"

        elif self.model_source == "local":
            self.model = joblib.load(resolve_path(model_cfg["path"]))

        else:
            raise ValueError(
                f"Unsupported model source: {self.model_source}"
            )

        self.preprocessor = Preprocessor()

        if 1 not in self.model.classes_:
            raise ValueError(
                "Model does not contain the late-delivery class (1)."
            )

        self.logger.info(
            "Predictor initialized | model_version=%s | model_source=%s",
            self.model_version,
            self.model_source,
        )

    def predict(self, orders: pd.DataFrame) -> pd.DataFrame:
        start_time = time.perf_counter()
        order_count = len(orders) if isinstance(orders, pd.DataFrame) else None

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
            probabilities = self.model.predict_proba(processed)[
                :, late_class_index
            ]
            predictions = self.model.predict(processed)

            results = validated.copy()
            results["late_probability"] = probabilities
            results["late_risk_percent"] = (probabilities * 100).round(2)
            results["predicted_is_late"] = predictions

            latency_ms = (time.perf_counter() - start_time) * 1000

            self.logger.info(
                "prediction_request_success | model_version=%s | "
                "model_source=%s | order_count=%s | latency_ms=%.2f | "
                "predicted_late_count=%s | mean_late_probability=%.4f",
                self.model_version,
                self.model_source,
                order_count,
                latency_ms,
                int((predictions == 1).sum()),
                float(probabilities.mean()),
            )

            return results

        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000

            self.logger.error(
                "prediction_request_failed | model_version=%s | "
                "model_source=%s | order_count=%s | latency_ms=%.2f | "
                "error_type=%s | error=%s",
                self.model_version,
                self.model_source,
                order_count,
                latency_ms,
                type(exc).__name__,
                str(exc),
            )

            raise
