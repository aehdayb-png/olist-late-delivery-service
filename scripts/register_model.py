import hashlib
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "final_model_logreg.pkl"
MODEL_NAME = "OlistLateDelivery"
EXPERIMENT_NAME = "Olist Late Delivery"

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}",
)
REGISTRY_URI = os.getenv("MLFLOW_REGISTRY_URI", TRACKING_URI)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model_hash = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(REGISTRY_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    client = MlflowClient(
        tracking_uri=TRACKING_URI,
        registry_uri=REGISTRY_URI,
    )

    existing_versions = client.search_model_versions(f"name='{MODEL_NAME}'")

    for version in existing_versions:
        run = client.get_run(version.run_id)

        if run.data.params.get("model_sha256") == model_hash:
            client.set_registered_model_alias(
                MODEL_NAME,
                "champion",
                version.version,
            )
            print("Existing model registration reused")
            print(f"Model name: {MODEL_NAME}")
            print(f"Model version: {version.version}")
            print("Alias: champion")
            print(f"Model SHA-256: {model_hash}")
            return

    model = joblib.load(MODEL_PATH)

    if not hasattr(model, "feature_names_in_"):
        raise ValueError("Model is missing feature_names_in_.")

    input_example = pd.DataFrame(
        [[0.0] * len(model.feature_names_in_)],
        columns=model.feature_names_in_,
    )
    output_example = model.predict_proba(input_example)
    signature = infer_signature(input_example, output_example)

    with mlflow.start_run(run_name="logistic-regression-baseline") as run:
        mlflow.log_params(
            {
                "model_type": type(model).__name__,
                "class_weight": "balanced",
                "max_iter": 1000,
                "random_state": 42,
                "sklearn_version": sklearn.__version__,
                "feature_count": len(model.feature_names_in_),
                "model_sha256": model_hash,
            }
        )

        mlflow.log_metrics(
            {
                "validation_roc_auc": 0.6931595894863017,
                "test_roc_auc": 0.7088474411566943,
            }
        )

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=MODEL_NAME,
        )

        versions = client.search_model_versions(f"name='{MODEL_NAME}'")

        registered_version = next(
            (version for version in versions if version.run_id == run.info.run_id),
            None,
        )

        if registered_version is None:
            raise RuntimeError("Could not find the registered model version.")

        client.set_registered_model_alias(
            MODEL_NAME,
            "champion",
            registered_version.version,
        )

        print("MLflow registration successful")
        print(f"Tracking URI: {TRACKING_URI}")
        print(f"Registry URI: {REGISTRY_URI}")
        print(f"Model name: {MODEL_NAME}")
        print(f"Model version: {registered_version.version}")
        print("Alias: champion")
        print(f"Model SHA-256: {model_hash}")
        print(f"Model URI: {model_info.model_uri}")


if __name__ == "__main__":
    main()
