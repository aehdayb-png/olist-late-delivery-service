from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import (
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    OrderRequest,
    PredictionResponse,
)
from src.predict import Predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once when the API starts."""
    app.state.predictor = Predictor()
    yield
    app.state.predictor = None


app = FastAPI(
    title="Olist Late Delivery Prediction API",
    description="Predicts the probability of late order delivery.",
    version="1.0.0",
    lifespan=lifespan,
)


def get_predictor() -> Predictor:
    predictor = getattr(app.state, "predictor", None)

    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction service is not ready.",
        )

    return predictor


@app.get("/health", response_model=HealthResponse)
def health():
    predictor = getattr(app.state, "predictor", None)

    return HealthResponse(
        status="ok" if predictor is not None else "unavailable",
        model_loaded=predictor is not None,
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    predictor = get_predictor()

    return ModelInfoResponse(
        model_version=predictor.model_version,
        model_type=type(predictor.model).__name__,
        features_count=len(predictor.preprocessor.feature_list),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_order(order: OrderRequest):
    predictor = get_predictor()

    try:
        orders = pd.DataFrame([order.model_dump()])
        result = predictor.predict(orders)

        return PredictionResponse(
            late_probability=float(result.iloc[0]["late_probability"]),
            late_risk_percent=float(result.iloc[0]["late_risk_percent"]),
            predicted_is_late=int(result.iloc[0]["predicted_is_late"]),
            model_version=predictor.model_version,
        )

    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(orders: list[OrderRequest]):
    predictor = get_predictor()

    if not orders:
        raise HTTPException(
            status_code=422,
            detail="At least one order is required.",
        )

    try:
        input_df = pd.DataFrame(
            [order.model_dump() for order in orders]
        )
        results = predictor.predict(input_df)

        predictions = [
            PredictionResponse(
                late_probability=float(row["late_probability"]),
                late_risk_percent=float(row["late_risk_percent"]),
                predicted_is_late=int(row["predicted_is_late"]),
                model_version=predictor.model_version,
            )
            for _, row in results.iterrows()
        ]

        return BatchPredictionResponse(
            predictions=predictions,
            total_orders=len(predictions),
            model_version=predictor.model_version,
        )

    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
