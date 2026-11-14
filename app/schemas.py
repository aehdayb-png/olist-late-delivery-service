from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class OrderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_purchase_timestamp: datetime
    order_estimated_delivery_date: datetime
    customer_state: str = Field(min_length=2, max_length=2)

    n_items: int | None = Field(default=None, ge=1, le=50)
    total_price: float | None = Field(default=None, ge=0, le=20000)
    total_freight: float | None = Field(default=None, ge=0, le=2000)
    n_distinct_products: int | None = Field(default=None, ge=1, le=50)
    n_distinct_sellers: int | None = Field(default=None, ge=1, le=20)
    total_payment: float | None = Field(default=None, ge=0, le=25000)
    n_payment_entries: int | None = Field(default=None, ge=1, le=50)
    max_installments: int | None = Field(default=None, ge=1, le=48)


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    late_probability: float
    late_risk_percent: float
    predicted_is_late: int
    model_version: str


class BatchPredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    predictions: list[PredictionResponse]
    total_orders: int
    model_version: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_version: str
    model_type: str
    features_count: int
