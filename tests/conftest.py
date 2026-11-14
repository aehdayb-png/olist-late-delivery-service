import pandas as pd
import pytest


@pytest.fixture
def sample_order():
    """Row 0 from Notebook 5 (known notebook output)."""
    return pd.DataFrame(
        [
            {
                "order_purchase_timestamp": "2017-10-02 10:56:33",
                "order_estimated_delivery_date": "2017-10-18",
                "customer_state": "SP",
                "n_items": 1,
                "total_price": 29.99,
                "total_freight": 8.72,
                "n_distinct_products": 1,
                "n_distinct_sellers": 1,
                "total_payment": 38.71,
                "n_payment_entries": 3,
                "max_installments": 1,
            }
        ]
    )
