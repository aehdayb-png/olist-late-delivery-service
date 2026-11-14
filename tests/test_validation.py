from src.validation import validate_orders


def test_valid_order_passes(sample_order):
    _, report = validate_orders(sample_order)
    assert report["valid"]


def test_unknown_state_rejected(sample_order):
    sample_order["customer_state"] = "XX"
    assert not validate_orders(sample_order)[1]["valid"]


def test_negative_price_rejected(sample_order):
    sample_order["total_price"] = -5
    assert not validate_orders(sample_order)[1]["valid"]


def test_estimated_before_purchase_rejected(sample_order):
    sample_order["order_estimated_delivery_date"] = "2017-09-01"
    assert not validate_orders(sample_order)[1]["valid"]


def test_missing_required_column_rejected(sample_order):
    bad = sample_order.drop(columns=["customer_state"])
    assert not validate_orders(bad)[1]["valid"]
