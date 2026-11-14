from src.features import build_features
from src.preprocessing import Preprocessor

FORBIDDEN = {
    "order_delivered_customer_date",
    "order_delivered_carrier_date",
    "order_approved_at",
    "review_score",
    "is_late",
}


def test_features_match_notebook(sample_order):
    row = build_features(sample_order).iloc[0].tolist()
    assert row == ["SP", 10, 0, 10, 15, 1, 29.99, 8.72, 1, 1, 38.71, 3, 1]


def test_no_leakage_columns(sample_order):
    assert not FORBIDDEN & set(build_features(sample_order).columns)
    assert not FORBIDDEN & set(Preprocessor().feature_list)
