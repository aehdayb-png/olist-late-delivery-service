import numpy as np
import pytest

from src.features import build_features
from src.preprocessing import Preprocessor


@pytest.fixture(scope="module")
def prep():
    return Preprocessor()


def test_output_matches_feature_list(prep, sample_order):
    X = prep.transform(build_features(sample_order))
    assert X.shape == (1, 39)
    assert list(X.columns) == prep.feature_list


def test_missing_numeric_is_imputed(prep, sample_order):
    sample_order["total_payment"] = np.nan
    X = prep.transform(build_features(sample_order))
    assert not X.isna().any().any()


def test_unknown_state_encodes_to_zeros(prep, sample_order):
    sample_order["customer_state"] = "ZZ"
    X = prep.transform(build_features(sample_order))
    state_cols = [c for c in X.columns if c.startswith("customer_state_")]
    assert X[state_cols].sum().sum() == 0
