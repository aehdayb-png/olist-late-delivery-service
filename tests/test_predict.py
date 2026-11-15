import pytest

from src.predict import Predictor


def test_predictor_loads_model():
    predictor = Predictor()

    assert predictor.model is not None
    assert predictor.preprocessor is not None
    assert predictor.model_version == "logreg_v1"


def test_prediction_output_shape(sample_order):
    predictor = Predictor()

    result = predictor.predict(sample_order)

    assert len(result) == 1
    assert "late_probability" in result.columns
    assert "late_risk_percent" in result.columns
    assert "predicted_is_late" in result.columns


def test_probability_is_between_zero_and_one(sample_order):
    predictor = Predictor()

    result = predictor.predict(sample_order)

    probability = result.loc[0, "late_probability"]

    assert 0 <= probability <= 1


def test_prediction_is_binary(sample_order):
    predictor = Predictor()

    result = predictor.predict(sample_order)

    assert result.loc[0, "predicted_is_late"] in [0, 1]


def test_empty_input_is_rejected():
    predictor = Predictor()

    with pytest.raises(ValueError, match="cannot be empty"):
        predictor.predict(__import__("pandas").DataFrame())


def test_invalid_input_is_rejected(sample_order):
    predictor = Predictor()

    invalid_order = sample_order.copy()
    invalid_order.loc[0, "customer_state"] = "XX"

    with pytest.raises(ValueError, match="Invalid order input"):
        predictor.predict(invalid_order)


def test_probability_matches_notebook(sample_order):
    result = Predictor().predict(sample_order)
    assert result.loc[0, "late_probability"] == pytest.approx(0.5, abs=1e-9)
