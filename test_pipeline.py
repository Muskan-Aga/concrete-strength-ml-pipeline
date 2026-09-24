import pytest
import joblib
from predict import predict, load_model

def test_model_can_be_loaded():
    """The saved model file loads without error."""
    model, feature_cols = load_model()
    assert model is not None
    assert len(feature_cols) > 0

def test_valid_sample_produces_expected_output():
    """A valid sample produces a numeric prediction of expected type."""
    sample_input = {
        "cement": 540.0,
        "blast_furnace_slag": 0.0,
        "fly_ash": 0.0,
        "water": 162.0,
        "superplasticizer": 2.5,
        "coarse_aggregate": 1040.0,
        "fine_aggregate": 676.0,
        "age": 28
    }
    result = predict(sample_input)
    assert isinstance(result, float)
    assert result > 0  # compressive strength should be positive

def test_missing_feature_is_rejected():
    """An input missing a required feature raises a clear error."""
    incomplete_input = {
        "cement": 540.0,
        "blast_furnace_slag": 0.0,
        "fly_ash": 0.0,
        "water": 162.0,
        # Missing: superplasticizer, coarse_aggregate, fine_aggregate, age
    }
    with pytest.raises(ValueError, match="Missing required feature"):
        predict(incomplete_input)
