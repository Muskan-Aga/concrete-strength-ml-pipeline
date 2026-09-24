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
        "Cement": 540.0,
        "Blast Furnace Slag": 0.0,
        "Fly Ash": 0.0,
        "Water": 162.0,
        "Superplasticizer": 2.5,
        "Coarse Aggregate": 1040.0,
        "Fine Aggregate": 676.0,
        "Age": 28
    }
    result = predict(sample_input)
    assert isinstance(result, float)
    assert result > 0  # compressive strength should be positive

def test_missing_feature_is_rejected():
    """An input missing a required feature raises a clear error."""
    incomplete_input = {
        "Cement": 540.0,
        "Blast Furnace Slag": 0.0,
        "Fly Ash": 0.0,
        "Water": 162.0,
        # Missing: Superplasticizer, Coarse Aggregate, Fine Aggregate, Age
    }
    with pytest.raises(ValueError, match="Missing required feature"):
        predict(incomplete_input)
