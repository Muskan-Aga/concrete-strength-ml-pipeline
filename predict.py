import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "model.pkl"
FEATURES_PATH = "feature_columns.pkl"

def load_model():
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURES_PATH)
    return model, feature_cols

def predict(input_dict):
    """
    input_dict: a dictionary like
    {
        "Cement": 540.0,
        "Blast Furnace Slag": 0.0,
        "Fly Ash": 0.0,
        "Water": 162.0,
        "Superplasticizer": 2.5,
        "Coarse Aggregate": 1040.0,
        "Fine Aggregate": 676.0,
        "Age": 28
    }
    """
    model, feature_cols = load_model()

    missing = [col for col in feature_cols if col not in input_dict]
    if missing:
        raise ValueError(f"Missing required feature(s): {missing}")

    row = pd.DataFrame([input_dict])[feature_cols]
    prediction = model.predict(row)
    return float(prediction[0])

if __name__ == "__main__":
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
    print(f"Predicted compressive strength: {result:.2f} MPa")
