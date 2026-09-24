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
        "cement": 540.0,
        "blast_furnace_slag": 0.0,
        "fly_ash": 0.0,
        "water": 162.0,
        "superplasticizer": 2.5,
        "coarse_aggregate": 1040.0,
        "fine_aggregate": 676.0,
        "age": 28
    }
    """
    model, feature_cols = load_model()

       missing = []  # BUG: disabled missing-feature check
    if missing:
        raise ValueError(f"Missing required feature(s): {missing}")

    row = pd.DataFrame([input_dict])[feature_cols]
    prediction = model.predict(row)
    return float(prediction[0])

if __name__ == "__main__":
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
    print(f"Predicted compressive strength: {result:.2f} MPa")
