import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import json
import sys

DATA_URL = "https://huggingface.co/spaces/sparsh007/ConcreteStrengthPrediction/raw/main/concrete_data.csv"

EXPECTED_COLUMNS = [
    "cement", "blast_furnace_slag", "fly_ash", "water",
    "superplasticizer", "coarse_aggregate", "fine_aggregate",
    "age", "concrete_compressive_strength"
]

def load_and_validate_data():
    df = pd.read_csv(DATA_URL)
    df.columns = [c.strip() for c in df.columns]

    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing expected columns: {missing_cols}")
        print(f"Actual columns found: {list(df.columns)}")
        sys.exit(1)

    print("Data validation passed. Columns present:", list(df.columns))
    return df

def main():
    df = load_and_validate_data()

    target_col = "concrete_compressive_strength"
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)
    baseline_preds = baseline.predict(X_test)
    baseline_mae = mean_absolute_error(y_test, baseline_preds)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    model_preds = model.predict(X_test)
    model_mae = mean_absolute_error(y_test, model_preds)

    print(f"Baseline MAE: {baseline_mae:.4f}")
    print(f"Model MAE: {model_mae:.4f}")

    joblib.dump(model, "model.pkl")
    joblib.dump(feature_cols, "feature_columns.pkl")

    margin = 2.0
    metrics = {
        "baseline_mae": baseline_mae,
        "model_mae": model_mae,
        "margin": margin,
        "gate_passed": bool(model_mae <= baseline_mae - margin)
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Metrics saved to metrics.json")
    print(f"Gate result: {'PASS' if metrics['gate_passed'] else 'FAIL'}")

if __name__ == "__main__":
    main()
