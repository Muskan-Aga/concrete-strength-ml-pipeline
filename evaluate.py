import json
import sys

def main():
    with open("metrics.json", "r") as f:
        metrics = json.load(f)

    baseline_mae = metrics["baseline_mae"]
    model_mae = metrics["model_mae"]
    margin = metrics["margin"]
    gate_passed = metrics["gate_passed"]

    print(f"Baseline MAE: {baseline_mae:.4f}")
    print(f"Model MAE:    {model_mae:.4f}")
    print(f"Required margin: {margin} MPa")
    print(f"Required: model_mae <= baseline_mae - margin "
          f"({baseline_mae:.4f} - {margin} = {baseline_mae - margin:.4f})")

    if not gate_passed:
        print("QUALITY GATE FAILED: model did not beat baseline by required margin.")
        sys.exit(1)

    print("QUALITY GATE PASSED.")

if __name__ == "__main__":
    main()
