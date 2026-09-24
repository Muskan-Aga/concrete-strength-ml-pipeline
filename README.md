# Concrete Compressive Strength Prediction — ML Pipeline

## Dataset

- **Source:** UCI Machine Learning Repository — Concrete Compressive Strength dataset,
  originally donated by I-Cheng Yeh (1998). Downloaded automatically at pipeline runtime from:
  https://huggingface.co/spaces/sparsh007/ConcreteStrengthPrediction/raw/main/concrete_data.csv
- **Task:** Regression
- **Target variable:** `concrete_compressive_strength` (in MPa)
- **Input features (8):** `cement`, `blast_furnace_slag`, `fly_ash`, `water`,
  `superplasticizer`, `coarse_aggregate`, `fine_aggregate`, `age`
- **Rows:** 1,030, no missing values

## Evaluation Metric

We use **Mean Absolute Error (MAE)**. It suits this task because:
- It's in the same unit as the target (MPa), so it's directly interpretable
  ("on average, predictions are off by X MPa")
- It's robust to a few large outliers compared to squared-error metrics like RMSE
- It's simple to explain to non-technical stakeholders

## Baseline and Quality Gate

- **Baseline model:** `DummyRegressor(strategy="mean")` — always predicts the mean strength
- **Candidate model:** `RandomForestRegressor(n_estimators=100, random_state=42)`
- **Split:** `train_test_split(test_size=0.2, random_state=42)` — reproducible, same split
  used throughout training, evaluation, and both failure demonstrations
- **Improvement margin:** 2.0 MPa
- **Quality gate rule:** `model_mae <= baseline_mae - margin`

**Why 2.0 MPa?** Concrete compressive strength typically ranges from ~10 to ~80 MPa, so a
2 MPa improvement represents a meaningful, practically useful gain rather than noise —
while still being achievable for a reasonably tuned model.

- If the margin were **too low** (e.g. 0.1 MPa), the gate would let through models that are
  barely better than just guessing the average — providing a false sense of "passing"
  quality control.
- If the margin were **too high** (e.g. 20 MPa), even a genuinely good model would fail the
  gate, blocking valid improvements from ever being published.

## Repository Structure

- `train.py` — downloads and validates data, trains baseline + candidate model, saves
  model, feature list, and metrics report
- `evaluate.py` — reads the metrics report and enforces the quality gate (exits with
  non-zero code on failure)
- `predict.py` — loads the trained model and produces predictions from an input dictionary;
  raises a clear error if required features are missing
- `test_pipeline.py` — automated tests: model loads correctly, a valid input produces a
  correctly-typed output, a missing-feature input is rejected
- `.github/workflows/pipeline.yml` — GitHub Actions workflow automating the full pipeline
- `requirements.txt` — dependency file

## Failure and Recovery Demonstrations

### Failure A — Model quality failure
**What I changed:** Temporarily weakened the model to
`RandomForestRegressor(n_estimators=1, max_depth=1, random_state=42)`, which cannot
capture enough signal to beat the baseline by the required margin.

**Result:** The `evaluate.py` step failed with exit code 1, printing the baseline MAE,
model MAE, and margin comparison. No model package was published (tests and upload steps
were skipped).

**Run link:** https://github.com/Muskan-Aga/concrete-strength-ml-pipeline/actions/runs/35985442884

**Which check caught it:** The quality gate check in `evaluate.py`.

### Failure B — Application failure
**What I changed:** Modified `predict.py` so `missing = []` instead of actually checking
for missing required features, disabling the validation logic.

**Result:** The test suite failed to collect/run correctly because of the broken code,
and `test_missing_feature_is_rejected` could not pass. No model package was published.

**Run link:** https://github.com/Muskan-Aga/concrete-strength-ml-pipeline/actions/runs/35986477653

**Which check caught it:** The automated test suite (`pytest test_pipeline.py`).

### Final Successful Run
Both issues were reverted (`predict.py` and `train.py` restored to their correct logic).
All steps passed and the model package was published successfully.

**Run link:** https://github.com/Muskan-Aga/concrete-strength-ml-pipeline/actions/runs/35987354051

**Published artifact:** `model-package-11` (downloaded locally as workflow artifacts expire)

## Reflection Questions

**1. Why does your evaluation metric suit your task?**
MAE is in the same units as the target (MPa), making it directly interpretable, and it
doesn't overly penalize occasional large errors the way squared-error metrics would.

**2. Why did you choose this improvement margin? What would happen if it were too low or high?**
2.0 MPa represents a practically meaningful improvement given the typical strength range
of concrete (10-80 MPa). Too low a margin would let weak models pass; too high a margin
would reject genuinely useful improvements.

**3. What caused each failed run? Which check prevented publication?**
Failure A was caused by deliberately weakening the model so it couldn't beat the baseline;
the quality gate check in `evaluate.py` caught it and blocked publication. Failure B was
caused by disabling missing-feature validation in `predict.py`; the automated test suite
caught it and blocked publication.

**4. Which parts of your workflow demonstrate continuous integration and artifact delivery?**
Every push to `main` automatically triggers checkout, dependency installation, training,
evaluation, and testing in a single job (continuous integration), and the trained model
package is only uploaded as a downloadable GitHub Actions artifact when every prior check
passes (continuous delivery of a validated artifact).

**5. Which MLOps maturity level best describes your implementation? Justify your answer.**
This implementation sits at a basic/early automation level: it automates training,
validation, testing, and packaging on every code push (which is more than manual,
notebook-driven ML work), but it lacks continuous training triggers based on new data,
automated deployment to a serving environment, and live monitoring/retraining feedback
loops. To reach the next level, the pipeline would need automated model deployment (e.g.
to a REST endpoint), scheduled or data-triggered retraining, and monitoring of live
prediction quality to detect drift.
