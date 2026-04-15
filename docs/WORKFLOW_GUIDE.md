# Housing Price Prediction — End-to-End Workflow

A step-by-step guide for building, validating, training, and deploying a new housing price model in this system.

---

## Prerequisites

Before starting, make sure the Docker stack is running. All pipeline scripts log to the Docker MLflow server when the tracking URI is set.

```powershell
# Start the full stack
docker compose up -d

# Verify all services are healthy
docker compose ps

# Set the tracking URI for this terminal session
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
```

You should see three healthy containers: `mlflow-server`, `housing-api`, and `housing-ui`.

---

## Step 1: Scrape & Clean

Scrape recent housing data from Realtor.com via HomeHarvest and apply initial cleaning (dedup, null removal, outlier filtering via IQR).

```powershell
python scripts/scrape_and_clean.py --location "Cincinnati, OH" --past_days 365 --mlflow
```

**Key flags:**

- `--location` — target geography (e.g., `"Cincinnati, OH"`, `"Hamilton County, OH"`)
- `--past_days` — how far back to scrape (365 = one year of sold listings)
- `--iqr_multiplier` — outlier sensitivity (default 1.5; use 2.0 for a more lenient filter)
- `--mlflow` — log cleaning lineage to MLflow (always recommended)

**Output:** A cleaned CSV in `data/cleaned/` named like `Cincinnati_OH_365d_20260415_140000.csv`.

**What gets logged to MLflow:** Row counts per cleaning step, IQR bounds, dataset hashes, removal percentages.

---

## Step 2: Validate

Run Great Expectations validation on the cleaned data. Files that pass move to `data/processed/`; files that fail are quarantined to `data/errors/`.

```powershell
python scripts/Data_validation.py --mlflow
```

This auto-discovers the most recent file in `data/cleaned/`. To validate a specific file:

```powershell
python scripts/Data_validation.py --file data/cleaned/Cincinnati_OH_365d_20260415_140000.csv --mlflow
```

**Output naming:**

- Pass → `data/processed/validated_Cincinnati_OH_365d_20260415_140000.csv`
- Fail → `data/errors/quarantined_Cincinnati_OH_365d_20260415_140000.csv`

### Handling Quarantined Files

If validation fails, you have three options:

**Review what failed:**
```powershell
python scripts/Data_validation.py --review
```

**Fix by removing bad rows (most common):**
```powershell
python scripts/Data_validation.py --fix data/errors/quarantined_Cincinnati_OH_365d_20260415_140000.csv --filter "beds>20" --filter "sold_price<10000" --reason "Removed outliers missed by IQR"
```

**Approve as-is (if failures are acceptable):**
```powershell
python scripts/Data_validation.py --approve data/errors/quarantined_Cincinnati_OH_365d_20260415_140000.csv --reason "Minor nulls in lot_sqft, acceptable for modeling"
```

**Reject (re-scrape needed):**
```powershell
python scripts/Data_validation.py --reject data/errors/quarantined_Cincinnati_OH_365d_20260415_140000.csv --reason "Sold_price distribution too skewed"
```

All review decisions are logged to MLflow with the reviewer name, reason, and audit trail.

---

## Step 3: Feature Engineering

Transform the validated data into model-ready features (imputation, one-hot encoding, derived features).

```powershell
python scripts/feature_engineering.py
```

This auto-discovers the most recent file in `data/processed/`. To use a specific file:

```powershell
python scripts/feature_engineering.py --input data/processed/validated_Cincinnati_OH_365d_20260415_140000.csv
```

**Output:** `data/training/engineered_Cincinnati_OH_365d_20260415_140000.csv`

**What gets logged to MLflow:** Input/output shapes, transformations applied, null counts, feature count delta.

---

## Step 4: Train & Tune

Run Optuna hyperparameter tuning for both Random Forest and XGBoost, then automatically register the best model.

```powershell
python scripts/train_and_tune.py --n_trials 50
```

This auto-discovers the most recent file in `data/training/`. To use a specific file:

```powershell
python scripts/train_and_tune.py --data data/training/engineered_Cincinnati_OH_365d_20260415_140000.csv --n_trials 50
```

**Key flags:**

- `--n_trials` — Optuna trials per model type (50 for quick iteration, 150-200 for a final push)
- `--cv_folds` — cross-validation folds (default 5)
- `--seed` — random seed for reproducibility (default 42)
- `--registered_model_name` — MLflow registry name (default `housing_price_model`)

**What happens during training:**

1. Runs a separate Optuna study for Random Forest and XGBoost
2. Compares the best CV Adjusted R² from each
3. Retrains both final models on the full dataset
4. Registers the session's best model in the MLflow Model Registry
5. Compares against the existing champion — only promotes if the new model scores higher
6. Assigns the `champion` alias to the winner; demoted model gets `previous_champion`

**What gets logged to MLflow:** Every trial's hyperparameters and CV metrics, best trial results, final model artifacts, feature importances, feature schema.

### Understanding the Champion System

The script uses a defend-the-title approach:

- **First run ever:** The best model automatically becomes champion.
- **Subsequent runs:** The new model must beat the existing champion's CV Adj R² to earn the alias. If it doesn't, it's registered as `challenger` and the incumbent keeps the crown.
- **Rollback:** The `previous_champion` alias always points to the last dethroned model, so you can roll back by manually reassigning the alias in MLflow if needed.

---

## Step 5: Deploy

Restart the API container to load the new champion model:

```powershell
docker compose restart api
```

Verify the new model loaded:

```powershell
curl http://localhost:8000/health
```

You should see `"model_loaded": true` and the updated feature count. The Streamlit UI at `http://localhost:8501` will reflect the new model immediately on the Model Dashboard tab.

---

## Quick Reference: Full Pipeline in One Go

```powershell
# Set tracking URI
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"

# Run the full pipeline
python scripts/scrape_and_clean.py --location "Cincinnati, OH" --past_days 365 --mlflow
python scripts/Data_validation.py --mlflow
python scripts/feature_engineering.py
python scripts/train_and_tune.py --n_trials 50

# Deploy the new champion
docker compose restart api
```

---

## Viewing Results

- **Streamlit UI** — `http://localhost:8501` — Predict tab for predictions, Model Dashboard for champion info and version history
- **MLflow UI** — `http://localhost:5000` — Full experiment tracking, run comparison, artifact inspection
- **FastAPI docs** — `http://localhost:8000/docs` — Interactive API documentation and testing

---

## File Naming Convention

Files carry their lineage through the pipeline via prefixed names:

| Stage | Example Filename |
|-------|-----------------|
| Scrape & Clean | `Cincinnati_OH_365d_20260415_140000.csv` |
| Validation Pass | `validated_Cincinnati_OH_365d_20260415_140000.csv` |
| Validation Fail | `quarantined_Cincinnati_OH_365d_20260415_140000.csv` |
| Fixed | `fixed_Cincinnati_OH_365d_20260415_140000.csv` |
| Feature Engineering | `engineered_Cincinnati_OH_365d_20260415_140000.csv` |

Each script auto-discovers the most recent file from the previous stage, so you rarely need to specify paths manually.

---

## Troubleshooting

**API shows `model_loaded: false` after restart:**
Check the API logs for the specific error: `docker logs housing-api --tail 20`. Common causes are a missing Python package (e.g., `xgboost`) or the champion alias not existing yet.

**Prediction returns a 500 error:**
Usually a schema mismatch between the model's expected input types and what the API sends. Check `docker logs housing-api --tail 30` for the full traceback.

**MLflow UI not loading in the Streamlit dashboard iframe:**
MLflow blocks iframe embedding by default. Use the "Open MLflow UI in a new tab" link, or access it directly at `http://localhost:5000`.

**Scripts not logging to MLflow:**
Make sure `$env:MLFLOW_TRACKING_URI = "http://localhost:5000"` is set in your current terminal session. Without it, scripts fall back to a local SQLite file.