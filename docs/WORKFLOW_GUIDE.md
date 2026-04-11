# Housing Price Predictor — Full Workflow Guide

This guide walks through every step of training a new model and deploying it for predictions. Follow this any time you retrain — whether you changed features, added data, or just want to tune with more trials.

---

## Architecture Overview

The system has three components, each running in its own Docker container:

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   Streamlit UI   │────▶│   FastAPI API     │────▶│   MLflow Server  │
│  localhost:8501   │     │  localhost:8000    │     │  localhost:5000   │
│                  │     │                  │     │                  │
│  Builds form     │     │  Loads champion   │     │  Stores models,  │
│  dynamically     │     │  model + schema   │     │  artifacts,      │
│  from /schema    │     │  from MLflow      │     │  metrics         │
└──────────────────┘     └──────────────────┘     └──────────────────┘
```

**Data flow for a prediction:**
1. User fills out the form in Streamlit (beds, baths, city, etc.)
2. Streamlit sends the raw inputs to FastAPI
3. FastAPI encodes the inputs into the 107-feature vector using the feature schema
4. FastAPI runs the model and returns the predicted price
5. Streamlit displays the result

---

## Step 1: Start the Docker Containers

Open PowerShell, navigate to your repo, and start the containers:

```powershell
cd C:\Users\Samg1\BANA_7075_FINAL\ML-Powered-Housing-Price-Prediction-System-
docker compose up --build -d
```

Wait about 15-20 seconds for all three containers to become healthy. You can verify with:

```powershell
docker compose ps
```

All three should show "healthy" or "running" status.

---

## Step 2: Train and Tune the Model

In the **same PowerShell window** (or a new one), point your local MLflow client at the Dockerized MLflow server and run the training script:

```powershell
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
python scripts/train_and_tune.py --data data/training/housing_engineered.csv --n_trials 50
```

**What this does:**
- Loads `housing_engineered.csv` and splits it into features (X) and target (sold_price)
- Runs 50 Optuna trials, each testing different Random Forest hyperparameters
- Each trial performs 5-fold cross-validation and is scored on **adjusted R²**
- Every trial's params and metrics are logged to MLflow as nested runs
- After tuning, the best hyperparameters retrain a final model on all 8,012 rows
- The final model, feature schema, feature importances, and feature names are logged as MLflow artifacts
- A local copy is saved to `models/best_rf_model.joblib`

**Timing:** ~30-40 minutes for 50 trials. Use `--n_trials 5` for a quick test.

**What gets logged to MLflow per trial:**
- `cv_adj_r2_mean` — the primary optimization metric
- `cv_r2_mean` — standard R² for comparison
- `cv_rmse_mean` — root mean squared error in dollars
- `cv_mae_mean` — mean absolute error in dollars
- All hyperparameters (n_estimators, max_depth, etc.)

**What gets logged for the final model:**
- Training metrics (train_rmse, train_r2, train_adj_r2, etc.)
- Best trial's CV metrics carried forward (best_cv_adj_r2_mean, etc.)
- `random_forest_model/` — the serialized model for serving
- `feature_schema.json` — the dynamic schema that drives the API and UI
- `feature_names.json` — ordered list of all feature columns
- `feature_importances.csv` — top 15 features by importance

---

## Step 3: Register the Model in MLflow

Open **http://localhost:5000** in your browser.

1. Click the **housing_price_rf** experiment (left sidebar)
2. Click the **▶ expand arrow** next to the latest `optuna_tuning_session` run
3. Click on the **final_model** nested run
4. Click the **Artifacts** tab
5. Verify you see: `feature_schema.json`, `feature_names.json`, `feature_importances.csv`, and `random_forest_model/`
6. Click on the **random_forest_model** folder
7. Click **Register Model**
8. If this is your first time: enter the name `housing_price_rf` and click Register
9. If a registered model already exists: select `housing_price_rf` from the dropdown

---

## Step 4: Set the Champion Alias

1. Click **Model registry** in the left sidebar
2. Click on **housing_price_rf**
3. Click on the **latest version** (e.g., Version 2)
4. Under **Aliases**, click **Add** → type `champion` → save

If an older version already has the `champion` alias, MLflow automatically moves it to the new version.

---

## Step 5: Restart the API

The API loads the model on startup, so you need to restart it to pick up the new champion:

```powershell
docker compose restart api
```

Wait 10 seconds, then verify:

```powershell
curl http://localhost:8000/health
```

You should see:
```json
{"status": "healthy", "model_loaded": true, "schema_loaded": true, ...}
```

---

## Step 6: Make Predictions

- **Streamlit UI**: Open **http://localhost:8501** in your browser
- **API directly**: Open **http://localhost:8000/docs** for the Swagger UI
- **Programmatic**: Send a POST request to `http://localhost:8000/predict`

Example API call with PowerShell:
```powershell
$body = @{
    numeric = @{
        beds = 3
        full_baths = 2
        half_baths = 1
        sqft = 1800
        year_built = 1975
        lot_sqft = 10000
        stories = 2
        hoa_fee = 0
        parking_garage = 2
    }
    categorical = @{
        style = "SINGLE_FAMILY"
        city = "Anderson Township"
        zip_code = "45230"
        county = "Hamilton"
    }
    binary = @{
        new_construction = $false
    }
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -Body $body -ContentType "application/json"
```

---

## Retraining After Changing Features

The system is modular. If you change your feature engineering (add columns, remove columns, rename them), just:

1. Update `scripts/feature_engineering.py` and regenerate `housing_engineered.csv`
2. Run the training script (Step 2) — the schema builder auto-detects the new features
3. Register and alias (Steps 3-4)
4. Restart the API (Step 5)

**No code changes needed in the API or Streamlit.** The schema drives everything dynamically.

---

## Useful Commands Reference

```powershell
# ── Docker ──
docker compose up --build -d      # Start all containers (background)
docker compose down                # Stop all containers
docker compose down -v             # Stop and wipe all data (fresh start)
docker compose restart api         # Restart just the API
docker compose logs -f             # Follow logs from all containers
docker compose logs -f api         # Follow just the API logs
docker compose ps                  # Check container status

# ── Training ──
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"    # Always set this first!

# Quick test (5 trials, ~5 min)
python scripts/train_and_tune.py --data data/training/housing_engineered.csv --n_trials 5

# Full tuning (50 trials, ~30-40 min)
python scripts/train_and_tune.py --data data/training/housing_engineered.csv --n_trials 50

# Extended tuning (100 trials, ~60-80 min)
python scripts/train_and_tune.py --data data/training/housing_engineered.csv --n_trials 100

# ── Debugging ──
curl http://localhost:8000/health   # Check API status
curl http://localhost:8000/schema   # View the loaded feature schema
```

---

## URLs

| Service | URL | Purpose |
|---------|-----|---------|
| Streamlit UI | http://localhost:8501 | Make predictions |
| FastAPI Docs | http://localhost:8000/docs | API documentation / testing |
| API Health | http://localhost:8000/health | Check model status |
| MLflow UI | http://localhost:5000 | View experiments, register models |

---

## Troubleshooting

**"model_loaded: false"** → The champion alias isn't set or the API needs a restart. Register the model, set the alias, and run `docker compose restart api`.

**"schema_loaded: false"** → The model was trained with an older script that doesn't generate `feature_schema.json`. Retrain with the current `train_and_tune.py`.

**"Could not connect to API"** → The API container isn't running or still starting. Check `docker compose ps` and `docker compose logs api`.

**Training errors about MLflow endpoints** → Version mismatch. Make sure `docker/requirements-api.txt` and `docker/Dockerfile.mlflow` use the same MLflow version as your local environment (`pip show mlflow`).

**Port conflicts** → Another process is using 5000, 8000, or 8501. Either stop it or change the port mapping in `docker-compose.yml` (e.g., `"5001:5000"`).

**Need a completely fresh start** → `docker compose down -v` then `docker compose up --build -d`. This wipes all MLflow data, so you'll need to retrain and re-register.
