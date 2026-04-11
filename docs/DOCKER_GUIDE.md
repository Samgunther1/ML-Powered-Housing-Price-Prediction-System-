# Docker Deployment Guide — Housing Price Predictor

## Overview

This setup runs three containers:

| Container | What it does | Local URL |
|-----------|-------------|-----------|
| **mlflow-server** | Stores models, artifacts, and the feature schema | http://localhost:5000 |
| **housing-api** | FastAPI backend — loads the champion model, serves predictions | http://localhost:8000 |
| **housing-ui** | Streamlit frontend — the user-facing prediction form | http://localhost:8501 |

---

## Prerequisites

### 1. Install Docker Desktop
- Download from: https://www.docker.com/products/docker-desktop/
- Run the installer and follow the prompts
- After install, **restart your computer**
- Open Docker Desktop and make sure it says "Docker Desktop is running" (green icon in system tray)

### 2. Verify Docker is working
Open PowerShell and run:
```powershell
docker --version
docker compose version
```
You should see version numbers for both. If not, Docker Desktop may need to finish starting up.

---

## Project File Setup

Your repo should have this structure after adding the new files:

```
your-repo/
├── docker-compose.yml          ← NEW (repo root)
├── docker/                     ← NEW folder
│   ├── Dockerfile.mlflow
│   ├── Dockerfile.api
│   ├── Dockerfile.streamlit
│   ├── requirements-api.txt
│   └── requirements-streamlit.txt
├── scripts/
│   ├── Api.py                  ← UPDATED (dynamic schema version)
│   ├── streamlit_app.py        ← UPDATED (dynamic schema version)
│   ├── train_and_tune.py       ← UPDATED (generates feature_schema.json)
│   ├── scrape_and_clean.py
│   ├── Data_validation.py
│   └── feature_engineering.py
├── data/
│   └── training/
│       └── housing_engineered.csv
├── models/
│   ├── best_rf_model.joblib
│   └── best_rf_model.meta.json
└── ...
```

---

## Step-by-Step Deployment

### Step 1: Build and start the containers

Open PowerShell, navigate to your repo root, and run:

```powershell
cd C:\path\to\your\repo
docker compose up --build
```

This will:
- Build three Docker images (takes 2-3 min the first time)
- Start MLflow, then the API, then Streamlit (in order, each waits for the previous one)

You'll see logs from all three containers streaming in your terminal. Wait until you see:
```
housing-ui  | You can now view your Streamlit app in your browser.
```

> **Tip:** Add `-d` to run in the background: `docker compose up --build -d`
> Then use `docker compose logs -f` to follow the logs.

### Step 2: Register your model in the Dockerized MLflow

The containers use their own MLflow instance (inside Docker), which is separate from the MLflow you've been using locally. You need to register your trained model there.

Open a **new PowerShell window** and run:

```powershell
cd C:\path\to\your\repo

# Point your local MLflow client at the Dockerized MLflow server
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"

# Run the training script — it will log everything to the Docker MLflow
python scripts/train_and_tune.py --data data/training/housing_engineered.csv --n_trials 50
```

This trains the model and logs it (along with `feature_schema.json`) to the Dockerized MLflow at http://localhost:5000.

### Step 3: Register the model and set the champion alias

Open the MLflow UI at **http://localhost:5000** in your browser:

1. Click on the **housing_price_rf** experiment
2. Expand the latest **optuna_tuning_session** run
3. Click on the **final_model** nested run
4. Scroll down to **Artifacts** → click on **random_forest_model**
5. Click **Register Model**
6. Name it: `housing_price_rf` → click **Register**
7. Go to the **Models** tab (top nav)
8. Click on **housing_price_rf**
9. Click on **Version 1**
10. Under **Aliases**, click **Add** → type `champion` → save

### Step 4: Restart the API to pick up the registered model

```powershell
docker compose restart api
```

The API container will restart, load the champion model and its schema, and be ready.

### Step 5: Use the app

- **Streamlit UI**: http://localhost:8501
- **FastAPI docs** (Swagger): http://localhost:8000/docs
- **MLflow UI**: http://localhost:5000

---

## Common Commands

```powershell
# Start everything
docker compose up --build

# Start in background
docker compose up --build -d

# View logs
docker compose logs -f              # all containers
docker compose logs -f api          # just the API

# Stop everything
docker compose down

# Stop and remove volumes (reset MLflow data)
docker compose down -v

# Restart a single container (e.g., after re-registering a model)
docker compose restart api

# Rebuild just one container
docker compose build api
docker compose up -d api
```

---

## Updating After Retraining

When you change features and retrain:

1. Run `train_and_tune.py` with `$env:MLFLOW_TRACKING_URI = "http://localhost:5000"`
2. Register the new model version in MLflow UI
3. Move the `champion` alias to the new version
4. Run `docker compose restart api`

The API reloads the new model + schema, and the Streamlit UI adapts automatically.

---

## Troubleshooting

### "Model not loaded" error in the API
- The model hasn't been registered with the `champion` alias yet
- Follow Steps 2-4 above

### "Could not connect to the prediction API" in Streamlit
- The API container might still be starting — wait 10 seconds and refresh
- Check `docker compose logs api` for errors

### Port already in use
- Something else is using port 5000, 8000, or 8501
- Either stop the other process, or change the ports in `docker-compose.yml`:
  ```yaml
  ports:
    - "5001:5000"  # maps to localhost:5001 instead
  ```

### Docker is slow on Windows
- Make sure WSL 2 is enabled (Docker Desktop → Settings → General → "Use the WSL 2 based engine")
- Allocate more RAM: Docker Desktop → Settings → Resources → increase memory

### Need to start fresh
```powershell
docker compose down -v    # removes containers AND stored data
docker compose up --build # rebuild everything
```
