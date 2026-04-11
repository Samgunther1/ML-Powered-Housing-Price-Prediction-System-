"""
api.py – FastAPI backend for Housing Price Prediction
======================================================
Loads the MLflow-registered model with alias 'champion' AND its
feature_schema.json artifact. All feature encoding is driven by the
schema — nothing is hardcoded. If you retrain with different features,
just register the new model as champion and restart the API.

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Optional

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ── Configuration ────────────────────────────────────────────────────────
MODEL_NAME = os.getenv("MODEL_NAME", "housing_price_rf")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")


# ── Request / Response Models ────────────────────────────────────────────
class PredictionRequest(BaseModel):
    """Dynamic input — numeric features as key/value pairs, categoricals as
    group_name: chosen_value, binaries as key: true/false.

    Example:
        {
            "numeric": {"beds": 3, "full_baths": 2, "sqft": 1800, ...},
            "categorical": {"city": "Anderson Township", "zip_code": "45230", ...},
            "binary": {"new_construction": false}
        }
    """
    numeric: dict[str, float] = {}
    categorical: dict[str, Optional[str]] = {}
    binary: dict[str, bool] = {}


class PredictionResponse(BaseModel):
    predicted_price: float
    predicted_price_formatted: str
    model_name: str
    model_alias: str
    input_summary: dict


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    schema_loaded: bool
    model_name: str
    model_alias: str
    feature_count: int


class SchemaResponse(BaseModel):
    numeric_features: list[str]
    numeric_defaults: dict[str, Any]
    binary_features: list[str]
    categorical_groups: dict[str, list[str]]
    auto_derived: dict[str, Any]


# ── Global State ─────────────────────────────────────────────────────────
model = None
schema = None


def load_schema_from_mlflow() -> dict:
    """Download feature_schema.json from the champion model's run artifacts."""
    client = mlflow.MlflowClient()

    # Get the model version for the alias
    mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    run_id = mv.run_id

    # Download the schema artifact
    artifact_path = client.download_artifacts(run_id, "feature_schema.json")
    with open(artifact_path, "r") as f:
        return json.load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the MLflow champion model and its feature schema on startup."""
    global model, schema
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Load model
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    print(f"Loading model: {model_uri}")
    try:
        model = mlflow.sklearn.load_model(model_uri)
        print("Model loaded successfully")
    except Exception as e:
        print(f"WARNING: Could not load model — {e}")

    # Load schema
    try:
        schema = load_schema_from_mlflow()
        print(f"Schema loaded: {len(schema['feature_columns'])} features, "
              f"{len(schema['numeric_features'])} numeric, "
              f"{len(schema['categorical_groups'])} categorical groups")
    except Exception as e:
        print(f"WARNING: Could not load feature schema — {e}")

    yield
    print("Shutting down...")


# ── App ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Housing Price Predictor",
    description="Predict Cincinnati-area housing prices using a tuned Random Forest model. "
                "Feature schema is loaded dynamically from MLflow.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Feature Encoding (fully schema-driven) ───────────────────────────────
def encode_features(req: PredictionRequest) -> pd.DataFrame:
    """Convert user input into the full feature vector using the loaded schema."""
    feature_columns = schema["feature_columns"]
    auto_derived = schema.get("auto_derived", {})
    auto_categorical = schema.get("auto_categorical_groups", {})
    numeric_defaults = schema.get("numeric_defaults", {})

    # Start with all zeros
    row = {col: 0 for col in feature_columns}

    # ── 1. Numeric features ──
    # Apply defaults first, then user overrides
    for col in schema["numeric_features"]:
        if col in auto_derived:
            continue  # will be filled in step 4
        row[col] = req.numeric.get(col, numeric_defaults.get(col, 0))

    # ── 2. Binary features (user-facing) ──
    for col in schema["binary_features"]:
        row[col] = int(req.binary.get(col, False))

    # ── 3. Categorical features (one-hot encoding) ──
    for group_name, valid_values in schema["categorical_groups"].items():
        chosen = req.categorical.get(group_name)
        if chosen and chosen in valid_values:
            col_name = f"{group_name}_{chosen}"
            if col_name in row:
                row[col_name] = 1
        # If not chosen or invalid, all columns in this group stay 0

    # ── 4. Auto-derived features ──
    now = datetime.now()

    for col, rule in auto_derived.items():
        if rule["rule"] == "greater_than_zero":
            source = rule["derived_from"]
            source_val = req.numeric.get(source, numeric_defaults.get(source, 0))
            derived_col = col  # e.g. "has_hoa"
            if derived_col in row:
                row[derived_col] = 1 if source_val > 0 else 0

        elif rule["rule"] == "current_month":
            if col in row:
                row[col] = now.month

        elif rule["rule"] == "current_season":
            # col is a group name like "sale_season"
            month_to_season = {
                12: "Winter", 1: "Winter", 2: "Winter",
                3: "Spring", 4: "Spring", 5: "Spring",
                6: "Summer", 7: "Summer", 8: "Summer",
                9: "Fall", 10: "Fall", 11: "Fall",
            }
            season = month_to_season[now.month]
            # Set the appropriate one-hot column
            if col in auto_categorical:
                for val in auto_categorical[col]:
                    onehot_col = f"{col}_{val}"
                    if onehot_col in row:
                        row[onehot_col] = 1 if val == season else 0

    df = pd.DataFrame([row], columns=feature_columns)
    return df


# ── Endpoints ────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check API health, model, and schema status."""
    return HealthResponse(
        status="healthy" if (model and schema) else "degraded",
        model_loaded=model is not None,
        schema_loaded=schema is not None,
        model_name=MODEL_NAME,
        model_alias=MODEL_ALIAS,
        feature_count=len(schema["feature_columns"]) if schema else 0,
    )


@app.get("/schema", response_model=SchemaResponse)
def get_schema():
    """Return the feature schema so the frontend can build its UI dynamically."""
    if schema is None:
        raise HTTPException(503, "Feature schema not loaded.")
    return SchemaResponse(
        numeric_features=schema["numeric_features"],
        numeric_defaults=schema.get("numeric_defaults", {}),
        binary_features=schema["binary_features"],
        categorical_groups=schema["categorical_groups"],
        auto_derived=schema.get("auto_derived", {}),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    """Generate a housing price prediction from raw property features."""
    if model is None:
        raise HTTPException(503, "Model not loaded. Check MLflow connection.")
    if schema is None:
        raise HTTPException(503, "Feature schema not loaded.")

    # Validate categorical inputs
    for group_name, chosen in req.categorical.items():
        if chosen is None:
            continue
        valid = schema["categorical_groups"].get(group_name)
        if valid is None:
            raise HTTPException(400, f"Unknown categorical group '{group_name}'. "
                                     f"Valid groups: {list(schema['categorical_groups'].keys())}")
        if chosen not in valid:
            raise HTTPException(400, f"Invalid value '{chosen}' for '{group_name}'. "
                                     f"Options: {valid}")

    # Encode and predict
    features_df = encode_features(req)
    prediction = model.predict(features_df)[0]
    prediction = float(np.clip(prediction, 0, None))

    # Build a readable summary
    summary = {}
    summary.update(req.numeric)
    summary.update(req.categorical)
    summary.update({k: v for k, v in req.binary.items()})

    return PredictionResponse(
        predicted_price=round(prediction, 2),
        predicted_price_formatted=f"${prediction:,.0f}",
        model_name=MODEL_NAME,
        model_alias=MODEL_ALIAS,
        input_summary=summary,
    )
