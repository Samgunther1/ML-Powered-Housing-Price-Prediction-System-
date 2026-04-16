"""
api.py – FastAPI backend for Housing Price Prediction
======================================================
Loads the MLflow-registered champion model (RF or XGBoost) AND its
feature_schema.json artifact using mlflow.pyfunc so any model flavor
works seamlessly. All feature encoding is driven by the schema —
nothing is hardcoded. If you retrain with different features, just
register the new model as champion and restart the API.

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Optional

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ── Configuration ────────────────────────────────────────────────────────
MODEL_NAME = os.getenv("MODEL_NAME", "housing_price_model")
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
    warnings: list[str] = []


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
    validity_bounds: dict[str, Any] = {}
    validity_rules: dict[str, Any] = {}


# ── Global State ─────────────────────────────────────────────────────────
model = None
schema = None


def _find_champion_version(client: mlflow.MlflowClient):
    """Find the champion model version using the search API (compatible with MLflow 3.x)."""
    from mlflow.exceptions import MlflowException

    # Try the direct alias lookup first (works in newer MLflow versions)
    try:
        return client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    except (MlflowException, Exception):
        pass

    # Fallback: search registered models and find the alias manually
    results = client.search_registered_models(filter_string=f"name='{MODEL_NAME}'")
    if not results:
        raise ValueError(f"Registered model '{MODEL_NAME}' not found")

    reg_model = results[0]
    for alias in getattr(reg_model, "aliases", {}).items() if hasattr(reg_model, "aliases") else []:
        alias_name, alias_version = alias
        if alias_name == MODEL_ALIAS:
            return client.get_model_version(MODEL_NAME, alias_version)

    # Last resort: parse aliases from the latest versions
    for mv in reg_model.latest_versions:
        if hasattr(mv, "aliases") and MODEL_ALIAS in (mv.aliases or []):
            return mv

    raise ValueError(f"No '{MODEL_ALIAS}' alias found for model '{MODEL_NAME}'")


def load_schema_from_mlflow() -> dict:
    """Download feature_schema.json from the champion model's run artifacts."""
    client = mlflow.MlflowClient()
    mv = _find_champion_version(client)
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
        model = mlflow.pyfunc.load_model(model_uri)
        # Log which version and type we loaded
        client = mlflow.MlflowClient()
        mv = _find_champion_version(client)
        tags = mv.tags if isinstance(mv.tags, dict) else {t.key: t.value for t in getattr(mv, 'tags', [])}
        model_type = tags.get("model_type", "unknown")
        cv_score = tags.get("cv_adj_r2", "N/A")
        print(f"Model loaded: v{mv.version} ({model_type}, CV Adj R² = {cv_score})")
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
    description="Predict Cincinnati-area housing prices using the MLflow champion model "
                "(Random Forest or XGBoost). Feature schema is loaded dynamically from MLflow.",
    version="2.1.0",
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

    # Cast columns to match the model's expected input schema exactly.
    # The pyfunc model exposes its signature — use it to determine which
    # columns need int64 vs float64.
    try:
        input_schema = model.metadata.get_input_schema()
        if input_schema:
            for col_spec in input_schema.inputs:
                col_name = col_spec.name
                if col_name in df.columns:
                    if col_spec.type.name in ("long", "integer"):
                        df[col_name] = df[col_name].astype(np.int64)
                    elif col_spec.type.name in ("double", "float"):
                        df[col_name] = df[col_name].astype(np.float64)
    except Exception:
        # Fallback: cast everything that looks like an integer column
        for col in df.columns:
            if df[col].dropna().apply(float.is_integer).all():
                df[col] = df[col].astype(np.int64)

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
        validity_bounds=schema.get("validity_bounds", {}),
        validity_rules=schema.get("validity_rules", {}),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    """Generate a housing price prediction from raw property features."""
    if model is None:
        raise HTTPException(503, "Model not loaded. Check MLflow connection.")
    if schema is None:
        raise HTTPException(503, "Feature schema not loaded.")

    warnings = []
    bounds = schema.get("validity_bounds", {})
    rules = schema.get("validity_rules", {})

    # ── Validity domain: numeric range checks ──
    for feat, value in req.numeric.items():
        if feat in bounds:
            b = bounds[feat]
            if value < b["min"] or value > b["max"]:
                raise HTTPException(
                    422,
                    f"'{feat.replace('_', ' ').title()}' value {value} is outside the "
                    f"training data range [{b['min']}, {b['max']}]. "
                    f"The model cannot reliably predict outside this range."
                )
            # Warn if in the extreme tails (outside 1st-99th percentile)
            if value < b["p01"] or value > b["p99"]:
                warnings.append(
                    f"{feat.replace('_', ' ').title()} ({value}) is in the extreme "
                    f"tail of the training data (1st–99th percentile: "
                    f"{b['p01']}–{b['p99']}). Prediction may be less reliable."
                )

    # ── Validity domain: new_construction + year_built logic ──
    if req.binary.get("new_construction", False):
        min_year = rules.get("new_construction_min_year")
        year_input = req.numeric.get("year_built")
        if min_year and year_input and year_input < min_year:
            raise HTTPException(
                422,
                f"New construction is checked but Year Built ({int(year_input)}) "
                f"is before {min_year}. New construction properties in the training "
                f"data were built in {min_year} or later."
            )

    # ── Validate categorical inputs ──
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
        warnings=warnings,
    )
