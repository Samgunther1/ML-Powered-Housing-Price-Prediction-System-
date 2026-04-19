"""
evaluate_business_metrics.py — Business metrics for the champion model
======================================================================
Computes business-oriented evaluation metrics on the engineered training
dataset using 5-fold cross-validated predictions. Each row is predicted
by a model that never saw it during training, so the numbers reflect
real-world generalization rather than in-sample fit.

Metrics computed:
  - Median absolute % error (MedAPE)
  - Mean absolute % error (MAPE)
  - Share of predictions within ±5%, ±10%, ±20% of the actual sale price
  - Median absolute error in dollars
  - Mean absolute error in dollars
  - RMSE in dollars

Optionally logs all metrics back to the champion's MLflow run with a
`business_` prefix so they show up alongside the training metrics.

Usage:
    # Print metrics using the most recent engineered CSV
    python scripts/evaluate_business_metrics.py

    # Log metrics back to the champion MLflow run
    python scripts/evaluate_business_metrics.py --log_to_mlflow

    # Use a specific engineered CSV
    python scripts/evaluate_business_metrics.py --data data/training/engineered_xxx.csv

    # Save a per-row predictions CSV for further analysis
    python scripts/evaluate_business_metrics.py --save_predictions predictions.csv
"""

import argparse
import json
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_val_predict


# ── Configuration ────────────────────────────────────────────────────────
MODEL_NAME = os.getenv("MODEL_NAME", "housing_price_model")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
TRAINING_DIR = Path("data/training")
TARGET = "sold_price"


# ── Helpers ──────────────────────────────────────────────────────────────
def find_latest_training_file() -> Path:
    """Return the most recent engineered_*.csv in data/training/."""
    csvs = sorted(TRAINING_DIR.glob("engineered_*.csv"),
                  key=lambda p: p.stat().st_mtime)
    if not csvs:
        raise FileNotFoundError(f"No engineered_*.csv files in {TRAINING_DIR}")
    return csvs[-1]


def load_champion_model(mv):
    """Load the raw sklearn/xgboost estimator (not pyfunc) so it can be
    cloned and re-fit during cross-validation."""
    model_type = mv.tags.get("model_type", "rf")
    uri = mv.source  # e.g., models:/m-<uuid>
    if model_type == "xgb":
        model = mlflow.xgboost.load_model(uri)
    else:
        model = mlflow.sklearn.load_model(uri)
    return model, model_type


def load_feature_columns(client: MlflowClient, mv) -> list:
    """Pull feature_schema.json from the champion run.
    Tries local mlruns/ first (fast path), then the tracking server.
    """
    run = client.get_run(mv.run_id)
    exp_id = run.info.experiment_id
    local = Path("mlruns") / exp_id / mv.run_id / "artifacts" / "feature_schema.json"
    if local.exists():
        with open(local) as f:
            return json.load(f)["feature_columns"]
    path = client.download_artifacts(mv.run_id, "feature_schema.json")
    with open(path) as f:
        return json.load(f)["feature_columns"]


def compute_business_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute business-oriented regression metrics."""
    abs_err = np.abs(y_pred - y_true)
    pct_err = abs_err / y_true * 100
    sq_err = (y_pred - y_true) ** 2
    return {
        "median_abs_pct_error":     float(np.median(pct_err)),
        "mean_abs_pct_error":       float(np.mean(pct_err)),
        "pct_within_5":             float(np.mean(pct_err <= 5) * 100),
        "pct_within_10":            float(np.mean(pct_err <= 10) * 100),
        "pct_within_20":            float(np.mean(pct_err <= 20) * 100),
        "median_abs_error_dollars": float(np.median(abs_err)),
        "mean_abs_error_dollars":   float(np.mean(abs_err)),
        "rmse_dollars":             float(np.sqrt(np.mean(sq_err))),
        "n_predictions":            int(len(y_true)),
    }


def print_metrics(m: dict, model_type: str, version: int, n_folds: int) -> None:
    display = "XGBoost" if model_type == "xgb" else "Random Forest"
    print("\n" + "─" * 58)
    print(f"  Business Metrics — {display} v{version} ({n_folds}-fold CV)")
    print("─" * 58)
    print(f"  Median absolute % error:    {m['median_abs_pct_error']:>6.2f}%")
    print(f"  Mean absolute % error:      {m['mean_abs_pct_error']:>6.2f}%")
    print(f"  Within ±5%  of actual:      {m['pct_within_5']:>6.1f}% of predictions")
    print(f"  Within ±10% of actual:      {m['pct_within_10']:>6.1f}% of predictions")
    print(f"  Within ±20% of actual:      {m['pct_within_20']:>6.1f}% of predictions")
    print(f"  Median dollar error:        ${m['median_abs_error_dollars']:>10,.0f}")
    print(f"  Mean dollar error (MAE):    ${m['mean_abs_error_dollars']:>10,.0f}")
    print(f"  RMSE:                       ${m['rmse_dollars']:>10,.0f}")
    print(f"  N predictions:               {m['n_predictions']}")
    print("─" * 58 + "\n")


# ── Main ─────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Compute business metrics for the champion model")
    ap.add_argument("--data", type=Path, default=None,
                    help="Engineered CSV path (default: most recent in data/training/)")
    ap.add_argument("--cv_folds", type=int, default=5,
                    help="Number of CV folds (default: 5, matches training)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for CV shuffle (default: 42)")
    ap.add_argument("--log_to_mlflow", action="store_true",
                    help="Log metrics back to the champion MLflow run")
    ap.add_argument("--save_predictions", type=Path, default=None,
                    help="Optional CSV path to save per-row predictions vs actuals")
    args = ap.parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    print(f"Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"Loading champion: {MODEL_NAME}@{MODEL_ALIAS}")
    mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    model, model_type = load_champion_model(mv)
    print(f"  v{mv.version}  type={model_type}  run_id={mv.run_id}")

    feature_columns = load_feature_columns(client, mv)

    data_path = args.data or find_latest_training_file()
    print(f"Loading data: {data_path}")
    df = pd.read_csv(data_path)

    missing = set(feature_columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"Data file is missing {len(missing)} feature columns expected by "
            f"the champion (first few: {sorted(missing)[:5]})."
        )
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in data file.")

    X = df[feature_columns].copy()
    y = df[TARGET].values
    print(f"  Shape: X={X.shape}, y={y.shape}")

    print(f"Running {args.cv_folds}-fold cross-validated predictions "
          f"(this retrains {args.cv_folds}× — may take a minute)...")
    cv = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    fresh = clone(model)
    y_pred = cross_val_predict(fresh, X, y, cv=cv, n_jobs=-1)
    print(f"  Done. {len(y_pred)} predictions generated.")

    metrics = compute_business_metrics(y, y_pred)
    print_metrics(metrics, model_type, mv.version, args.cv_folds)

    if args.save_predictions:
        out = pd.DataFrame({
            "actual": y,
            "predicted": y_pred,
            "abs_error": np.abs(y_pred - y),
            "abs_pct_error": np.abs(y_pred - y) / y * 100,
        })
        out.to_csv(args.save_predictions, index=False)
        print(f"Predictions saved to: {args.save_predictions}")

    if args.log_to_mlflow:
        print("Logging metrics to champion run...")
        with mlflow.start_run(run_id=mv.run_id):
            for k, v in metrics.items():
                mlflow.log_metric(f"business_{k}", v)
            mlflow.set_tag("business_metrics_evaluated", "true")
            mlflow.set_tag("business_metrics_cv_folds", str(args.cv_folds))
        print(f"  Logged to run {mv.run_id}")


if __name__ == "__main__":
    main()
