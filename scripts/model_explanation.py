"""
model_explanation.py – SHAP feature importance for the champion model
======================================================================
Downloads the MLflow champion model (RF or XGBoost), computes SHAP values
on a sample of the engineered training data, aggregates one-hot encoded
columns back to their source categorical groups (e.g. `city_Cincinnati`,
`city_Norwood`, ... → `city`), and logs PNG plots back to the champion
run as artifacts under `explanations/`.

The FastAPI backend serves these PNGs to the Streamlit UI for display on
the Model Dashboard. Re-run any time a new model is promoted to champion.

Usage:
    # Explain the current champion using the most recent engineered CSV
    python scripts/model_explanation.py

    # Use a specific data file
    python scripts/model_explanation.py --data data/training/engineered_20260418.csv

    # Increase SHAP sample size (default: 500)
    python scripts/model_explanation.py --sample 1000
"""

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import shap
from mlflow import MlflowClient


# ── Configuration ────────────────────────────────────────────────────────
MODEL_NAME = os.getenv("MODEL_NAME", "housing_price_model")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
TRAINING_DIR = Path("data/training")


# ── Helpers ──────────────────────────────────────────────────────────────
def find_latest_training_file() -> Path:
    """Return the most recent engineered_*.csv in data/training/."""
    csvs = sorted(TRAINING_DIR.glob("engineered_*.csv"), key=lambda p: p.stat().st_mtime)
    if not csvs:
        raise FileNotFoundError(f"No engineered_*.csv files found in {TRAINING_DIR}")
    return csvs[-1]


def _extract_model_artifact_path(mv_source: str) -> str:
    """Parse the artifact sub-path out of a `runs:/...` source URI, if that's
    what we were given. Returns None if the source isn't a runs:/ URI.
    """
    if mv_source and mv_source.startswith("runs:/"):
        parts = mv_source.split("/", 2)
        if len(parts) >= 3 and parts[2]:
            return parts[2]
    return None


def _find_model_dir_in_run(client: MlflowClient, run_id: str) -> str | None:
    """Scan run artifacts for a subdirectory containing an MLmodel file.
    Returns the artifact path if found, else None.
    """
    try:
        for a in client.list_artifacts(run_id):
            if a.is_dir:
                for sub in client.list_artifacts(run_id, a.path):
                    if sub.path.endswith("MLmodel"):
                        return a.path
    except Exception:
        pass
    return None


def _download_model_artifacts(client: MlflowClient, mv) -> str:
    """Resolve the champion model to a local directory.

    Strategies tried in order:
      1. `mlflow.artifacts.download_artifacts(artifact_uri=mv.source)` — handles
         MLflow 3.x logged-model URIs (`models:/m-<uuid>`) and other schemes.
      2. If `mv.source` is a `runs:/` URI, parse out the artifact sub-path and
         use `client.download_artifacts` on the run (classic MLflow 2.x layout).
      3. List the run's artifacts and look for any subdirectory containing an
         `MLmodel` file — catches cases where the model was logged under a
         non-standard name.
    """
    run_id = mv.run_id
    errors = []

    # Strategy 1: direct source URI (works for MLflow 3.x logged models)
    try:
        return mlflow.artifacts.download_artifacts(
            artifact_uri=mv.source,
            tracking_uri=MLFLOW_TRACKING_URI,
        )
    except Exception as e:
        errors.append(f"  [1] artifact_uri='{mv.source}' → {type(e).__name__}: {e}")

    # Strategy 2: runs:/ parsing (MLflow 2.x layout)
    subpath = _extract_model_artifact_path(mv.source)
    if subpath:
        try:
            return client.download_artifacts(run_id, subpath)
        except Exception as e:
            errors.append(f"  [2] run artifact_path='{subpath}' → {type(e).__name__}: {e}")

    # Strategy 3: scan run artifacts for an MLmodel file
    scanned_path = _find_model_dir_in_run(client, run_id)
    if scanned_path:
        try:
            return client.download_artifacts(run_id, scanned_path)
        except Exception as e:
            errors.append(f"  [3] run artifact_path='{scanned_path}' → {type(e).__name__}: {e}")
    else:
        errors.append("  [3] No MLmodel-bearing directory found in run artifacts.")

    raise RuntimeError(
        "Could not download model artifacts. This usually happens when the MLflow "
        "server stores model files at a container-local path that isn't exposed via "
        "the HTTP artifact proxy. Attempts:\n"
        + "\n".join(errors)
        + "\n\nFixes to try:\n"
          "  • Run this script inside the housing-api container (which has access "
          "to the mlflow-data volume):\n"
          "      docker compose exec api python scripts/model_explanation.py\n"
          "    (requires adding 'shap' + 'matplotlib' to requirements-api.txt and "
          "rebuilding the API image)\n"
          "  • Or enable artifact serving on the MLflow server by adding "
          "`--serve-artifacts` to its startup command."
    )


def _locate_schema(client: MlflowClient, run_id: str,
                   schema_override: Optional[Path] = None) -> dict:
    """Resolve feature_schema.json via (1) an explicit override path,
    (2) the host's local mlruns directory using the run's experiment_id,
    (3) the tracking server's HTTP artifact API. Each strategy's error
    is captured and surfaced if all three fail.
    """
    errors = []

    # Strategy 1: user-provided path
    if schema_override is not None:
        if schema_override.exists():
            with open(schema_override) as f:
                return json.load(f)
        errors.append(f"  [1] --schema path does not exist: {schema_override}")

    # Strategy 2: local mlruns directory (common when model artifacts are
    # already synced to the host, as they clearly are for this model)
    try:
        run = client.get_run(run_id)
        exp_id = run.info.experiment_id
        local_candidate = Path("mlruns") / exp_id / run_id / "artifacts" / "feature_schema.json"
        if local_candidate.exists():
            print(f"  Found schema locally: {local_candidate}")
            with open(local_candidate) as f:
                return json.load(f)
        errors.append(f"  [2] Not in local mlruns: {local_candidate}")
    except Exception as e:
        errors.append(f"  [2] Local lookup failed: {type(e).__name__}: {e}")

    # Strategy 3: HTTP via tracking server
    try:
        path = client.download_artifacts(run_id, "feature_schema.json")
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        errors.append(f"  [3] HTTP download failed: {type(e).__name__}: {e}")

    raise RuntimeError(
        "Could not load feature_schema.json. Attempts:\n"
        + "\n".join(errors)
        + "\n\nFixes:\n"
          "  • Copy the file from the mlflow-server container and pass it via --schema:\n"
          f"      docker cp mlflow-server:/mlflow/mlartifacts/<exp>/{run_id}/artifacts/feature_schema.json .\n"
          "      python scripts/model_explanation.py --schema feature_schema.json\n"
          "  • Or run this script inside the housing-api container (see README)."
    )


def load_champion(client: MlflowClient, schema_override: Optional[Path] = None):
    """Load the raw sklearn/xgboost model (not pyfunc) so SHAP can introspect
    the tree structure. Also returns the model version and its feature schema.
    """
    mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    run_id = mv.run_id
    model_type = mv.tags.get("model_type", "rf")
    print(f"  mv.source:    {mv.source}")
    print(f"  mv.run_id:    {run_id}")
    print(f"  model_type:   {model_type}")

    local_model_dir = _download_model_artifacts(client, mv)
    print(f"  Downloaded to: {local_model_dir}")

    if model_type == "xgb":
        model = mlflow.xgboost.load_model(local_model_dir)
    else:
        model = mlflow.sklearn.load_model(local_model_dir)

    schema = _locate_schema(client, run_id, schema_override)

    return model, mv, schema, run_id


def build_column_to_group_map(schema: dict) -> dict:
    """Map each feature column to its logical group.

    Categorical one-hot columns map to their group name (e.g.
    `city_Cincinnati` → `city`). Numeric and binary columns map to
    themselves. Longer group prefixes are matched first so that
    `zip_code_*` isn't accidentally grouped under a hypothetical `zip`.
    """
    feature_columns = schema["feature_columns"]
    categorical_groups = schema["categorical_groups"]

    col_to_group: dict[str, str] = {}
    for group_name in sorted(categorical_groups.keys(), key=len, reverse=True):
        prefix = f"{group_name}_"
        for col in feature_columns:
            if col not in col_to_group and col.startswith(prefix):
                col_to_group[col] = group_name
    for col in feature_columns:
        col_to_group.setdefault(col, col)
    return col_to_group


def collapse_shap_to_groups(
    shap_values: np.ndarray,
    feature_columns: list,
    col_to_group: dict,
) -> tuple[np.ndarray, list]:
    """Sum signed SHAP values across one-hot columns within each group.

    Summing signed values is the correct aggregation per SHAP's additivity
    property: the group's contribution to a prediction equals the sum of
    its members' contributions.
    """
    group_order: list[str] = []
    seen: set[str] = set()
    for col in feature_columns:
        g = col_to_group[col]
        if g not in seen:
            group_order.append(g)
            seen.add(g)

    idx = {g: i for i, g in enumerate(group_order)}
    collapsed = np.zeros((shap_values.shape[0], len(group_order)))
    for col_i, col in enumerate(feature_columns):
        collapsed[:, idx[col_to_group[col]]] += shap_values[:, col_i]
    return collapsed, group_order


# ── Plotting ─────────────────────────────────────────────────────────────
def plot_category_bar(
    collapsed_shap: np.ndarray,
    group_names: list,
    top_n: int,
    title: str,
    out_path: Path,
) -> None:
    """Horizontal bar chart of mean |summed SHAP| per group."""
    mean_abs = np.mean(np.abs(collapsed_shap), axis=0)
    order = np.argsort(mean_abs)[::-1][:top_n]
    labels = [group_names[i].replace("_", " ").title() for i in order]
    values = mean_abs[order]

    fig, ax = plt.subplots(figsize=(9, max(5, 0.4 * len(order))))
    y = np.arange(len(order))
    ax.barh(y, values, color="#047857", edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|  (impact on predicted price, $)")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_category_beeswarm(
    collapsed_shap: np.ndarray,
    group_names: list,
    top_n: int,
    title: str,
    out_path: Path,
    seed: int = 42,
) -> None:
    """Beeswarm-style scatter of per-row group SHAP values (signed)."""
    rng = np.random.default_rng(seed)
    mean_abs = np.mean(np.abs(collapsed_shap), axis=0)
    order = np.argsort(mean_abs)[::-1][:top_n][::-1]  # reversed so biggest ends on top

    fig, ax = plt.subplots(figsize=(10, max(5, 0.45 * len(order))))
    for plot_y, feat_idx in enumerate(order):
        vals = collapsed_shap[:, feat_idx]
        jitter = rng.uniform(-0.32, 0.32, size=len(vals))
        ax.scatter(
            vals,
            np.full_like(vals, plot_y, dtype=float) + jitter,
            alpha=0.35, s=10, color="#047857", edgecolors="none",
        )

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([group_names[i].replace("_", " ").title() for i in order])
    ax.axvline(0, color="#666", linestyle="--", linewidth=0.8)
    ax.set_xlabel("SHAP value  (impact on predicted price, $)")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute category-level SHAP feature importance for the champion model"
    )
    parser.add_argument("--data", type=Path, default=None,
                        help="Engineered CSV path (default: most recent in data/training/)")
    parser.add_argument("--schema", type=Path, default=None,
                        help="Local path to feature_schema.json (use if auto-download fails)")
    parser.add_argument("--sample", type=int, default=500,
                        help="Rows to sample for SHAP (default: 500)")
    parser.add_argument("--top_n", type=int, default=20,
                        help="Top N groups to plot (default: 20)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling and jitter (default: 42)")
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    print(f"Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"Loading champion: {MODEL_NAME}@{MODEL_ALIAS}")
    model, mv, schema, run_id = load_champion(client, schema_override=args.schema)
    model_type = mv.tags.get("model_type", "rf")
    print(f"  v{mv.version}  type={model_type}  run_id={run_id}")

    data_path = args.data or find_latest_training_file()
    print(f"Loading data: {data_path}")
    df = pd.read_csv(data_path)
    feature_columns = schema["feature_columns"]
    missing = set(feature_columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"Data file is missing {len(missing)} feature columns expected by "
            f"the champion (first few: {sorted(missing)[:5]}). "
            f"Re-run feature_engineering.py or pass --data with a compatible file."
        )
    X = df[feature_columns].copy()
    print(f"  Dataset shape: {X.shape}")

    sample_size = min(args.sample, len(X))
    X_sample = X.sample(n=sample_size, random_state=args.seed).reset_index(drop=True)
    print(f"  SHAP sample: {sample_size} rows")

    print("Computing SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):  # defensive — shouldn't happen for regression
        shap_values = shap_values[0]
    print(f"  SHAP values shape: {shap_values.shape}")

    col_to_group = build_column_to_group_map(schema)
    collapsed_shap, group_names = collapse_shap_to_groups(
        shap_values, feature_columns, col_to_group
    )
    print(f"  Collapsed to {len(group_names)} groups "
          f"(from {len(feature_columns)} columns)")

    display_type = "Random Forest" if model_type == "rf" else "XGBoost"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        bar_path = tmp / "shap_bar_category.png"
        bee_path = tmp / "shap_beeswarm_category.png"

        plot_category_bar(
            collapsed_shap, group_names, args.top_n,
            title=f"Category-Level Feature Importance — {display_type} v{mv.version}",
            out_path=bar_path,
        )
        plot_category_beeswarm(
            collapsed_shap, group_names, args.top_n,
            title=f"Category-Level SHAP Distribution — {display_type} v{mv.version}",
            out_path=bee_path,
            seed=args.seed,
        )
        print(f"  Plots written: {bar_path.name}, {bee_path.name}")

        # Log to the champion run (resume existing run instead of creating new one)
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifact(str(bar_path), artifact_path="explanations")
            mlflow.log_artifact(str(bee_path), artifact_path="explanations")
            mlflow.set_tag("shap_explained", "true")
            mlflow.set_tag("shap_sample_size", str(sample_size))

    print(f"\nDone. Artifacts logged to run {run_id} under 'explanations/'.")
    print("To refresh the API with the new artifacts:")
    print("  docker compose restart api")


if __name__ == "__main__":
    main()
