"""
train_and_tune.py – Housing Price Prediction: RF + XGBoost Training & Tuning
=============================================================================
Uses Optuna (TPE-based Bayesian optimization) to tune both a Random Forest
and an XGBoost regressor with 5-fold cross-validation. Every trial is
tracked in MLflow. After tuning, the champion model (best CV Adj R²) is
retrained on the full dataset, registered in the MLflow Model Registry,
and automatically assigned the "champion" alias.

Usage:
    python train_and_tune.py                          # defaults
    python train_and_tune.py --n_trials 100           # more tuning budget
    python train_and_tune.py --data path/to/data.csv  # custom data path
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
from mlflow.models.signature import infer_signature
from xgboost import XGBRegressor


def adjusted_r2(y_true, y_pred, n_features):
    """Compute adjusted R² given the number of features."""
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)

warnings.filterwarnings("ignore", category=FutureWarning)


# ── Feature Schema Builder ───────────────────────────────────────────────
def build_feature_schema(feature_names: list[str], X=None, y=None) -> dict:
    """Auto-detect feature types from column naming conventions and compute
    validity domain bounds from the training data.

    Parameters
    ----------
    feature_names : list[str]
        Column names of the feature matrix.
    X : array-like, optional
        Training feature matrix. If provided, computes min/max bounds and
        percentiles for each numeric feature to define the validity domain.
    y : array-like, optional
        Training target values. If provided, stores target range for
        prediction sanity checks.
    """
    from collections import defaultdict

    # ── Step 1: Find all candidate prefix groups ──
    # For each column, generate all possible prefixes by splitting on "_"
    # e.g. "zip_code_45208" → candidates: "zip", "zip_code"
    prefix_candidates = defaultdict(set)  # prefix → set of (column, value)
    for col in feature_names:
        parts = col.split("_")
        for i in range(1, len(parts)):
            prefix = "_".join(parts[:i])
            value = "_".join(parts[i:])
            prefix_candidates[prefix].add((col, value))

    # ── Step 2: Pick the LONGEST prefix that has 2+ members ──
    # Sort by prefix length descending so longer prefixes win
    assigned = set()
    categorical_groups = {}  # prefix → sorted list of values
    categorical_columns = {}  # prefix → list of column names

    for prefix in sorted(prefix_candidates.keys(), key=len, reverse=True):
        members = prefix_candidates[prefix]
        # Only form a group if 2+ columns AND none already assigned
        unassigned = [(col, val) for col, val in members if col not in assigned]
        if len(unassigned) >= 2:
            categorical_groups[prefix] = sorted([val for _, val in unassigned])
            categorical_columns[prefix] = sorted([col for col, _ in unassigned])
            for col, _ in unassigned:
                assigned.add(col)

    # ── Step 3: Classify remaining columns ──
    remaining = [col for col in feature_names if col not in assigned]

    # Known binary patterns (0/1 flags)
    binary_features = []
    numeric_features = []
    for col in remaining:
        # Heuristics for binary: starts with "has_", "is_", "new_", or known flags
        if (col.startswith("has_") or col.startswith("is_") or
                col in ("new_construction",)):
            binary_features.append(col)
        else:
            numeric_features.append(col)

    # ── Also rescue has_* columns that got grouped categorically ──
    # e.g. has_garage + has_hoa form a "has" group, but they're really
    # independent binary flags that should be auto-derived
    rescue_prefixes = []
    for prefix, values in list(categorical_groups.items()):
        cols_in_group = categorical_columns[prefix]
        if all(c.startswith("has_") or c.startswith("is_") for c in cols_in_group):
            # These are binary flags, not a real categorical
            binary_features.extend(cols_in_group)
            for c in cols_in_group:
                assigned.discard(c)
            rescue_prefixes.append(prefix)
    for p in rescue_prefixes:
        del categorical_groups[p]
        del categorical_columns[p]

    # ── Step 4: Detect auto-derived features ──
    auto_derived = {}

    # has_X flags: derive from corresponding numeric feature
    for col in binary_features:
        if col.startswith("has_"):
            target = col[4:]  # e.g. "has_hoa" → "hoa"
            # Look for a numeric feature containing this word
            for num_col in numeric_features:
                if target in num_col:
                    auto_derived[col] = {
                        "derived_from": num_col,
                        "rule": "greater_than_zero",
                    }
                    break

    # sale_month: derive from current date
    if "sale_month" in numeric_features:
        auto_derived["sale_month"] = {
            "derived_from": "current_date",
            "rule": "current_month",
        }

    # sale_season group: derive from current date
    if "sale_season" in categorical_groups:
        auto_derived["sale_season"] = {
            "derived_from": "current_date",
            "rule": "current_season",
        }

    # Split binary into user-input vs auto-derived
    user_binary = [f for f in binary_features if f not in auto_derived]
    auto_columns = [col for col in feature_names if col in auto_derived or
                    any(col in categorical_columns.get(grp, [])
                        for grp in auto_derived if grp in categorical_groups)]

    # ── Step 5: Build defaults for numeric features ──
    numeric_defaults = {col: 0 for col in numeric_features}
    default_overrides = {
        "beds": 3, "full_baths": 2, "half_baths": 0, "sqft": 1500,
        "year_built": 1960, "lot_sqft": 8000, "stories": 2, "hoa_fee": 0,
        "parking_garage": 1, "sale_month": 1,
    }
    for col, val in default_overrides.items():
        if col in numeric_defaults:
            numeric_defaults[col] = val

    # ── Step 6: Identify user-facing categorical groups ──
    # (exclude auto-derived groups like sale_season)
    user_categorical_groups = {
        k: v for k, v in categorical_groups.items()
        if k not in auto_derived
    }
    auto_categorical_groups = {
        k: v for k, v in categorical_groups.items()
        if k in auto_derived
    }

    schema = {
        "feature_columns": feature_names,
        "numeric_features": numeric_features,
        "binary_features": user_binary,
        "categorical_groups": user_categorical_groups,
        "auto_derived": auto_derived,
        "auto_categorical_groups": auto_categorical_groups,
        "numeric_defaults": numeric_defaults,
        "target": "sold_price",
    }

    # ── Step 7: Compute validity domain from training data ──
    if X is not None:
        df_train = pd.DataFrame(X, columns=feature_names) if not isinstance(X, pd.DataFrame) else X

        # Numeric bounds: min, max, 1st and 99th percentile
        validity_bounds = {}
        for col in numeric_features:
            if col in df_train.columns:
                series = pd.to_numeric(df_train[col], errors="coerce").dropna()
                if len(series) > 0:
                    validity_bounds[col] = {
                        "min": round(float(series.min()), 2),
                        "max": round(float(series.max()), 2),
                        "p01": round(float(series.quantile(0.01)), 2),
                        "p99": round(float(series.quantile(0.99)), 2),
                        "median": round(float(series.median()), 2),
                    }

        schema["validity_bounds"] = validity_bounds

        # Compute the max year_built in training data for new_construction logic
        if "year_built" in df_train.columns:
            max_year = int(df_train["year_built"].max())
            schema["validity_rules"] = {
                "new_construction_min_year": max_year - 2,
            }

    if y is not None:
        y_series = pd.Series(y)
        schema["target_range"] = {
            "min": round(float(y_series.min()), 2),
            "max": round(float(y_series.max()), 2),
            "median": round(float(y_series.median()), 2),
        }

    return schema

# ── CLI ──────────────────────────────────────────────────────────────────
TRAINING_DIR = Path("data/training")


def find_latest_training_file(directory: Path = TRAINING_DIR) -> Path:
    """Find the most recently modified CSV in data/training/."""
    csv_files = sorted(directory.glob("*.csv"), key=lambda f: f.stat().st_mtime)
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {directory}. "
            f"Run feature_engineering.py first."
        )
    latest = csv_files[-1]
    print(f"Auto-discovered training file: {latest}")
    return latest


def parse_args():
    parser = argparse.ArgumentParser(description="Train & tune RF + XGBoost housing models")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to the engineered CSV (default: most recent in data/training/)")
    parser.add_argument("--models", type=str, default="both",
                        choices=["rf", "xgb", "both"],
                        help="Which model(s) to train: 'rf', 'xgb', or 'both' (default: both)")
    parser.add_argument("--n_trials", type=int, default=50,
                        help="Number of Optuna trials per model (default: 50)")
    parser.add_argument("--cv_folds", type=int, default=5,
                        help="Number of CV folds (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--experiment_name", type=str,
                        default="housing_price_prediction",
                        help="MLflow experiment name")
    parser.add_argument("--registered_model_name", type=str,
                        default="housing_price_model",
                        help="MLflow Model Registry name for champion model")
    parser.add_argument("--model_output", type=str, default="models/best_model.joblib",
                        help="Path to save the final model locally")
    args = parser.parse_args()

    # Auto-discover if no --data provided
    if args.data is None:
        args.data = str(find_latest_training_file())

    return args


# ── Data Loading ─────────────────────────────────────────────────────────
def load_data(path: str):
    """Load CSV and split into features / target."""
    df = pd.read_csv(path)
    target_col = "sold_price"

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    y = df[target_col].values
    X = df.drop(columns=[target_col])
    feature_names = X.columns.tolist()

    print(f"Loaded {X.shape[0]:,} rows × {X.shape[1]} features")
    print(f"Target range: ${y.min():,.0f} – ${y.max():,.0f}  "
          f"(mean ${y.mean():,.0f})")
    return X, y, feature_names


# ── Optuna Objective ─────────────────────────────────────────────────────
def create_objective(X, y, cv_folds, seed, n_features, model_type="rf"):
    """Return an Optuna objective function that logs each trial to MLflow.

    Optimises adjusted R² (maximise) to penalise complexity from the
    large number of one-hot encoded features.

    Parameters
    ----------
    model_type : str
        "rf" for Random Forest, "xgb" for XGBoost.
    """
    # Custom scorer: adjusted R² needs n_features, so we bake it in
    adj_r2_scorer = make_scorer(
        adjusted_r2, greater_is_better=True, n_features=n_features
    )

    def objective(trial: optuna.Trial) -> float:
        if model_type == "rf":
            params = {
                "n_estimators":      trial.suggest_int("n_estimators", 100, 800, step=50),
                "max_depth":         trial.suggest_int("max_depth", 5, 50),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 15),
                "max_features":      trial.suggest_categorical("max_features",
                                                               ["sqrt", "log2", 0.3, 0.5, 0.7, 1.0]),
                "random_state":      seed,
                "n_jobs":            -1,
            }
            model = RandomForestRegressor(**params)

        elif model_type == "xgb":
            params = {
                "n_estimators":   trial.suggest_int("n_estimators", 100, 1000, step=50),
                "max_depth":      trial.suggest_int("max_depth", 3, 12),
                "learning_rate":  trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample":      trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_alpha":      trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda":     trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "gamma":          trial.suggest_float("gamma", 1e-8, 5.0, log=True),
                "random_state":   seed,
                "n_jobs":         -1,
                "verbosity":      0,
            }
            model = XGBRegressor(**params)

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        label = "RF" if model_type == "rf" else "XGB"

        # 5-fold CV with adjusted R² as the primary tuning metric
        adj_r2_scores = cross_val_score(
            model, X, y,
            cv=cv_folds,
            scoring=adj_r2_scorer,
            n_jobs=-1,
        )
        mean_adj_r2 = adj_r2_scores.mean()
        std_adj_r2  = adj_r2_scores.std()

        # Also compute RMSE, MAE, and standard R² for richer tracking
        neg_rmse_scores = cross_val_score(
            model, X, y,
            cv=cv_folds,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        mae_scores = cross_val_score(
            model, X, y,
            cv=cv_folds,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )
        r2_scores = cross_val_score(
            model, X, y,
            cv=cv_folds,
            scoring="r2",
            n_jobs=-1,
        )

        mean_rmse = -neg_rmse_scores.mean()
        std_rmse  = neg_rmse_scores.std()
        mean_mae  = -mae_scores.mean()
        mean_r2   = r2_scores.mean()

        # ── Log trial to MLflow ──
        with mlflow.start_run(nested=True, run_name=f"{label}_trial_{trial.number}"):
            mlflow.log_params(params)
            mlflow.log_metrics({
                "cv_adj_r2_mean": round(mean_adj_r2, 4),
                "cv_adj_r2_std":  round(std_adj_r2, 4),
                "cv_r2_mean":     round(mean_r2, 4),
                "cv_rmse_mean":   round(mean_rmse, 2),
                "cv_rmse_std":    round(std_rmse, 2),
                "cv_mae_mean":    round(mean_mae, 2),
            })
            mlflow.set_tag("trial_number", trial.number)
            mlflow.set_tag("model_type", model_type)

        print(f"  {label} Trial {trial.number:>3d}  |  Adj R²: {mean_adj_r2:.4f}  "
              f"(±{std_adj_r2:.4f})  |  RMSE: ${mean_rmse:>10,.2f}  |  R²: {mean_r2:.4f}")

        return mean_adj_r2   # Optuna maximises this

    return objective


# ── Final Model Training ─────────────────────────────────────────────────
def train_final_model(X, y, best_params, feature_names, model_output, seed,
                      best_trial_metrics, model_type="rf"):
    """Retrain on full dataset with best hyperparameters and log to MLflow.

    Returns the trained model, the MLflow run ID, and the logged model URI.
    """
    label = "Random Forest" if model_type == "rf" else "XGBoost"
    print("\n" + "=" * 65)
    print(f"RETRAINING FINAL {label.upper()} MODEL ON FULL DATASET")
    print("=" * 65)

    if model_type == "rf":
        best_params["random_state"] = seed
        best_params["n_jobs"] = -1
        final_model = RandomForestRegressor(**best_params)
    else:
        best_params["random_state"] = seed
        best_params["n_jobs"] = -1
        best_params["verbosity"] = 0
        final_model = XGBRegressor(**best_params)

    final_model.fit(X, y)

    # ── Model signature for MLflow serving ──
    sample_input = pd.DataFrame(X[:5], columns=feature_names)
    sample_output = final_model.predict(sample_input)
    signature = infer_signature(sample_input, sample_output)

    # ── Log final model to MLflow ──
    artifact_path = f"{model_type}_model"
    log_func = mlflow.sklearn.log_model if model_type == "rf" else mlflow.xgboost.log_model

    with mlflow.start_run(nested=True, run_name=f"final_{model_type}_model") as run:
        mlflow.log_params(best_params)

        # In-sample metrics (sanity check, not evaluation)
        train_preds = final_model.predict(X)
        train_rmse = np.sqrt(np.mean((y - train_preds) ** 2))
        train_mae  = np.mean(np.abs(y - train_preds))
        n_features = X.shape[1] if hasattr(X, 'shape') else len(feature_names)
        train_r2     = r2_score(y, train_preds)
        train_adj_r2 = adjusted_r2(y, train_preds, n_features)

        mlflow.log_metrics({
            "train_rmse":   round(train_rmse, 2),
            "train_mae":    round(train_mae, 2),
            "train_r2":     round(train_r2, 4),
            "train_adj_r2": round(train_adj_r2, 4),
        })

        # Carry forward the best trial's CV metrics for easy reference
        mlflow.log_metrics(best_trial_metrics)

        # Log the model with signature and feature list
        if model_type == "rf":
            log_func(
                sk_model=final_model,
                artifact_path=artifact_path,
                signature=signature,
                input_example=sample_input.iloc[:1],
            )
        else:
            log_func(
                xgb_model=final_model,
                artifact_path=artifact_path,
                signature=signature,
                input_example=sample_input.iloc[:1],
            )

        # Log feature names as artifact for downstream use
        feature_artifact = {"feature_names": feature_names}
        feature_path = Path("feature_names.json")
        feature_path.write_text(json.dumps(feature_artifact, indent=2))
        mlflow.log_artifact(str(feature_path))
        feature_path.unlink()

        # ── Build & log feature schema for dynamic API/UI ──
        schema = build_feature_schema(feature_names, X=X, y=y)
        schema_path = Path("feature_schema.json")
        schema_path.write_text(json.dumps(schema, indent=2))
        mlflow.log_artifact(str(schema_path))
        schema_path.unlink()
        print(f"  Feature schema logged: {len(schema['numeric_features'])} numeric, "
              f"{len(schema['categorical_groups'])} categorical groups, "
              f"{len(schema['binary_features'])} binary")

        # Feature importance top 15
        importances = final_model.feature_importances_
        feat_imp = sorted(zip(feature_names, importances),
                          key=lambda x: x[1], reverse=True)
        imp_df = pd.DataFrame(feat_imp[:15], columns=["feature", "importance"])
        imp_path = Path("feature_importances.csv")
        imp_df.to_csv(imp_path, index=False)
        mlflow.log_artifact(str(imp_path))
        imp_path.unlink()

        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("stage", "final_production")
        final_run_id = run.info.run_id
        model_uri = f"runs:/{final_run_id}/{artifact_path}"

    # ── Save locally with joblib ──
    output_path = Path(model_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, output_path)

    # Also save feature schema alongside the model for local use
    meta_path = output_path.with_suffix(".meta.json")
    schema = build_feature_schema(feature_names, X=X, y=y)
    schema["n_features"] = len(feature_names)
    schema["training_rows"] = X.shape[0]
    schema["model_type"] = model_type
    meta_path.write_text(json.dumps(schema, indent=2))

    print(f"\n{label} model saved to: {output_path}")
    print(f"Feature metadata:     {meta_path}")
    print(f"MLflow run ID:        {final_run_id}")

    print(f"\nIn-sample metrics (sanity check):")
    print(f"  RMSE:    ${train_rmse:>10,.2f}")
    print(f"  MAE:     ${train_mae:>10,.2f}")
    print(f"  R²:      {train_r2:.4f}")
    print(f"  Adj R²:  {train_adj_r2:.4f}")

    print(f"\nTop 10 features by importance:")
    for feat, imp in feat_imp[:10]:
        print(f"  {feat:<30s}  {imp:.4f}")

    return final_model, final_run_id, model_uri


# ── Helper: Run Optuna Study for a Single Model Type ────────────────────
def run_optuna_study(X, y, model_type, n_trials, cv_folds, seed, n_features):
    """Run an Optuna study for a given model type and return the study."""
    label = "Random Forest" if model_type == "rf" else "XGBoost"
    study_name = f"{model_type}_housing_tuning"

    print(f"\n{'=' * 65}")
    print(f"  TUNING {label.upper()}: {n_trials} trials, {cv_folds}-fold CV")
    print(f"{'=' * 65}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        study_name=study_name,
    )
    study.optimize(
        create_objective(X, y, cv_folds, seed, n_features, model_type=model_type),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_trial
    print(f"\n  BEST {label.upper()} TRIAL: #{best.number}")
    print(f"  CV Adj R²: {best.value:.4f}")
    print(f"  Params:    {json.dumps(best.params, indent=4)}")

    return study


def retrieve_best_trial_cv_metrics(best_trial_number, model_type):
    """Look up the CV metrics logged in MLflow for the best Optuna trial."""
    best_trial_run = mlflow.search_runs(
        filter_string=(
            f"tags.trial_number = '{best_trial_number}' "
            f"and tags.model_type = '{model_type}'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    best_trial_metrics = {}
    for col in best_trial_run.columns:
        if col.startswith("metrics.cv_"):
            metric_name = col.replace("metrics.", "best_")
            val = best_trial_run[col].iloc[0]
            if pd.notna(val):
                best_trial_metrics[metric_name] = round(val, 4)
    return best_trial_metrics


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    X, y, feature_names = load_data(args.data)

    # Determine which model types to train
    if args.models == "both":
        model_types = ["rf", "xgb"]
    else:
        model_types = [args.models]

    model_labels = {"rf": "Random Forest", "xgb": "XGBoost"}
    run_label = " + ".join(model_labels[m] for m in model_types)

    # ── MLflow setup ──
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=f"tuning_{'+'.join(model_types)}") as parent_run:
        mlflow.set_tag("stage", "hyperparameter_tuning")
        mlflow.set_tag("model_types", ",".join(model_types))
        mlflow.log_params({
            "n_trials_per_model": args.n_trials,
            "cv_folds":           args.cv_folds,
            "seed":               args.seed,
            "data_path":          args.data,
            "n_rows":             X.shape[0],
            "n_features":         X.shape[1],
            "model_types":        ",".join(model_types),
        })

        n_features = X.shape[1]

        # ── Run Optuna for selected model types ──
        studies = {}
        for mtype in model_types:
            studies[mtype] = run_optuna_study(
                X, y, mtype, args.n_trials, args.cv_folds, args.seed, n_features
            )

        # ── Log best results to parent run ──
        for mtype, study in studies.items():
            best = study.best_trial
            mlflow.log_metrics({
                f"{mtype}_best_cv_adj_r2": round(best.value, 4),
                f"{mtype}_best_trial_num": best.number,
            })

        # ── Determine this session's best model ──
        if len(studies) == 1:
            champion_type = model_types[0]
        else:
            champion_type = max(studies, key=lambda m: studies[m].best_value)

        champion_study = studies[champion_type]
        champion_best = champion_study.best_trial
        champion_label = model_labels[champion_type]

        print(f"\n{'=' * 65}")
        print(f"  SESSION BEST: {champion_label}  (CV Adj R² = {champion_best.value:.4f})")
        for mtype, study in studies.items():
            print(f"  {model_labels[mtype]} best: {study.best_value:.4f}")
        print(f"{'=' * 65}")

        mlflow.set_tag("champion_model_type", champion_type)

        # ── Train final models (logged as nested runs) ──
        results = {}
        for mtype, study in studies.items():
            best = study.best_trial
            cv_metrics = retrieve_best_trial_cv_metrics(best.number, mtype)

            suffix = f"_{mtype}"
            base = Path(args.model_output)
            output_path = str(base.parent / f"{base.stem}{suffix}.joblib")

            model, run_id, model_uri = train_final_model(
                X, y, best.params.copy(), feature_names,
                output_path, args.seed, cv_metrics, model_type=mtype,
            )
            results[mtype] = {
                "model": model,
                "run_id": run_id,
                "model_uri": model_uri,
                "cv_adj_r2": best.value,
            }

        # ── Register & conditionally promote to champion ────────────
        champion_uri = results[champion_type]["model_uri"]
        new_score = round(champion_best.value, 4)

        client = mlflow.tracking.MlflowClient()

        # Check if there's an existing champion to defend its title
        existing_champion_score = None
        try:
            existing_mv = client.get_model_version_by_alias(
                name=args.registered_model_name,
                alias="champion",
            )
            existing_champion_score_str = existing_mv.tags.get("cv_adj_r2")
            if existing_champion_score_str is not None:
                existing_champion_score = float(existing_champion_score_str)
                existing_type = existing_mv.tags.get("model_type", "unknown")
                print(f"\nExisting champion: v{existing_mv.version} "
                      f"({existing_type}, CV Adj R² = {existing_champion_score:.4f})")
        except Exception:
            # No registered model or no champion alias yet — first run
            print(f"\nNo existing champion found — this will be the first.")

        # Always register the new version for auditability
        print(f"\nRegistering this run's best model ({champion_label}) in "
              f"'{args.registered_model_name}' ...")
        mv = mlflow.register_model(
            model_uri=champion_uri,
            name=args.registered_model_name,
        )
        print(f"  Registered version: {mv.version}")

        # Tag the new version regardless of promotion
        client.set_model_version_tag(
            name=args.registered_model_name,
            version=mv.version,
            key="model_type",
            value=champion_type,
        )
        client.set_model_version_tag(
            name=args.registered_model_name,
            version=mv.version,
            key="cv_adj_r2",
            value=str(new_score),
        )

        # Only promote if the new model actually beats the incumbent
        if existing_champion_score is not None and new_score <= existing_champion_score:
            print(f"\n  New model CV Adj R² ({new_score:.4f}) does NOT beat "
                  f"existing champion ({existing_champion_score:.4f}).")
            print(f"  Champion alias stays on v{existing_mv.version}. "
                  f"New model registered as v{mv.version} (challenger).")
            client.set_registered_model_alias(
                name=args.registered_model_name,
                alias="challenger",
                version=mv.version,
            )
        else:
            client.set_registered_model_alias(
                name=args.registered_model_name,
                alias="champion",
                version=mv.version,
            )
            if existing_champion_score is not None:
                print(f"\n  NEW CHAMPION! CV Adj R² improved: "
                      f"{existing_champion_score:.4f} → {new_score:.4f}")
                # Reassign old champion as the previous alias for rollback
                client.set_registered_model_alias(
                    name=args.registered_model_name,
                    alias="previous_champion",
                    version=existing_mv.version,
                )
                print(f"  Previous champion (v{existing_mv.version}) aliased "
                      f"as 'previous_champion'")
            else:
                print(f"\n  First champion set! "
                      f"v{mv.version} ({champion_label}, CV Adj R² = {new_score:.4f})")
            print(f"  Alias 'champion' → v{mv.version}")

        parent_id = parent_run.info.run_id
        print(f"\nParent MLflow run ID: {parent_id}")

    print("\nDone! To view results:")
    print("  mlflow ui --port 5000")
    print(f"  Then open http://localhost:5000")
    print(f"\nTo load the champion model:")
    print(f"  import mlflow")
    print(f"  model = mlflow.pyfunc.load_model("
          f"'models:/{args.registered_model_name}@champion')")


if __name__ == "__main__":
    main()
