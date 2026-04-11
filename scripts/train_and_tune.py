"""
train_model.py – Housing Price Prediction: Random Forest Training & Tuning
==========================================================================
Uses Optuna (TPE-based Bayesian optimization) to tune a Random Forest
regressor with 5-fold cross-validation. Every trial is tracked in MLflow.
After tuning, the best model is retrained on the full dataset and logged
as an MLflow artifact ready for serving via FastAPI / Streamlit.

Usage:
    python train_model.py                          # defaults
    python train_model.py --n_trials 100           # more tuning budget
    python train_model.py --data path/to/data.csv  # custom data path
"""

import argparse
import json
import warnings
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
from mlflow.models.signature import infer_signature


def adjusted_r2(y_true, y_pred, n_features):
    """Compute adjusted R² given the number of features."""
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)

warnings.filterwarnings("ignore", category=FutureWarning)


# ── Feature Schema Builder ───────────────────────────────────────────────
def build_feature_schema(feature_names: list[str]) -> dict:
    """Auto-detect feature types from column naming conventions.

    Strategy: try progressively longer underscore-delimited prefixes to find
    groups of 2+ columns that share the same prefix. For example:
      - city_Cincinnati, city_Madeira  →  group "city" with values
      - zip_code_45208, zip_code_45209 →  group "zip_code" with values
      - sale_season_Spring, sale_season_Summer → group "sale_season" with values
      - sale_month (standalone) → not grouped, classified separately
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
    return schema

# ── CLI ──────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Train & tune RF housing model")
    parser.add_argument("--data", type=str, default="data/housing_engineered.csv",
                        help="Path to the engineered CSV")
    parser.add_argument("--n_trials", type=int, default=50,
                        help="Number of Optuna trials (default: 50)")
    parser.add_argument("--cv_folds", type=int, default=5,
                        help="Number of CV folds (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--experiment_name", type=str,
                        default="housing_price_rf",
                        help="MLflow experiment name")
    parser.add_argument("--model_output", type=str, default="models/best_rf_model.joblib",
                        help="Path to save the final model locally")
    return parser.parse_args()


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
def create_objective(X, y, cv_folds, seed, n_features):
    """Return an Optuna objective function that logs each trial to MLflow.
    
    Optimises adjusted R² (maximise) to penalise complexity from the
    large number of one-hot encoded features.
    """
    # Custom scorer: adjusted R² needs n_features, so we bake it in
    adj_r2_scorer = make_scorer(
        adjusted_r2, greater_is_better=True, n_features=n_features
    )

    def objective(trial: optuna.Trial) -> float:
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

        rf = RandomForestRegressor(**params)

        # 5-fold CV with adjusted R² as the primary tuning metric
        adj_r2_scores = cross_val_score(
            rf, X, y,
            cv=cv_folds,
            scoring=adj_r2_scorer,
            n_jobs=-1,
        )
        mean_adj_r2 = adj_r2_scores.mean()
        std_adj_r2  = adj_r2_scores.std()

        # Also compute RMSE, MAE, and standard R² for richer tracking
        neg_rmse_scores = cross_val_score(
            rf, X, y,
            cv=cv_folds,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        mae_scores = cross_val_score(
            rf, X, y,
            cv=cv_folds,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )
        r2_scores = cross_val_score(
            rf, X, y,
            cv=cv_folds,
            scoring="r2",
            n_jobs=-1,
        )

        mean_rmse = -neg_rmse_scores.mean()
        std_rmse  = neg_rmse_scores.std()
        mean_mae  = -mae_scores.mean()
        mean_r2   = r2_scores.mean()

        # ── Log trial to MLflow ──
        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
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

        print(f"  Trial {trial.number:>3d}  |  Adj R²: {mean_adj_r2:.4f}  "
              f"(±{std_adj_r2:.4f})  |  RMSE: ${mean_rmse:>10,.2f}  |  R²: {mean_r2:.4f}")

        return mean_adj_r2   # Optuna maximises this

    return objective


# ── Final Model Training ─────────────────────────────────────────────────
def train_final_model(X, y, best_params, feature_names, model_output, seed,
                      best_trial_metrics):
    """Retrain on full dataset with best hyperparameters and log to MLflow."""
    print("\n" + "=" * 65)
    print("RETRAINING FINAL MODEL ON FULL DATASET")
    print("=" * 65)

    best_params["random_state"] = seed
    best_params["n_jobs"] = -1

    final_rf = RandomForestRegressor(**best_params)
    final_rf.fit(X, y)

    # ── Model signature for MLflow serving ──
    sample_input = pd.DataFrame(X[:5], columns=feature_names)
    sample_output = final_rf.predict(sample_input)
    signature = infer_signature(sample_input, sample_output)

    # ── Log final model to MLflow ──
    with mlflow.start_run(nested=True, run_name="final_model"):
        mlflow.log_params(best_params)

        # In-sample metrics (sanity check, not evaluation)
        train_preds = final_rf.predict(X)
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
        mlflow.sklearn.log_model(
            sk_model=final_rf,
            artifact_path="random_forest_model",
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
        schema = build_feature_schema(feature_names)
        schema_path = Path("feature_schema.json")
        schema_path.write_text(json.dumps(schema, indent=2))
        mlflow.log_artifact(str(schema_path))
        schema_path.unlink()
        print(f"  Feature schema logged: {len(schema['numeric_features'])} numeric, "
              f"{len(schema['categorical_groups'])} categorical groups, "
              f"{len(schema['binary_features'])} binary")

        # Feature importance top 15
        importances = final_rf.feature_importances_
        feat_imp = sorted(zip(feature_names, importances),
                          key=lambda x: x[1], reverse=True)
        imp_df = pd.DataFrame(feat_imp[:15], columns=["feature", "importance"])
        imp_path = Path("feature_importances.csv")
        imp_df.to_csv(imp_path, index=False)
        mlflow.log_artifact(str(imp_path))
        imp_path.unlink()

        mlflow.set_tag("model_type", "final_production")
        final_run_id = mlflow.active_run().info.run_id

    # ── Save locally with joblib ──
    output_path = Path(model_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_rf, output_path)

    # Also save feature schema alongside the model for local use
    meta_path = output_path.with_suffix(".meta.json")
    schema = build_feature_schema(feature_names)
    schema["n_features"] = len(feature_names)
    schema["training_rows"] = X.shape[0]
    meta_path.write_text(json.dumps(schema, indent=2))

    print(f"\nFinal model saved to: {output_path}")
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

    return final_rf, final_run_id


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    X, y, feature_names = load_data(args.data)

    # ── MLflow setup ──
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name="optuna_tuning_session") as parent_run:
        mlflow.set_tag("stage", "hyperparameter_tuning")
        mlflow.log_params({
            "n_trials":    args.n_trials,
            "cv_folds":    args.cv_folds,
            "seed":        args.seed,
            "data_path":   args.data,
            "n_rows":      X.shape[0],
            "n_features":  X.shape[1],
        })

        # ── Optuna study ──
        print(f"\nStarting Optuna tuning: {args.n_trials} trials, "
              f"{args.cv_folds}-fold CV\n" + "-" * 65)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=args.seed),
            study_name="rf_housing_tuning",
        )
        study.optimize(
            create_objective(X, y, args.cv_folds, args.seed, X.shape[1]),
            n_trials=args.n_trials,
            show_progress_bar=True,
        )

        # ── Log best trial results to parent run ──
        best = study.best_trial
        mlflow.log_metrics({
            "best_cv_adj_r2": round(best.value, 4),
            "best_trial_num": best.number,
        })
        mlflow.log_params({f"best_{k}": v for k, v in best.params.items()})

        print(f"\n{'=' * 65}")
        print(f"BEST TRIAL: #{best.number}")
        print(f"  CV Adj R²: {best.value:.4f}")
        print(f"  Params:    {json.dumps(best.params, indent=4)}")

        # ── Train final model ──
        # Retrieve the best trial's CV metrics from its MLflow run
        best_trial_run = mlflow.search_runs(
            filter_string=f"tags.trial_number = '{best.number}'",
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

        final_model, final_run_id = train_final_model(
            X, y, best.params.copy(), feature_names,
            args.model_output, args.seed, best_trial_metrics,
        )

        # Log the parent run ID for easy lookup
        parent_id = parent_run.info.run_id
        print(f"\nParent MLflow run ID: {parent_id}")

    print("\nDone! To view results:")
    print("  mlflow ui --port 5000")
    print(f"  Then open http://localhost:5000")


if __name__ == "__main__":
    main()
