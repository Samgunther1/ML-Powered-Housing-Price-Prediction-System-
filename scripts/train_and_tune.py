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
from mlflow.models.signature import infer_signature

warnings.filterwarnings("ignore", category=FutureWarning)

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
def create_objective(X, y, cv_folds, seed):
    """Return an Optuna objective function that logs each trial to MLflow."""

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

        # 5-fold CV with negative RMSE (sklearn convention)
        neg_rmse_scores = cross_val_score(
            rf, X, y,
            cv=cv_folds,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )

        mean_rmse = -neg_rmse_scores.mean()
        std_rmse  = neg_rmse_scores.std()

        # Also compute MAE & R² for richer tracking
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

        mean_mae = -mae_scores.mean()
        mean_r2  = r2_scores.mean()

        # ── Log trial to MLflow ──
        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
            mlflow.log_params(params)
            mlflow.log_metrics({
                "cv_rmse_mean": round(mean_rmse, 2),
                "cv_rmse_std":  round(std_rmse, 2),
                "cv_mae_mean":  round(mean_mae, 2),
                "cv_r2_mean":   round(mean_r2, 4),
            })
            mlflow.set_tag("trial_number", trial.number)

        print(f"  Trial {trial.number:>3d}  |  RMSE: ${mean_rmse:>10,.2f}  "
              f"(±${std_rmse:>8,.2f})  |  R²: {mean_r2:.4f}")

        return mean_rmse   # Optuna minimises by default

    return objective


# ── Final Model Training ─────────────────────────────────────────────────
def train_final_model(X, y, best_params, feature_names, model_output, seed):
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
        ss_res = np.sum((y - train_preds) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        train_r2 = 1 - ss_res / ss_tot

        mlflow.log_metrics({
            "train_rmse": round(train_rmse, 2),
            "train_mae":  round(train_mae, 2),
            "train_r2":   round(train_r2, 4),
        })

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

    # Also save feature names alongside the model
    meta_path = output_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps({
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "training_rows": X.shape[0],
        "target": "sold_price",
    }, indent=2))

    print(f"\nFinal model saved to: {output_path}")
    print(f"Feature metadata:     {meta_path}")
    print(f"MLflow run ID:        {final_run_id}")

    print(f"\nIn-sample metrics (sanity check):")
    print(f"  RMSE:  ${train_rmse:>10,.2f}")
    print(f"  MAE:   ${train_mae:>10,.2f}")
    print(f"  R²:    {train_r2:.4f}")

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
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=args.seed),
            study_name="rf_housing_tuning",
        )
        study.optimize(
            create_objective(X, y, args.cv_folds, args.seed),
            n_trials=args.n_trials,
            show_progress_bar=True,
        )

        # ── Log best trial results to parent run ──
        best = study.best_trial
        mlflow.log_metrics({
            "best_cv_rmse":   round(best.value, 2),
            "best_trial_num": best.number,
        })
        mlflow.log_params({f"best_{k}": v for k, v in best.params.items()})

        print(f"\n{'=' * 65}")
        print(f"BEST TRIAL: #{best.number}")
        print(f"  CV RMSE:  ${best.value:,.2f}")
        print(f"  Params:   {json.dumps(best.params, indent=4)}")

        # ── Train final model ──
        final_model, final_run_id = train_final_model(
            X, y, best.params.copy(), feature_names,
            args.model_output, args.seed,
        )

        # Log the parent run ID for easy lookup
        parent_id = parent_run.info.run_id
        print(f"\nParent MLflow run ID: {parent_id}")

    print("\nDone! To view results:")
    print("  mlflow ui --port 5000")
    print(f"  Then open http://localhost:5000")


if __name__ == "__main__":
    main()