"""
feature_engineering.py
─────────────────────
Reads validated/cleaned housing data from the processed/ folder,
applies feature engineering transformations, logs metadata to MLflow,
and writes the engineered dataset to the training/ folder.

Auto-discovers the most recent file in data/processed/ by default,
and derives the output filename from the input (e.g.
validated_Cincinnati_OH_365d_20250414_183022.csv → engineered_Cincinnati_OH_365d_20250414_183022.csv).

Usage:
    python feature_engineering.py                              # auto-discover latest
    python feature_engineering.py --input data/processed/my_file.csv  # specific file

MLflow Tracking (Light):
    - Parameters: input file, output file, number of raw/engineered features, row count
    - Tags: list of transformations applied, dataset version, source file
    - Metrics: null counts, feature count delta
"""

import os
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import mlflow

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

PROCESSED_DIR = Path("data/processed")
TRAINING_DIR = Path("data/training")

MLFLOW_EXPERIMENT_NAME = "housing-feature-engineering"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"  # SQLite backend (recommended over filesystem)


# ──────────────────────────────────────────────
# FILE DISCOVERY
# ──────────────────────────────────────────────

def find_latest_processed_file(directory: Path = PROCESSED_DIR) -> Path:
    """Find the most recently modified CSV in data/processed/."""
    csv_files = sorted(directory.glob("*.csv"), key=lambda f: f.stat().st_mtime)
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {directory}. "
            f"Run Data_validation.py first."
        )
    latest = csv_files[-1]
    logger.info(f"Found latest processed file: {latest}")
    return latest


def build_output_filename(source_name: str) -> str:
    """Derive output name from input: validated_X.csv → engineered_X.csv.

    Strips any known verb prefix from the source and replaces it with 'engineered_'.
    """
    KNOWN_PREFIXES = ("validated_", "approved_", "fixed_",
                      "quarantined_", "rejected_", "removed_")
    clean_name = source_name
    for prefix in KNOWN_PREFIXES:
        if clean_name.startswith(prefix):
            clean_name = clean_name[len(prefix):]
            break
    return f"engineered_{clean_name}"

# ──────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# TRANSFORMATION REGISTRY
# ──────────────────────────────────────────────
# Each transformation is a function that takes a DataFrame and returns
# the modified DataFrame. Register new transformations here so they
# are automatically tracked in MLflow.

TRANSFORMATIONS_APPLIED: list[str] = []


def register(name: str):
    """Decorator that logs the transformation name when it runs."""
    def decorator(func):
        def wrapper(df: pd.DataFrame, *args, **kwargs):
            logger.info(f"Applying transformation: {name}")
            result = func(df, *args, **kwargs)
            TRANSFORMATIONS_APPLIED.append(name)
            logger.info(f"  ✓ {name} complete — shape now {result.shape}")
            return result
        return wrapper
    return decorator


# ──────────────────────────────────────────────
# FEATURE ENGINEERING TRANSFORMATIONS
# ──────────────────────────────────────────────
# Placeholder examples — replace/extend with your actual transforms.
# Each function should accept and return a DataFrame.


@register("drop_non_feature_columns")
def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop identifiers, URLs, agent/broker info, and other non-predictive columns."""
    drop_cols = [
        # Identifiers & URLs
        "property_url", "property_id", "listing_id", "permalink",
        "mls", "mls_id",
        # Status fields (target leakage / not useful)
        "status", "mls_status",
        # Free text & address components (redundant or unstructured)
        "text", "formatted_address", "full_street_line", "street", "unit",
        # List price variants (target leakage — correlated directly with sold_price)
        "list_price", "list_price_min", "list_price_max",
        # Date fields (will extract features separately if needed)
        "pending_date", "last_sold_date", "last_sold_price",
        "last_status_change_date", "last_update_date",
        # Valuation fields (target leakage)
        "assessed_value", "estimated_value",
        # Tax fields
        "tax", "tax_history",
        # Derived field that may leak or duplicate
        "price_per_sqft",
        # Geospatial (dropping for now — could revisit for clustering)
        "latitude", "longitude",
        # Neighborhood / geo codes
        "neighborhoods", "fips_code",
        # Agent & broker info (not predictive of price)
        "agent_id", "agent_name", "agent_email", "agent_phones",
        "agent_mls_set", "agent_nrds_id",
        "broker_id", "broker_name",
        "builder_id", "builder_name",
        "office_id", "office_mls_set", "office_name",
        "office_email", "office_phones",
        # Non-tabular / media
        "nearby_schools", "primary_photo", "alt_photos",
    ]
    existing = [c for c in drop_cols if c in df.columns]
    missing = [c for c in drop_cols if c not in df.columns]

    if existing:
        df = df.drop(columns=existing)
        logger.info(f"    Dropped {len(existing)} columns")
    if missing:
        logger.warning(f"    {len(missing)} columns not found (skipped): {missing}")

    return df


@register("drop_days_on_mls")
def drop_days_on_mls(df: pd.DataFrame) -> pd.DataFrame:
    """Drop days_on_mls — 86% missing and potential leakage."""
    if "days_on_mls" in df.columns:
        df = df.drop(columns=["days_on_mls"])
        logger.info("    Dropped days_on_mls (86% missing)")
    return df


@register("extract_list_date_features")
def extract_list_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract sale_month and sale_season from list_date, then drop the raw column."""
    if "list_date" not in df.columns:
        return df

    df["list_date_parsed"] = pd.to_datetime(df["list_date"], errors="coerce")

    # Extract month (1-12)
    df["sale_month"] = df["list_date_parsed"].dt.month

    # Map month to season
    season_map = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Fall", 10: "Fall", 11: "Fall",
    }
    df["sale_season"] = df["sale_month"].map(season_map)

    # Fill missing with mode (most common month/season)
    mode_month = df["sale_month"].mode().iloc[0] if not df["sale_month"].mode().empty else 6
    mode_season = df["sale_season"].mode().iloc[0] if not df["sale_season"].mode().empty else "Summer"
    nulls_before = df["sale_month"].isnull().sum()
    df["sale_month"] = df["sale_month"].fillna(mode_month).astype(int)
    df["sale_season"] = df["sale_season"].fillna(mode_season)
    logger.info(f"    Extracted sale_month & sale_season — filled {nulls_before} nulls with mode (month={mode_month}, season={mode_season})")

    # Drop raw date columns
    df = df.drop(columns=["list_date", "list_date_parsed"])
    return df


@register("impute_high_null_columns")
def impute_high_null_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Handle high-null columns: hoa_fee, half_baths, parking_garage.

    Strategy: NaN → 0 (absence is the real value) + binary flag columns.
    """
    # hoa_fee: NaN means no HOA → $0
    if "hoa_fee" in df.columns:
        df["has_hoa"] = df["hoa_fee"].notna().astype(int)
        nulls = df["hoa_fee"].isnull().sum()
        df["hoa_fee"] = df["hoa_fee"].fillna(0)
        logger.info(f"    hoa_fee: filled {nulls} nulls → 0, created has_hoa flag")

    # half_baths: NaN means no half bath → 0
    if "half_baths" in df.columns:
        nulls = df["half_baths"].isnull().sum()
        df["half_baths"] = df["half_baths"].fillna(0).astype(int)
        logger.info(f"    half_baths: filled {nulls} nulls → 0")

    # parking_garage: NaN means no garage → 0
    if "parking_garage" in df.columns:
        df["has_garage"] = df["parking_garage"].notna().astype(int)
        nulls = df["parking_garage"].isnull().sum()
        df["parking_garage"] = df["parking_garage"].fillna(0).astype(int)
        logger.info(f"    parking_garage: filled {nulls} nulls → 0, created has_garage flag")

    return df


@register("drop_near_zero_null_rows")
def drop_near_zero_null_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the handful of rows where style or city is null (<0.1% of data)."""
    before = len(df)
    df = df.dropna(subset=["style", "city"])
    dropped = before - len(df)
    if dropped:
        logger.info(f"    Dropped {dropped} rows with null style or city")
    return df


def _grouped_median_impute(
    df: pd.DataFrame,
    col: str,
    group_cols: list[str],
) -> pd.DataFrame:
    """Impute nulls in `col` using grouped median, falling back to global median."""
    if col not in df.columns or df[col].isnull().sum() == 0:
        return df

    nulls_before = df[col].isnull().sum()

    # Grouped median
    grouped_median = df.groupby(group_cols)[col].transform("median")
    df[col] = df[col].fillna(grouped_median)

    # Fallback: global median for any remaining nulls (groups with all-null)
    remaining = df[col].isnull().sum()
    if remaining > 0:
        global_median = df[col].median()
        df[col] = df[col].fillna(global_median)

    nulls_after = df[col].isnull().sum()
    logger.info(
        f"    {col}: imputed {nulls_before - nulls_after} nulls "
        f"via grouped median ({group_cols}), {nulls_after} remain"
    )
    return df


@register("grouped_median_impute_low_null")
def grouped_median_impute_low_null(df: pd.DataFrame) -> pd.DataFrame:
    """Impute low-null numeric columns using grouped median by style/beds.

    Columns & grouping logic:
        - year_built:  by style
        - beds:        by style (can't group by beds since it's the target)
        - sqft:        by style + beds
        - stories:     by style (mode-like, but median works for 1/2/3)
        - full_baths:  by style + beds
        - lot_sqft:    by style
    """
    df = _grouped_median_impute(df, "year_built", ["style"])
    df = _grouped_median_impute(df, "beds", ["style"])

    # Now that beds is filled, use it as a grouper for sqft and full_baths
    df = _grouped_median_impute(df, "sqft", ["style", "beds"])
    df = _grouped_median_impute(df, "stories", ["style"])
    df = _grouped_median_impute(df, "full_baths", ["style", "beds"])
    df = _grouped_median_impute(df, "lot_sqft", ["style"])

    return df


@register("one_hot_encode_categoricals")
def one_hot_encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode all categorical columns.

    Columns encoded:
        - style        (6 categories: SINGLE_FAMILY, CONDOS, TOWNHOMES, etc.)
        - city         (38 categories)
        - state        (1 category — OH only for now; kept for future multi-state data)
        - zip_code     (45 categories — treated as categorical, not numeric)
        - county       (6 categories)
        - sale_season  (4 categories: Winter, Spring, Summer, Fall)
        - new_construction (bool → int, no one-hot needed)

    Uses drop_first=True to avoid multicollinearity.
    """
    # Convert new_construction bool → int (0/1) directly
    if "new_construction" in df.columns:
        df["new_construction"] = df["new_construction"].astype(int)
        logger.info("    new_construction: bool → int")

    # Ensure zip_code is treated as string for one-hot (not numeric)
    if "zip_code" in df.columns:
        df["zip_code"] = df["zip_code"].astype(str)

    # Identify categorical columns to one-hot encode
    ohe_cols = [c for c in ["style", "city", "state", "zip_code", "county", "sale_season"]
                if c in df.columns]

    before_cols = df.shape[1]
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=True, dtype=int)
    after_cols = df.shape[1]

    logger.info(f"    One-hot encoded {len(ohe_cols)} columns: {ohe_cols}")
    logger.info(f"    Features: {before_cols} → {after_cols} (+{after_cols - before_cols})")

    return df


@register("final_null_audit")
def final_null_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Log any remaining nulls as a safety check."""
    remaining = df.isnull().sum()
    remaining = remaining[remaining > 0]
    if len(remaining) == 0:
        logger.info("    ✓ Zero nulls remaining — dataset is clean")
    else:
        logger.warning(f"    Remaining nulls:\n{remaining.to_string()}")
    return df


# ──────────────────────────────────────────────
# PIPELINE — ordered list of transforms to apply
# ──────────────────────────────────────────────

PIPELINE = [
    drop_unnecessary_columns,        # 1. Remove 46 non-feature columns
    drop_days_on_mls,                # 2. Drop 86%-null days_on_mls
    extract_list_date_features,      # 3. list_date → sale_month + sale_season, mode-fill
    drop_near_zero_null_rows,        # 4. Drop ~7 rows with null style/city
    impute_high_null_columns,        # 5. hoa_fee/half_baths/parking → 0, add flags
    grouped_median_impute_low_null,  # 6. Grouped median for beds/sqft/baths/etc.
    one_hot_encode_categoricals,     # 7. One-hot encode all categorical columns
    final_null_audit,                # 8. Safety check — should be zero nulls
]


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def run_feature_engineering(input_file: str | None = None):
    """Execute the full feature engineering pipeline with MLflow tracking."""

    input_path = Path(input_file) if input_file else find_latest_processed_file()
    output_name = build_output_filename(input_path.name)
    output_path = TRAINING_DIR / output_name

    # ── Validate paths ──
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    raw_shape = df.shape
    raw_columns = df.columns.tolist()
    raw_dtypes = df.dtypes.astype(str).to_dict()
    logger.info(f"Loaded {raw_shape[0]} rows × {raw_shape[1]} columns")

    # ── MLflow setup ──
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    source_label = input_path.stem
    with mlflow.start_run(run_name=f"feat_eng_{source_label}"):

        # Log input metadata
        mlflow.log_param("input_file", input_path.name)
        mlflow.log_param("input_path", str(input_path))
        mlflow.log_param("output_file", output_name)
        mlflow.log_param("raw_rows", raw_shape[0])
        mlflow.log_param("raw_features", raw_shape[1])

        # ── Apply pipeline ──
        for transform_fn in PIPELINE:
            df = transform_fn(df)

        # ── Post-transform metadata ──
        eng_shape = df.shape
        eng_columns = df.columns.tolist()
        null_counts = df.isnull().sum().sum()

        logger.info(f"Engineering complete: {eng_shape[0]} rows × {eng_shape[1]} columns")
        logger.info(f"Total remaining nulls: {null_counts}")

        # ── Log to MLflow ──
        mlflow.log_param("engineered_rows", eng_shape[0])
        mlflow.log_param("engineered_features", eng_shape[1])
        mlflow.log_param("transformations_applied", json.dumps(TRANSFORMATIONS_APPLIED))

        mlflow.log_metric("null_count_total", null_counts)
        mlflow.log_metric("feature_count_delta", eng_shape[1] - raw_shape[1])
        mlflow.log_metric("row_count_delta", eng_shape[0] - raw_shape[0])

        mlflow.set_tag("pipeline_stage", "feature_engineering")
        mlflow.set_tag("dataset_version", datetime.now().strftime("%Y%m%d"))
        mlflow.set_tag("source_file", input_path.name)

        # ── Save engineered dataset ──
        df.to_csv(output_path, index=False)
        logger.info(f"Saved engineered dataset to {output_path}")

        # ── Summary ──
        logger.info("─" * 50)
        logger.info("Feature Engineering Summary")
        logger.info(f"  Input:      {input_path}")
        logger.info(f"  Raw:        {raw_shape[0]} rows × {raw_shape[1]} cols")
        logger.info(f"  Engineered: {eng_shape[0]} rows × {eng_shape[1]} cols")
        logger.info(f"  Transforms: {TRANSFORMATIONS_APPLIED}")
        logger.info(f"  Nulls:      {null_counts}")
        logger.info(f"  Output:     {output_path}")
        logger.info("─" * 50)

    logger.info("MLflow run logged successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run feature engineering pipeline.")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input CSV (default: most recent file in data/processed/)",
    )
    args = parser.parse_args()
    run_feature_engineering(input_file=args.input)