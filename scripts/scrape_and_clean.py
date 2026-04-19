"""
scrape_and_clean.py

Scrapes housing data from Realtor.com via HomeHarvest, then applies cleaning steps:
    1. Drop completely empty rows
    2. Deduplicate rows
    3. Remove rows missing sold_price (target variable)
    4. Filter to valid property styles
    5. Remove outliers via IQR method for numeric columns

Optionally logs full cleaning lineage to MLflow:
    - Raw vs cleaned row counts
    - Rows dropped per step
    - IQR bounds for each column
    - Dataset hashes (raw + cleaned) for versioning
    - Cleaned parquet as an artifact

Usage:
    python scripts/scrape_and_clean.py --location "Cincinnati, OH" --past_days 365
    python scripts/scrape_and_clean.py --location "Cincinnati, OH" --past_days 365 --mlflow
    python scripts/scrape_and_clean.py --location "Cincinnati, OH" --past_days 365 --mlflow --iqr_multiplier 2.0
"""

import argparse
import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTLIER_COLUMNS = [
    "sold_price",
    "sqft",
    "beds",
    #"full_baths",
    #"half_baths",
    "lot_sqft",
    #"stories",
    #"parking_garage",
    #"days_on_mls",
]

VALID_STYLES = [
    "SINGLE_FAMILY",
    "CONDO",
    "CONDOS",
    "TOWNHOMES",
    "APARTMENT",
]

REPORT_DIR = Path("cleaning_reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Filename builder
# ---------------------------------------------------------------------------

def build_output_filename(location: str, past_days: int) -> str:
    """Build a readable filename: <city_state>_<past_days>d_<YYYYMMDD_HHMMSS>.csv

    Examples:
        "Cincinnati, OH" / 365  →  Cincinnati_OH_365d_20250414_183022.csv
        "New York, NY"   / 90   →  New_York_NY_90d_20250414_183022.csv
    """
    # Normalize location: strip whitespace, replace commas/spaces with underscores
    clean_loc = location.strip()
    clean_loc = clean_loc.replace(",", "")   # "Cincinnati, OH" → "Cincinnati OH"
    clean_loc = clean_loc.replace(" ", "_")   # "Cincinnati OH"  → "Cincinnati_OH"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{clean_loc}_{past_days}d_{ts}.csv"


# ---------------------------------------------------------------------------
# Data hashing
# ---------------------------------------------------------------------------

def compute_dataset_hash(df: pd.DataFrame) -> str:
    """Compute a deterministic SHA-256 hash of a DataFrame for versioning."""
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()


# ---------------------------------------------------------------------------
# Scraping
# ---------------------------------------------------------------------------

def scrape_data(location: str, listing_type: str, past_days: int) -> pd.DataFrame:
    """Scrape housing data using HomeHarvest."""
    try:
        from homeharvest import scrape_property
    except ImportError:
        raise ImportError("homeharvest is required. Install with: pip install homeharvest")

    logger.info(f"Scraping: location='{location}' | type={listing_type} | past_days={past_days}")
    df = scrape_property(
        location=location,
        listing_type=listing_type,
        past_days=past_days,
    )
    logger.info(f"Scraped {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
# Cleaning steps (each returns df + count of rows dropped)
# ---------------------------------------------------------------------------

def drop_empty_rows(df: pd.DataFrame) -> tuple:
    """Drop rows where every column is NaN."""
    before = len(df)
    df = df.dropna(how="all")
    dropped = before - len(df)
    logger.info(f"Empty rows: dropped {dropped}")
    return df, dropped


def deduplicate(df: pd.DataFrame) -> tuple:
    """Remove duplicate rows, skipping unhashable columns (lists, dicts)."""
    before = len(df)

    hashable_cols = []
    for col in df.columns:
        sample = df[col].dropna()
        if len(sample) == 0:
            hashable_cols.append(col)
            continue
        first_val = sample.iloc[0]
        if not isinstance(first_val, (list, dict, set)):
            hashable_cols.append(col)
        else:
            logger.info(f"  Skipping column '{col}' during dedup (contains {type(first_val).__name__})")

    df = df.drop_duplicates(subset=hashable_cols)
    dropped = before - len(df)
    logger.info(f"Duplicates: dropped {dropped} (checked {len(hashable_cols)}/{len(df.columns)} columns)")
    return df, dropped


def remove_missing_target(df: pd.DataFrame) -> tuple:
    """Remove rows where sold_price is null."""
    before = len(df)
    df = df.dropna(subset=["sold_price"])
    dropped = before - len(df)
    logger.info(f"Missing sold_price: dropped {dropped}")
    return df, dropped


def filter_style(df: pd.DataFrame, valid_styles: list = VALID_STYLES) -> tuple:
    """Keep only rows with property styles relevant to price prediction."""
    before = len(df)
    df = df[df["style"].isin(valid_styles) | df["style"].isna()]
    dropped = before - len(df)
    logger.info(f"Style filter: dropped {dropped} (kept: {valid_styles})")
    return df, dropped


def remove_outliers_iqr(
    df: pd.DataFrame,
    columns: list,
    multiplier: float = 1.5,
) -> tuple:
    """
    Remove outliers using the IQR method.

    Returns:
        (cleaned_df, total_dropped, outlier_report)
        outlier_report is a dict of col -> {outliers_found, lower_bound, upper_bound, Q1, Q3}
    """
    before = len(df)
    mask = pd.Series(True, index=df.index)
    outlier_report = {}

    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found — skipping outlier check")
            continue

        series = pd.to_numeric(df[col], errors="coerce")
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr

        col_mask = (series >= lower) & (series <= upper) | series.isna()
        col_outliers = (~col_mask).sum()

        outlier_report[col] = {
            "outliers_found": int(col_outliers),
            "lower_bound": round(float(lower), 2),
            "upper_bound": round(float(upper), 2),
            "Q1": round(float(q1), 2),
            "Q3": round(float(q3), 2),
            "IQR": round(float(iqr), 2),
        }

        mask = mask & col_mask

    df = df[mask]
    total_dropped = before - len(df)

    logger.info(f"Outlier removal (IQR x{multiplier}): dropped {total_dropped} rows total")
    for col, info in outlier_report.items():
        if info["outliers_found"] > 0:
            logger.info(
                f"  {col}: {info['outliers_found']} outliers | "
                f"bounds=[{info['lower_bound']}, {info['upper_bound']}]"
            )

    return df, total_dropped, outlier_report


# ---------------------------------------------------------------------------
# Full cleaning pipeline
# ---------------------------------------------------------------------------

def clean_data(
    df: pd.DataFrame,
    iqr_multiplier: float = 1.5,
) -> tuple:
    """
    Full cleaning pipeline. Returns (cleaned_df, cleaning_log).

    cleaning_log captures every step for MLflow logging.
    """
    cleaning_log = {
        "raw_rows": len(df),
        "raw_columns": len(df.columns),
        "raw_hash": compute_dataset_hash(df),
        "iqr_multiplier": iqr_multiplier,
        "valid_styles": VALID_STYLES,
        "outlier_columns": OUTLIER_COLUMNS,
        "steps": {},
    }

    logger.info(f"{'='*60}")
    logger.info(f"Starting cleaning pipeline — {len(df)} rows in")
    logger.info(f"{'='*60}")

    # Step 1: Empty rows
    df, dropped = drop_empty_rows(df)
    cleaning_log["steps"]["empty_rows_dropped"] = dropped

    # Step 2: Dedup
    df, dropped = deduplicate(df)
    cleaning_log["steps"]["duplicates_dropped"] = dropped

    # Step 3: Missing target
    df, dropped = remove_missing_target(df)
    cleaning_log["steps"]["missing_target_dropped"] = dropped

    # Step 4: Style filter
    df, dropped = filter_style(df)
    cleaning_log["steps"]["style_filter_dropped"] = dropped

    # Step 5: Outliers
    df, dropped, outlier_report = remove_outliers_iqr(
        df, columns=OUTLIER_COLUMNS, multiplier=iqr_multiplier
    )
    cleaning_log["steps"]["outliers_dropped"] = dropped
    cleaning_log["outlier_details"] = outlier_report

    df = df.reset_index(drop=True)

    # Final stats
    cleaning_log["cleaned_rows"] = len(df)
    cleaning_log["cleaned_columns"] = len(df.columns)
    cleaning_log["cleaned_hash"] = compute_dataset_hash(df)
    cleaning_log["total_rows_removed"] = cleaning_log["raw_rows"] - len(df)
    cleaning_log["removal_pct"] = round(
        100 * cleaning_log["total_rows_removed"] / max(cleaning_log["raw_rows"], 1), 2
    )
    cleaning_log["timestamp"] = datetime.now().isoformat()

    logger.info(f"{'='*60}")
    logger.info(f"Cleaning complete — {len(df)} rows remaining ({cleaning_log['removal_pct']}% removed)")
    logger.info(f"{'='*60}")

    return df, cleaning_log


# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------

def log_cleaning_to_mlflow(
    cleaning_log: dict,
    cleaned_file: str,
    experiment_name: str = "housing_data_cleaning",
    run_id: str = None,
):
    """
    Log the full cleaning lineage to MLflow.

    Logs:
        - Params: location, listing_type, past_days, iqr_multiplier, dataset hashes
        - Metrics: row counts, rows dropped per step, removal %
        - Artifacts: cleaning report JSON, cleaned parquet
        - Tags: raw/cleaned hashes for cross-run linking
    """
    try:
        import mlflow
    except ImportError:
        raise ImportError("mlflow is required. Install with: pip install mlflow")

    # Save cleaning report to disk for artifact logging
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = REPORT_DIR / f"cleaning_report_{ts}.json"
    with open(report_file, "w") as f:
        json.dump(cleaning_log, f, indent=2, default=str)

    def _log():
        # --- Params: scrape context ---
        scrape = cleaning_log.get("scrape_params", {})
        if scrape:
            mlflow.log_param("location", scrape.get("location", ""))
            mlflow.log_param("listing_type", scrape.get("listing_type", ""))
            mlflow.log_param("past_days", scrape.get("past_days", ""))

        # --- Params: cleaning config ---
        mlflow.log_param("iqr_multiplier", cleaning_log["iqr_multiplier"])
        mlflow.log_param("raw_hash", cleaning_log["raw_hash"][:16])
        mlflow.log_param("cleaned_hash", cleaning_log["cleaned_hash"][:16])
        mlflow.log_param("valid_styles", ", ".join(cleaning_log["valid_styles"]))
        mlflow.log_param("outlier_columns", ", ".join(cleaning_log["outlier_columns"]))
        mlflow.log_param("output_file", Path(cleaned_file).name)

        # --- Metrics: row counts ---
        mlflow.log_metrics({
            "raw_rows": cleaning_log["raw_rows"],
            "cleaned_rows": cleaning_log["cleaned_rows"],
            "total_rows_removed": cleaning_log["total_rows_removed"],
            "removal_pct": cleaning_log["removal_pct"],
        })

        # --- Metrics: rows dropped per step ---
        for step_name, count in cleaning_log["steps"].items():
            mlflow.log_metric(f"step_{step_name}", count)

        # --- Metrics: outlier bounds per column ---
        for col, details in cleaning_log.get("outlier_details", {}).items():
            mlflow.log_metric(f"outlier_{col}_count", details["outliers_found"])
            mlflow.log_metric(f"outlier_{col}_lower", details["lower_bound"])
            mlflow.log_metric(f"outlier_{col}_upper", details["upper_bound"])

        # --- Tags for cross-run linking ---
        mlflow.set_tag("raw_dataset_hash", cleaning_log["raw_hash"])
        mlflow.set_tag("cleaned_dataset_hash", cleaning_log["cleaned_hash"])
        mlflow.set_tag("pipeline_stage", "data_cleaning")

        # --- Artifacts ---
        mlflow.log_artifact(str(report_file), artifact_path="data_cleaning")
        mlflow.log_artifact(cleaned_file, artifact_path="data_cleaning")

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    if run_id:
        with mlflow.start_run(run_id=run_id):
            _log()
    else:
        mlflow.set_experiment(experiment_name)
        scrape = cleaning_log.get("scrape_params", {})
        loc_label = scrape.get("location", "").replace(",", "").replace(" ", "_")
        run_label = f"cleaning_{loc_label}_{ts}" if loc_label else f"data_cleaning_{ts}"
        with mlflow.start_run(run_name=run_label):
            _log()

    logger.info(f"Cleaning lineage logged to MLflow (experiment: {experiment_name})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Scrape housing data from Realtor.com and clean it for modeling."
    )
    parser.add_argument("--location", type=str, default="Cincinnati, OH",
                        help="Location string for HomeHarvest scrape")
    parser.add_argument("--listing_type", type=str, default="sold",
                        help="Listing type: sold, for_sale, for_rent")
    parser.add_argument("--past_days", type=int, default=365,
                        help="Number of past days to scrape")
    parser.add_argument("--iqr_multiplier", type=float, default=1.5,
                        help="IQR multiplier for outlier removal (1.5=standard, 2.0=lenient)")
    parser.add_argument("--output", type=str, default="data/cleaned",
                        help="Output directory for the cleaned parquet file")
    parser.add_argument("--mlflow", action="store_true",
                        help="Log cleaning lineage to MLflow")
    parser.add_argument("--experiment", type=str, default="housing_data_cleaning",
                        help="MLflow experiment name")

    args = parser.parse_args()

    # --- Scrape ---
    df_raw = scrape_data(args.location, args.listing_type, args.past_days)

    # --- Clean ---
    df_clean, cleaning_log = clean_data(df_raw, iqr_multiplier=args.iqr_multiplier)

    # Add scrape params to the log (before MLflow so they're available for logging)
    cleaning_log["scrape_params"] = {
        "location": args.location,
        "listing_type": args.listing_type,
        "past_days": args.past_days,
    }

    # --- Save ---
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = build_output_filename(args.location, args.past_days)
    csv_file = output_dir / filename
    df_clean.to_csv(csv_file, index=False)

    # --- MLflow ---
    if args.mlflow:
        log_cleaning_to_mlflow(
            cleaning_log,
            cleaned_file=str(csv_file),
            experiment_name=args.experiment,
        )

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  Location:      {args.location}")
    print(f"  Raw rows:      {cleaning_log['raw_rows']}")
    print(f"  Cleaned rows:  {cleaning_log['cleaned_rows']}")
    print(f"  Removed:       {cleaning_log['total_rows_removed']} ({cleaning_log['removal_pct']}%)")
    print(f"  Breakdown:")
    for step, count in cleaning_log["steps"].items():
        print(f"    {step}: {count}")
    print(f"  Raw hash:      {cleaning_log['raw_hash'][:16]}...")
    print(f"  Cleaned hash:  {cleaning_log['cleaned_hash'][:16]}...")
    print(f"  CSV:           {csv_file}")
    if args.mlflow:
        print(f"  MLflow:        ✅ logged to '{args.experiment}'")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
