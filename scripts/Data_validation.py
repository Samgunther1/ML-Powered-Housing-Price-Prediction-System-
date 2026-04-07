"""
validate_housing_data.py

Great Expectations validation module for the HomeHarvest housing price prediction pipeline.
Validates raw scraped data from Realtor.com before feature engineering and model training.

Workflow (quarantine pattern):
    Incoming data (API scrape)
        |
    Validation checks
        |
        +---> PASS --> data/raw/        (ready for feature engineering)
        +---> FAIL --> data/errors/     (quarantined for manual review)
                            |
                       Manual review
                            |
                       +---> Approved  --> moved to data/raw/
                       +---> Rejected  --> archived or deleted

Usage:
    # Standalone scrape + validate + route:
    python validate_housing_data.py --location "Cincinnati, OH" --past_days 365

    # In your pipeline:
    from validate_housing_data import validate_and_route, review_quarantined

    df_valid = validate_and_route(df)            # routes to raw/ or errors/
    review_quarantined("data/errors/some_file")  # approve or reject
"""

import hashlib
import json
import logging
import shutil
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

try:
    import great_expectations as gx
    from great_expectations.expectations import (
    ExpectColumnToExist,
    ExpectColumnValuesToBeBetween,
    ExpectColumnValuesToBeInSet,
    ExpectColumnValuesToNotBeNull,
    ExpectTableColumnCountToBeBetween,
    ExpectTableRowCountToBeBetween,       # <-- changed
)
except ImportError:
    raise ImportError(
        "great_expectations is required. Install with: pip install great_expectations"
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Directory structure for quarantine workflow
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"         # validated, clean data lands here
ERRORS_DIR = DATA_DIR / "errors"   # failed validation, quarantined for review
ARCHIVE_DIR = DATA_DIR / "archive" # rejected data after manual review
REPORT_DIR = Path("validation_reports")

for _dir in [RAW_DIR, ERRORS_DIR, ARCHIVE_DIR, REPORT_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Configuration: columns and validation rules
# ---------------------------------------------------------------------------

# Columns that MUST exist in the raw DataFrame
REQUIRED_COLUMNS = [
    "style",          
    "list_price",
    "sold_price",
    "year_built",
    "days_on_mls",
    "sqft",
    "beds",
    "full_baths",
    "half_baths",
    "formatted_address",
    "new_construction",
    "city",
    "lot_sqft",
    "stories",
    "parking_garage",
    "hoa_fee",
    "county",
    "state",
    "zip_code",
    "status",
]

# Columns that should not be null (column -> mostly threshold)
# "mostly" = fraction of non-null values required (1.0 = zero nulls allowed)
NOT_NULL_EXPECTATIONS = {
    "sold_price": 1.0,         # target variable — no nulls
    "formatted_address": 0.95,
    "city": 0.95,
    "county": 0.95,
    "state": 0.99,
    "zip_code": 0.95,
    "status": 0.99,
    "style": 0.95,
    "sqft": 0.80,
    "beds": 0.85,
    "full_baths": 0.85,        
    "half_baths": 0.70,
}

# Numeric range checks: column -> (min, max, mostly)
RANGE_EXPECTATIONS = {
    "list_price": (10_000, 10_000_000, 1.0),
    "sold_price": (10_000, 10_000_000, 1.0),
    "sqft": (100, 50_000, 0.95),
    "beds": (0, 20, 0.99),
    "full_baths": (0, 20, 0.99),  # <-- fixed from "baths"
    "half_baths": (0, 10, 0.99),
    "year_built": (1800, 2027, 0.90),
    "lot_sqft": (0, 5_000_000, 0.80),
    "stories": (0, 10, 0.90),
}

# Categorical value checks: column -> (allowed_values, mostly)
CATEGORICAL_EXPECTATIONS = {
    "style": (
        [
            "SINGLE_FAMILY",
            "CONDOS",
            "LAND",
            "MULTI_FAMILY",
            "APARTMENT",
            "OTHER",     
            "CONDO",
            "TOWNHOMES",
        ],
        0.90,
    ),
    "status": (
        ["SOLD"],
        0.99,
    ),
}

# Minimum number of rows expected from a scrape
MIN_ROW_COUNT = 1


# ---------------------------------------------------------------------------
# Data hashing for MLflow versioning
# ---------------------------------------------------------------------------

def compute_dataset_hash(df: pd.DataFrame) -> str:
    """Compute a deterministic SHA-256 hash of a DataFrame for versioning."""
    # Sort columns for consistency, then hash the CSV bytes
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()


def build_schema_snapshot(df: pd.DataFrame) -> dict:
    """Capture schema metadata: column names, dtypes, null counts."""
    return {
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "null_counts": df.isnull().sum().to_dict(),
        "shape": list(df.shape),
    }


# ---------------------------------------------------------------------------
# Core validation logic
# ---------------------------------------------------------------------------

def build_expectation_suite(
    context: gx.data_context,
    suite_name: str = "housing_ingestion_suite",
) -> gx.ExpectationSuite:
    """Build and return the Expectation Suite for housing data validation."""

    # Remove existing suite if present (idempotent rebuilds)
    try:
        existing = context.suites.get(suite_name)
        if existing:
            context.suites.delete(suite_name)
    except Exception:
        pass

    suite = context.suites.add(gx.ExpectationSuite(name=suite_name))

    # --- Row count ---
    suite.add_expectation(
    ExpectTableRowCountToBeBetween(min_value=MIN_ROW_COUNT)
)

    # --- Column count sanity check ---
    suite.add_expectation(
        ExpectTableColumnCountToBeBetween(min_value=len(REQUIRED_COLUMNS))
    )

    # --- Required columns exist ---
    for col in REQUIRED_COLUMNS:
        suite.add_expectation(ExpectColumnToExist(column=col))

    # --- Not-null checks ---
    for col, mostly in NOT_NULL_EXPECTATIONS.items():
        suite.add_expectation(
            ExpectColumnValuesToNotBeNull(column=col, mostly=mostly)
        )

    # --- Numeric range checks ---
    for col, (min_val, max_val, mostly) in RANGE_EXPECTATIONS.items():
        suite.add_expectation(
            ExpectColumnValuesToBeBetween(
                column=col,
                min_value=min_val,
                max_value=max_val,
                mostly=mostly,
            )
        )

    # --- Categorical checks ---
    for col, (value_set, mostly) in CATEGORICAL_EXPECTATIONS.items():
        suite.add_expectation(
            ExpectColumnValuesToBeInSet(
                column=col,
                value_set=value_set,
                mostly=mostly,
            )
        )

    logger.info(
        f"Built suite '{suite_name}' with {len(suite.expectations)} expectations"
    )
    return suite


def run_validation(
    df: pd.DataFrame,
    suite_name: str = "housing_ingestion_suite",
    context: Optional[gx.data_context] = None,
) -> Tuple[bool, object, dict]:
    """
    Run Great Expectations validation on a raw housing DataFrame.

    Returns:
        (success, results, summary)
    """
    if context is None:
        context = gx.get_context()

    suite = build_expectation_suite(context, suite_name)

    # Wire up the DataFrame as a pandas data source
    ds_name = "housing_pandas_ds"
    try:
        datasource = context.data_sources.get(ds_name)
    except Exception:
        datasource = context.data_sources.add_pandas(ds_name)

    asset_name = "raw_listings"
    try:
        data_asset = datasource.get_asset(asset_name)
    except Exception:
        data_asset = datasource.add_dataframe_asset(name=asset_name)

    batch_def_name = "full_batch"
    try:
        batch_definition = data_asset.get_batch_definition(batch_def_name)
    except Exception:
        batch_definition = data_asset.add_batch_definition_whole_dataframe(
            batch_def_name
        )

    # Create validation definition
    vd_name = f"validate_{suite_name}"
    try:
        context.validation_definitions.delete(vd_name)
    except Exception:
        pass

    validation_definition = context.validation_definitions.add(
        gx.ValidationDefinition(
            name=vd_name,
            data=batch_definition,
            suite=suite,
        )
    )

    results = validation_definition.run(batch_parameters={"dataframe": df})

    # Build summary
    summary = {
        "success": results.success,
        "evaluated": results.statistics["evaluated_expectations"],
        "passed": results.statistics["successful_expectations"],
        "failed": results.statistics["unsuccessful_expectations"],
        "timestamp": datetime.now().isoformat(),
        "row_count": len(df),
        "column_count": len(df.columns),
        "dataset_hash": compute_dataset_hash(df),
    }

    if not results.success:
        summary["failures"] = []
        for r in results.results:
            if not r.success:
                summary["failures"].append(
                    {
                        "expectation": r.expectation_config.type,
                        "kwargs": r.expectation_config.kwargs,
                        "observed": str(r.result)[:200],
                    }
                )

    return results.success, results, summary


# ---------------------------------------------------------------------------
# Quarantine workflow: route data based on validation outcome
# ---------------------------------------------------------------------------

def _save_parquet(df: pd.DataFrame, directory: Path, label: str) -> Path:
    """Save a DataFrame as a timestamped parquet file."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = directory / f"{label}_{ts}.parquet"
    df.to_parquet(filepath, index=False)
    logger.info(f"Saved {len(df)} rows to {filepath}")
    return filepath


def _save_report(summary: dict) -> Path:
    """Save validation report JSON to the reports directory."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = REPORT_DIR / f"validation_{ts}.json"
    with open(report_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Validation report saved to {report_file}")
    return report_file


def validate_and_route(
    df: pd.DataFrame,
    strict: bool = False,
    label: str = "listings",
) -> Tuple[bool, Path, dict]:
    """
    Validate a housing DataFrame and route it to the appropriate directory.

    Pass -> data/raw/
    Fail -> data/errors/ (quarantined)

    Args:
        df: Raw DataFrame from HomeHarvest
        strict: If True, raise on validation failure
        label: Prefix for the saved parquet filename

    Returns:
        (passed: bool, filepath: Path, summary: dict)
    """
    logger.info(f"Validating housing data: {len(df)} rows, {len(df.columns)} columns")

    success, results, summary = run_validation(df)
    report_path = _save_report(summary)
    summary["report_path"] = str(report_path)

    if success:
        filepath = _save_parquet(df, RAW_DIR, label)
        logger.info(
            f"✅ PASSED — {summary['passed']}/{summary['evaluated']} expectations met. "
            f"Data saved to {filepath}"
        )
    else:
        filepath = _save_parquet(df, ERRORS_DIR, label)
        msg = (
            f"❌ FAILED — {summary['failed']}/{summary['evaluated']} expectations failed. "
            f"Data quarantined to {filepath}\n"
        )
        for failure in summary.get("failures", []):
            msg += f"  • {failure['expectation']} | {failure['kwargs']}\n"

        if strict:
            logger.error(msg)
            raise ValueError(msg)
        else:
            logger.warning(msg)

    summary["data_path"] = str(filepath)
    return success, filepath, summary


def review_quarantined(
    quarantined_path: str,
    approve: bool = True,
) -> Path:
    """
    Manual review step: approve or reject a quarantined dataset.

    Args:
        quarantined_path: Path to a file in data/errors/
        approve: True to move to data/raw/, False to move to data/archive/

    Returns:
        New file path after move
    """
    src = Path(quarantined_path)
    if not src.exists():
        raise FileNotFoundError(f"Quarantined file not found: {src}")

    if approve:
        dest = RAW_DIR / src.name
        action = "APPROVED → moved to data/raw/"
    else:
        dest = ARCHIVE_DIR / src.name
        action = "REJECTED → archived to data/archive/"

    shutil.move(str(src), str(dest))
    logger.info(f"Review: {action} ({dest})")
    return dest


# ---------------------------------------------------------------------------
# MLflow integration with data versioning
# ---------------------------------------------------------------------------

def validate_and_log_to_mlflow(
    df: pd.DataFrame,
    run_id: Optional[str] = None,
    experiment_name: Optional[str] = None,
    strict: bool = False,
    label: str = "listings",
) -> Tuple[bool, Path, dict]:
    """
    Full pipeline step: validate, route (quarantine pattern), and log to MLflow.

    MLflow logs include:
        - Data quality metrics (pass rate, row count, etc.)
        - Dataset hash for reproducibility / versioning
        - Schema snapshot (columns, dtypes, null counts)
        - Validation report JSON as artifact
        - Data file path as a tag

    Args:
        df: Raw DataFrame from HomeHarvest
        run_id: Existing MLflow run ID (if None, creates new run)
        experiment_name: MLflow experiment name (used when run_id is None)
        strict: Raise on validation failure
        label: Prefix for saved parquet filename

    Returns:
        (passed, filepath, summary)
    """
    try:
        import mlflow
    except ImportError:
        raise ImportError("mlflow is required. Install with: pip install mlflow")

    # Step 1: validate and route to raw/ or errors/
    success, filepath, summary = validate_and_route(df, strict=False, label=label)

    # Step 2: build versioning artifacts
    dataset_hash = summary["dataset_hash"]
    schema = build_schema_snapshot(df)

    schema_file = REPORT_DIR / f"schema_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(schema_file, "w") as f:
        json.dump(schema, f, indent=2, default=str)

    # Step 3: log everything to MLflow
    def _log():
        # Metrics
        mlflow.log_metrics(
            {
                "dq_expectations_total": summary["evaluated"],
                "dq_expectations_passed": summary["passed"],
                "dq_expectations_failed": summary["failed"],
                "dq_pass_rate": summary["passed"] / max(summary["evaluated"], 1),
                "dq_row_count": summary["row_count"],
                "dq_column_count": summary["column_count"],
            }
        )

        # Tags for quick filtering in the MLflow UI
        mlflow.set_tag("data_validation_passed", str(success))
        mlflow.set_tag("dataset_hash", dataset_hash)
        mlflow.set_tag("data_path", str(filepath))
        mlflow.set_tag("quarantined", str(not success))

        # Artifacts
        mlflow.log_artifact(summary["report_path"], artifact_path="data_validation")
        mlflow.log_artifact(str(schema_file), artifact_path="data_validation")

        # Log the dataset hash as a param for easy comparison across runs
        mlflow.log_param("dataset_hash", dataset_hash[:16])  # first 16 chars as param
        mlflow.log_param("dataset_rows", summary["row_count"])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if run_id:
        with mlflow.start_run(run_id=run_id):
            _log()
    else:
        if experiment_name:
            mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=f"data_validation_{ts}"):
            _log()

    status = "✅ PASSED" if success else "❌ FAILED (quarantined)"
    logger.info(f"{status} — results logged to MLflow (hash: {dataset_hash[:12]}...)")

    if not success and strict:
        msg = f"Validation failed: {summary['failed']} expectations did not pass"
        raise ValueError(msg)

    return success, filepath, summary


# ---------------------------------------------------------------------------
# CLI entrypoint: scrape + validate + route (+ optional MLflow logging)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Scrape housing data, validate, and route via quarantine workflow."
    )
    parser.add_argument("--location", type=str, default="Cincinnati, OH",
                        help="Location string for HomeHarvest scrape")
    parser.add_argument("--listing_type", type=str, default="sold",
                        help="Listing type: sold, for_sale, for_rent")
    parser.add_argument("--past_days", type=int, default=365,
                        help="Number of past days to scrape")
    parser.add_argument("--mlflow", action="store_true",
                        help="Log results to MLflow")
    parser.add_argument("--experiment", type=str, default="housing_data_validation",
                        help="MLflow experiment name")
    parser.add_argument("--strict", action="store_true",
                        help="Raise error on validation failure")

    args = parser.parse_args()

    # --- Scrape ---
    try:
        from homeharvest import scrape_property
    except ImportError:
        raise ImportError("homeharvest is required. Install with: pip install homeharvest")

    logger.info(f"Scraping {args.location} | type={args.listing_type} | past_days={args.past_days}")
    df = scrape_property(
        location=args.location,
        listing_type=args.listing_type,
        past_days=args.past_days,
    )
    logger.info(f"Scraped {df.shape[0]} rows, {df.shape[1]} columns")

    # --- Validate + Route (+ optional MLflow) ---
    if args.mlflow:
        success, filepath, summary = validate_and_log_to_mlflow(
            df,
            experiment_name=args.experiment,
            strict=args.strict,
        )
    else:
        success, filepath, summary = validate_and_route(
            df,
            strict=args.strict,
        )

    print(f"\n{'='*60}")
    print(f"  Result:    {'PASSED' if success else 'FAILED (quarantined)'}")
    print(f"  Data:      {filepath}")
    print(f"  Rows:      {summary['row_count']}")
    print(f"  Hash:      {summary['dataset_hash'][:16]}...")
    print(f"  Report:    {summary.get('report_path', 'N/A')}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()