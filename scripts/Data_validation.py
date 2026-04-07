"""
validate_housing_data.py

Great Expectations validation module for the HomeHarvest housing price prediction pipeline.
Validates CLEANED data (output of scrape_and_clean.py) before feature engineering and model training.

Workflow (quarantine pattern):
    Cleaned data (from scrape_and_clean.py)
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
    # Validate the most recent cleaned file (default):
    python scripts/Data_validation.py --mlflow

    # Validate a specific file:
    python scripts/Data_validation.py --file data/cleaned/cleaned_20260406_150000.parquet --mlflow

    # Review quarantined files:
    python scripts/Data_validation.py --review
    python scripts/Data_validation.py --approve data/errors/listings_20260406_150000.parquet
    python scripts/Data_validation.py --reject data/errors/listings_20260406_150000.parquet
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
        ExpectTableRowCountToBeBetween,
    )
except ImportError:
    raise ImportError(
        "great_expectations is required. Install with: pip install great_expectations"
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Directory structure
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
CLEANED_DIR = DATA_DIR / "cleaned"   # input: output of scrape_and_clean.py
RAW_DIR = DATA_DIR / "raw"           # validated, ready for modeling
ERRORS_DIR = DATA_DIR / "errors"     # failed validation, quarantined
ARCHIVE_DIR = DATA_DIR / "archive"   # rejected after manual review
REPORT_DIR = Path("validation_reports")

for _dir in [CLEANED_DIR, RAW_DIR, ERRORS_DIR, ARCHIVE_DIR, REPORT_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Configuration: columns and validation rules (tuned for CLEANED data)
# ---------------------------------------------------------------------------

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

# Tighter null thresholds since data is already cleaned
NOT_NULL_EXPECTATIONS = {
    "sold_price": 1.0,
    "formatted_address": 0.98,
    "city": 0.98,
    "county": 0.98,
    "state": 1.0,
    "zip_code": 0.98,
    "status": 1.0,
    "style": 0.98,
    "sqft": 0.85,
    "beds": 0.90,
    "full_baths": 0.90,
    "half_baths": 0.80,
}

# Range checks — outliers should already be removed, so these act as a safety net
RANGE_EXPECTATIONS = {
    "list_price": (10_000, 100_000_000, 1.0),
    "sold_price": (10_000, 100_000_000, 1.0),
    "sqft": (100, 50_000, 0.98),
    "beds": (0, 20, 1.0),
    "full_baths": (0, 20, 1.0),
    "half_baths": (0, 10, 1.0),
    "year_built": (1800, 2027, 0.95),
    "lot_sqft": (0, 5_000_000, 0.85),
    "stories": (0, 10, 0.95),
}

CATEGORICAL_EXPECTATIONS = {
    "style": (
        [
            "SINGLE_FAMILY",
            "CONDO",
            "CONDOS",
            "TOWNHOMES",
            "MULTI_FAMILY",
        ],
        0.98,
    ),
    "status": (
        ["SOLD"],
        0.99,
    ),
}

MIN_ROW_COUNT = 50  # cleaned data should have a reasonable number of rows


# ---------------------------------------------------------------------------
# Find the most recent cleaned file
# ---------------------------------------------------------------------------

def find_latest_cleaned_file(directory: Path = CLEANED_DIR) -> Path:
    """Find the most recently created parquet file in the cleaned directory."""
    parquet_files = sorted(directory.glob("*.parquet"), key=lambda f: f.stat().st_mtime)

    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found in {directory}. "
            f"Run scrape_and_clean.py first."
        )

    latest = parquet_files[-1]
    logger.info(f"Found latest cleaned file: {latest}")
    return latest


# ---------------------------------------------------------------------------
# Data hashing for MLflow versioning
# ---------------------------------------------------------------------------

def compute_dataset_hash(df: pd.DataFrame) -> str:
    """Compute a deterministic SHA-256 hash of a DataFrame for versioning."""
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()


def build_schema_snapshot(df: pd.DataFrame) -> dict:
    """Capture schema metadata: column names, dtypes, null counts."""
    return {
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "null_counts": df.isnull().sum().to_dict(),
        "null_pcts": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        "shape": list(df.shape),
    }


# ---------------------------------------------------------------------------
# Core validation logic
# ---------------------------------------------------------------------------

def build_expectation_suite(
    context: gx.DataContext,
    suite_name: str = "housing_cleaned_suite",
) -> gx.ExpectationSuite:
    """Build and return the Expectation Suite for cleaned housing data."""

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

    # --- Column count ---
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
    suite_name: str = "housing_cleaned_suite",
    context: Optional[gx.DataContext] = None,
) -> Tuple[bool, object, dict]:
    """Run Great Expectations validation on a cleaned housing DataFrame."""
    if context is None:
        context = gx.get_context()

    suite = build_expectation_suite(context, suite_name)

    ds_name = "housing_pandas_ds"
    try:
        datasource = context.data_sources.get(ds_name)
    except Exception:
        datasource = context.data_sources.add_pandas(ds_name)

    asset_name = "cleaned_listings"
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
# Quarantine workflow
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
    Validate a cleaned DataFrame and route to raw/ or errors/.

    Pass -> data/raw/
    Fail -> data/errors/ (quarantined)
    """
    logger.info(f"Validating cleaned data: {len(df)} rows, {len(df.columns)} columns")

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
    """Manual review: approve (move to raw/) or reject (move to archive/)."""
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


def list_quarantined() -> list:
    """List all files currently in the errors/ quarantine directory."""
    files = sorted(ERRORS_DIR.glob("*.parquet"), key=lambda f: f.stat().st_mtime)
    if not files:
        print("\nNo quarantined files found in data/errors/\n")
        return files

    print(f"\n{'='*60}")
    print(f"  Quarantined files ({len(files)})")
    print(f"{'='*60}")

    # Find all validation reports to match against quarantined files
    reports = sorted(REPORT_DIR.glob("validation_*.json"), key=lambda f: f.stat().st_mtime)
    report_data = {}
    for rpt_path in reports:
        try:
            with open(rpt_path) as rpt:
                data = json.load(rpt)
            if not data.get("success", True):
                report_data[rpt_path.name] = data
        except Exception:
            pass

    for f in files:
        size_kb = f.stat().st_size / 1024
        print(f"\n  File: {f}")
        print(f"  Size: {size_kb:.0f} KB")

        # Try to find matching report by looking for closest timestamp
        for rpt_name, rpt_data in report_data.items():
            failures = rpt_data.get("failures", [])
            if failures:
                print(f"  Failures ({len(failures)}):")
                for fail in failures:
                    print(f"    ❌ {fail['expectation']}")
                    print(f"       {fail['kwargs']}")
                break

    print(f"\n{'='*60}")
    print(f"  To approve:  python scripts/Data_validation.py --approve <filepath>")
    print(f"  To reject:   python scripts/Data_validation.py --reject <filepath>")
    print(f"{'='*60}\n")

    return files


# ---------------------------------------------------------------------------
# MLflow integration with data versioning
# ---------------------------------------------------------------------------

def validate_and_log_to_mlflow(
    df: pd.DataFrame,
    source_file: str = "",
    run_id: Optional[str] = None,
    experiment_name: Optional[str] = None,
    strict: bool = False,
    label: str = "listings",
) -> Tuple[bool, Path, dict]:
    """
    Full pipeline step: validate, route (quarantine), and log to MLflow.

    MLflow logs:
        - Data quality metrics (pass rate, row/col count)
        - Dataset hash for versioning
        - Schema snapshot
        - Validation report as artifact
        - Source file path
    """
    try:
        import mlflow
    except ImportError:
        raise ImportError("mlflow is required. Install with: pip install mlflow")

    # Step 1: validate and route
    success, filepath, summary = validate_and_route(df, strict=False, label=label)

    # Step 2: build versioning artifacts
    dataset_hash = summary["dataset_hash"]
    schema = build_schema_snapshot(df)

    schema_file = REPORT_DIR / f"schema_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(schema_file, "w") as f:
        json.dump(schema, f, indent=2, default=str)

    # Step 3: log to MLflow
    def _log():
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

        mlflow.set_tag("data_validation_passed", str(success))
        mlflow.set_tag("dataset_hash", dataset_hash)
        mlflow.set_tag("data_path", str(filepath))
        mlflow.set_tag("source_file", source_file)
        mlflow.set_tag("quarantined", str(not success))

        mlflow.log_artifact(summary["report_path"], artifact_path="data_validation")
        mlflow.log_artifact(str(schema_file), artifact_path="data_validation")

        mlflow.log_param("dataset_hash", dataset_hash[:16])
        mlflow.log_param("dataset_rows", summary["row_count"])
        mlflow.log_param("source_file", str(source_file))

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
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate cleaned housing data and route via quarantine workflow."
    )

    # Input source
    parser.add_argument("--file", type=str, default=None,
                        help="Path to a specific cleaned parquet file. "
                             "If not provided, uses the most recent file in data/cleaned/")

    # MLflow
    parser.add_argument("--mlflow", action="store_true",
                        help="Log results to MLflow")
    parser.add_argument("--experiment", type=str, default="housing_data_validation",
                        help="MLflow experiment name")
    parser.add_argument("--strict", action="store_true",
                        help="Raise error on validation failure")

    # Review mode
    parser.add_argument("--review", action="store_true",
                        help="List quarantined files and their failure reasons")
    parser.add_argument("--approve", type=str, default=None,
                        help="Approve a quarantined file (move to data/raw/)")
    parser.add_argument("--reject", type=str, default=None,
                        help="Reject a quarantined file (move to data/archive/)")

    args = parser.parse_args()

    # --- Review mode ---
    if args.review:
        list_quarantined()
        return

    if args.approve:
        review_quarantined(args.approve, approve=True)
        return

    if args.reject:
        review_quarantined(args.reject, approve=False)
        return

    # --- Validation mode ---
    if args.file:
        source_file = Path(args.file)
        if not source_file.exists():
            raise FileNotFoundError(f"File not found: {source_file}")
    else:
        source_file = find_latest_cleaned_file()

    logger.info(f"Loading: {source_file}")
    df = pd.read_parquet(source_file)
    logger.info(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns from {source_file}")

    # --- Validate + Route (+ optional MLflow) ---
    if args.mlflow:
        success, filepath, summary = validate_and_log_to_mlflow(
            df,
            source_file=str(source_file),
            experiment_name=args.experiment,
            strict=args.strict,
        )
    else:
        success, filepath, summary = validate_and_route(
            df,
            strict=args.strict,
        )

    print(f"\n{'='*60}")
    print(f"  Source:    {source_file}")
    print(f"  Result:    {'PASSED' if success else 'FAILED (quarantined)'}")
    print(f"  Data:      {filepath}")
    print(f"  Rows:      {summary['row_count']}")
    print(f"  Hash:      {summary['dataset_hash'][:16]}...")
    print(f"  Report:    {summary.get('report_path', 'N/A')}")
    if not success:
        print(f"  Failures:  {summary['failed']}")
        for f in summary.get("failures", []):
            print(f"    ❌ {f['expectation']} | {f['kwargs']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
