"""
validate_housing_data.py

Great Expectations validation module for the HomeHarvest housing price prediction pipeline.
Validates CLEANED data (output of scrape_and_clean.py) before feature engineering and model training.

Workflow (quarantine pattern):
    Cleaned data (from scrape_and_clean.py)
        |
    Validation checks
        |
        +---> PASS --> data/processed/        (ready for feature engineering)
        +---> FAIL --> data/errors/     (quarantined for manual review)
                            |
                       Manual review
                            |
                       +---> Approved  --> moved to data/processed/
                       +---> Rejected  --> archived or deleted

Usage:
    # Validate the most recent cleaned file (default):
    python scripts/Data_validation.py --mlflow

    # Validate a specific file:
    python scripts/Data_validation.py --file data/cleaned/cleaned_20260406_150000.parquet --mlflow

    # Review quarantined files:
    python scripts/Data_validation.py --review
    python scripts/Data_validation.py --approve data/errors/listings_20260406_150000.parquet --reason "Minor nulls in lot_sqft, acceptable for modeling"
    python scripts/Data_validation.py --reject data/errors/listings_20260406_150000.parquet --reason "Sold_price distribution too skewed, re-scrape needed"
"""

import hashlib
import json
import logging
import shutil
import argparse
import getpass
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
PROCESSED_DIR = DATA_DIR / "processed"           # validated, ready for modeling
ERRORS_DIR = DATA_DIR / "errors"     # failed validation, quarantined
ARCHIVE_DIR = DATA_DIR / "archive"   # rejected after manual review
REPORT_DIR = Path("validation_reports")

for _dir in [CLEANED_DIR, PROCESSED_DIR, ERRORS_DIR, ARCHIVE_DIR, REPORT_DIR]:
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
         "APARTMENT",
        ],
        0.98,
    ),
    "status": (
        ["SOLD"],
        1.0,
    ),
}

MIN_ROW_COUNT = 50  # cleaned data should have a reasonable number of rows


# ---------------------------------------------------------------------------
# MLflow review experiment config
# ---------------------------------------------------------------------------
REVIEW_EXPERIMENT_NAME = "housing_quarantine_reviews"


# ---------------------------------------------------------------------------
# Find the most recent cleaned file
# ---------------------------------------------------------------------------

def find_latest_cleaned_file(directory: Path = CLEANED_DIR) -> Path:
    """Find the most recently created CSV file in the cleaned directory."""
    csv_files = sorted(directory.glob("*.csv"), key=lambda f: f.stat().st_mtime)

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {directory}. "
            f"Run scrape_and_clean.py first."
        )

    latest = csv_files[-1]
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
    context: gx.data_context,
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
    context: Optional[gx.data_context] = None,
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

def _save_csv(df: pd.DataFrame, directory: Path, label: str) -> Path:
    """Save a DataFrame as a timestamped CSV file."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = directory / f"{label}_{ts}.csv"
    df.to_csv(filepath, index=False)
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
    Validate a cleaned DataFrame and route to processed/ or errors/.

    Pass -> data/processed/
    Fail -> data/errors/ (quarantined)
    """
    logger.info(f"Validating cleaned data: {len(df)} rows, {len(df.columns)} columns")

    success, results, summary = run_validation(df)
    report_path = _save_report(summary)
    summary["report_path"] = str(report_path)

    if success:
        filepath = _save_csv(df, PROCESSED_DIR, label)
        logger.info(
            f"✅ PASSED — {summary['passed']}/{summary['evaluated']} expectations met. "
            f"Data saved to {filepath}"
        )
    else:
        filepath = _save_csv(df, ERRORS_DIR, label)
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


def _find_matching_validation_report(quarantined_path: Path) -> Optional[dict]:
    """
    Find the original validation report for a quarantined file by matching
    the data_path field in saved reports.
    """
    reports = sorted(REPORT_DIR.glob("validation_*.json"), key=lambda f: f.stat().st_mtime)
    for rpt_path in reversed(reports):
        try:
            with open(rpt_path) as f:
                data = json.load(f)
            # Match by data_path or by filename substring
            saved_data_path = data.get("data_path", "")
            if (
                str(quarantined_path) in saved_data_path
                or quarantined_path.name in saved_data_path
            ):
                return data
        except Exception:
            continue
    return None


def _save_review_report(
    quarantined_path: Path,
    decision: str,
    dest: Path,
    reason: str,
    reviewer: str,
    original_report: Optional[dict],
) -> Path:
    """Save a review decision report as JSON."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    review_report = {
        "review_timestamp": datetime.now().isoformat(),
        "decision": decision,
        "reason": reason,
        "reviewer": reviewer,
        "quarantined_file": str(quarantined_path),
        "destination_file": str(dest),
    }

    # Carry forward context from the original validation report
    if original_report:
        review_report["original_validation"] = {
            "timestamp": original_report.get("timestamp"),
            "evaluated": original_report.get("evaluated"),
            "passed": original_report.get("passed"),
            "failed": original_report.get("failed"),
            "failures": original_report.get("failures", []),
            "dataset_hash": original_report.get("dataset_hash"),
            "row_count": original_report.get("row_count"),
        }

    report_file = REPORT_DIR / f"review_{decision}_{ts}.json"
    with open(report_file, "w") as f:
        json.dump(review_report, f, indent=2, default=str)
    logger.info(f"Review report saved to {report_file}")
    return report_file


def _log_review_to_mlflow(
    quarantined_path: Path,
    decision: str,
    dest: Path,
    reason: str,
    reviewer: str,
    review_report_path: Path,
    original_report: Optional[dict],
    experiment_name: str = REVIEW_EXPERIMENT_NAME,
) -> None:
    """Log the quarantine review decision to MLflow."""
    try:
        import mlflow
    except ImportError:
        logger.warning(
            "mlflow not installed — review logged to JSON only. "
            "Install with: pip install mlflow"
        )
        return

    mlflow.set_experiment(experiment_name)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"review_{decision}_{quarantined_path.stem}_{ts}"

    with mlflow.start_run(run_name=run_name):
        # --- Tags: decision metadata ---
        mlflow.set_tag("review_decision", decision)
        mlflow.set_tag("review_reason", reason)
        mlflow.set_tag("reviewer", reviewer)
        mlflow.set_tag("quarantined_file", str(quarantined_path))
        mlflow.set_tag("destination_file", str(dest))

        # --- Carry forward original validation context ---
        if original_report:
            dataset_hash = original_report.get("dataset_hash", "unknown")
            mlflow.set_tag("dataset_hash", dataset_hash)
            mlflow.log_param("dataset_hash", dataset_hash[:16])
            mlflow.log_param("dataset_rows", original_report.get("row_count", 0))

            mlflow.log_metrics({
                "dq_expectations_total": original_report.get("evaluated", 0),
                "dq_expectations_passed": original_report.get("passed", 0),
                "dq_expectations_failed": original_report.get("failed", 0),
                "dq_pass_rate": (
                    original_report.get("passed", 0)
                    / max(original_report.get("evaluated", 1), 1)
                ),
            })

            # Log count of failures that were manually accepted
            failures = original_report.get("failures", [])
            if decision == "approved":
                mlflow.log_metric("dq_failures_manually_accepted", len(failures))

        # --- Log the review report as artifact ---
        mlflow.log_artifact(str(review_report_path), artifact_path="quarantine_reviews")

    logger.info(f"Review decision logged to MLflow experiment '{experiment_name}'")


def review_quarantined(
    quarantined_path: str,
    approve: bool = True,
    reason: str = "",
    reviewer: str = "",
    log_to_mlflow: bool = True,
) -> Path:
    """
    Manual review: approve (move to processed/) or reject (move to archive/).

    Logs the decision, reason, and reviewer to both a JSON report and MLflow
    so there is a full audit trail for the team.
    """
    src = Path(quarantined_path)
    if not src.exists():
        raise FileNotFoundError(f"Quarantined file not found: {src}")

    decision = "approved" if approve else "rejected"

    # Default reviewer to the current OS user
    if not reviewer:
        try:
            reviewer = getpass.getuser()
        except Exception:
            reviewer = "unknown"

    # Prompt for reason if not provided (interactive usage)
    if not reason:
        reason = f"Manual {decision} — no reason provided"

    if approve:
        dest = PROCESSED_DIR / src.name
    else:
        dest = ARCHIVE_DIR / src.name

    # Look up the original validation report for context
    original_report = _find_matching_validation_report(src)

    # Save JSON review report
    review_report_path = _save_review_report(
        quarantined_path=src,
        decision=decision,
        dest=dest,
        reason=reason,
        reviewer=reviewer,
        original_report=original_report,
    )

    # Log to MLflow
    if log_to_mlflow:
        _log_review_to_mlflow(
            quarantined_path=src,
            decision=decision,
            dest=dest,
            reason=reason,
            reviewer=reviewer,
            review_report_path=review_report_path,
            original_report=original_report,
        )

    # Move the file
    shutil.move(str(src), str(dest))
    action = "APPROVED → data/processed/" if approve else "REJECTED → data/archive/"
    logger.info(f"Review: {action} ({dest})")
    logger.info(f"  Reviewer: {reviewer}")
    logger.info(f"  Reason:   {reason}")

    return dest


def list_quarantined() -> list:
    """List all files currently in the errors/ quarantine directory."""
    files = sorted(ERRORS_DIR.glob("*.csv"), key=lambda f: f.stat().st_mtime)
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
    print(f"  To approve:  python scripts/Data_validation.py --approve <filepath> --reason \"...\"")
    print(f"  To reject:   python scripts/Data_validation.py --reject <filepath> --reason \"...\"")
    print(f"  Add --reviewer <name> to tag who reviewed (defaults to OS user)")
    print(f"  Add --no-mlflow to skip MLflow logging")
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
                        help="Approve a quarantined file (move to data/processed/)")
    parser.add_argument("--reject", type=str, default=None,
                        help="Reject a quarantined file (move to data/archive/)")
    parser.add_argument("--reason", type=str, default="",
                        help="Reason for approve/reject decision (logged to MLflow)")
    parser.add_argument("--reviewer", type=str, default="",
                        help="Name of the reviewer (defaults to OS username)")
    parser.add_argument("--no-mlflow", action="store_true",
                        help="Skip MLflow logging for review decisions")

    args = parser.parse_args()

    # --- Review mode ---
    if args.review:
        list_quarantined()
        return

    if args.approve:
        review_quarantined(
            args.approve,
            approve=True,
            reason=args.reason,
            reviewer=args.reviewer,
            log_to_mlflow=not args.no_mlflow,
        )
        return

    if args.reject:
        review_quarantined(
            args.reject,
            approve=False,
            reason=args.reason,
            reviewer=args.reviewer,
            log_to_mlflow=not args.no_mlflow,
        )
        return

    # --- Validation mode ---
    if args.file:
        source_file = Path(args.file)
        if not source_file.exists():
            raise FileNotFoundError(f"File not found: {source_file}")
    else:
        source_file = find_latest_cleaned_file()

    logger.info(f"Loading: {source_file}")
    df = pd.read_csv(source_file)
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