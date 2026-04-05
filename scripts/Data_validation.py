"""
validate_housing_data.py
 
Great Expectations validation module for the HomeHarvest housing price prediction pipeline.
Validates raw scraped data from Realtor.com before feature engineering and model training.
 
Usage:
    # In your pipeline script or notebook:
    from validate_housing_data import validate_ingestion, validate_and_log_to_mlflow
 
    df = homeharvest.scrape(...)
    df_validated = validate_ingestion(df)  # raises on failure
 
    # Or with MLflow integration:
    df_validated = validate_and_log_to_mlflow(df, run_id="abc123")
"""
 
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from homeharvest import scrape_property
import pandas as pd
 
try:
    import great_expectations as gx
    from great_expectations.expectations import (
        ExpectColumnToExist,
        ExpectColumnValuesToBeBetween,
        ExpectColumnValuesToBeInSet,
        ExpectColumnValuesToNotBeNull,
        ExpectTableColumnCountToBeBetween,
        ExpectTableRowCountToBeGreater,
    )
except ImportError:
    raise ImportError(
        "great_expectations is required. Install with: pip install great_expectations"
    )
 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
df = scrape_property(
    location="Cincinnati, OH",
    listing_type="sold",
    past_days=365
)
print(data.shape)
data.head()


# ---------------------------------------------------------------------------
# Configuration: adjust these to match your HomeHarvest schema & tolerances
# ---------------------------------------------------------------------------
 
# Columns that MUST exist in the raw DataFrame
REQUIRED_COLUMNS = [
    "style"
    "list_price",
    "sold_price",
    "year_built",
    "days_on_mls",
    "sqft",
    "beds",
    "full_baths",
    "half_baths"
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
    "sold_price": 1.0,       # target variable — no nulls
    "formatted_address": 0.95,
    "city": 0.95,
    "county":0.95,
    "state": 0.99,
    "zip_code": 0.95,
    "status": 0.99,
    "style": 0.95,
    "sqft": 0.80,            
    "beds": 0.85,
    "baths": 0.85,
}
 
# Numeric range checks: column -> (min, max, mostly)
RANGE_EXPECTATIONS = {
    "list_price": (10_000, 100_000_000, 1.0),
    "sqft": (100, 50_000, 0.95),
    "beds": (0, 20, 0.99),
    "baths": (0, 20, 0.99),
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
            "OTHER"
            "CONDO"
        ],
        0.90,
    ),
}
 
# Minimum number of rows expected from a scrape
MIN_ROW_COUNT = 1
 
 
# ---------------------------------------------------------------------------
# Core validation logic
# ---------------------------------------------------------------------------
 
 
def build_expectation_suite(
    context: gx.DataContext,
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
 
    # --- Row count -----------------------------------------------------------
    suite.add_expectation(
        ExpectTableRowCountToBeGreater(value=MIN_ROW_COUNT)
    )
 
    # --- Column count sanity check -------------------------------------------
    suite.add_expectation(
        ExpectTableColumnCountToBeBetween(min_value=len(REQUIRED_COLUMNS))
    )
 
    # --- Required columns exist ----------------------------------------------
    for col in REQUIRED_COLUMNS:
        suite.add_expectation(ExpectColumnToExist(column=col))
 
    # --- Not-null checks -----------------------------------------------------
    for col, mostly in NOT_NULL_EXPECTATIONS.items():
        suite.add_expectation(
            ExpectColumnValuesToNotBeNull(column=col, mostly=mostly)
        )
 
    # --- Numeric range checks ------------------------------------------------
    for col, (min_val, max_val, mostly) in RANGE_EXPECTATIONS.items():
        suite.add_expectation(
            ExpectColumnValuesToBeBetween(
                column=col,
                min_value=min_val,
                max_value=max_val,
                mostly=mostly,
            )
        )
 
    # --- Categorical checks --------------------------------------------------
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
    context: Optional[gx.DataContext] = None,
):
    """
    Run Great Expectations validation on a raw housing DataFrame.
 
    Returns:
        (success: bool, results: ValidationResult, summary: dict)
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
 
    # Build a human-readable summary
    summary = {
        "success": results.success,
        "evaluated": results.statistics["evaluated_expectations"],
        "passed": results.statistics["successful_expectations"],
        "failed": results.statistics["unsuccessful_expectations"],
        "timestamp": datetime.now().isoformat(),
        "row_count": len(df),
        "column_count": len(df.columns),
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
# Pipeline-ready wrapper
# ---------------------------------------------------------------------------
 
 
def validate_ingestion(
    df: pd.DataFrame,
    strict: bool = True,
    save_report: bool = True,
    report_dir: str = "validation_reports",
) -> pd.DataFrame:
    """
    Validate a housing DataFrame. Drop into your pipeline right after scraping.
 
    Args:
        df: Raw DataFrame from HomeHarvest
        strict: If True, raise on validation failure. If False, log warnings.
        save_report: Save a JSON report to disk
        report_dir: Directory for validation reports
 
    Returns:
        The original DataFrame (pass-through for chaining)
 
    Raises:
        ValueError: If strict=True and validation fails
    """
    logger.info(f"Validating housing data: {len(df)} rows, {len(df.columns)} columns")
 
    success, results, summary = run_validation(df)
 
    # Save report
    if save_report:
        report_path = Path(report_dir)
        report_path.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = report_path / f"validation_{ts}.json"
        with open(report_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Validation report saved to {report_file}")
 
    if success:
        logger.info(
            f"✅ PASSED — {summary['passed']}/{summary['evaluated']} expectations met"
        )
    else:
        msg = (
            f"❌ FAILED — {summary['failed']}/{summary['evaluated']} expectations failed\n"
        )
        for failure in summary.get("failures", []):
            msg += f"  • {failure['expectation']} | {failure['kwargs']}\n"
 
        if strict:
            logger.error(msg)
            raise ValueError(msg)
        else:
            logger.warning(msg)
 
    return df
 
 
# ---------------------------------------------------------------------------
# MLflow integration
# ---------------------------------------------------------------------------
 
 
def validate_and_log_to_mlflow(
    df: pd.DataFrame,
    run_id: Optional[str] = None,
    experiment_name: Optional[str] = None,
    strict: bool = True,
) -> pd.DataFrame:
    """
    Validate data and log the results as MLflow artifacts/metrics.
    Ties data quality to each training run for full reproducibility.
 
    Args:
        df: Raw DataFrame from HomeHarvest
        run_id: Existing MLflow run ID to log under (if None, creates new run)
        experiment_name: MLflow experiment name (used if run_id is None)
        strict: Raise on failure if True
 
    Returns:
        The original DataFrame
    """
    try:
        import mlflow
    except ImportError:
        raise ImportError("mlflow is required. Install with: pip install mlflow")
 
    success, results, summary = run_validation(df)
 
    # Save report to a temp file for artifact logging
    report_dir = Path("validation_reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = report_dir / f"validation_{ts}.json"
    with open(report_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
 
    # Log to MLflow
    def _log(run):
        mlflow.log_artifact(str(report_file), artifact_path="data_validation")
        mlflow.log_metrics(
            {
                "dq_expectations_total": summary["evaluated"],
                "dq_expectations_passed": summary["passed"],
                "dq_expectations_failed": summary["failed"],
                "dq_pass_rate": summary["passed"] / max(summary["evaluated"], 1),
                "dq_row_count": summary["row_count"],
            }
        )
        mlflow.set_tag("data_validation_passed", str(success))
 
    if run_id:
        with mlflow.start_run(run_id=run_id):
            _log(None)
    else:
        if experiment_name:
            mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=f"data_validation_{ts}"):
            _log(None)
 
    status = "✅ PASSED" if success else "❌ FAILED"
    logger.info(f"{status} — results logged to MLflow")
 
    if not success and strict:
        msg = f"Validation failed: {summary['failed']} expectations did not pass"
        raise ValueError(msg)
 
    return df