"""
scrape_and_clean.py

Scrapes housing data from Realtor.com via HomeHarvest, then applies cleaning steps:
    1. Drop completely empty rows
    2. Deduplicate rows
    3. Remove rows missing sold_price (target variable)
    4. Remove outliers via IQR method for numeric columns

Usage:
    python scripts/scrape_and_clean.py --location "Cincinnati, OH" --past_days 365
    python scripts/scrape_and_clean.py --location "Cincinnati, OH" --past_days 365 --iqr_multiplier 2.0
    python scripts/scrape_and_clean.py --location "Cincinnati, OH" --listing_type sold --past_days 180 --output data/cleaned
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

# Columns to check for outliers via IQR
OUTLIER_COLUMNS = [
    "sold_price",
    "sqft",
    #"beds",
    #"full_baths",
    #"half_baths",
    #"lot_sqft",
    #"stories",
    #"parking_garage",
    #"days_on_mls",
]

VALID_STYLES = [
    "SINGLE_FAMILY",
    "CONDO",
    "CONDOS",
    "TOWNHOMES",
    "MULTI_FAMILY",
    "APARTMENNT",
]

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


def drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where every column is NaN."""
    before = len(df)
    df = df.dropna(how="all")
    dropped = before - len(df)
    if dropped > 0:
        logger.info(f"Dropped {dropped} completely empty rows")
    else:
        logger.info("No completely empty rows found")
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows, keeping the first occurrence.
    
    Excludes columns containing unhashable types (lists, dicts)
    that HomeHarvest sometimes returns (e.g. nearby_schools, agents, photos).
    """
    before = len(df)

    # Find columns that only contain hashable values
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
            logger.info(f"Skipping column '{col}' during dedup (contains {type(first_val).__name__})")

    df = df.drop_duplicates(subset=hashable_cols)
    dropped = before - len(df)
    if dropped > 0:
        logger.info(f"Dropped {dropped} duplicate rows (checked {len(hashable_cols)}/{len(df.columns)} columns)")
    else:
        logger.info("No duplicate rows found")
    return df


def remove_missing_target(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where sold_price is null."""
    before = len(df)
    df = df.dropna(subset=["sold_price"])
    dropped = before - len(df)
    if dropped > 0:
        logger.info(f"Dropped {dropped} rows with missing sold_price")
    else:
        logger.info("No rows with missing sold_price")
    return df

def filter_style(df: pd.DataFrame, valid_styles: list = VALID_STYLES) -> pd.DataFrame:
    """Keep only rows with property styles relevant to price prediction."""
    before = len(df)
    df = df[df["style"].isin(valid_styles) | df["style"].isna()]
    dropped = before - len(df)
    if dropped > 0:
        logger.info(f"Dropped {dropped} rows with excluded styles (kept: {valid_styles})")
    else:
        logger.info("No rows dropped by style filter")
    return df

def remove_outliers_iqr(
    df: pd.DataFrame,
    columns: list,
    multiplier: float = 1.5,
) -> pd.DataFrame:
    """
    Remove outliers using the IQR method.

    For each column, values outside [Q1 - multiplier*IQR, Q3 + multiplier*IQR]
    are considered outliers. Rows with an outlier in ANY of the specified columns
    are removed.

    NaN values in a given column are ignored (not treated as outliers).

    Args:
        df: Input DataFrame
        columns: Columns to check for outliers
        multiplier: IQR multiplier (1.5 = standard, 2.0 = more lenient)

    Returns:
        DataFrame with outlier rows removed
    """
    before = len(df)
    mask = pd.Series(True, index=df.index)

    outlier_report = {}

    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found in DataFrame — skipping outlier check")
            continue

        # Only consider non-null values for IQR calculation
        series = pd.to_numeric(df[col], errors="coerce")

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr

        # Keep rows that are within bounds OR are NaN (don't penalize missing data here)
        col_mask = (series >= lower) & (series <= upper) | series.isna()
        col_outliers = (~col_mask).sum()

        if col_outliers > 0:
            outlier_report[col] = {
                "outliers_found": int(col_outliers),
                "lower_bound": round(lower, 2),
                "upper_bound": round(upper, 2),
                "Q1": round(q1, 2),
                "Q3": round(q3, 2),
            }

        mask = mask & col_mask

    df = df[mask]
    total_dropped = before - len(df)

    logger.info(f"Outlier removal (IQR x{multiplier}): dropped {total_dropped} rows total")
    for col, info in outlier_report.items():
        logger.info(
            f"  {col}: {info['outliers_found']} outliers | "
            f"bounds=[{info['lower_bound']}, {info['upper_bound']}]"
        )

    return df


def clean_data(
    df: pd.DataFrame,
    iqr_multiplier: float = 1.5,
) -> pd.DataFrame:
    """
    Full cleaning pipeline:
        1. Drop completely empty rows
        2. Deduplicate
        3. Remove rows without sold_price
        4. Remove land,other styles
        5. Remove outliers (IQR)

    Returns:
        Cleaned DataFrame
    """
    logger.info(f"{'='*60}")
    logger.info(f"Starting cleaning pipeline — {len(df)} rows in")
    logger.info(f"{'='*60}")

    df = drop_empty_rows(df)
    df = deduplicate(df)
    df = remove_missing_target(df)
    df = filter_style(df)
    df = remove_outliers_iqr(df, columns=OUTLIER_COLUMNS, multiplier=iqr_multiplier)

    df = df.reset_index(drop=True)

    logger.info(f"{'='*60}")
    logger.info(f"Cleaning complete — {len(df)} rows remaining")
    logger.info(f"{'='*60}")

    return df


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

    args = parser.parse_args()

    # --- Scrape ---
    df_raw = scrape_data(args.location, args.listing_type, args.past_days)

    # --- Clean ---
    df_clean = clean_data(df_raw, iqr_multiplier=args.iqr_multiplier)

    # --- Save ---
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    

    # Also save a CSV for easy inspection
    csv_file = output_dir / f"cleaned_{ts}.csv"
    df_clean.to_csv(csv_file, index=False)

    print(f"\n{'='*60}")
    print(f"  Raw rows:      {len(df_raw)}")
    print(f"  Cleaned rows:  {len(df_clean)}")
    print(f"  Removed:       {len(df_raw) - len(df_clean)} ({100*(len(df_raw)-len(df_clean))/max(len(df_raw),1):.1f}%)")
    print(f"  CSV:           {csv_file}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()