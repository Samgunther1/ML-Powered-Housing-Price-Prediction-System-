# Data Version Log

## Examples

## Raw Data (Raw data scrapes no processing)

### v1 — 2026-03-27
- **Source**: HomeHarvest (Realtor.com)
- **Locations**: Cincinnati OH, Columbus OH, Dayton OH
- **Listing type**: Sold
- **Date range**: Past 365 days
- **Records**: 8,412
- **Passed validation**: 8,410
- **Failed valiation**: 2
- **Scraped by**: Sam

## Error Data (Failed validation phase)

### fails from v1 - 2026-03-27
- Details about what failed and why
- Manual review (what was manually passed or failed)
- Recordsmoved to archive
- Records moved back into raw data


## Processed Data (Transformed data)

### v1 — 2026-04-05
- **Based on**: Raw v2
- **Changes**: Dropped rows missing sqft or price, removed outliers (price > $2M),
  created price_per_sqft feature, one-hot encoded property type
- **Records**: 13,210
- **Processed by**: Partner A


## Data Version Log Starts Here

## Raw Data


## Processed Data