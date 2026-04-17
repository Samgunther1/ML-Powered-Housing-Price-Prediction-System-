# ML-Powered Housing Price Prediction System
### BANA 7075 — Final Project

## 🚀 Project Overview
This project addresses real estate price volatility by providing ML-powered valuations. By scraping data via the `HomeHarvest` API, we provide data-driven estimates for home buyers, sellers, and lenders.

## 👥 Contributors (Group 2)
* **Samuel Gunther**
* **Ryan O'Connor**
* **Robin Cohen**
* **BJ Osibogun**

## 🛠️ Tech Stack & DataOps
* **Model:** Random Forest Regressor (chosen for resistance to outliers).
* **Data Source:** `HomeHarvest` (Real-time Realtor.com scraping).
* **Validation:** `great_expectations` (Ensures data completeness and schema accuracy).
* **Versioning:** `MLflow` (Tracking experiments and data lineage).

## 📊 Performance Metrics
* **Technical:** RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error).
* **Business:** Sold Price Accuracy Rate and Revenue Impact.

## 🏗️ Data Pipeline Architecture
1. **Ingestion:** Scrape raw data via `scrape_property`.
2. **Validation:** Pass data through `great_expectations`.
3. **Tracking:** Record metadata and versioning in `MLflow`.
4. **Training:** Feature engineering and Random Forest modeling.


