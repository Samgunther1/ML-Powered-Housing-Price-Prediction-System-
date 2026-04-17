# ML-Powered Housing Price Prediction System
### BANA 7075 — Final Project (Group 2)

## 👥 Contributors
* **Samuel Gunther**
* **Ryan O'Connor**
* **Robin Cohen**
* **BJ Osibogun**

---

## 🚀 Project Overview
This project addresses real estate price volatility by providing ML-powered, data-driven valuations. By scraping real-time data via the `HomeHarvest` API, the system provides accurate estimates to help stakeholders navigate volatile market conditions with confidence.

### 🎯 Target Stakeholders
* **Primary:** Home buyers, sellers, and mortgage lenders requiring high-accuracy valuations for financial decisions.
* **Secondary:** Local government agencies analyzing housing affordability and market trends.

---

## ✨ Value Proposition
* **Market Transparency:** Provides accessible pricing insights to reduce miscommunication between buyers and sellers.
* **Operational Efficiency:** Reduces reliance on manual appraisals, cutting down administrative costs and delays.
* **Risk Mitigation:** Allows lenders to better assess collateral value, lowering the risk of over-lending and foreclosures.

---

## 🛠️ Tech Stack & Design Principles
* **Model:** **Random Forest Regressor**
    * *Justification:* Chosen for its resistance to outliers in volatile markets and its ability to capture non-linear patterns (e.g., the complex impact of school districts on value).
* **Data Source:** `HomeHarvest` (Real-time Realtor.com scraping).
* **Validation:** `great_expectations` (Ensures data completeness and schema accuracy).
* **Versioning:** `MLflow` (Tracking experiments, metadata, and data lineage).

### Key Design Principles
1.  **Adaptability:** The pipeline can be retrained for any geographical location (e.g., California vs. Midwest) by simply updating the API query.
2.  **Automation:** Model tuning and hyperparameter optimization are automated to adapt to new data without manual code changes.
3.  **Modularity:** Each stage (Ingestion, Validation, Training) is self-contained, ensuring that updates to one do not break the others.

---

## 🏗️ Data Pipeline Architecture
The system follows a modular "DataOps" approach to ensure model integrity:

1.  **Ingestion:** Raw data is scraped via `scrape_property` and logged in MLflow.
2.  **Validation (Great Expectations):**
    * **Completeness:** Identifies missing values; triggers imputation or exclusion.
    * **Accuracy:** Range checks (e.g., Year Built $\geq$ 1900, Stories between 1–5).
    * **Schema:** Ensures Zip Codes are 5-digit strings and bedrooms are integers.
3.  **Processing:** Valid

4.  **Training & Tracking:** Random Forest modeling with performance logging in MLflow.

---

## 📊 Performance Metrics

| Category | Metric | Significance |
| :--- | :--- | :--- |
| **Technical** | **RMSE / MAE** | Measures average error and penalizes large pricing mistakes. |
| **Technical** | **Latency** | Ensures the user interface returns predictions quickly. |
| **Business** | **Sold Price Accuracy** | Tracks how often predicted prices match final sale prices. |
| **Business** | **Loan Default Rate** | Monitors if better valuations lead to safer lending decisions. |

---

## 📅 Development Roadmap

| Phase | Tasks | Deliverables |
| :--- | :--- | :--- |
| **1. Proposal** | Define problem, identify dataset, confirm approach. | Project Proposal |
| **2. Data & Setup** | Load data, clean dataset, explore features. | Cleaned Dataset |
| **3. Prototype** | Train initial model, build basic user interface. | Working Prototype |
| **4. Refinement** | Tune hyperparameters, improve UI/UX. | Optimized System |
| **5. Delivery** | Final testing, documentation, and demo. | Final Report & Video |

---
