# ML-Powered Housing Price Prediction System
### BANA 7075 — Final Project (Group 2)

## 👥 Contributors
* **Samuel Gunther**
* **Ryan O'Connor**
* **Robin Cohen**
* **BJ Osibogun**

---

## 🚀 Project Overview
This project addresses real estate price volatility by providing ML-powered, data-driven valuations. By scraping real-time data via the `HomeHarvest` API, the system provides estimates to help stakeholders navigate volatile market conditions.

### 🎯 Target Stakeholders
* **Primary:** Home buyers, sellers, and mortgage lenders requiring high-accuracy valuations for financial decisions
* **Secondary:** Local government agencies analyzing housing affordability and market trends

---

## ✨ Value Proposition

* **Financial Accuracy:** The ML system reducing pricing errors caused by market volatility by using data-driven predictions will help homebuyers avoid overpaying and enable sellers to list at competitive prices, leading to higher return on investments.
* **Operational Efficiency:** By reducing reliance on manual appraisals and other assessments, financial institutions, local government agencies, homebuyers and sellers can cut down operational and administrative costs.
* **Risk Mitigation:** Accurate property valuations allow banks and mortgage lenders to better assess collateral value before issuing loans. This lowers the risk of over-lending and foreclosures.
* **Market Transparency:** The model will make pricing insights more accessible and understandable to all stakeholders, which will reduce miscommunications and allow buyers, sellers, and regulators to make more informed decisions.
* **Policy Support:** Local governments can use data from the model to better understand housing affordability trends, which will allow them to make better informed decisions in setting up policies.
* **Consumer Confidence:** Consistent, data-backed price estimates will help users feel more confident in their decisions, which will improve trust in the overall real estate process. It can also increase market participation in volatile markets. 

---

## 🛠️ Model & Tech Stack
* **Model:** **Random Forest Regressor**
    * *Justification:* Chosen for its resistance to outliers in volatile markets and its ability to capture non-linear patterns (e.g., the impact of school districts on value)
* **Data Source:** `HomeHarvest` API
* **Validation:** `great_expectations`
* **Versioning:** `MLflow`

### Key Design Principles
1.  **Adaptability:** The pipeline can be retrained for any geographical location by simply updating the API query
2.  **Automation:** Model tuning and hyperparameter optimization are automated to adapt to new data without manual code changes
3.  **Modularity:** Each stage (Ingestion, Validation, Training) is self-contained, ensuring that updates to one do not break the others

---

## 🏗️ Data Pipeline Architecture
The system follows a modular "DataOps" approach to ensure model integrity:

1.  **Ingestion:** Raw data is scraped via `scrape_property` and logged in MLflow
2.  **Validation (Great Expectations):**
    * **Completeness:** Identifies missing values; triggers imputation or exclusion
    * **Accuracy:** Ensures fields contain appropriate values based on context, eg: years built are not in the future
    * **Schema:** Ensures each column contains values in the format assignd to the column, eg: ZIP codes are 5 digit strings
3.  **Processing:** Validated data undergoes feature engineering
4.  **Training & Tracking:** Random Forest modeling with performance logging in MLflow

---

## 📊 Performance Metrics

| Category | Metric | Significance |
| :--- | :--- | :--- |
| **Technical** | **RMSE** | Gives more weight to large errors, which is important for avoiding big pricing mistakes that can impact financial decisions |
| **Technical** | **MAE** | Will help show how far off predictions are on average by measuring the average difference between predicted and actual housing prices. |
| **Technical** | **Latency** | Ensures the user interface returns predictions quickly |
| **Business** | **Sold Price Accuracy** | Tracks how often predicted prices match final sale prices |
| **Business** | **Loan Default Rate** | Monitors if better valuations lead to safer lending decisions |

---

## 📅 Development Roadmap

| Phase | Tasks | Deliverable |
| :--- | :--- | :--- |
| **1. Proposal** | Define problem, identify dataset, confirm approach | Project Proposal |
| **2. Data & Setup** | Load data, clean dataset, explore features | Cleaned Dataset |
| **3. Prototype** | Train initial model, build basic user interface | Working Prototype |
| **4. Refinement** | Tune hyperparameters, improve UI/UX | Optimized System |
| **5. Delivery** | Final testing, documentation, and demo | Final Report & Video |

---
