"""
streamlit_app.py – Housing Price Prediction UI
===============================================
Fully dynamic Streamlit frontend. Fetches the feature schema from the
FastAPI backend and builds input fields accordingly. If features change
after retraining, the UI adapts automatically — no code changes needed.

Usage:
    streamlit run streamlit_app.py --server.port 8501
"""

import os

import requests
import streamlit as st

# ── Configuration ────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")


# ── Page Setup ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cincinnati Housing Price Predictor",
    page_icon="🏠",
    layout="centered",
)

st.title("🏠 Cincinnati Housing Price Predictor")
st.markdown(
    "Enter property details below to get an estimated sale price. "
    "Fields marked *(optional)* can be left at their defaults."
)
st.divider()


# ── Load Schema from API ────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_schema():
    """Fetch feature schema from the backend."""
    try:
        resp = requests.get(f"{API_URL}/schema", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(
            f"Could not connect to the prediction API at `{API_URL}`. "
            f"Make sure the backend is running.\n\n**Error:** {e}"
        )
        return None


schema = load_schema()

if schema is None:
    st.stop()

# ── Helpers ──────────────────────────────────────────────────────────────
# Features the user doesn't need to input (auto-derived by the API)
auto_derived_keys = set(schema.get("auto_derived", {}).keys())

# Separate numeric features into user-facing vs auto-derived
user_numeric = [f for f in schema["numeric_features"] if f not in auto_derived_keys]


def pretty_label(col_name: str) -> str:
    """Convert snake_case column names to readable labels."""
    return col_name.replace("_", " ").title()


# ── Input Form: Numeric Features ────────────────────────────────────────
st.subheader("Property Details")

# Lay out numeric fields in 3 columns
defaults = schema.get("numeric_defaults", {})
numeric_inputs = {}

cols = st.columns(3)
for i, feat in enumerate(user_numeric):
    default_val = defaults.get(feat, 0)
    with cols[i % 3]:
        # Use integer input for features that are naturally whole numbers
        if feat in ("beds", "half_baths", "parking_garage"):
            numeric_inputs[feat] = float(st.number_input(
                pretty_label(feat),
                min_value=0, value=int(default_val), step=1,
                key=f"num_{feat}",
            ))
        elif feat in ("full_baths",):
            numeric_inputs[feat] = float(st.number_input(
                pretty_label(feat),
                min_value=1, value=int(default_val), step=1,
                key=f"num_{feat}",
            ))
        elif feat in ("year_built",):
            numeric_inputs[feat] = float(st.number_input(
                pretty_label(feat),
                min_value=1800, max_value=2026, value=int(default_val), step=1,
                key=f"num_{feat}",
            ))
        elif feat in ("sqft", "lot_sqft"):
            numeric_inputs[feat] = float(st.number_input(
                pretty_label(feat),
                min_value=0, value=int(default_val), step=50,
                key=f"num_{feat}",
            ))
        elif feat in ("hoa_fee",):
            numeric_inputs[feat] = st.number_input(
                f"{pretty_label(feat)} ($/mo)",
                min_value=0.0, value=float(default_val), step=25.0,
                key=f"num_{feat}",
            )
        elif feat in ("stories",):
            numeric_inputs[feat] = st.number_input(
                pretty_label(feat),
                min_value=0.0, max_value=5.0, value=float(default_val), step=0.5,
                key=f"num_{feat}",
            )
        else:
            # Generic numeric input for any new features
            numeric_inputs[feat] = st.number_input(
                pretty_label(feat),
                value=float(default_val),
                key=f"num_{feat}",
            )


# ── Input Form: Binary Features ─────────────────────────────────────────
binary_inputs = {}
if schema["binary_features"]:
    for feat in schema["binary_features"]:
        binary_inputs[feat] = st.checkbox(
            pretty_label(feat), value=False, key=f"bin_{feat}"
        )


# ── Input Form: Categorical Features ────────────────────────────────────
st.subheader("Location & Type")

categorical_inputs = {}
cat_groups = schema["categorical_groups"]

cols2 = st.columns(2)
for i, (group_name, values) in enumerate(cat_groups.items()):
    with cols2[i % 2]:
        # Determine a sensible default
        default_map = {
            "style": "SINGLE_FAMILY",
            "county": "Hamilton",
        }
        default_val = default_map.get(group_name)

        # Some categoricals should be optional (city, zip_code)
        optional_groups = {"city", "zip_code"}
        if group_name in optional_groups:
            options_list = ["(Not specified)"] + sorted(values)
            chosen = st.selectbox(
                f"{pretty_label(group_name)} *(optional)*",
                options=options_list,
                index=0,
                key=f"cat_{group_name}",
            )
            categorical_inputs[group_name] = None if chosen == "(Not specified)" else chosen
        else:
            sorted_vals = sorted(values)
            default_idx = sorted_vals.index(default_val) if default_val in sorted_vals else 0
            chosen = st.selectbox(
                pretty_label(group_name),
                options=sorted_vals,
                index=default_idx,
                key=f"cat_{group_name}",
            )
            categorical_inputs[group_name] = chosen

st.divider()


# ── Prediction ───────────────────────────────────────────────────────────
if st.button("💰 Predict Price", type="primary", use_container_width=True):
    payload = {
        "numeric": numeric_inputs,
        "categorical": categorical_inputs,
        "binary": binary_inputs,
    }

    with st.spinner("Getting prediction..."):
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            resp.raise_for_status()
            result = resp.json()

            st.success("Prediction complete!")
            st.metric(
                label="Estimated Sale Price",
                value=result["predicted_price_formatted"],
            )

            with st.expander("Input Summary"):
                for k, v in result["input_summary"].items():
                    display_val = v if v is not None else "—"
                    st.text(f"{pretty_label(k)}: {display_val}")

            with st.expander("Model Info"):
                st.text(f"Model: {result['model_name']}")
                st.text(f"Alias: {result['model_alias']}")

        except requests.exceptions.ConnectionError:
            st.error(
                f"Could not connect to the prediction API at `{API_URL}`. "
                "Make sure the backend is running."
            )
        except requests.exceptions.HTTPError as e:
            detail = e.response.json().get("detail", str(e))
            st.error(f"Prediction failed: {detail}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")


# ── Footer ───────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Powered by a Random Forest model tuned with Optuna and tracked in MLflow. "
    "The UI is built dynamically from the model's feature schema — "
    "retrain with new features and the UI updates automatically."
)
