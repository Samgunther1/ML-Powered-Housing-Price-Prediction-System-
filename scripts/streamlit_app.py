"""
streamlit_app.py – Housing Price Prediction UI
===============================================
Two-view Streamlit frontend:
    🏠 Predict  — Dynamic prediction form driven by the API's feature schema
    📊 MLflow   — Champion model summary card + embedded MLflow UI

Fetches the feature schema from the FastAPI backend and builds input
fields accordingly. If features change after retraining, the UI adapts
automatically — no code changes needed.

Usage:
    streamlit run streamlit_app.py --server.port 8501
"""

import os
from datetime import datetime

import requests
import streamlit as st

# ── Configuration ────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://localhost:5000")
MLFLOW_IFRAME_URL = os.getenv("MLFLOW_IFRAME_URL", MLFLOW_URL)
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "housing_price_model")


# ── Page Setup ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cincinnati Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ── API Helpers ──────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_schema():
    """Fetch feature schema from the backend."""
    try:
        resp = requests.get(f"{API_URL}/schema", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


@st.cache_data(ttl=60)
def load_health():
    """Fetch API health status."""
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


@st.cache_data(ttl=30)
def load_champion_info():
    """Fetch champion model details from the MLflow REST API.

    Uses the /registered-models/search endpoint (works across MLflow versions)
    to find the champion alias, then fetches the corresponding run metrics.
    """
    try:
        # Step 1: Get registered model info (includes aliases)
        resp = requests.get(
            f"{MLFLOW_URL}/api/2.0/mlflow/registered-models/search",
            params={"filter": f"name='{REGISTERED_MODEL_NAME}'"},
            timeout=5,
        )
        resp.raise_for_status()
        models = resp.json().get("registered_models", [])
        if not models:
            return None

        reg_model = models[0]
        aliases = reg_model.get("aliases", [])

        # Find the champion alias version
        champion_version = None
        for a in aliases:
            if a.get("alias") == "champion":
                champion_version = a.get("version")
                break

        if champion_version is None:
            return None

        # Step 2: Search model versions to get the champion's details
        mv_resp = requests.get(
            f"{MLFLOW_URL}/api/2.0/mlflow/model-versions/search",
            params={
                "filter": f"name='{REGISTERED_MODEL_NAME}'",
                "order_by": "version_number DESC",
                "max_results": 20,
            },
            timeout=5,
        )
        mv_resp.raise_for_status()
        all_versions = mv_resp.json().get("model_versions", [])

        # Find the champion version in the list
        mv = None
        for v in all_versions:
            if v.get("version") == champion_version:
                mv = v
                break

        if mv is None:
            return None

        # Attach alias info to each version for the history table
        alias_map = {}  # version -> list of alias names
        for a in aliases:
            ver = a.get("version")
            alias_map.setdefault(ver, []).append(a.get("alias"))
        for v in all_versions:
            v["aliases"] = alias_map.get(v.get("version"), [])

        # Parse tags
        raw_tags = mv.get("tags", {})
        if isinstance(raw_tags, list):
            tags = {t["key"]: t["value"] for t in raw_tags}
        else:
            tags = raw_tags

        # Step 3: Get the run details for metrics
        run_id = mv.get("run_id", "")
        run_metrics = {}
        run_params = {}
        if run_id:
            run_resp = requests.get(
                f"{MLFLOW_URL}/api/2.0/mlflow/runs/get",
                params={"run_id": run_id},
                timeout=5,
            )
            if run_resp.ok:
                run_data = run_resp.json().get("run", {}).get("data", {})
                run_metrics = {m["key"]: m["value"] for m in run_data.get("metrics", [])}
                run_params = {p["key"]: p["value"] for p in run_data.get("params", [])}

        return {
            "version": mv.get("version", "?"),
            "model_type": tags.get("model_type", "unknown"),
            "cv_adj_r2": tags.get("cv_adj_r2", "N/A"),
            "run_id": run_id,
            "creation_timestamp": mv.get("creation_timestamp", ""),
            "aliases": mv.get("aliases", []),
            "metrics": run_metrics,
            "params": run_params,
            "tags": tags,
            "all_versions": all_versions,
        }
    except Exception:
        return None


@st.cache_data(ttl=300)
def load_feature_importance(plot: str = "bar"):
    """Fetch the SHAP feature importance PNG from the API.

    `plot` is 'bar' (category-level mean |SHAP|) or 'beeswarm'
    (per-row distribution).
    """
    try:
        resp = requests.get(
            f"{API_URL}/feature-importance",
            params={"plot": plot},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.content
    except Exception:
        return None


def pretty_label(col_name: str) -> str:
    """Convert snake_case column names to readable labels."""
    return col_name.replace("_", " ").title()


def format_timestamp(ts_ms) -> str:
    """Convert an MLflow millisecond timestamp to a readable string."""
    try:
        ts_sec = int(ts_ms) / 1000
        return datetime.fromtimestamp(ts_sec).strftime("%b %d, %Y  %I:%M %p")
    except Exception:
        return "—"


def format_model_type(model_type: str) -> str:
    """Return a display-friendly model type name."""
    labels = {"rf": "Random Forest", "xgb": "XGBoost"}
    return labels.get(model_type, model_type)


# ── Title ────────────────────────────────────────────────────────────────
st.title("🏠 Cincinnati Housing Price Predictor")

# ── Tabs ─────────────────────────────────────────────────────────────────
tab_predict, tab_mlflow = st.tabs(["🏠 Predict", "📊 Model Dashboard"])


# ══════════════════════════════════════════════════════════════════════════
# TAB 1: PREDICT
# ══════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown(
        "Enter property details below to get an estimated sale price. "
        "Fields marked *(optional)* can be left at their defaults."
    )
    st.divider()

    schema = load_schema()

    if schema is None:
        st.error(
            f"Could not connect to the prediction API at `{API_URL}`. "
            "Make sure the backend is running."
        )
    else:
        # Features the user doesn't need to input (auto-derived by the API)
        auto_derived_keys = set(schema.get("auto_derived", {}).keys())
        user_numeric = [f for f in schema["numeric_features"] if f not in auto_derived_keys]

        # ── Numeric Features ─────────────────────────────────────────────
        st.subheader("Property Details")

        defaults = schema.get("numeric_defaults", {})
        bounds = schema.get("validity_bounds", {})
        rules = schema.get("validity_rules", {})
        numeric_inputs = {}

        cols = st.columns(3)
        for i, feat in enumerate(user_numeric):
            default_val = defaults.get(feat, 0)
            b = bounds.get(feat, {})
            # Use training data bounds for min/max when available
            feat_min = b.get("min", None)
            feat_max = b.get("max", None)

            with cols[i % 3]:
                label = pretty_label(feat)
                if feat_min is not None and feat_max is not None:
                    label += f"  ({feat_min:g}–{feat_max:g})"

                if feat in ("beds", "half_baths", "parking_garage"):
                    numeric_inputs[feat] = float(st.number_input(
                        label,
                        min_value=int(feat_min) if feat_min is not None else 0,
                        max_value=int(feat_max) if feat_max is not None else 99,
                        value=int(default_val), step=1,
                        key=f"num_{feat}",
                    ))
                elif feat in ("full_baths",):
                    numeric_inputs[feat] = float(st.number_input(
                        label,
                        min_value=int(feat_min) if feat_min is not None else 1,
                        max_value=int(feat_max) if feat_max is not None else 99,
                        value=int(default_val), step=1,
                        key=f"num_{feat}",
                    ))
                elif feat in ("year_built",):
                    numeric_inputs[feat] = float(st.number_input(
                        label,
                        min_value=int(feat_min) if feat_min is not None else 1800,
                        max_value=int(feat_max) if feat_max is not None else 2027,
                        value=int(default_val), step=1,
                        key=f"num_{feat}",
                    ))
                elif feat in ("sqft", "lot_sqft"):
                    numeric_inputs[feat] = float(st.number_input(
                        label,
                        min_value=int(feat_min) if feat_min is not None else 0,
                        max_value=int(feat_max) if feat_max is not None else 99999999,
                        value=int(default_val), step=50,
                        key=f"num_{feat}",
                    ))
                elif feat in ("hoa_fee",):
                    numeric_inputs[feat] = st.number_input(
                        f"{pretty_label(feat)} ($/mo)  ({feat_min:g}–{feat_max:g})" if feat_min is not None else f"{pretty_label(feat)} ($/mo)",
                        min_value=float(feat_min) if feat_min is not None else 0.0,
                        max_value=float(feat_max) if feat_max is not None else 99999.0,
                        value=float(default_val), step=25.0,
                        key=f"num_{feat}",
                    )
                elif feat in ("stories",):
                    numeric_inputs[feat] = st.number_input(
                        label,
                        min_value=float(feat_min) if feat_min is not None else 0.0,
                        max_value=float(feat_max) if feat_max is not None else 5.0,
                        value=float(default_val), step=0.5,
                        key=f"num_{feat}",
                    )
                else:
                    numeric_inputs[feat] = st.number_input(
                        label,
                        value=float(default_val),
                        key=f"num_{feat}",
                    )

        # ── Binary Features ──────────────────────────────────────────────
        binary_inputs = {}
        if schema["binary_features"]:
            for feat in schema["binary_features"]:
                binary_inputs[feat] = st.checkbox(
                    pretty_label(feat), value=False, key=f"bin_{feat}"
                )

        # ── New construction + year_built validation ─────────────────────
        new_construction_min_year = rules.get("new_construction_min_year")
        if binary_inputs.get("new_construction", False) and new_construction_min_year:
            year_input = numeric_inputs.get("year_built", 0)
            if year_input < new_construction_min_year:
                st.warning(
                    f"⚠️ New Construction is checked but Year Built ({int(year_input)}) "
                    f"is before {new_construction_min_year}. New construction properties "
                    f"in the training data were built in {new_construction_min_year} or later."
                )

        # ── Categorical Features ─────────────────────────────────────────
        st.subheader("Location & Type")

        categorical_inputs = {}
        cat_groups = schema["categorical_groups"]

        cols2 = st.columns(2)
        for i, (group_name, values) in enumerate(cat_groups.items()):
            with cols2[i % 2]:
                default_map = {
                    "style": "SINGLE_FAMILY",
                    "county": "Hamilton",
                }
                default_val = default_map.get(group_name)

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

        # ── Prediction Button ────────────────────────────────────────────
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

                    # Show warnings from validity domain checks
                    if result.get("warnings"):
                        for w in result["warnings"]:
                            st.warning(f"⚠️ {w}")

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
                    try:
                        detail = e.response.json().get("detail", str(e))
                    except Exception:
                        detail = str(e)
                    # Show 422 validity domain errors as warnings, others as errors
                    if e.response.status_code == 422:
                        st.error(f"🚫 Outside training domain: {detail}")
                    else:
                        st.error(f"Prediction failed: {detail}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")


# ══════════════════════════════════════════════════════════════════════════
# TAB 2: MODEL DASHBOARD
# ══════════════════════════════════════════════════════════════════════════
with tab_mlflow:

    champion = load_champion_info()
    health = load_health()

    # ── Champion Summary Card ────────────────────────────────────────────
    if champion:
        model_type_display = format_model_type(champion["model_type"])
        cv_score = champion["cv_adj_r2"]
        version = champion["version"]
        created = format_timestamp(champion["creation_timestamp"])

        # Banner
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #065f46 0%, #047857 100%);
            border: 1px solid #10b981;
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1rem;
        ">
            <h3 style="color: #ecfdf5; margin: 0 0 0.25rem 0;">
                🏆 Champion Model — v{version}
            </h3>
            <p style="color: #a7f3d0; margin: 0; font-size: 0.95rem;">
                {model_type_display} &nbsp;·&nbsp; Registered {created}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Metric cards row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Model Type", model_type_display)
        with col2:
            st.metric("Version", f"v{version}")
        with col3:
            st.metric("CV Adj R²", cv_score)
        with col4:
            feature_count = health.get("feature_count", "—") if health else "—"
            st.metric("Features", feature_count)

        # ── Training Metrics ─────────────────────────────────────────────
        metrics = champion.get("metrics", {})
        params = champion.get("params", {})

        if metrics:
            st.subheader("Training Run Metrics")
            metric_cols = st.columns(4)

            display_metrics = [
                ("train_rmse", "Train RMSE", "${:,.0f}"),
                ("train_mae", "Train MAE", "${:,.0f}"),
                ("train_r2", "Train R²", "{:.4f}"),
                ("train_adj_r2", "Train Adj R²", "{:.4f}"),
                ("best_cv_rmse_mean", "Best CV RMSE", "${:,.0f}"),
                ("best_cv_mae_mean", "Best CV MAE", "${:,.0f}"),
                ("best_cv_r2_mean", "Best CV R²", "{:.4f}"),
                ("best_cv_adj_r2_mean", "Best CV Adj R²", "{:.4f}"),
            ]

            shown = 0
            for key, label, fmt in display_metrics:
                if key in metrics:
                    with metric_cols[shown % 4]:
                        try:
                            formatted = fmt.format(float(metrics[key]))
                        except (ValueError, KeyError):
                            formatted = str(metrics[key])
                        st.metric(label, formatted)
                    shown += 1

        # ── Feature Importance (SHAP) ────────────────────────────────────
        st.subheader("Feature Importance (SHAP)")

        fi_bar = load_feature_importance("bar")
        if fi_bar is None:
            st.info(
                "Feature importance artifacts are not available yet. "
                "Run `python scripts/model_explanation.py` and then "
                "`docker compose restart api` to refresh."
            )
        else:
            view_choice = st.radio(
                "SHAP view",
                options=["Bar", "Distribution"],
                horizontal=True,
                label_visibility="collapsed",
                key="fi_view",
            )
            if view_choice == "Bar":
                st.image(
                    fi_bar,
                    use_container_width=True,
                    caption="Mean |SHAP value| per feature group — the average "
                            "impact each group has on the predicted sale price.",
                )
            else:
                fi_bee = load_feature_importance("beeswarm")
                if fi_bee is None:
                    st.warning("Beeswarm plot is not available.")
                else:
                    st.image(
                        fi_bee,
                        use_container_width=True,
                        caption="Per-row SHAP distribution — points to the right "
                                "pushed the prediction higher, points to the left "
                                "pushed it lower.",
                    )

        # ── Hyperparameters ──────────────────────────────────────────────
        if params:
            with st.expander("Hyperparameters"):
                skip_params = {"random_state", "n_jobs", "verbosity",
                               "dataset_hash", "dataset_rows", "source_file",
                               "output_file", "input_file", "input_path"}
                filtered = {k: v for k, v in params.items() if k not in skip_params}
                if filtered:
                    param_cols = st.columns(3)
                    for i, (k, v) in enumerate(sorted(filtered.items())):
                        with param_cols[i % 3]:
                            st.text(f"{pretty_label(k)}: {v}")

        # ── Version History ──────────────────────────────────────────────
        all_versions = champion.get("all_versions", [])
        if all_versions:
            st.subheader("Model Version History")

            history_rows = []
            for v in all_versions:
                raw_tags = v.get("tags", {})
                if isinstance(raw_tags, list):
                    vtags = {t["key"]: t["value"] for t in raw_tags}
                else:
                    vtags = raw_tags

                aliases = v.get("aliases", [])
                alias_str = ", ".join(aliases) if aliases else "—"

                history_rows.append({
                    "Version": f"v{v.get('version', '?')}",
                    "Type": format_model_type(vtags.get("model_type", "?")),
                    "CV Adj R²": vtags.get("cv_adj_r2", "—"),
                    "Alias": alias_str,
                    "Created": format_timestamp(v.get("creation_timestamp", "")),
                })

            st.dataframe(
                history_rows,
                use_container_width=True,
                hide_index=True,
            )

        st.divider()

    else:
        st.warning(
            f"Could not connect to MLflow at `{MLFLOW_URL}`. "
            "Make sure the MLflow tracking server is running."
        )

    # ── API Health ───────────────────────────────────────────────────────
    if health:
        with st.expander("API Health"):
            status_icon = "🟢" if health.get("status") == "healthy" else "🟡"
            st.text(f"{status_icon} Status: {health.get('status', 'unknown')}")
            st.text(f"Model loaded: {health.get('model_loaded', False)}")
            st.text(f"Schema loaded: {health.get('schema_loaded', False)}")
            st.text(f"Model: {health.get('model_name', '—')} @ {health.get('model_alias', '—')}")

    # ── MLflow Link ────────────────────────────────────────────────────────
    st.subheader("MLflow Tracking Server")

    st.markdown(
        f"🔗 **[Open MLflow UI in a new tab]({MLFLOW_IFRAME_URL})**"
    )
    st.caption(
        "Browse experiments, compare runs, and inspect artifacts in the full MLflow UI."
    )


# ── Footer ───────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Powered by a champion model (Random Forest or XGBoost) tuned with Optuna "
    "and tracked in MLflow. The UI is built dynamically from the model's feature "
    "schema — retrain with new features and the UI updates automatically."
)
