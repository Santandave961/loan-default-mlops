"""
dashboard.py
Streamlit MLOps dashboard: shows current registered model versions and stages,
lets you compare champion vs challenger metrics, shows drift status, and
provides a manual "promote to production" button (simulating an MLOps
approval workflow before flipping traffic to a new model).

Run:
    streamlit run app/dashboard.py
"""

import json
import os
from pathlib import Path

import mlflow
import pandas as pd
import streamlit as st

# MLflow 3.x deprecated the local filesystem tracking backend in favor of
# database backends (e.g. SQLite). This project intentionally uses the
# filesystem backend (mlruns/) for simplicity, so we explicitly opt back in.
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

REGISTERED_MODEL_NAME = "sme-loan-default-classifier"
ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = ROOT / "artifacts"
MLRUNS_DIR = ROOT / "mlruns"

# Explicitly point MLflow at the committed mlruns folder using an absolute
# path. Without this, MLflow resolves paths relative to the process's
# working directory, which isn't guaranteed to be the repo root on
# Streamlit Cloud. Path.as_uri() is used (not a manual f-string) because it
# correctly formats file:// URIs on both Windows (file:///C:/...) and
# Linux (file:///home/...) -- a manual f"file://{path}" breaks on Windows.
mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
mlflow.set_registry_uri(MLRUNS_DIR.as_uri())

st.set_page_config(page_title="SME Loan Default - MLOps Dashboard", layout="wide")
st.title("🇳🇬 SME Loan Default — MLOps Dashboard")
st.caption("Model registry, champion/challenger comparison, and drift monitoring")

client = mlflow.MlflowClient()


def get_all_versions():
    try:
        return client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    except Exception:
        return []


def get_aliases_by_version():
    """Returns {version: [alias1, alias2, ...]} for the registered model.
    Version keys are normalized to strings since different MLflow versions
    return version numbers as either str or int."""
    try:
        rm = client.get_registered_model(REGISTERED_MODEL_NAME)
        by_version = {}
        for alias, version in rm.aliases.items():
            by_version.setdefault(str(version), []).append(alias)
        return by_version
    except Exception:
        return {}


# ---- Section 1: Registry Overview ----
st.header("1. Model Registry")
versions = get_all_versions()

if not versions:
    st.warning(
        f"No registered versions found for '{REGISTERED_MODEL_NAME}'. "
        "Run `python src/train.py` first."
    )
else:
    aliases_by_version = get_aliases_by_version()
    reg_data = []
    for v in versions:
        run = client.get_run(v.run_id)
        version_str = str(v.version)
        aliases = aliases_by_version.get(version_str, [])
        reg_data.append({
            "version": version_str,
            "aliases": ", ".join(aliases) if aliases else "-",
            "run_name": run.data.tags.get("mlflow.runName", "-"),
            "roc_auc": round(run.data.metrics.get("roc_auc", 0), 4),
            "f1": round(run.data.metrics.get("f1", 0), 4),
        })
    reg_df = pd.DataFrame(reg_data).sort_values("version", ascending=False)
    st.dataframe(reg_df, use_container_width=True)

    # ---- Section 2: Champion / Challenger comparison ----
    st.header("2. Champion vs Challenger")
    col1, col2 = st.columns(2)

    version_options = reg_df["version"].tolist()
    champion_versions = [v for v, aliases in aliases_by_version.items() if "champion" in aliases]
    default_champion_idx = 0
    if champion_versions and champion_versions[0] in version_options:
        default_champion_idx = version_options.index(champion_versions[0])
    with col1:
        champion_v = st.selectbox(
            "Champion (current alias: champion)", version_options, index=default_champion_idx
        )
    with col2:
        challenger_options = [v for v in version_options if v != champion_v]
        challenger_v = st.selectbox(
            "Challenger", challenger_options if challenger_options else version_options
        )

    champ_row = reg_df[reg_df["version"] == champion_v].iloc[0]
    chal_row = reg_df[reg_df["version"] == challenger_v].iloc[0] if challenger_options else None

    m1, m2, m3 = st.columns(3)
    m1.metric("Champion ROC AUC", champ_row["roc_auc"])
    if chal_row is not None:
        delta = round(chal_row["roc_auc"] - champ_row["roc_auc"], 4)
        m2.metric("Challenger ROC AUC", chal_row["roc_auc"], delta=delta)
    m3.metric("Champion F1", champ_row["f1"])

    st.markdown("---")

    # ---- Section 3: Manual promotion ----
    st.header("3. Promote to Production")
    st.write(
        "Simulates the manual approval step before an MLOps pipeline routes "
        "live traffic to a new model version."
    )
    promote_choice = st.selectbox("Select version to promote", version_options, key="promote_select")
    if st.button(f"🚀 Promote version {promote_choice} to Production"):
        client.set_registered_model_alias(
            name=REGISTERED_MODEL_NAME,
            alias="production",
            version=promote_choice,
        )
        st.success(f"Version {promote_choice} promoted (alias 'production' set). Refresh to see the update.")

st.markdown("---")

# ---- Section 4: Drift Status ----
st.header("4. Drift Monitoring Status")
drift_path = ARTIFACT_DIR / "drift_report.json"
if drift_path.exists():
    with open(drift_path) as f:
        drift = json.load(f)

    status_col, ratio_col = st.columns(2)
    if drift["retrain_needed"]:
        status_col.error("⚠️ Retrain recommended — drift threshold exceeded")
    else:
        status_col.success("✅ No significant drift detected")
    ratio_col.metric("Features drifted", f"{drift['drifted_ratio']:.1%}")

    drift_df = pd.DataFrame([
        {"feature": k, **v} for k, v in drift["features"].items()
    ])
    st.dataframe(drift_df, use_container_width=True)
else:
    st.info("No drift report yet. Run `python src/drift_monitor.py` to generate one.")

st.markdown("---")
st.caption(
    "Built with MLflow (tracking + model registry), scikit-learn/XGBoost/LightGBM, "
    "and Streamlit. Part of a portfolio project demonstrating production ML lifecycle "
    "management for fintech credit risk use cases."
)