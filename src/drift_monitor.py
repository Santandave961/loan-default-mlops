"""
drift_monitor.py
Checks incoming ("production") data against the original training distribution
and logs drift metrics as an MLflow run. If drift crosses the configured
threshold, flags that a retrain is needed.

This simulates a real MLOps pattern: periodically scoring how much your live
traffic has diverged from what the model was trained on, so you catch model
decay before it shows up in business metrics.

Usage:
    python src/drift_monitor.py
"""

import json
import os
from pathlib import Path

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import mlflow
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "sme_loan_data.csv"
ARTIFACT_DIR = ROOT / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

EXPERIMENT_NAME = "sme-loan-drift-monitoring"

# KS statistic threshold above which a feature is considered "drifted"
DRIFT_THRESHOLD = 0.1
# Fraction of numeric features drifted before we flag "retrain needed"
RETRAIN_TRIGGER_RATIO = 0.3

NUMERIC_COLS_TO_MONITOR = [
    "monthly_revenue",
    "monthly_expenses",
    "loan_amount",
    "avg_monthly_txn_count",
    "avg_txn_value",
    "mobile_money_balance_volatility",
    "utility_payment_consistency",
    "prior_default_count",
]


def simulate_production_batch(reference_df: pd.DataFrame, drift_strength: float = 0.0) -> pd.DataFrame:
    """
    Simulates a new incoming batch of scored loan applications.
    drift_strength=0.0 -> same distribution as training (no drift).
    drift_strength>0.0 -> shifts key features to simulate real-world drift
    (e.g. macroeconomic shock raising loan amounts and volatility).
    """
    rng = np.random.default_rng()
    batch = reference_df.sample(n=1000, replace=True, random_state=None).copy()

    if drift_strength > 0:
        batch["monthly_revenue"] *= (1 - 0.3 * drift_strength)  # revenues drop
        batch["loan_amount"] *= (1 + 0.4 * drift_strength)  # loan sizes grow
        batch["mobile_money_balance_volatility"] = np.clip(
            batch["mobile_money_balance_volatility"] * (1 + 0.8 * drift_strength), 0, 1
        )
        batch["utility_payment_consistency"] = np.clip(
            batch["utility_payment_consistency"] * (1 - 0.3 * drift_strength), 0, 1
        )

    return batch


def compute_drift(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    drift_report = {}
    for col in NUMERIC_COLS_TO_MONITOR:
        stat, p_value = ks_2samp(reference[col], current[col])
        drift_report[col] = {
            "ks_statistic": round(float(stat), 4),
            "p_value": round(float(p_value), 4),
            "drifted": bool(stat > DRIFT_THRESHOLD),
        }
    return drift_report


def main(drift_strength: float = 0.0):
    mlflow.set_experiment(EXPERIMENT_NAME)

    reference_df = pd.read_csv(DATA_PATH)
    current_batch = simulate_production_batch(reference_df, drift_strength=drift_strength)

    drift_report = compute_drift(reference_df, current_batch)

    n_drifted = sum(1 for v in drift_report.values() if v["drifted"])
    n_total = len(drift_report)
    drifted_ratio = n_drifted / n_total
    retrain_needed = drifted_ratio >= RETRAIN_TRIGGER_RATIO

    with mlflow.start_run(run_name="drift_check"):
        mlflow.log_param("drift_strength_simulated", drift_strength)
        mlflow.log_param("drift_threshold_ks", DRIFT_THRESHOLD)
        mlflow.log_metric("features_drifted", n_drifted)
        mlflow.log_metric("features_monitored", n_total)
        mlflow.log_metric("drifted_ratio", drifted_ratio)
        mlflow.log_metric("retrain_needed", int(retrain_needed))

        for col, stats in drift_report.items():
            mlflow.log_metric(f"ks_{col}", stats["ks_statistic"])

        report_path = ARTIFACT_DIR / "drift_report.json"
        with open(report_path, "w") as f:
            json.dump(
                {"retrain_needed": retrain_needed, "drifted_ratio": drifted_ratio, "features": drift_report},
                f,
                indent=2,
            )
        mlflow.log_artifact(str(report_path))

    print(f"Drifted features: {n_drifted}/{n_total} ({drifted_ratio:.1%})")
    print(f"Retrain needed: {retrain_needed}")
    for col, stats in drift_report.items():
        flag = "DRIFTED" if stats["drifted"] else "ok"
        print(f"  {col:35s} KS={stats['ks_statistic']:.4f}  [{flag}]")


if __name__ == "__main__":
    import sys
    strength = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    main(drift_strength=strength)