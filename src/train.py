"""
train.py
Trains multiple model families on the SME loan default dataset, logging every
run (params, metrics, artifacts) to MLflow, then registers the best model to
the MLflow Model Registry under stage "Staging".

Usage:
    python src/train.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "sme_loan_data.csv"
ARTIFACT_DIR = ROOT / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

EXPERIMENT_NAME = "sme-loan-default"
REGISTERED_MODEL_NAME = "sme-loan-default-classifier"

CATEGORICAL_COLS = ["sector", "state", "collateral_type"]
TARGET_COL = "default"


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    numeric_cols = [c for c in X.columns if c not in CATEGORICAL_COLS]
    return X, y, numeric_cols


def build_preprocessor(numeric_cols):
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )


def evaluate(model, X_test, y_test, threshold=0.5):
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, preds, average="binary", zero_division=0
    )

    return {
        "roc_auc": auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }, proba, preds


def log_confusion_matrix(y_test, preds, run_name):
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["No Default", "Default"]).plot(ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix - {run_name}")
    path = ARTIFACT_DIR / f"confusion_matrix_{run_name}.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def log_shap_summary(model, preprocessor, X_sample, feature_names, run_name):
    """Best-effort SHAP summary plot; skipped gracefully if the explainer fails
    for a given model type (e.g. non tree-based models with large one-hot dims)."""
    try:
        X_transformed = preprocessor.transform(X_sample)
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()

        explainer = shap.Explainer(model, X_transformed)
        shap_values = explainer(X_transformed[:200])

        fig = plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_values, X_transformed[:200], feature_names=feature_names, show=False)
        path = ARTIFACT_DIR / f"shap_summary_{run_name}.png"
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path
    except Exception as e:
        print(f"  [SHAP skipped for {run_name}: {e}]")
        return None


def get_feature_names(preprocessor, numeric_cols):
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_COLS))
    return numeric_cols + cat_names


def main():
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y, numeric_cols = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced", random_state=42
        ),
    }
    if HAS_XGB:
        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            eval_metric="logloss",
            scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
            random_state=42,
        )
    if HAS_LGBM:
        models["lightgbm"] = LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=42,
            verbosity=-1,
        )

    results = []

    for name, clf in models.items():
        print(f"\n=== Training {name} ===")
        preprocessor = build_preprocessor(numeric_cols)
        pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])

        with mlflow.start_run(run_name=name) as run:
            pipeline.fit(X_train, y_train)

            metrics, proba, preds = evaluate(pipeline, X_test, y_test)

            # Log params
            mlflow.log_param("model_type", name)
            for p, v in clf.get_params().items():
                # keep param logging concise; skip huge/non-scalar params
                if isinstance(v, (int, float, str, bool)) or v is None:
                    mlflow.log_param(p, v)

            # Log metrics
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            # Confusion matrix artifact
            cm_path = log_confusion_matrix(y_test, preds, name)
            mlflow.log_artifact(str(cm_path))

            # SHAP summary (tree models only, to keep this fast/reliable)
            if name in ("random_forest", "xgboost", "lightgbm"):
                fitted_preprocessor = pipeline.named_steps["preprocess"]
                feature_names = get_feature_names(fitted_preprocessor, numeric_cols)
                shap_path = log_shap_summary(
                    pipeline.named_steps["model"],
                    fitted_preprocessor,
                    X_test,
                    feature_names,
                    name,
                )
                if shap_path:
                    mlflow.log_artifact(str(shap_path))

            # Log the full pipeline (preprocessing + model) as one MLflow model
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            print(f"  ROC AUC: {metrics['roc_auc']:.4f} | PR AUC: {metrics['pr_auc']:.4f} "
                  f"| F1: {metrics['f1']:.4f}")

            results.append({
                "name": name,
                "run_id": run.info.run_id,
                "metrics": metrics,
            })

    # ---- Pick champion model by ROC AUC and register it ----
    best = max(results, key=lambda r: r["metrics"]["roc_auc"])
    print(f"\n=== Champion model: {best['name']} (ROC AUC = {best['metrics']['roc_auc']:.4f}) ===")

    model_uri = f"runs:/{best['run_id']}/model"
    registered = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)

    # Aliases replace the deprecated Staging/Production "stages" API in
    # current MLflow versions. "champion" = current best/serving model,
    # "challenger" = a candidate being evaluated against it.
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="champion",
        version=registered.version,
    )
    print(f"Registered '{REGISTERED_MODEL_NAME}' v{registered.version} -> alias: champion")

    # Save a summary file for the dashboard / README
    summary = {
        "champion": best["name"],
        "champion_run_id": best["run_id"],
        "registered_version": registered.version,
        "all_results": [
            {"name": r["name"], "run_id": r["run_id"], **r["metrics"]} for r in results
        ],
    }
    with open(ARTIFACT_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {ARTIFACT_DIR / 'training_summary.json'}")


if __name__ == "__main__":
    main()