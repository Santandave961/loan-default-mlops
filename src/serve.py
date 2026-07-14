"""
serve.py
FastAPI service that loads the current "Production" stage model from the
MLflow Model Registry and exposes a /predict endpoint.

Run:
    uvicorn src.serve:app --reload --port 8000

Then:
    curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @sample_request.json
"""

from pathlib import Path

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

REGISTERED_MODEL_NAME = "sme-loan-default-classifier"
MODEL_ALIAS = "production"  # falls back to "champion" if no "production" alias is set

app = FastAPI(title="SME Loan Default Scoring API", version="1.0.0")

_model = None
_model_version = None


class LoanApplication(BaseModel):
    sector: str = Field(..., examples=["Trading"])
    state: str = Field(..., examples=["Lagos"])
    collateral_type: str = Field(..., examples=["None"])
    business_age_months: int
    monthly_revenue: float
    monthly_expenses: float
    loan_amount: float
    loan_tenor_months: int
    interest_rate: float
    avg_monthly_txn_count: int
    avg_txn_value: float
    airtime_topup_freq_monthly: int
    mobile_money_balance_volatility: float
    utility_payment_consistency: float
    prior_loans_count: int
    prior_default_count: int
    bvn_linked: int
    has_registered_business: int
    owner_age: int


def load_production_model():
    global _model, _model_version
    client = mlflow.MlflowClient()

    for alias in (MODEL_ALIAS, "champion"):
        try:
            version_info = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, alias)
            _model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}@{alias}")
            _model_version = version_info.version
            print(f"Loaded model version {_model_version} from alias '{alias}'")
            return
        except mlflow.exceptions.MlflowException:
            continue
    raise RuntimeError(
        f"No model found under alias 'production' or 'champion' for '{REGISTERED_MODEL_NAME}'. "
        "Run src/train.py first."
    )


@app.on_event("startup")
def startup_event():
    load_production_model()


@app.get("/health")
def health():
    return {"status": "ok", "model_version": _model_version}


@app.post("/predict")
def predict(application: LoanApplication):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = pd.DataFrame([application.model_dump()])
    proba = float(_model.predict_proba(df)[:, 1][0])
    prediction = int(proba >= 0.5)

    return {
        "model_version": _model_version,
        "default_probability": round(proba, 4),
        "prediction": "default" if prediction == 1 else "no_default",
        "risk_band": (
            "high" if proba >= 0.5 else "medium" if proba >= 0.2 else "low"
        ),
    }