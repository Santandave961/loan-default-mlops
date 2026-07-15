# SME Loan Default — Full MLflow Model Lifecycle Project

A production-style MLOps portfolio project for Nigerian fintech/SME lending,
demonstrating the full model lifecycle — not just training a model, but
managing it: experiment tracking, model registry, champion/challenger
comparison, drift detection, and deployment.

This targets the gap most data science portfolios have: showing you can
operate models in production, which is what ML Engineer job descriptions
(banks, fintechs, telecoms) actually ask for.

## Why this project

Most portfolios show "I trained a model and got 0.85 AUC." This shows:
- Experiment tracking across multiple model families
- Model versioning and registry (Staging → Production → Archived)
- A champion/challenger workflow, the pattern real fintechs use before
  fully swapping a production model
- Drift detection — recognizing when a model is starting to decay
- A served API + a dashboard an ops/risk team could actually use

## Project structure

```
loan-default-mlops/
├── data/
│   ├── generate_data.py      # synthetic Nigerian SME loan dataset
│   └── sme_loan_data.csv     # generated dataset (8,000 rows)
├── src/
│   ├── train.py              # trains 4 model families, logs to MLflow, registers champion
│   ├── drift_monitor.py      # KS-test drift detection vs training distribution
│   └── serve.py              # FastAPI serving layer for the Production-stage model
├── app/
│   └── dashboard.py          # Streamlit MLOps dashboard
├── artifacts/                # confusion matrices, SHAP plots, drift/training reports
├── requirements.txt
└── README.md
```

## Dataset

Synthetic but realistically modeled: 8,000 SME loan applications with
alternative data signals used by Nigerian fintech lenders — mobile money
transaction patterns, airtime top-up frequency, utility payment
consistency, BVN linkage — alongside traditional loan and business
features. Base default rate ≈ 20%, calibrated to reflect a higher-risk SME
lending segment.

Regenerate anytime:
```bash
python data/generate_data.py
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**1. Start the MLflow tracking UI** (in its own terminal):
```bash
mlflow ui --port 5000
```
Open http://localhost:5000 to browse runs.

**2. Train models** — trains Logistic Regression, Random Forest, XGBoost,
and LightGBM; logs params/metrics/artifacts (confusion matrix, SHAP summary)
for each; registers the best model (by ROC AUC) to the Model Registry at
stage `Staging`:
```bash
python src/train.py
```

**3. Check for drift** — simulates a new batch of scored applications and
compares its distribution to the training data using a KS test. Pass a
drift strength (0.0 = no drift, higher = more simulated shock) to test the
retrain trigger:
```bash
python src/drift_monitor.py 0.0   # no drift
python src/drift_monitor.py 0.6   # simulated macro shock (revenue drop, higher volatility)
```

**4. Launch the dashboard** — registry overview, champion/challenger
comparison, manual promotion button, and drift status:
```bash
streamlit run app/dashboard.py
```

**5. Serve the production model** as an API:
```bash
uvicorn src.serve:app --reload --port 8000
```
Then:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sector": "Trading", "state": "Lagos", "collateral_type": "None",
    "business_age_months": 18, "monthly_revenue": 850000, "monthly_expenses": 620000,
    "loan_amount": 1200000, "loan_tenor_months": 12, "interest_rate": 0.28,
    "avg_monthly_txn_count": 55, "avg_txn_value": 15000,
    "airtime_topup_freq_monthly": 6, "mobile_money_balance_volatility": 0.4,
    "utility_payment_consistency": 0.7, "prior_loans_count": 1,
    "prior_default_count": 0, "bvn_linked": 1, "has_registered_business": 1,
    "owner_age": 34
  }'
```

Note: `serve.py` looks for a `Production`-stage model first, falling back to
`Staging`. Promote a version to `Production` via the dashboard (Section 3)
before serving, or `train.py`'s registered `Staging` version will be used.

## Deployment notes

Same pattern as other projects in this portfolio: pin `requirements.txt`,
avoid force-adding large artifacts to git, deploy the FastAPI service to
Render and the Streamlit dashboard to Streamlit Cloud. MLflow's tracking
store (`mlruns/`) should point at a persistent backend (e.g. a hosted
Postgres + S3-compatible artifact store) rather than local SQLite/files for
a real deployment — local file store is fine for portfolio/demo purposes.

## Talking points for interviews

- Why ROC AUC + PR AUC together (class imbalance at ~20% default rate)
- Why `class_weight="balanced"` / `scale_pos_weight` matter here
- What champion/challenger buys you operationally vs. just redeploying
- Why KS-test drift on feature distributions, not just tracking accuracy
  (accuracy lags — distribution drift is often the earlier signal)
- Trade-offs of the 1-year cliff between "SHAP for explainability" and
  "regulatory audit trail" — connects your accounting background to why
  interpretability matters for lending decisions, not just accuracy
