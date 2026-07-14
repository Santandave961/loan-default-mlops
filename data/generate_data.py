"""
generate_data.py
Generates a synthetic Nigerian SME loan dataset for the Loan Default MLOps project.

Simulates realistic features used by Nigerian fintech lenders (Moniepoint, Kuda,
Carbon, FairMoney-style) to score SME loan applicants, including alternative
data signals beyond traditional credit history.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
N_SAMPLES = 8000

SECTORS = ["Trading", "Agric", "Retail", "Tech", "Transport", "Fashion", "Food Services"]
COLLATERAL_TYPES = ["None", "Inventory", "Equipment", "Property", "Vehicle"]
STATES = ["Lagos", "Rivers", "Abuja", "Kano", "Oyo", "Enugu", "Kaduna", "Delta"]


def generate_dataset(n_samples: int = N_SAMPLES, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    sector = rng.choice(SECTORS, size=n_samples)
    state = rng.choice(STATES, size=n_samples)
    collateral_type = rng.choice(
        COLLATERAL_TYPES, size=n_samples, p=[0.35, 0.25, 0.15, 0.15, 0.10]
    )

    business_age_months = rng.integers(1, 180, size=n_samples)
    monthly_revenue = rng.lognormal(mean=13.5, sigma=0.9, size=n_samples).round(2)  # Naira
    monthly_expenses = (monthly_revenue * rng.uniform(0.5, 0.95, size=n_samples)).round(2)

    loan_amount = (monthly_revenue * rng.uniform(0.5, 3.0, size=n_samples)).round(2)
    loan_tenor_months = rng.choice([3, 6, 9, 12, 18, 24], size=n_samples)
    interest_rate = rng.uniform(0.15, 0.45, size=n_samples).round(3)  # annualized

    # Alternative data signals
    avg_monthly_txn_count = rng.poisson(lam=60, size=n_samples)
    avg_txn_value = (monthly_revenue / np.maximum(avg_monthly_txn_count, 1)).round(2)
    airtime_topup_freq_monthly = rng.poisson(lam=8, size=n_samples)
    mobile_money_balance_volatility = rng.uniform(0.05, 0.9, size=n_samples).round(3)
    utility_payment_consistency = rng.uniform(0.2, 1.0, size=n_samples).round(3)  # 1 = always on time
    prior_loans_count = rng.poisson(lam=1.5, size=n_samples)
    prior_default_count = np.minimum(
        rng.poisson(lam=0.3, size=n_samples), prior_loans_count
    )

    bvn_linked = rng.choice([1, 0], size=n_samples, p=[0.85, 0.15])
    has_registered_business = rng.choice([1, 0], size=n_samples, p=[0.55, 0.45])

    owner_age = rng.integers(21, 65, size=n_samples)

    df = pd.DataFrame(
        {
            "sector": sector,
            "state": state,
            "collateral_type": collateral_type,
            "business_age_months": business_age_months,
            "monthly_revenue": monthly_revenue,
            "monthly_expenses": monthly_expenses,
            "loan_amount": loan_amount,
            "loan_tenor_months": loan_tenor_months,
            "interest_rate": interest_rate,
            "avg_monthly_txn_count": avg_monthly_txn_count,
            "avg_txn_value": avg_txn_value,
            "airtime_topup_freq_monthly": airtime_topup_freq_monthly,
            "mobile_money_balance_volatility": mobile_money_balance_volatility,
            "utility_payment_consistency": utility_payment_consistency,
            "prior_loans_count": prior_loans_count,
            "prior_default_count": prior_default_count,
            "bvn_linked": bvn_linked,
            "has_registered_business": has_registered_business,
            "owner_age": owner_age,
        }
    )

    # ---- Simulate default probability from a latent risk score ----
    # Lower revenue-to-loan ratio, high volatility, poor utility payment history,
    # prior defaults, and no collateral all raise default risk.
    revenue_to_loan_ratio = df["monthly_revenue"] / np.maximum(df["loan_amount"], 1)
    cashflow_margin = (df["monthly_revenue"] - df["monthly_expenses"]) / np.maximum(
        df["monthly_revenue"], 1
    )

    risk_score = (
        -1.2 * np.log1p(revenue_to_loan_ratio)
        + 2.0 * df["mobile_money_balance_volatility"]
        - 1.5 * df["utility_payment_consistency"]
        - 1.0 * cashflow_margin
        + 1.8 * (df["prior_default_count"] > 0).astype(float)
        + 0.6 * (df["collateral_type"] == "None").astype(float)
        - 0.4 * df["bvn_linked"]
        - 0.3 * df["has_registered_business"]
        - 0.005 * df["business_age_months"]
        + rng.normal(0, 0.7, size=n_samples)  # noise
    )

    # Convert risk score to probability via logistic function, then sample outcome.
    # Shift by mean + offset so the base default rate lands around 12-15%,
    # in line with typical SME loan book performance.
    shifted_score = risk_score - risk_score.mean() - 1.8
    default_prob = 1 / (1 + np.exp(-shifted_score))
    df["default"] = rng.binomial(1, default_prob)

    return df


def main():
    df = generate_dataset()
    out_dir = Path(__file__).resolve().parent
    out_path = out_dir / "sme_loan_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows -> {out_path}")
    print(f"Default rate: {df['default'].mean():.2%}")
    print(df.head())


if __name__ == "__main__":
    main()