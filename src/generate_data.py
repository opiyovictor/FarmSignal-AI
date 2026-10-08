"""
FarmSignal AI — synthetic data generator.

Produces a realistic (but fully synthetic) farmer-level dataset modelled on the
kinds of variables a smallholder agri-finance program like Tupande would hold:
loan repayment history, farm visit cadence, input redemption, training
attendance, and basic demographics. No real farmer data is used anywhere —
this exists purely to give the modelling and orchestration pipeline something
real to run against.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_FARMERS = 4000

REGIONS = ["Kakamega", "Bungoma", "Busia", "Vihiga", "Trans Nzoia", "Siaya"]


def generate_farmers(n=N_FARMERS) -> pd.DataFrame:
    df = pd.DataFrame({
        "farmer_id": [f"F{100000 + i}" for i in range(n)],
        "region": RNG.choice(REGIONS, size=n),
        "gender": RNG.choice(["F", "M"], size=n, p=[0.62, 0.38]),
        "age": RNG.integers(20, 70, size=n),
        "years_with_program": RNG.integers(1, 9, size=n),
        "farm_size_acres": np.round(RNG.gamma(2.0, 0.6, size=n), 2),
        "loan_amount_kes": RNG.integers(3000, 25000, size=n),
        "distance_to_agrodealer_km": np.round(RNG.exponential(4.0, size=n), 1),
        "days_since_last_farm_visit": RNG.integers(3, 180, size=n),
        "input_redemption_rate": np.round(RNG.beta(5, 2, size=n), 3),
        "agronomic_training_attendance": np.round(RNG.beta(4, 2, size=n), 3),
        "group_repayment_avg": np.round(RNG.beta(6, 2, size=n), 3),
        "mobile_money_active": RNG.choice([0, 1], size=n, p=[0.2, 0.8]),
        "prior_season_yield_index": np.round(RNG.normal(1.0, 0.25, size=n), 3),
    })

    # Synthetic outcome-generating process.
    # The portfolio dataset is deliberately designed as a benchmarkable ML task:
    # a latent farm-resilience score combines observable programme signals, and
    # farmers in the highest-risk tail are more likely to experience default /
    # dropout. A small label-noise component prevents a perfectly separable,
    # unrealistic toy problem.
    risk_signal = (
        3.2 * (df["days_since_last_farm_visit"] / 180)
        - 3.4 * df["input_redemption_rate"]
        - 3.0 * df["agronomic_training_attendance"]
        - 3.5 * df["group_repayment_avg"]
        - 1.5 * df["prior_season_yield_index"].clip(lower=0)
        + 1.4 * (
            df["distance_to_agrodealer_km"]
            / df["distance_to_agrodealer_km"].quantile(0.99)
        ).clip(upper=1)
        + 1.0 * (1 - df["mobile_money_active"])
        + 0.7 * (df["loan_amount_kes"] / df["loan_amount_kes"].max())
        - 0.5 * (df["years_with_program"] / df["years_with_program"].max())
    )

    # Define the outcome from the upper quartile of the latent risk signal,
    # then introduce 2% label noise. This produces a realistic-but-learnable
    # benchmark for a portfolio model without claiming real-world accuracy.
    cutoff = risk_signal.quantile(0.75)
    outcome = (risk_signal >= cutoff).astype(int)
    label_noise = RNG.random(n) < 0.01
    outcome = np.where(label_noise, 1 - outcome, outcome)
    df["defaulted_or_dropped_out"] = outcome.astype(int)

    return df


if __name__ == "__main__":
    farmers = generate_farmers()
    out_path = "data/farmers.csv"
    farmers.to_csv(out_path, index=False, encoding="utf-8")
    rate = farmers["defaulted_or_dropped_out"].mean()
    print(f"Generated {len(farmers)} synthetic farmer records -> {out_path}")
    print(f"Base rate of default/dropout in synthetic data: {rate:.1%}")
