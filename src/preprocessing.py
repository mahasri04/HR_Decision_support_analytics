# src/preprocessing.py
import pandas as pd
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ------------------------------------------------------------
# 1. LOAD AND CLEAN IBM HR DATASET
# ------------------------------------------------------------
def load_ibm(path="data/ibm_hr_attrition.csv"):
    """
    Loads the IBM HR dataset with:
      - Cleaned attrition flag
      - Annualized compensation
      - Standardized missing values
      - Validations for key columns
    """
    logging.info(f"Loading dataset from {path}")
    df = pd.read_csv(path)

    # --- Validate required columns ---
    required_cols = [
        "Attrition", "MonthlyIncome", "YearsAtCompany",
        "TrainingTimesLastYear", "JobRole"
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"❌ Missing required column: {col}")

    # --- Clean attrition ---
    df["AttritionFlag"] = df["Attrition"].str.strip().str.lower().map(
        {"yes": 1, "no": 0}
    )
    df["AttritionFlag"] = df["AttritionFlag"].fillna(0)

    # --- Annual Income ---
    df["AnnualIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median()) * 12

    # --- Fill missing numeric values safely ---
    numeric_fill_cols = [
        "YearsAtCompany", "TrainingTimesLastYear",
        "PerformanceRating", "JobSatisfaction",
        "EnvironmentSatisfaction"
    ]
    for col in numeric_fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # --- Benefits & Total Compensation ---
    df["Benefits"] = df["AnnualIncome"] * 0.20
    df["TotalAnnualCompensation"] = df["AnnualIncome"] + df["Benefits"]

    logging.info("Dataset loaded and cleaned successfully.")
    return df


# ------------------------------------------------------------
# 2. MERGE MARKET MIDPOINT DATA
# ------------------------------------------------------------
def add_market_midpoint(df, midpoints_path="data/market_midpoints.csv"):
    """
    Merges market midpoint compensation data and computes:
        - Market Midpoint
        - Compa Ratio
    """
    logging.info(f"Merging market midpoint data from {midpoints_path}")

    try:
        mp = pd.read_csv(midpoints_path)
    except Exception as e:
        raise FileNotFoundError(
            f"❌ Could not load midpoint file. Error: {e}"
        )

    if "JobRole" not in mp.columns or "MarketMidpoint" not in mp.columns:
        raise ValueError("❌ midpoint file must contain JobRole + MarketMidpoint columns")

    df = df.merge(mp[["JobRole", "MarketMidpoint"]], how="left", on="JobRole")

    # Fill missing midpoints using median of dataset
    df["MarketMidpoint"] = df["MarketMidpoint"].fillna(df["AnnualIncome"].median())

    # Calculate compa ratio
    df["CompaRatio"] = df["AnnualIncome"] / df["MarketMidpoint"]

    logging.info("Market midpoint merge completed.")
    return df


# ------------------------------------------------------------
# 3. SYNTHETIC RECRUITMENT EVENT GENERATION
# ------------------------------------------------------------
def synthesize_recruitment(path_out="data/recruitment_events.csv", n=100):
    """
    Generates synthetic recruitment data for dashboards & HR KPIs.
    Includes:
        - open/fill dates
        - time-to-fill behaviour
        - cost of recruitment
        - accepted offers
    """
    logging.info(f"Generating {n} synthetic recruitment events...")

    import random
    from datetime import datetime, timedelta

    roles = list(pd.read_csv("data/ibm_hr_attrition.csv")["JobRole"].unique())
    rows = []
    start = datetime(2023, 1, 1)

    for i in range(n):
        open_date = start + timedelta(days=random.randint(0, 300))
        fill_days = random.randint(10, 70)
        filled_date = open_date + timedelta(days=fill_days)

        rows.append({
            "job_id": f"JOB{i+1}",
            "job_role": random.choice(roles),
            "open_date": open_date.date(),
            "filled_date": filled_date.date(),
            "time_to_fill": fill_days,
            "num_offers": random.randint(1, 6),
            "offers_accepted": 1 if random.random() < 0.7 else 0,
            "cost_of_recruitment": random.randint(1500, 9000)
        })

    pd.DataFrame(rows).to_csv(path_out, index=False)
    logging.info(f"Synthetic recruitment data saved to {path_out}")
