# src/metrics.py

import pandas as pd
import numpy as np

# ------------------------------------------------------------
# Helper: safe metric calculation
# ------------------------------------------------------------
def safe_div(num, den):
    return (num / den * 100) if den not in (0, None) else None

def exists(df, cols):
    """Check if required columns exist in the dataset."""
    return all(col in df.columns for col in cols)

# ------------------------------------------------------------
# POPULATION METRICS
# ------------------------------------------------------------
def headcount(df):
    return df.shape[0]

def employees_left(df):
    if not exists(df, ['AttritionFlag']):
        return None
    return df[df['AttritionFlag'] == 1].shape[0]

def employees_stayed(df):
    if not exists(df, ['AttritionFlag']):
        return None
    return df[df['AttritionFlag'] == 0].shape[0]

# ------------------------------------------------------------
# TURNOVER & RETENTION
# ------------------------------------------------------------
def turnover_rate(df):
    left = employees_left(df)
    hc = headcount(df)
    return safe_div(left, hc)

def retention_rate(df):
    stayed = employees_stayed(df)
    hc = headcount(df)
    return safe_div(stayed, hc)

def early_attrition_rate(df):
    """Employees leaving within <1 year."""
    if not exists(df, ['AttritionFlag', 'YearsAtCompany']):
        return None
    left_early = df[(df['AttritionFlag'] == 1) & (df['YearsAtCompany'] < 1)].shape[0]
    return safe_div(left_early, headcount(df))

def high_performer_attrition(df):
    """Attrition among employees with high ratings."""
    if not exists(df, ['AttritionFlag', 'PerformanceRating']):
        return None
    high_perf = df[df['PerformanceRating'] >= 4]
    return safe_div(high_perf[high_perf['AttritionFlag'] == 1].shape[0], high_perf.shape[0])

def stability_index(df):
    if not exists(df, ['YearsAtCompany']):
        return None
    stable = df[df['YearsAtCompany'] >= 1].shape[0]
    return safe_div(stable, headcount(df))

def avg_years_of_stay(df):
    if not exists(df, ['YearsAtCompany']):
        return None
    return df['YearsAtCompany'].mean()

# ------------------------------------------------------------
# RECRUITMENT METRICS
# ------------------------------------------------------------
def time_to_fill(recruit_df):
    if not exists(recruit_df, ['open_date', 'filled_date']):
        return None
    recruit_df = recruit_df.copy()
    recruit_df['open_date'] = pd.to_datetime(recruit_df['open_date'])
    recruit_df['filled_date'] = pd.to_datetime(recruit_df['filled_date'])
    recruit_df['days_to_fill'] = (recruit_df['filled_date'] - recruit_df['open_date']).dt.days
    return recruit_df['days_to_fill'].mean()

def cost_per_hire(recruit_df):
    if not exists(recruit_df, ['cost_of_recruitment']):
        return None
    return recruit_df['cost_of_recruitment'].mean()

def offer_acceptance_rate(recruit_df):
    if not exists(recruit_df, ['num_offers', 'offers_accepted']):
        return None
    return safe_div(recruit_df['offers_accepted'].sum(), recruit_df['num_offers'].sum())

def internal_mobility_rate(df):
    if not exists(df, ['JobLevel', 'PreviousJobLevel']):
        return None
    moved = df[df['JobLevel'] > df['PreviousJobLevel']].shape[0]
    return safe_div(moved, headcount(df))

# ------------------------------------------------------------
# COMPENSATION METRICS
# ------------------------------------------------------------
def labour_cost_per_fte(df):
    if not exists(df, ['TotalAnnualCompensation']):
        return None
    return df['TotalAnnualCompensation'].mean()

def labour_cost_total(df):
    if not exists(df, ['TotalAnnualCompensation']):
        return None
    return df['TotalAnnualCompensation'].sum()

def compa_ratio_avg(df):
    if not exists(df, ['CompaRatio']):
        return None
    return df['CompaRatio'].mean()

def gender_pay_gap(df):
    if not exists(df, ['Gender', 'MonthlyIncome']):
        return None
    male = df[df['Gender'] == "Male"]['MonthlyIncome'].mean()
    female = df[df['Gender'] == "Female"]['MonthlyIncome'].mean()
    return safe_div(male - female, male)

# ------------------------------------------------------------
# TRAINING & PERFORMANCE METRICS
# ------------------------------------------------------------
def training_coverage(df):
    if not exists(df, ['TrainingTimesLastYear']):
        return None
    trained = df[df['TrainingTimesLastYear'] > 0].shape[0]
    return safe_div(trained, headcount(df))

def training_effectiveness(df):
    """Correlation between training and performance."""
    if not exists(df, ['TrainingTimesLastYear', 'PerformanceRating']):
        return None
    return df['TrainingTimesLastYear'].corr(df['PerformanceRating'])

def avg_performance_rating(df):
    if not exists(df, ['PerformanceRating']):
        return None
    return df['PerformanceRating'].mean()

# ------------------------------------------------------------
# TURNOVER COSTS
# ------------------------------------------------------------
def cost_of_turnover_one_employee(avg_hiring_cost=20000, onboarding_loss=10000, productivity_loss=15000):
    return avg_hiring_cost + onboarding_loss + productivity_loss

def total_cost_of_turnover(df, avg_cost_replacement=45000):
    left = employees_left(df)
    if left is None:
        return None
    return left * avg_cost_replacement

# ------------------------------------------------------------
# FULL SUMMARY FOR DASHBOARD
# ------------------------------------------------------------
def hr_summary(df, recruit_df=None):
    """Returns a dictionary of all important HR metrics for dashboard."""
    summary = {
        # Workforce Overview
        "Headcount": headcount(df),
        "Employees Left": employees_left(df),
        "Employees Stayed": employees_stayed(df),

        # Retention & Turnover
        "Turnover Rate (%)": turnover_rate(df),
        "Early Attrition Rate (%)": early_attrition_rate(df),
        "Retention Rate (%)": retention_rate(df),
        "High Performer Attrition (%)": high_performer_attrition(df),
        "Avg Years of Stay": avg_years_of_stay(df),
        "Stability Index (%)": stability_index(df),

        # Compensation
        "Avg Labour Cost per FTE": labour_cost_per_fte(df),
        "Total Labour Cost": labour_cost_total(df),
        "Compa Ratio Avg": compa_ratio_avg(df),
        "Gender Pay Gap (%)": gender_pay_gap(df),

        # Performance & Training
        "Training Coverage (%)": training_coverage(df),
        "Training Effectiveness (Corr)": training_effectiveness(df),
        "Avg Performance Rating": avg_performance_rating(df),

        # Turnover Cost
        "Total Cost of Turnover": total_cost_of_turnover(df),

        # Recruitment Metrics (if provided)
        "Recruitment: Time to Fill (days)": time_to_fill(recruit_df) if recruit_df is not None else None,
        "Recruitment: Cost per Hire": cost_per_hire(recruit_df) if recruit_df is not None else None,
        "Recruitment: Offer Acceptance Rate (%)": offer_acceptance_rate(recruit_df) if recruit_df is not None else None,
    }

    return summary
