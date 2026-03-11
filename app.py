import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
import json
from src.preprocessing import load_ibm, add_market_midpoint, synthesize_recruitment
from src.metrics import *
from src.rules_engine import generate_recommendations
from src.ml_model import prepare_features

st.set_page_config(layout="wide", page_title="HR Decision Support Dashboard")

# -----------------------------------------
# LOAD DATA
# -----------------------------------------
DATA_PATH = "data/ibm_hr_attrition.csv"
MID_PATH = "data/market_midpoints.csv"
RECRUIT_PATH = "data/recruitment_events.csv"

df = load_ibm(DATA_PATH)
df = add_market_midpoint(df, MID_PATH)

if not os.path.exists(RECRUIT_PATH):
    synthesize_recruitment(RECRUIT_PATH, n=120)

recruit_df = pd.read_csv(RECRUIT_PATH)

# -----------------------------------------
# SIDEBAR
# -----------------------------------------
st.sidebar.header("Filters")
dept_filter = st.sidebar.multiselect(
    "Department", 
    options=df['Department'].unique(), 
    default=list(df['Department'].unique())
)
df_f = df[df['Department'].isin(dept_filter)]

st.sidebar.header("Scenario Simulation")
attrition_reduction_pct = st.sidebar.slider("Reduce attrition by (%)", 0, 50, 0)
training_increase_pct = st.sidebar.slider("Increase training coverage by (%)", 0, 50, 0)

# -----------------------------------------
# KPI METRICS
# -----------------------------------------
metrics = {
    'turnover_rate': turnover_rate(df_f),
    'retention_rate': retention_rate(df_f),
    'stability_index': stability_index(df_f),
    'avg_years': avg_years_of_stay(df_f),
    'labour_cost_per_fte': labour_cost_per_fte(df_f),
    'labour_cost_total': labour_cost_total(df_f),
    'CompaRatio': compa_ratio_avg(df_f),
    'training_coverage': training_coverage(df_f),
    'avg_performance': avg_performance_rating(df_f),
    'time_to_fill': time_to_fill(recruit_df),
    'cost_per_hire': cost_per_hire(recruit_df),
    'offer_acceptance_rate': offer_acceptance_rate(recruit_df)
}

# Scenario-adjusted metrics
sim_metrics = metrics.copy()
if attrition_reduction_pct:
    sim_metrics['turnover_rate'] *= (1 - attrition_reduction_pct / 100)
if training_increase_pct:
    sim_metrics['training_coverage'] *= (1 + training_increase_pct / 100)

# -----------------------------------------
# DASHBOARD HEADER
# -----------------------------------------
st.title("HR Decision Support Dashboard")

# KPI CARDS
k1, k2, k3, k4 = st.columns(4)
k1.metric("Turnover Rate", f"{sim_metrics['turnover_rate']:.2f}%", 
          f"{metrics['turnover_rate'] - sim_metrics['turnover_rate']:.2f}%")
k2.metric("Retention Rate", f"{sim_metrics['retention_rate']:.2f}%")
k3.metric("Avg Compa Ratio", f"{sim_metrics['CompaRatio']:.2f}")
k4.metric("Training Coverage", f"{sim_metrics['training_coverage']:.2f}%")

k5, k6, k7 = st.columns(3)
k5.metric("Time to Fill (days)", f"{sim_metrics['time_to_fill']:.1f}")
k6.metric("Cost per Hire", f"{sim_metrics['cost_per_hire']:.0f}")
k7.metric("Offer Acceptance Rate", f"{sim_metrics['offer_acceptance_rate']:.2f}%")

# -----------------------------------------
# CHARTS
# -----------------------------------------
st.subheader("Attrition by Department (Simulated)")
if "AttritionFlag" in df_f.columns:
    adj = df_f.copy()
    adj['AttritionFlag'] *= (1 - attrition_reduction_pct/100)
    st.bar_chart(adj.groupby("Department")["AttritionFlag"].sum())
else:
    st.warning("AttritionFlag missing for chart.")

st.subheader("Compa Ratio Distribution")
fig, ax = plt.subplots()
ax.hist(df_f['CompaRatio'].dropna(), bins=20)
st.pyplot(fig)

# -----------------------------------------
# RECOMMENDATIONS
# -----------------------------------------
st.subheader("Automated Recommendations")
recs = generate_recommendations({
    'turnover_rate': sim_metrics['turnover_rate'],
    'CompaRatio': sim_metrics['CompaRatio'],
    'training_coverage': sim_metrics['training_coverage'],
    'avg_performance': sim_metrics['avg_performance'],
})
for r in recs:
    st.write("- ", r)

# -----------------------------------------
# SCENARIO COST SAVINGS
# -----------------------------------------
st.subheader("Scenario Cost Savings")
if attrition_reduction_pct > 0:
    base_cost = total_cost_of_turnover(df_f)
    new_left = int((sim_metrics['turnover_rate']/100) * headcount(df_f))
    new_cost = new_left * 45000
    savings = base_cost - new_cost
    st.success(f"Estimated savings: ₹{savings:,.0f}")

# -----------------------------------------
# ML MODEL – ATTRITION RISK
# -----------------------------------------
st.header("📌 Attrition Prediction Model")

model_path = "outputs/rf_attrition.joblib"
metrics_path = "outputs/metrics.json"

if os.path.exists(model_path):
    clf = joblib.load(model_path)

    X_for_pred = prepare_features(df)
    df['attrition_proba'] = clf.predict_proba(X_for_pred)[:,1]

    st.subheader("Top 10 High-Risk Employees")
    st.dataframe(
        df.sort_values("attrition_proba", ascending=False)[
            ['EmployeeNumber', 'JobRole', 'Department', 'attrition_proba']
        ].head(10)
    )

else:
    st.warning("Train the model first using train_ml.py")

# -----------------------------------------
# ML METRICS (LOADED FROM FILE)
# -----------------------------------------
st.header("📊 Model Performance")

if os.path.exists(metrics_path):
    with open(metrics_path, "r") as f:
        m = json.load(f)

    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{m['accuracy']:.3f}")
    c2.metric("ROC-AUC", f"{m['roc_auc']:.3f}")

    st.write("### Classification Report")
    st.text(m['classification_report'])

    st.write("### Confusion Matrix")
    st.dataframe(pd.DataFrame(
        m['confusion_matrix'],
        columns=["Pred No", "Pred Yes"],
        index=["Actual No", "Actual Yes"]
    ))
else:
    st.info("metrics.json not found — run train_ml.py")

# -----------------------------------------
# SHAP & FEATURE IMPORTANCE
# -----------------------------------------
st.header("📊 Explainability (SHAP & Feature Importance)")

if os.path.exists("outputs/shap_summary.png"):
    st.subheader("SHAP Summary Plot")
    st.image("outputs/shap_summary.png")

if os.path.exists("outputs/feature_importance.png"):
    st.subheader("Feature Importance Chart")
    st.image("outputs/feature_importance.png")

# -----------------------------------------
# SURVEY IMPORT
# -----------------------------------------
st.subheader("Import Engagement Survey")
survey_file = st.file_uploader("Upload survey CSV", type=["csv"])

if survey_file:
    s_df = pd.read_csv(survey_file)
    st.write("Survey Sample:")
    st.dataframe(s_df.head())

    for col in [
        'JobSatisfaction_Score',
        'TrainingEffectiveness_Score',
        'CompensationFairness_Score'
    ]:
        if col in s_df.columns:
            st.write(f"Avg {col}: {s_df[col].mean():.2f}")
