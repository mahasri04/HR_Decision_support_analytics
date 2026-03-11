# src/ml_model.py
import pandas as pd
import numpy as np
import os
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix
)
from xgboost import XGBClassifier

# ---------------------------------------------------
# FEATURE PREPARATION
# ---------------------------------------------------
def prepare_features(df_input):
    """
    Cleans & encodes features consistently for both
    training and prediction.
    """
    df = df_input.copy()

    # Core numeric features
    cols = [
        'Age', 'MonthlyIncome', 'YearsAtCompany',
        'JobSatisfaction', 'EnvironmentSatisfaction',
        'TrainingTimesLastYear'
    ]

    X = df[cols].copy()

    # OverTime flag handling
    if 'OverTime' in df.columns:
        X['OverTimeFlag'] = df['OverTime'].str.lower().eq("yes").astype(int)
    else:
        # Default to 0 if missing
        X['OverTimeFlag'] = 0

    # Handle missing values
    return X.fillna(0)


# ---------------------------------------------------
# MODEL TRAINING WRAPPER
# ---------------------------------------------------
def train_models(data_path="data/ibm_hr_attrition.csv"):
    df = pd.read_csv(data_path)

    # Encode target
    df['AttritionFlag'] = df['Attrition'].map({'Yes': 1, 'No': 0})

    X = prepare_features(df)
    y = df['AttritionFlag']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    # ---------------------------------------------------
    # Define candidate models
    # ---------------------------------------------------
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=500),
        "GradientBoosting": GradientBoostingClassifier(),
        "XGBoost": XGBClassifier(
            eval_metric="logloss",
            learning_rate=0.05,
            max_depth=5,
            n_estimators=250
        )
    }

    results = {}
    best_model = None
    best_auc = -1

    # ---------------------------------------------------
    # Train & evaluate all models
    # ---------------------------------------------------
    for name, model in models.items():
        print(f"\n🔹 Training {name}...")

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_proba)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "model": model,
            "auc": auc,
            "report": report,
            "confusion_matrix": cm
        }

        print(f"{name} ROC–AUC: {auc:.4f}")

        # Keep best model
        if auc > best_auc:
            best_auc = auc
            best_model = model
            best_name = name

    print(f"\n🏆 Best Model Selected: **{best_name}** with AUC={best_auc:.4f}")

    # ---------------------------------------------------
    # Save model
    # ---------------------------------------------------
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(best_model, "outputs/best_model.joblib")
    print("✅ Saved best model → outputs/best_model.joblib")

    # ---------------------------------------------------
    # SHAP EXPLAINABILITY
    # ---------------------------------------------------
    print("\n📌 Generating SHAP explainability plots...")

    explainer = shap.Explainer(best_model, X_train)
    shap_values = explainer(X_train)

    # SHAP summary plot
    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, X_train, show=False)
    plt.tight_layout()
    plt.savefig("outputs/shap_summary.png")
    plt.close()

    print("📊 SHAP summary saved → outputs/shap_summary.png")

    # Feature importance (model-based)
    if hasattr(best_model, "feature_importances_"):
        plt.figure(figsize=(8, 5))
        plt.barh(X_train.columns, best_model.feature_importances_)
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.savefig("outputs/feature_importance.png")
        plt.close()

        print("📊 Feature importance saved → outputs/feature_importance.png")

    return best_model, results


# ---------------------------------------------------
# SCRIPT ENTRY POINT
# ---------------------------------------------------
if __name__ == "__main__":
    train_models()
