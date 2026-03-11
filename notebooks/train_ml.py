import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve
)

# Ensure output directory
os.makedirs("outputs", exist_ok=True)

# =====================================================
# 1. Load & Prepare Data
# =====================================================
df = pd.read_csv("data/ibm_hr_attrition.csv")
df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

FEATURES = [
    'Age', 'MonthlyIncome', 'YearsAtCompany',
    'JobSatisfaction', 'EnvironmentSatisfaction',
    'TrainingTimesLastYear'
]

X = df[FEATURES].fillna(0)
y = df["Attrition"]

# Save dataset info for dashboard
dataset_info = {
    "rows": len(df),
    "columns": df.shape[1],
    "features_used": FEATURES
}
with open("outputs/dataset_info.json", "w") as f:
    json.dump(dataset_info, f, indent=4)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =====================================================
# 2. Hyperparameter Tuning (GridSearchCV)
# =====================================================
param_grid = {
    "n_estimators": [150, 200, 300],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 4],
}

grid = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)
best_model = grid.best_estimator_

print("\n🚀 Best Parameters:", grid.best_params_)

# =====================================================
# 3. Model Evaluation
# =====================================================
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
report = classification_report(y_test, y_pred)
matrix = confusion_matrix(y_test, y_pred)

metrics_out = {
    "accuracy": accuracy,
    "roc_auc": roc_auc,
    "classification_report": report,
    "confusion_matrix": matrix.tolist()
}

with open("outputs/metrics.json", "w") as f:
    json.dump(metrics_out, f, indent=4)

print("📁 Saved metrics to outputs/metrics.json")

# =====================================================
# 4. Confusion Matrix Heatmap
# =====================================================
plt.figure(figsize=(6, 5))
sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png")
plt.close()

# =====================================================
# 5. ROC Curve Plot
# =====================================================
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/roc_curve.png")
plt.close()

# =====================================================
# 6. SHAP Explainability
# =====================================================
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_train)

np.save("outputs/shap_values.npy", shap_values[1])
np.save("outputs/shap_features.npy", X_train.values)

plt.figure()
shap.summary_plot(shap_values[1], X_train, show=False)
plt.tight_layout()
plt.savefig("outputs/shap_summary.png")
plt.close()

# =====================================================
# 7. Feature Importance
# =====================================================
importances = best_model.feature_importances_
importance_df = pd.DataFrame({
    "feature": FEATURES,
    "importance": importances
}).sort_values("importance", ascending=False)

plt.figure(figsize=(6, 4))
plt.barh(importance_df["feature"], importance_df["importance"])
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png")
plt.close()

importance_df.to_csv("outputs/feature_importance.csv", index=False)

# =====================================================
# 8. Save Final Model
# =====================================================
joblib.dump(best_model, "outputs/rf_attrition.joblib")

print("\n✅ Enhanced model saved successfully!")
print("📊 All metrics + plots + SHAP exported to /outputs folder.")
