"""
=============================================================
Customer Churn Analysis & Prediction
=============================================================
Goal : Identify key drivers of customer churn in a telecom
       dataset, build predictive models, and surface retention
       strategies via dashboards.

Sections
--------
1. Import & Load Data
2. Exploratory Data Analysis (EDA)
3. Feature Engineering & Preprocessing
4. Customer Segmentation
5. Predictive Modeling (Logistic Regression, Random Forest, KNN)
6. Model Evaluation (confusion matrix, ROC-AUC, classification report)
7. Feature-Importance & Retention Recommendations
8. Export Results for Dashboard

Dependencies
------------
pip install pandas numpy matplotlib seaborn scikit-learn
=============================================================
"""

# ─── 1. IMPORTS & SYNTHETIC DATA  ─────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, ConfusionMatrixDisplay
)
from sklearn.impute import SimpleImputer

sns.set_style("whitegrid")
np.random.seed(42)


def generate_telecom_data(n: int = 5000) -> pd.DataFrame:
    """
    Generates a realistic synthetic telecom-churn dataset.
    """
    tenure        = np.random.exponential(scale=30, size=n).astype(int).clip(0, 72)
    monthly_chgs  = np.round(np.random.normal(65, 25, n), 2).clip(20, 150)
    total_chgs    = np.round(tenure * monthly_chgs * np.random.uniform(0.85, 1.15, n), 2)
    contract      = np.random.choice(["Month-to-month", "One year", "Two year"], n,
                                     p=[0.52, 0.28, 0.20])
    internet      = np.random.choice(["Fiber optic", "DSL", "No"], n, p=[0.44, 0.37, 0.19])
    tech_support  = np.random.choice([0, 1], n, p=[0.55, 0.45])
    online_backup = np.random.choice([0, 1], n, p=[0.50, 0.50])
    gender        = np.random.choice(["Male", "Female"], n)
    senior        = np.random.choice([0, 1], n, p=[0.84, 0.16])
    partner       = np.random.choice([0, 1], n, p=[0.48, 0.52])
    dependents    = np.random.choice([0, 1], n, p=[0.70, 0.30])
    calls_support = np.random.poisson(lam=1.8, size=n)

    # Churn probability
    churn_prob = (
        0.30
        - 0.004 * tenure
        + 0.003 * monthly_chgs
        + 0.12  * (contract == "Month-to-month").astype(float)
        - 0.08  * (contract == "Two year").astype(float)
        + 0.06  * (internet == "Fiber optic").astype(float)
        - 0.10  * tech_support
        - 0.07  * online_backup
        + 0.02  * senior
        - 0.03  * partner
        + 0.04  * calls_support
    )
    churn_prob = np.clip(churn_prob, 0.02, 0.95)
    churn      = (np.random.rand(n) < churn_prob).astype(int)

    return pd.DataFrame({
        "CustomerID":     range(1, n + 1),
        "Gender":         gender,
        "SeniorCitizen":  senior,
        "Partner":        partner,
        "Dependents":     dependents,
        "Tenure":         tenure,
        "MonthlyCharges": monthly_chgs,
        "TotalCharges":   total_chgs,
        "Contract":       contract,
        "InternetService":internet,
        "TechSupport":    tech_support,
        "OnlineBackup":   online_backup,
        "SupportCalls":   calls_support,
        "Churn":          churn,
    })


# ─── 2. LOAD DATA ────────────────────────────────────────────
df = generate_telecom_data(n=5000)
print("Dataset shape:", df.shape)
print(df.head())
print("\nChurn distribution:\n", df["Churn"].value_counts())


# ─── 3. EDA ──────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Exploratory Data Analysis – Churn Drivers", fontsize=15, y=1.02)

# 3a) Churn rate by Contract type
ct = df.groupby("Contract")["Churn"].mean().sort_values(ascending=False)
axes[0, 0].bar(ct.index, ct.values, color=["#ff6b6b", "#ffa94d", "#00e5a0"])
axes[0, 0].set_title("Churn Rate by Contract")
axes[0, 0].set_ylabel("Churn Rate")

# 3b) Tenure distribution per churn class
for val, label, col in [(0, "Retained", "#00e5a0"), (1, "Churned", "#ff6b6b")]:
    subset = df.loc[df["Churn"] == val, "Tenure"]
    axes[0, 1].hist(subset, bins=30, alpha=0.6, label=label, color=col)
axes[0, 1].set_title("Tenure by Churn Status")
axes[0, 1].set_xlabel("Tenure (months)")
axes[0, 1].legend()

# 3c) Monthly Charges boxplot
df.boxplot(column="MonthlyCharges", by="Churn", ax=axes[0, 2])
axes[0, 2].set_title("Monthly Charges by Churn")
axes[0, 2].set_xlabel("Churn (0 = No, 1 = Yes)")
plt.suptitle("") 

# 3d) Internet Service churn rate
isc = df.groupby("InternetService")["Churn"].mean().sort_values(ascending=False)
axes[1, 0].bar(isc.index, isc.values, color=["#ff6b6b", "#ffa94d", "#00e5a0"])
axes[1, 0].set_title("Churn Rate by Internet Service")
axes[1, 0].set_ylabel("Churn Rate")

# 3e) Support calls vs churn
sc = df.groupby("SupportCalls")["Churn"].mean()
axes[1, 1].plot(sc.index, sc.values, marker="o", color="#b388ff", linewidth=2)
axes[1, 1].set_title("Churn Rate vs Support Calls")
axes[1, 1].set_xlabel("Number of Support Calls")
axes[1, 1].set_ylabel("Churn Rate")

# 3f) Senior Citizen churn
sen = df.groupby("SeniorCitizen")["Churn"].mean()
axes[1, 2].bar(["Non-Senior", "Senior"], sen.values,
               color=["#00bcd4", "#ff6b6b"])
axes[1, 2].set_title("Churn Rate – Senior vs Non-Senior")
axes[1, 2].set_ylabel("Churn Rate")

plt.tight_layout()
plt.savefig("eda_churn_drivers.png", dpi=150, bbox_inches="tight")
plt.show()


# ─── 4. FEATURE ENGINEERING & PREPROCESSING ─────────────────
# One-hot encode categoricals
df_model = pd.get_dummies(df.drop(columns=["CustomerID"]),
                          columns=["Gender", "Contract", "InternetService"],
                          drop_first=True)

X = df_model.drop(columns=["Churn"])
y = df_model["Churn"]

# Train / test split  (80 / 20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\nTrain: {X_train_sc.shape}  |  Test: {X_test_sc.shape}")


# ─── 5. MODEL TRAINING ───────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=8,
                                                  random_state=42),
    "KNN (k=7)":           KNeighborsClassifier(n_neighbors=7),
}

results = {}
for name, clf in models.items():
    clf.fit(X_train_sc, y_train)
    y_pred  = clf.predict(X_test_sc)
    y_prob  = clf.predict_proba(X_test_sc)[:, 1]
    auc     = roc_auc_score(y_test, y_prob)
    results[name] = {"clf": clf, "y_pred": y_pred, "y_prob": y_prob, "auc": auc}
    print(f"\n{'='*50}\n{name}  –  ROC-AUC: {auc:.4f}\n{'='*50}")
    print(classification_report(y_test, y_pred))


# ─── 6. MODEL EVALUATION VISUALS ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC curves
colors = {"Logistic Regression": "#00e5a0",
          "Random Forest":       "#ffa94d",
          "KNN (k=7)":           "#b388ff"}
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    axes[0].plot(fpr, tpr, color=colors[name],
                 label=f"{name} (AUC = {res['auc']:.3f})", linewidth=2)
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curves – Churn Models")
axes[0].legend(loc="lower right")

# Confusion matrix for best model
best_name = max(results, key=lambda k: results[k]["auc"])
ConfusionMatrixDisplay.from_predictions(
    y_test, results[best_name]["y_pred"], ax=axes[1],
    cmap="Blues", colorbar=False
)
axes[1].set_title(f"Confusion Matrix – {best_name}")

plt.tight_layout()
plt.savefig("model_evaluation.png", dpi=150, bbox_inches="tight")
plt.show()


# ─── 7. FEATURE IMPORTANCE (Random Forest) ──────────────────
rf = results["Random Forest"]["clf"]
importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 6))
importance.tail(10).plot(kind="barh", color="#00e5a0", ax=ax)
ax.set_title("Top-10 Churn Drivers (Random Forest Feature Importance)")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()


# ─── 8. MONTHLY TREND EXPORT (for dashboard) ────────────────
# Simulated monthly churn-rate trend
months = np.arange(1, 13)
trend_churn   = np.round(0.28 - 0.015 * months + np.random.normal(0, 0.01, 12), 3).clip(0.05, 0.35)
trend_retain  = 1 - trend_churn
trend_df = pd.DataFrame({
    "Month":         months,
    "Churn_Rate":    trend_churn,
    "Retention_Rate": trend_retain,
})
trend_df.to_csv("monthly_churn_trend.csv", index=False)
print("\n✓ Saved monthly_churn_trend.csv for dashboard import.")
print("\n─── DONE ───")
