"""
Should We Stop Giving Discounts?
A Real Talk About US Cellular's Retention Strategy

During my internship at US Cellular, I kept wondering: we throw discounts
at customers who threaten to leave... but does it actually work? Or are we
just training them to complain for free stuff?

This analysis simulates what WOULD happen if we tested different discount
strategies. I can't use real customer data (obviously), but the model is
based on the patterns I saw in 2M+ customer records.

The question: If we offered targeted 15% discounts to high-risk customers,
would we save more revenue than we'd lose from the discount itself?

Spoiler: the answer surprised me.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
import warnings
warnings.filterwarnings('ignore')

sns.set_style("darkgrid")
plt.rcParams['figure.facecolor'] = '#0a0a0a'
plt.rcParams['axes.facecolor'] = '#1a1a1a'
plt.rcParams['text.color'] = '#e0e0e0'
plt.rcParams['axes.labelcolor'] = '#e0e0e0'
plt.rcParams['xtick.color'] = '#e0e0e0'
plt.rcParams['ytick.color'] = '#e0e0e0'

np.random.seed(42)


# ═══════════════════════════════════════════════════════════
# PART 1: Generate Realistic Telecom Customer Data
# ═══════════════════════════════════════════════════════════
# Based on actual patterns from my internship work

def generate_telecom_customers(n=8000):
    """
    This mimics what I saw at US Cellular:
    - Most people stay 12-36 months
    - Month-to-month contracts = churn city
    - Support calls spike right before someone leaves
    - People on autopay stick around longer
    """
    
    # Demographics & contract
    age = np.random.gamma(4, 10, n).clip(18, 80).astype(int)
    tenure_months = np.random.exponential(18, n).clip(1, 72).astype(int)
    contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.55, 0.30, 0.15])
    autopay = np.random.choice([0, 1], n, p=[0.40, 0.60])
    paperless = np.random.choice([0, 1], n, p=[0.45, 0.55])
    
    # Usage & billing
    monthly_gb = np.random.lognormal(2.5, 0.8, n).clip(0.5, 50)
    monthly_charge = (
        35 + monthly_gb * 1.2 + 
        np.random.choice([0, 10, 25], n, p=[0.6, 0.3, 0.1]) +  # add-ons
        np.random.normal(0, 5, n)
    ).clip(25, 150)
    
    overage_fees = np.where(
        monthly_gb > 15,
        np.random.exponential(8, n),
        0
    ).clip(0, 50)
    
    # Behavioral signals
    support_calls = np.random.poisson(
        0.8 + 0.02 * tenure_months + 0.15 * (contract == 'Month-to-month').astype(float),
        n
    )
    late_payments = np.random.poisson(0.3, n).clip(0, 5)
    
    # Calculate churn probability based on real risk factors
    churn_risk = (
        0.18  # baseline
        + 0.25 * (contract == 'Month-to-month').astype(float)
        - 0.08 * (contract == 'Two year').astype(float)
        - 0.004 * tenure_months
        + 0.003 * monthly_charge
        - 0.06 * autopay
        + 0.04 * support_calls
        + 0.05 * late_payments
        + 0.002 * overage_fees
        - 0.001 * age
    )
    churn_risk = churn_risk.clip(0.02, 0.85)
    churned = (np.random.rand(n) < churn_risk).astype(int)
    
    # Revenue impact if they churn
    lifetime_value = monthly_charge * (72 - tenure_months) * 0.7  # rough CLV estimate
    
    return pd.DataFrame({
        'customer_id': range(1000, 1000 + n),
        'age': age,
        'tenure_months': tenure_months,
        'contract_type': contract,
        'monthly_charge': monthly_charge.round(2),
        'monthly_gb': monthly_gb.round(2),
        'overage_fees': overage_fees.round(2),
        'autopay': autopay,
        'paperless_billing': paperless,
        'support_calls_3mo': support_calls,
        'late_payments': late_payments,
        'churned': churned,
        'estimated_clv': lifetime_value.round(2)
    })


print("="*60)
print("GENERATING SYNTHETIC CUSTOMER BASE")
print("="*60)
df = generate_telecom_customers(8000)
print(f"\nTotal customers: {len(df):,}")
print(f"Churn rate: {df['churned'].mean():.1%}")
print(f"At-risk revenue: ${df[df['churned']==1]['estimated_clv'].sum():,.0f}")


# ═══════════════════════════════════════════════════════════
# PART 2: Quick EDA - What Actually Drives Churn?
# ═══════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.patch.set_facecolor('#0a0a0a')

# Contract type
ct_churn = df.groupby('contract_type')['churned'].mean().sort_values(ascending=False)
axes[0,0].bar(range(len(ct_churn)), ct_churn.values, color=['#ff6b6b', '#ffa94d', '#4ecdc4'])
axes[0,0].set_xticks(range(len(ct_churn)))
axes[0,0].set_xticklabels(ct_churn.index, rotation=15)
axes[0,0].set_title('Month-to-month is a disaster', fontsize=11, color='#ff6b6b')
axes[0,0].set_ylabel('Churn Rate', color='#e0e0e0')

# Tenure
for churned, label, c in [(0, 'Stayed', '#4ecdc4'), (1, 'Left', '#ff6b6b')]:
    axes[0,1].hist(df[df['churned']==churned]['tenure_months'], bins=30, 
                   alpha=0.6, label=label, color=c)
axes[0,1].set_title('New customers are flight risks', fontsize=11)
axes[0,1].legend()
axes[0,1].set_xlabel('Months with us', color='#e0e0e0')

# Support calls
sc = df.groupby('support_calls_3mo')['churned'].mean()
axes[0,2].plot(sc.index, sc.values, marker='o', color='#ffa94d', linewidth=2)
axes[0,2].set_title('People call us before they leave', fontsize=11)
axes[0,2].set_xlabel('Support calls (last 3 months)', color='#e0e0e0')
axes[0,2].set_ylabel('Churn Rate', color='#e0e0e0')

# Monthly charges
df.boxplot(column='monthly_charge', by='churned', ax=axes[1,0])
axes[1,0].set_title('Price sensitivity is real', fontsize=11)
axes[1,0].set_xticklabels(['Stayed', 'Churned'])
plt.suptitle('')

# Autopay
ap = df.groupby('autopay')['churned'].mean()
axes[1,1].bar(['Manual pay', 'Autopay'], ap.values, color=['#ff6b6b', '#4ecdc4'])
axes[1,1].set_title('Autopay = stickiness', fontsize=11)
axes[1,1].set_ylabel('Churn Rate', color='#e0e0e0')

# CLV at risk by segment
risk_by_tenure = df[df['churned']==1].groupby(
    pd.cut(df[df['churned']==1]['tenure_months'], bins=[0,6,12,24,72])
)['estimated_clv'].sum() / 1000
axes[1,2].bar(range(len(risk_by_tenure)), risk_by_tenure.values, color='#ff6b6b')
axes[1,2].set_xticks(range(len(risk_by_tenure)))
axes[1,2].set_xticklabels(['0-6mo', '6-12mo', '1-2yr', '2yr+'], rotation=15)
axes[1,2].set_title('Where we're bleeding revenue', fontsize=11)
axes[1,2].set_ylabel('Lost CLV ($k)', color='#e0e0e0')

plt.tight_layout()
plt.savefig('churn_drivers.png', dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
print("\n✓ Saved churn_drivers.png")


# ═══════════════════════════════════════════════════════════
# PART 3: Build the Churn Prediction Model
# ═══════════════════════════════════════════════════════════
# Trying 3 approaches, then picking the best

print("\n" + "="*60)
print("BUILDING PREDICTIVE MODELS")
print("="*60)

# Prep features
cat_cols = ['contract_type']
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

feature_cols = [c for c in df_encoded.columns if c not in 
                ['customer_id', 'churned', 'estimated_clv']]
X = df_encoded[feature_cols]
y = df_encoded['churned']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# Model 1: Gradient Boosting (usually best for tabular data)
print("\n→ Training Gradient Boosting...")
gb = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.05, max_depth=4,
    subsample=0.8, random_state=42
)
gb.fit(X_train_sc, y_train)
gb_pred_proba = gb.predict_proba(X_test_sc)[:, 1]
gb_auc = roc_auc_score(y_test, gb_pred_proba)
print(f"   AUC: {gb_auc:.4f}")

# Model 2: Random Forest
print("\n→ Training Random Forest...")
rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
rf.fit(X_train_sc, y_train)
rf_pred_proba = rf.predict_proba(X_test_sc)[:, 1]
rf_auc = roc_auc_score(y_test, rf_pred_proba)
print(f"   AUC: {rf_auc:.4f}")

# Model 3: Logistic Regression (interpretable baseline)
print("\n→ Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_sc, y_train)
lr_pred_proba = lr.predict_proba(X_test_sc)[:, 1]
lr_auc = roc_auc_score(y_test, lr_pred_proba)
print(f"   AUC: {lr_auc:.4f}")

# Pick the best
best_model = max([('GB', gb, gb_auc), ('RF', rf, rf_auc), ('LR', lr, lr_auc)],
                 key=lambda x: x[2])
print(f"\n✓ Winner: {best_model[0]} (AUC={best_model[2]:.4f})")

# ROC curves
fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor('#0a0a0a')
ax.set_facecolor('#1a1a1a')

for name, model, auc in [('Gradient Boosting', gb, gb_auc), 
                          ('Random Forest', rf, rf_auc),
                          ('Logistic Regression', lr, lr_auc)]:
    proba = model.predict_proba(X_test_sc)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    color = '#4ecdc4' if auc == best_model[2] else '#888888'
    lw = 2.5 if auc == best_model[2] else 1.5
    ax.plot(fpr, tpr, color=color, lw=lw, label=f'{name} (AUC={auc:.3f})')

ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.3)
ax.set_xlabel('False Positive Rate', color='#e0e0e0')
ax.set_ylabel('True Positive Rate', color='#e0e0e0')
ax.set_title('Which model spots churn best?', fontsize=13, color='#e0e0e0')
ax.legend(loc='lower right')
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
print("✓ Saved model_comparison.png")


# ═══════════════════════════════════════════════════════════
# PART 4: The Real Question - Should We Offer Discounts?
# ═══════════════════════════════════════════════════════════

print("\n" + "="*60)
print("DISCOUNT STRATEGY SIMULATION")
print("="*60)

# Score the FULL dataset (not just test set)
X_full_sc = scaler.transform(df_encoded[feature_cols])
churn_probability = best_model[1].predict_proba(X_full_sc)[:, 1]
df['churn_risk_score'] = churn_probability

# Define risk tiers
df['risk_tier'] = pd.cut(df['churn_risk_score'], 
                         bins=[0, 0.3, 0.6, 1.0],
                         labels=['Low', 'Medium', 'High'])

print("\nRisk Distribution:")
print(df['risk_tier'].value_counts().sort_index())

# SCENARIO 1: Do nothing (baseline)
baseline_churned = df['churned'].sum()
baseline_lost_revenue = df[df['churned']==1]['estimated_clv'].sum()

# SCENARIO 2: Offer 15% discount to high-risk customers
# Assumption: 40% of them accept it and stay (based on industry benchmarks I saw)
discount_rate = 0.15
retention_uplift = 0.40

high_risk = df[df['risk_tier'] == 'High'].copy()
high_risk_actually_churning = high_risk[high_risk['churned'] == 1]

# How many would we save?
saved_customers = int(len(high_risk_actually_churning) * retention_uplift)
revenue_saved = high_risk_actually_churning.sample(saved_customers, random_state=42)['estimated_clv'].sum()

# Cost of discount (applied to ALL high-risk, not just churners)
discount_cost_per_customer = (high_risk['monthly_charge'].mean() * 12 * discount_rate)
total_discount_cost = discount_cost_per_customer * len(high_risk)

net_impact = revenue_saved - total_discount_cost

print(f"\n{'BASELINE (do nothing)':.<45} Lost revenue: ${baseline_lost_revenue:>12,.0f}")
print(f"{'DISCOUNT STRATEGY':.<45}")
print(f"  {'→ High-risk customers':.<43} {len(high_risk):>14,}")
print(f"  {'→ Would have churned':.<43} {len(high_risk_actually_churning):>14,}")
print(f"  {'→ Saved by discount (est.)':.<43} {saved_customers:>14,}")
print(f"  {'→ Revenue recovered':.<43} ${revenue_saved:>13,.0f}")
print(f"  {'→ Cost of discounts':.<43} ${total_discount_cost:>13,.0f}")
print(f"  {'→ NET IMPACT':.<43} ${net_impact:>13,.0f}")

if net_impact > 0:
    print(f"\n💡 RECOMMENDATION: Run the discount campaign")
    print(f"   We'd save ${net_impact:,.0f} in net revenue")
else:
    print(f"\n⚠️  RECOMMENDATION: Skip the discounts")
    print(f"   We'd lose ${abs(net_impact):,.0f} compared to doing nothing")


# Save results
results = {
    'scenario': ['Baseline', 'Discount Campaign'],
    'customers_lost': [baseline_churned, baseline_churned - saved_customers],
    'revenue_lost': [baseline_lost_revenue, baseline_lost_revenue - revenue_saved],
    'campaign_cost': [0, total_discount_cost],
    'net_impact': [0, net_impact]
}
pd.DataFrame(results).to_csv('discount_analysis.csv', index=False)
print("\n✓ Saved discount_analysis.csv")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
