"""
EDA and Feature Engineering for Auto Insurance Churn Prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# 1. LOAD DATA
# ============================================================================

print("=" * 80)
print("LOADING DATA")
print("=" * 80)

df = pd.read_csv('autoinsurance_churn.csv')

print(f"\nDataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

# ============================================================================
# 2. BASIC DATA EXPLORATION
# ============================================================================

print("\n" + "=" * 80)
print("BASIC DATA EXPLORATION")
print("=" * 80)

print(f"\nData types:\n{df.dtypes}")
print(f"\nBasic statistics:\n{df.describe()}")

# ============================================================================
# 3. TARGET VARIABLE ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("TARGET VARIABLE ANALYSIS")
print("=" * 80)

churn_counts = df['Churn'].value_counts()
churn_rate = df['Churn'].mean()

print(f"\nChurn distribution:")
print(churn_counts)
print(f"\nChurn rate: {churn_rate:.2%}")
print(f"Class imbalance ratio: 1:{churn_counts[0]/churn_counts[1]:.2f}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Count plot
axes[0].bar(['No Churn', 'Churn'], churn_counts.values, color=['green', 'red'])
axes[0].set_title('Churn Distribution', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Count')
for i, v in enumerate(churn_counts.values):
    axes[0].text(i, v + 1000, str(v), ha='center', fontweight='bold')

# Pie chart
axes[1].pie(churn_counts.values, labels=['No Churn', 'Churn'], 
            autopct='%1.1f%%', colors=['green', 'red'], startangle=90)
axes[1].set_title('Churn Percentage', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('original_data_visualisation/churn_distribution.png', dpi=300, bbox_inches='tight')
print("\n Saved: churn_distribution.png")

# ============================================================================
# 4. MISSING VALUES ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("MISSING VALUES ANALYSIS")
print("=" * 80)

missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Column': missing.index,
    'Missing_Count': missing.values,
    'Missing_Percentage': missing_pct.values
})
missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

print("\nMissing values summary:")
print(missing_df)

# Visualize
if len(missing_df) > 0:
    plt.figure(figsize=(12, 6))
    plt.barh(missing_df['Column'], missing_df['Missing_Percentage'])
    plt.xlabel('Missing Percentage (%)')
    plt.title('Missing Values by Feature', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('original_data_visualisation/missing_values.png', dpi=300, bbox_inches='tight')
    print("\n Saved: missing_values.png")
else:
    print("\n No missing values found!")

# ============================================================================
# 5. NUMERICAL FEATURES ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("NUMERICAL FEATURES ANALYSIS")
print("=" * 80)

numerical_cols = ['curr_ann_amt', 'days_tenure', 'age_in_years', 'income', 
                  'length_of_residence', 'latitude', 'longitude']

# Filter to existing columns
numerical_cols = [col for col in numerical_cols if col in df.columns]

print(f"\nNumerical features: {numerical_cols}")
print(f"\nNumerical statistics:")
print(df[numerical_cols].describe())

# Churn vs No Churn comparison
print("\n" + "-" * 80)
print("Numerical Features by Churn Status")
print("-" * 80)

for col in numerical_cols:
    churn_yes = df[df['Churn'] == 1][col].mean()
    churn_no = df[df['Churn'] == 0][col].mean()
    print(f"\n{col}:")
    print(f"  Churn=0 (mean): {churn_no:.2f}")
    print(f"  Churn=1 (mean): {churn_yes:.2f}")
    print(f"  Difference: {abs(churn_yes - churn_no):.2f}")

# Visualize distributions
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.flatten()

for idx, col in enumerate(numerical_cols):
    if idx < len(axes):
        df[df['Churn'] == 0][col].hist(bins=30, alpha=0.6, label='No Churn', 
                                        color='green', ax=axes[idx])
        df[df['Churn'] == 1][col].hist(bins=30, alpha=0.6, label='Churn', 
                                        color='red', ax=axes[idx])
        axes[idx].set_title(f'{col} Distribution', fontweight='bold')
        axes[idx].legend()
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')

# Hide unused subplots
for idx in range(len(numerical_cols), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('original_data_visualisation/numerical_distributions.png', dpi=300, bbox_inches='tight')
print("\n Saved: numerical_distributions.png")

# ============================================================================
# 6. CATEGORICAL FEATURES ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("CATEGORICAL FEATURES ANALYSIS")
print("=" * 80)

categorical_cols = ['marital_status', 'home_owner', 'college_degree', 
                    'good_credit', 'has_children', 'state', 'home_market_value']

# Filter to existing columns
categorical_cols = [col for col in categorical_cols if col in df.columns]

print(f"\nCategorical features: {categorical_cols}")

for col in categorical_cols:
    print(f"\n{col} - Value counts:")
    print(df[col].value_counts())
    
    # Churn rate by category
    print(f"\nChurn rate by {col}:")
    churn_by_cat = df.groupby(col)['Churn'].agg(['sum', 'count', 'mean'])
    churn_by_cat.columns = ['Churn_Count', 'Total_Count', 'Churn_Rate']
    print(churn_by_cat.sort_values('Churn_Rate', ascending=False))

# Visualize key categorical features
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

plot_cols = ['marital_status', 'home_owner', 'college_degree', 
             'good_credit', 'has_children']
plot_cols = [col for col in plot_cols if col in df.columns]

for idx, col in enumerate(plot_cols[:6]):
    churn_rate_by_cat = df.groupby(col)['Churn'].mean().sort_values(ascending=False)
    churn_rate_by_cat.plot(kind='bar', ax=axes[idx], color='coral')
    axes[idx].set_title(f'Churn Rate by {col}', fontweight='bold')
    axes[idx].set_ylabel('Churn Rate')
    axes[idx].set_xlabel(col)
    axes[idx].tick_params(axis='x', rotation=45)

# Hide unused subplots
for idx in range(len(plot_cols), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('original_data_visualisation/categorical_churn_rates.png', dpi=300, bbox_inches='tight')
print("\n Saved: categorical_churn_rates.png")

# ============================================================================
# 7. CORRELATION ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("CORRELATION ANALYSIS")
print("=" * 80)

# Select numerical columns for correlation
corr_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Remove ID columns
corr_cols = [col for col in corr_cols if 'id' not in col.lower()]

corr_matrix = df[corr_cols].corr()

print("\nCorrelation with Churn:")
churn_corr = corr_matrix['Churn'].sort_values(ascending=False)
print(churn_corr)

# Visualize correlation matrix
plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1)
plt.title('Correlation Matrix', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('original_data_visualisation/correlation_matrix.png', dpi=300, bbox_inches='tight')
print("\n Saved: correlation_matrix.png")

# ============================================================================
# 8. FEATURE ENGINEERING
# ============================================================================

print("\n" + "=" * 80)
print("FEATURE ENGINEERING")
print("=" * 80)

# Create a copy for feature engineering
df_fe = df.copy()

# ---- Date Features ----
print("\n[1] Creating date-based features...")

if 'cust_orig_date' in df_fe.columns:
    df_fe['cust_orig_date'] = pd.to_datetime(df_fe['cust_orig_date'])
    df_fe['tenure_years'] = df_fe['days_tenure'] / 365.25
    df_fe['tenure_months'] = df_fe['days_tenure'] / 30.44
    
    # Extract year and month from customer origin date
    df_fe['cust_orig_year'] = df_fe['cust_orig_date'].dt.year
    df_fe['cust_orig_month'] = df_fe['cust_orig_date'].dt.month
    
    print("   Created: tenure_years, tenure_months, cust_orig_year, cust_orig_month")

# ---- Premium-related Features ----
print("\n[2] Creating premium-related features...")

if 'curr_ann_amt' in df_fe.columns and 'income' in df_fe.columns:
    # Premium to income ratio
    df_fe['premium_to_income_ratio'] = df_fe['curr_ann_amt'] / (df_fe['income'] + 1)
    
    # Monthly premium
    df_fe['monthly_premium'] = df_fe['curr_ann_amt'] / 12
    
    # Premium affordability score (lower is more affordable)
    df_fe['premium_affordability'] = (df_fe['curr_ann_amt'] / (df_fe['income'] + 1)) * 100
    
    print("   Created: premium_to_income_ratio, monthly_premium, premium_affordability")

# ---- Age Buckets ----
print("\n[3] Creating age buckets...")

if 'age_in_years' in df_fe.columns:
    df_fe['age_group'] = pd.cut(df_fe['age_in_years'], 
                                 bins=[0, 25, 35, 45, 55, 65, 100],
                                 labels=['<25', '25-35', '35-45', '45-55', '55-65', '65+'])
    print("   Created: age_group")

# ---- Tenure Buckets ----
print("\n[4] Creating tenure buckets...")

if 'days_tenure' in df_fe.columns:
    df_fe['tenure_bucket'] = pd.cut(df_fe['days_tenure'], 
                                     bins=[0, 365, 730, 1095, 1825, 10000],
                                     labels=['<1yr', '1-2yr', '2-3yr', '3-5yr', '5yr+'])
    print("   Created: tenure_bucket")

# ---- Income Buckets ----
print("\n[5] Creating income buckets...")

if 'income' in df_fe.columns:
    df_fe['income_bracket'] = pd.cut(df_fe['income'], 
                                      bins=[0, 30000, 50000, 75000, 100000, 1000000],
                                      labels=['Low', 'Lower-Mid', 'Mid', 'Upper-Mid', 'High'])
    print("   Created: income_bracket")

# ---- Customer Profile Score ----
print("\n[6] Creating customer profile score...")

# Composite score based on positive indicators
score_components = []

if 'good_credit' in df_fe.columns:
    score_components.append(df_fe['good_credit'])
if 'college_degree' in df_fe.columns:
    score_components.append(df_fe['college_degree'])
if 'home_owner' in df_fe.columns:
    score_components.append(df_fe['home_owner'])

if len(score_components) > 0:
    df_fe['customer_quality_score'] = sum(score_components)
    print("   Created: customer_quality_score")

# ---- Residence Stability ----
print("\n[7] Creating residence stability features...")

if 'length_of_residence' in df_fe.columns:
    df_fe['is_long_term_resident'] = (df_fe['length_of_residence'] >= 10).astype(int)
    print("   Created: is_long_term_resident")

# ---- Geographic Features ----
print("\n[8] Creating geographic features...")

if 'state' in df_fe.columns:
    # Churn rate by state (encoding)
    state_churn_rate = df_fe.groupby('state')['Churn'].mean()
    df_fe['state_churn_rate'] = df_fe['state'].map(state_churn_rate)
    print("   Created: state_churn_rate")

# ---- Family Status ----
print("\n[9] Creating family status features...")

if 'has_children' in df_fe.columns and 'marital_status' in df_fe.columns:
    df_fe['family_size_proxy'] = df_fe['has_children'].fillna(0) + \
                                  (df_fe['marital_status'] == 'Married').astype(int)
    print("   Created: family_size_proxy")

# ============================================================================
# 9. SAVE ENGINEERED DATA
# ============================================================================

print("\n" + "=" * 80)
print("SAVING ENGINEERED DATA")
print("=" * 80)

print(f"\nOriginal shape: {df.shape}")
print(f"Engineered shape: {df_fe.shape}")
print(f"New features created: {df_fe.shape[1] - df.shape[1]}")

# Save
df_fe.to_csv('autoinsurance_churn_engineered.csv', index=False)
print("\n Saved: autoinsurance_churn_engineered.csv")

print("\n" + "=" * 80)
print("EDA AND FEATURE ENGINEERING COMPLETE!")
print("=" * 80)

print("\n📊 Summary:")
print(f"  - Total records: {len(df_fe):,}")
print(f"  - Total features: {df_fe.shape[1]}")
print(f"  - Churn rate: {df_fe['Churn'].mean():.2%}")
print(f"  - New features: {df_fe.shape[1] - df.shape[1]}")

print("\n📁 Files created:")
print("  1. churn_distribution.png")
print("  2. missing_values.png (if applicable)")
print("  3. numerical_distributions.png")
print("  4. categorical_churn_rates.png")
print("  5. correlation_matrix.png")
print("  6. autoinsurance_churn_engineered.csv")

print("\n✅ Ready for modeling!")