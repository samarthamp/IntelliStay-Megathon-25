"""
Machine Learning Pipeline for Churn Prediction (GPU-OPTIMIZED)
Includes: Data Preprocessing, SMOTE, Model Training
Optimized for NVIDIA RTX 4060
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
warnings.filterwarnings('ignore')

# ============================================================================
# 0. GPU CONFIGURATION CHECK
# ============================================================================

print("=" * 80)
print("GPU CONFIGURATION CHECK")
print("=" * 80)

print(f"\nXGBoost version: {xgb.__version__}")

# Check for GPU availability
try:
    # Try to get GPU info
    gpu_available = True
    device = 'cuda'
    tree_method = 'hist'
    print(f"✓ GPU training enabled!")
    print(f"  Device: {device}")
    print(f"  Tree method: {tree_method} (GPU-accelerated)")
except Exception as e:
    gpu_available = False
    device = 'cpu'
    tree_method = 'hist'
    print(f"⚠️  GPU not available, using CPU")
    print(f"  Reason: {str(e)}")

# ============================================================================
# 1. LOAD ENGINEERED DATA
# ============================================================================

print("\n" + "=" * 80)
print("LOADING ENGINEERED DATA")
print("=" * 80)

df = pd.read_csv('autoinsurance_churn_engineered.csv')
print(f"\nDataset shape: {df.shape}")
print(f"Churn rate: {df['Churn'].mean():.2%}")

# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================

print("\n" + "=" * 80)
print("DATA PREPROCESSING")
print("=" * 80)

# ---- Drop unnecessary columns ----
drop_cols = ['individual_id', 'address_id', 'date_of_birth', 'cust_orig_date',
             'acct_suspd_date', 'city', 'county']
drop_cols = [col for col in drop_cols if col in df.columns]

print(f"\nDropping columns: {drop_cols}")
df_model = df.drop(columns=drop_cols)

# ---- Handle missing values ----
print("\n[1] Handling missing values...")

# Fill numerical missing values with median
numerical_cols = df_model.select_dtypes(include=[np.number]).columns.tolist()
numerical_cols.remove('Churn')  # Don't fill target

for col in numerical_cols:
    if df_model[col].isnull().sum() > 0:
        median_val = df_model[col].median()
        df_model[col].fillna(median_val, inplace=True)
        print(f"  ✓ Filled {col} with median: {median_val:.2f}")

# Fill categorical missing values with mode
categorical_cols = df_model.select_dtypes(include=['object', 'category']).columns.tolist()

for col in categorical_cols:
    if df_model[col].isnull().sum() > 0:
        mode_val = df_model[col].mode()[0]
        df_model[col].fillna(mode_val, inplace=True)
        print(f"  ✓ Filled {col} with mode: {mode_val}")

# ---- Encode categorical variables ----
print("\n[2] Encoding categorical variables...")

label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col].astype(str))
    label_encoders[col] = le
    print(f"  ✓ Encoded {col} ({len(le.classes_)} categories)")

# Save encoders
joblib.dump(label_encoders, 'label_encoders.pkl')
print("\n✓ Saved: label_encoders.pkl")

# ---- Separate features and target ----
X = df_model.drop('Churn', axis=1)
y = df_model['Churn']

print(f"\n✓ Features shape: {X.shape}")
print(f"✓ Target shape: {y.shape}")
print(f"✓ Feature names: {list(X.columns)}")

# ---- Train-test split ----
print("\n[3] Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  ✓ Training set: {X_train.shape[0]:,} samples")
print(f"  ✓ Test set: {X_test.shape[0]:,} samples")
print(f"  ✓ Train churn rate: {y_train.mean():.2%}")
print(f"  ✓ Test churn rate: {y_test.mean():.2%}")

# ---- Feature Scaling ----
print("\n[4] Scaling features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame to keep feature names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

joblib.dump(scaler, 'scaler.pkl')
print("  ✓ Saved: scaler.pkl")

# ============================================================================
# 3. HANDLE CLASS IMBALANCE WITH SMOTE
# ============================================================================

print("\n" + "=" * 80)
print("HANDLING CLASS IMBALANCE WITH SMOTE")
print("=" * 80)

print(f"\nBefore SMOTE:")
print(f"  Class 0: {(y_train == 0).sum():,}")
print(f"  Class 1: {(y_train == 1).sum():,}")
print(f"  Ratio: 1:{(y_train == 0).sum() / (y_train == 1).sum():.2f}")

smote_start = time.time()
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
smote_time = time.time() - smote_start

print(f"\nAfter SMOTE:")
print(f"  Class 0: {(y_train_resampled == 0).sum():,}")
print(f"  Class 1: {(y_train_resampled == 1).sum():,}")
print(f"  Ratio: 1:1 (balanced)")
print(f"  ⏱️  SMOTE time: {smote_time:.2f} seconds")

# ============================================================================
# 4. MODEL TRAINING
# ============================================================================

print("\n" + "=" * 80)
print("MODEL TRAINING")
print("=" * 80)

train_start = time.time()

model = xgb.XGBClassifier(
    random_state=42,
    device=device,
    tree_method=tree_method
)
model.fit(X_train_resampled, y_train_resampled)

train_time = time.time() - train_start

print(f"\n✓ Model trained successfully!")
print(f"  ⏱️  Training time: {train_time:.2f} seconds")


# ============================================================================
# 5. MODEL EVALUATION
# ============================================================================

print("\n" + "=" * 80)
print("MODEL EVALUATION")
print("=" * 80)

y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("\n📊 Performance Metrics:")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")
print(f"  ROC-AUC:   {roc_auc:.4f}")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'])
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('original_data_visualisation/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: confusion_matrix.png")

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve', fontsize=14, fontweight='bold')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('results/roc_curve.png', dpi=300, bbox_inches='tight')
print("✓ Saved: roc_curve.png")

# Feature Importance
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n📈 Top 20 Most Important Features:")
print(feature_importance.head(20))

plt.figure(figsize=(12, 8))
top_features = feature_importance.head(20)
plt.barh(top_features['Feature'], top_features['Importance'])
plt.xlabel('Importance')
plt.title('Top 20 Feature Importances', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('original_data_visualisation/feature_importance.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: feature_importance.png")

# ============================================================================
# 6. SAVE MODEL AND ARTIFACTS
# ============================================================================

print("\n" + "=" * 80)
print("SAVING MODEL AND ARTIFACTS")
print("=" * 80)

# Save model
joblib.dump(model, 'final_xgboost_model.pkl')
print("✓ Saved: final_xgboost_model.pkl")

# Save feature names
joblib.dump(list(X_train.columns), 'feature_names.pkl')
print("✓ Saved: feature_names.pkl")

# Save test data for dashboard
test_data = pd.DataFrame(X_test_scaled, columns=X_test.columns)
test_data['Churn_True'] = y_test.values
test_data['Churn_Predicted'] = y_pred
test_data['Churn_Probability'] = y_proba
test_data.to_csv('test_data_with_predictions.csv', index=False)
print("✓ Saved: test_data_with_predictions.csv")

# Save performance metrics
metrics_dict = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'roc_auc': roc_auc,
    'training_time': train_time,
    'gpu_used': gpu_available
}
joblib.dump(metrics_dict, 'model_metrics.pkl')
print("✓ Saved: model_metrics.pkl")


# ============================================================================
# 7. TIMING SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("TIMING SUMMARY")
print("=" * 80)

total_time = smote_time + train_time

print(f"\n⏱️  Performance Breakdown:")
print(f"  SMOTE resampling:    {smote_time:.2f}s")
print(f"  Model training:      {train_time:.2f}s")
print(f"  {'─'*40}")
print(f"  TOTAL PIPELINE TIME: {total_time:.2f}s")

if gpu_available:
    print(f"\n⚡ GPU acceleration was used")
else:
    print(f"\n💻 CPU mode was used")

# ============================================================================
# 8. SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("MODELING PIPELINE COMPLETE!")
print("=" * 80)

print("\n📊 Final Model Summary:")
print(f"  - Algorithm: XGBoost Classifier (GPU: {gpu_available})")
print(f"  - Training samples: {X_train_resampled.shape[0]:,} (after SMOTE)")
print(f"  - Test samples: {X_test.shape[0]:,}")
print(f"  - Features: {X_train.shape[1]}")
print(f"  - ROC-AUC Score: {roc_auc:.4f}")
print(f"  - F1-Score: {f1:.4f}")
print(f"  - Total time: {total_time:.2f} seconds")

print("\n📁 Files created:")
print("  1. label_encoders.pkl")
print("  2. scaler.pkl")
print("  3. final_xgboost_model.pkl")
print("  4. feature_names.pkl")
print("  5. test_data_with_predictions.csv")
print("  6. model_metrics.pkl")
print("  7. confusion_matrix.png")
print("  8. roc_curve.png")
print("  9. feature_importance.png")

print("\n✅ Ready for SHAP explainability and dashboard!")