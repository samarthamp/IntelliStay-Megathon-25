"""
DiCE (Diverse Counterfactual Explanations) Integration
Generates actionable retention strategies by finding minimal changes needed
"""

import pandas as pd
import numpy as np
import joblib
import dice_ml
from dice_ml import Dice
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("DiCE COUNTERFACTUAL EXPLANATIONS SETUP")
print("=" * 80)

# ============================================================================
# 1. LOAD MODEL AND DATA
# ============================================================================

print("\n[1] Loading model and data...")

# Load model
model = joblib.load('final_xgboost_model.pkl')
print("Loaded model")

# Load feature names
feature_names = joblib.load('feature_names.pkl')
print(f"Loaded {len(feature_names)} features")

# Load scaler
scaler = joblib.load('scaler.pkl')
print("Loaded scaler")

# Load test data
test_data = pd.read_csv('test_data_with_predictions.csv')
print(f"Loaded {len(test_data)} test samples")

# Load original engineered data (before scaling) for DiCE
original_data = pd.read_csv('autoinsurance_churn_engineered.csv')

# Drop columns that were dropped in modeling
drop_cols = ['individual_id', 'address_id', 'date_of_birth', 'cust_orig_date', 
             'acct_suspd_date', 'city', 'county', 'Churn']
drop_cols = [col for col in drop_cols if col in original_data.columns]
original_data = original_data.drop(columns=drop_cols)

print(f"Loaded original data: {original_data.shape}")

# ============================================================================
# 2. DEFINE ACTIONABLE FEATURES
# ============================================================================

print("\n[2] Defining actionable vs immutable features...")

# Features the company CAN change through retention actions
ACTIONABLE_FEATURES = [
    'curr_ann_amt',           # Offer premium discount
    'monthly_premium',        # Derived from annual premium
    'premium_affordability',  # Changes with premium
    'premium_to_income_ratio' # Changes with premium
]

# Features that CANNOT be changed
IMMUTABLE_FEATURES = [
    'age_in_years', 'age_group',
    'days_tenure', 'tenure_years', 'tenure_months', 'tenure_bucket',
    'latitude', 'longitude', 'state', 'state_churn_rate',
    'income', 'income_bracket',
    'has_children', 'marital_status', 'family_size_proxy',
    'length_of_residence', 'is_long_term_resident',
    'home_market_value', 'home_owner',
    'college_degree', 'good_credit', 'customer_quality_score',
    'cust_orig_year', 'cust_orig_month'
]

print(f"\nActionable features ({len(ACTIONABLE_FEATURES)}):")
for feat in ACTIONABLE_FEATURES:
    print(f"  • {feat}")

print(f"\nImmutable features ({len(IMMUTABLE_FEATURES)}):")
print(f"  (Cannot be changed by retention actions)")

# ============================================================================
# 3. PREPARE DATA FOR DiCE
# ============================================================================

print("\n[3] Preparing data for DiCE...")

# DiCE needs unscaled data, but ALL features must be numeric
# We need to use the preprocessed (encoded) data, not the raw original

# Load the engineered data and apply same preprocessing as modeling
dice_df = pd.read_csv('autoinsurance_churn_engineered.csv')

# Drop the same columns
drop_cols = ['individual_id', 'address_id', 'date_of_birth', 'cust_orig_date', 
             'acct_suspd_date', 'city', 'county']
drop_cols = [col for col in drop_cols if col in dice_df.columns]
dice_df = dice_df.drop(columns=drop_cols)

# Handle missing values (same as modeling)
# For numerical columns, fill with median
numerical_cols = dice_df.select_dtypes(include=[np.number]).columns.tolist()
if 'Churn' in numerical_cols:
    numerical_cols.remove('Churn')

for col in numerical_cols:
    if dice_df[col].isnull().sum() > 0:
        median_val = dice_df[col].median()
        dice_df[col].fillna(median_val, inplace=True)

# Define categorical columns
categorical_cols = dice_df.select_dtypes(include=['object', 'category']).columns.tolist()

# --- FIX STARTS HERE ---
# Handle missing categorical values by filling with the mode
for col in categorical_cols:
    if dice_df[col].isnull().sum() > 0:
        # Get the most frequent value (mode)
        mode_val = dice_df[col].mode()[0] 
        dice_df[col].fillna(mode_val, inplace=True)
# --- FIX ENDS HERE ---

# Encode categorical variables (same as modeling)
# Load the label encoders we saved during training
label_encoders = joblib.load('label_encoders.pkl')

for col in categorical_cols:
    if col in label_encoders:
        # Use the same encoder from training
        le = label_encoders[col]
        # The .astype(str) is still good practice in case of mixed types
        dice_df[col] = le.transform(dice_df[col].astype(str))
    else:
        # If for some reason it's not in encoders, create new one
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        dice_df[col] = le.fit_transform(dice_df[col].astype(str))

print(f"All features are now numeric")
print(f"Data types: {dice_df.dtypes.value_counts().to_dict()}")

# Now filter to test set indices
test_indices = test_data.index
dice_data = dice_df.loc[test_indices].copy()

# Separate features and target
X_dice = dice_data[feature_names]
y_dice = dice_data['Churn']

# Create final DataFrame for DiCE
dice_data_final = X_dice.copy()
dice_data_final['Churn'] = y_dice

print(f"DiCE dataset prepared: {dice_data_final.shape}")
print(f"All columns numeric: {dice_data_final.select_dtypes(include=[np.number]).shape[1] == dice_data_final.shape[1]}")

# ============================================================================
# 4. CREATE DiCE DATA INTERFACE
# ============================================================================

print("\n[4] Creating DiCE data interface...")

# Verify all features are numeric
assert dice_data_final.select_dtypes(include=[np.number]).shape[1] == dice_data_final.shape[1], \
    "Not all features are numeric!"

# Create DiCE data object - ALL features are continuous (numeric)
dice_data_obj = dice_ml.Data(
    dataframe=dice_data_final,
    continuous_features=feature_names,  # All are numeric now
    outcome_name='Churn'
)

print("DiCE data interface created")

# ============================================================================
# 5. CREATE DiCE MODEL WRAPPER
# ============================================================================

print("\n[5] Wrapping XGBoost model for DiCE...")

# DiCE needs a specific model interface
# We'll create a custom wrapper since we have preprocessing (scaling)

class ScaledXGBoostWrapper:
    """Wrapper to handle scaling before XGBoost prediction"""
    
    def __init__(self, model, scaler, feature_names):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.backend = "sklearn"
        
    def predict(self, X):
        """Predict method for DiCE"""
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names].values
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Predict probabilities
        return self.model.predict_proba(X_scaled)
    
    def predict_proba(self, X):
        """For compatibility"""
        return self.predict(X)

# Create wrapped model
wrapped_model = ScaledXGBoostWrapper(model, scaler, feature_names)

# Create DiCE model object
dice_model = dice_ml.Model(model=wrapped_model, backend="sklearn", model_type='classifier')

print("Model wrapped for DiCE")

# ============================================================================
# 6. INITIALIZE DiCE EXPLAINER
# ============================================================================

print("\n[6] Initializing DiCE explainer...")

dice_explainer = Dice(
    dice_data_obj,
    dice_model,
    method='genetic'  # Can also use 'genetic' for better optimization
)

print(" DiCE explainer initialized!")

# Save for dashboard use
joblib.dump({
    'dice_explainer': dice_explainer,
    'dice_data': dice_data,
    'actionable_features': ACTIONABLE_FEATURES,
    'immutable_features': IMMUTABLE_FEATURES,
    'wrapped_model': wrapped_model
}, 'dice_explainer.pkl')

print("\nSaved: dice_explainer.pkl")

# ============================================================================
# 7. GENERATE SAMPLE COUNTERFACTUALS
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING SAMPLE COUNTERFACTUALS")
print("=" * 80)

# Select high-risk customers
high_risk_customers = test_data[test_data['Churn_Probability'] > 0.7].head(3)

print(f"\n📊 Generating counterfactuals for {len(high_risk_customers)} high-risk customers...")

sample_counterfactuals = []

for idx, row in high_risk_customers.iterrows():
    # Get customer data from dice_data_final (not dice_data)
    customer_data = dice_data_final.loc[idx:idx].drop('Churn', axis=1)
    original_prob = row['Churn_Probability']
    
    print(f"\n{'='*60}")
    print(f"Customer {idx} | Original Churn Risk: {original_prob:.1%}")
    print(f"{'='*60}")
    
    try:
        # Generate counterfactuals
        dice_cf = dice_explainer.generate_counterfactuals(
            query_instances=customer_data,
            total_CFs=3,  # Generate 3 diverse options
            desired_class=0,  # Want to change to "No Churn"
            features_to_vary=ACTIONABLE_FEATURES,  # Only change these
            permitted_range={
                'curr_ann_amt': [
                    customer_data['curr_ann_amt'].values[0] * 0.5,  # Max 50% discount
                    customer_data['curr_ann_amt'].values[0]   # Cannot increase
                ],
                'monthly_premium': [
                    customer_data['monthly_premium'].values[0] * 0.5,
                    customer_data['monthly_premium'].values[0]
                ],
                'premium_affordability': [
                    customer_data['premium_affordability'].values[0] * 0.5,
                    customer_data['premium_affordability'].values[0]
                ],
                'premium_to_income_ratio': [
                    customer_data['premium_to_income_ratio'].values[0] * 0.5,
                    customer_data['premium_to_income_ratio'].values[0]
                ]
            },
            proximity_weight=0.5,  # Balance between minimal change and success
            diversity_weight=1.0   # Want diverse options
        )
        
        # Extract counterfactuals
        # Extract counterfactuals
        cf_examples = dice_cf.cf_examples_list[0]
        
        # Check if counterfactuals were found
        if cf_examples.final_cfs_df is None or len(cf_examples.final_cfs_df) == 0:
            print("\n⚠️  No valid counterfactuals found for this customer")
            continue
        
        cf_df = cf_examples.final_cfs_df
        
        print("\n✅ RETENTION STRATEGY OPTIONS:")
        print(f"   Found {len(cf_df)} counterfactual scenarios\n")
        
        # Iterate through counterfactuals (usually 3)
        option_num = 1
        for cf_idx in range(len(cf_df)):
            cf_row = cf_df.iloc[cf_idx]
            
            # Get predicted churn probability for this counterfactual
            cf_features = cf_row[feature_names].values.reshape(1, -1)
            cf_scaled = scaler.transform(cf_features)
            cf_prob = model.predict_proba(cf_scaled)[0][1]
            
            print(f"\n{'─'*60}")
            print(f"OPTION {option_num}:")
            print(f"  New Churn Risk: {cf_prob:.1%} (was {original_prob:.1%})")
            print(f"  Risk Reduction: {(original_prob - cf_prob):.1%}")
            print(f"{'─'*60}")
            
            # Show what changed
            has_changes = False
            for feat in ACTIONABLE_FEATURES:
                orig_val = customer_data[feat].values[0]
                new_val = cf_row[feat]
                
                # Check for significant change (more than 0.1%)
                if abs(new_val - orig_val) / (abs(orig_val) + 1e-10) > 0.001:
                    has_changes = True
                    pct_change = ((new_val - orig_val) / (abs(orig_val) + 1e-10)) * 100
                    
                    # Format based on feature type
                    if 'premium' in feat.lower() or 'curr_ann_amt' in feat.lower():
                        print(f"  💰 {feat}:")
                        print(f"     Original: ${orig_val:.2f}")
                        print(f"     New:      ${new_val:.2f}")
                        print(f"     Change:   {pct_change:+.1f}%")
                    elif 'ratio' in feat.lower() or 'affordability' in feat.lower():
                        print(f"  📊 {feat}:")
                        print(f"     Original: {orig_val:.4f}")
                        print(f"     New:      {new_val:.4f}")
                        print(f"     Change:   {pct_change:+.1f}%")
                    else:
                        print(f"  {feat}:")
                        print(f"     Original: {orig_val:.2f}")
                        print(f"     New:      {new_val:.2f}")
                        print(f"     Change:   {pct_change:+.1f}%")
            
            if not has_changes:
                print("  ⚠️  No significant changes in actionable features")
            
            option_num += 1
        
        print(f"\n{'='*60}\n")
        
        # Save for dashboard
        sample_counterfactuals.append({
            'customer_index': idx,
            'original_probability': original_prob,
            'counterfactuals': cf_df.to_dict('records')
        })
        
        print(f"\nSuccessfully generated counterfactuals")
        
    except Exception as e:
        print(f"\n⚠️  Could not generate counterfactuals: {str(e)}")
        continue

# Save sample counterfactuals
if sample_counterfactuals:
    joblib.dump(sample_counterfactuals, 'sample_counterfactuals.pkl')
    print(f"\nSaved: sample_counterfactuals.pkl ({len(sample_counterfactuals)} examples)")

# ============================================================================
# 8. SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("DiCE SETUP COMPLETE!")
print("=" * 80)

print("\n📊 Configuration:")
print(f"  - Actionable features: {len(ACTIONABLE_FEATURES)}")
print(f"  - Immutable features: {len(IMMUTABLE_FEATURES)}")
print(f"  - Sample counterfactuals: {len(sample_counterfactuals)}")

print("\n📁 Files created:")
print("  1. dice_explainer.pkl")
print("  2. sample_counterfactuals.pkl")

print("\n✅ Ready for dashboard integration!")
print("\n💡 DiCE will generate:")
print("   - Minimal changes needed to reduce churn")
print("   - Multiple diverse retention strategies")
print("   - Only actionable features (premium discounts)")
print("   - Feasible, realistic recommendations")