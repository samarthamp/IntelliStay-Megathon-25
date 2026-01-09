"""
SHAP Explainability Analysis with Local LLM Integration
- Global explanations (Summary plots)
- Local explanations (Force plots per customer)
- LLM-powered insights using Qwen via LM Studio
"""

import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import requests
import json
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend
plt.switch_backend('Agg')

# ============================================================================
# 0. LM STUDIO CONFIGURATION
# ============================================================================

print("=" * 80)
print("LM STUDIO CONFIGURATION")
print("=" * 80)

# LM Studio default endpoint
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"

# Test connection
def test_lm_studio():
    try:
        response = requests.get("http://localhost:1234/v1/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print(f"LM Studio connected successfully!")
            print(f"  Available model: {models.get('data', [{}])[0].get('id', 'Unknown')}")
            return True
        else:
            print(f"⚠️  LM Studio connection failed: Status {response.status_code}")
            return False
    except Exception as e:
        print(f"⚠️  Cannot connect to LM Studio: {str(e)}")
        print(f"   Make sure LM Studio is running on http://localhost:1234")
        return False

LM_STUDIO_AVAILABLE = test_lm_studio()

# ============================================================================
# 1. LOAD MODEL AND DATA
# ============================================================================

print("\n" + "=" * 80)
print("LOADING MODEL AND DATA FOR SHAP ANALYSIS")
print("=" * 80)

# Load model
model = joblib.load('final_xgboost_model.pkl')
print("Loaded: final_xgboost_model.pkl")

# Load feature names
feature_names = joblib.load('feature_names.pkl')
print(f"Loaded: feature_names.pkl ({len(feature_names)} features)")

# Load test data
test_data = pd.read_csv('test_data_with_predictions.csv')
print(f"Loaded: test_data_with_predictions.csv ({len(test_data)} samples)")

# Prepare data for SHAP (remove prediction columns)
X_test_shap = test_data[feature_names]
y_test = test_data['Churn_True']
y_pred = test_data['Churn_Predicted']
y_proba = test_data['Churn_Probability']

print(f"\nTest data shape: {X_test_shap.shape}")

# ============================================================================
# 2. CREATE SHAP EXPLAINER
# ============================================================================

print("\n" + "=" * 80)
print("CREATING SHAP EXPLAINER")
print("=" * 80)

print("\n🔍 Initializing SHAP TreeExplainer...")
print("   This may take a few minutes for large datasets...")




X_shap_sample = X_test_shap
y_proba_sample = y_proba
print(f"\nUsing full test set ({len(X_test_shap):,} rows)")

# Create explainer
explainer = shap.TreeExplainer(model)
print("SHAP TreeExplainer created successfully!")

# Calculate SHAP values
print("\n🔍 Calculating SHAP values...")
shap_values = explainer.shap_values(X_shap_sample)
print("SHAP values calculated!")

# Save explainer and SHAP values
joblib.dump(explainer, 'shap_explainer.pkl')
print("\nSaved: shap_explainer.pkl")

np.save('shap_values.npy', shap_values)
np.save('shap_sample_indices.npy', X_shap_sample.index.values)
print("Saved: shap_values.npy")
print("Saved: shap_sample_indices.npy")

# ============================================================================
# 3. GLOBAL SHAP EXPLANATIONS (For Strategists)
# ============================================================================

print("\n" + "=" * 80)
print("GLOBAL SHAP EXPLANATIONS - THE BIG PICTURE")
print("=" * 80)

print("\n[1] Creating SHAP Summary Plot (Global Feature Importance)...")

# Summary plot - shows feature importance and impact direction
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values, X_shap_sample, plot_type="dot", show=False)
plt.title('SHAP Summary Plot - What Drives Customer Churn?', 
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('results/shap_summary_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: shap_summary_plot.png")

print("\n[2] Creating SHAP Feature Importance Bar Plot...")

# Bar plot - shows mean absolute SHAP values
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values, X_shap_sample, plot_type="bar", show=False)
plt.title('Top Churn Drivers - Feature Importance Rankings', 
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('results/shap_feature_importance_bar.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: shap_feature_importance_bar.png")

# Calculate global feature importance
mean_abs_shap = np.abs(shap_values).mean(axis=0)
global_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Mean_Abs_SHAP': mean_abs_shap,
}).sort_values('Mean_Abs_SHAP', ascending=False).reset_index(drop=True)

print("\n📊 Top 10 Global Churn Drivers:")
print(global_importance_df.head(10).to_string(index=False))

# Save global importance
global_importance_df.to_csv('shap_global_importance.csv', index=False)
print("\nSaved: shap_global_importance.csv")

# ============================================================================
# 4. LOCAL SHAP EXPLANATIONS (For Retention Agents)
# ============================================================================

print("\n" + "=" * 80)
print("LOCAL SHAP EXPLANATIONS - INDIVIDUAL CUSTOMER INSIGHTS")
print("=" * 80)

# Create a DataFrame to store individual SHAP explanations
individual_explanations = []

print("\n🔍 Generating individual customer explanations...")

for idx in range(min(100, len(X_shap_sample))):  # First 100 customers
    customer_shap = shap_values[idx]
    customer_data = X_shap_sample.iloc[idx]
    customer_proba = y_proba_sample.iloc[idx]
    
    # Get top 5 contributing features (both positive and negative)
    shap_contributions = pd.DataFrame({
        'Feature': feature_names,
        'SHAP_Value': customer_shap,
        'Feature_Value': customer_data.values
    }).sort_values('SHAP_Value', key=abs, ascending=False)
    
    top_drivers = shap_contributions.head(5)
    
    individual_explanations.append({
        'Customer_Index': X_shap_sample.index[idx],
        'Churn_Probability': customer_proba,
        'Top_Driver_1': top_drivers.iloc[0]['Feature'],
        'Top_Driver_1_SHAP': top_drivers.iloc[0]['SHAP_Value'],
        'Top_Driver_1_Value': top_drivers.iloc[0]['Feature_Value'],
        'Top_Driver_2': top_drivers.iloc[1]['Feature'],
        'Top_Driver_2_SHAP': top_drivers.iloc[1]['SHAP_Value'],
        'Top_Driver_2_Value': top_drivers.iloc[1]['Feature_Value'],
        'Top_Driver_3': top_drivers.iloc[2]['Feature'],
        'Top_Driver_3_SHAP': top_drivers.iloc[2]['SHAP_Value'],
        'Top_Driver_3_Value': top_drivers.iloc[2]['Feature_Value'],
    })

individual_explanations_df = pd.DataFrame(individual_explanations)
individual_explanations_df.to_csv('shap_individual_explanations.csv', index=False)
print(f"Generated explanations for {len(individual_explanations)} customers")
print("Saved: shap_individual_explanations.csv")

# ============================================================================
# 5. GENERATE SAMPLE WATERFALL PLOTS (Cleaner than Force Plots)
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING SAMPLE WATERFALL PLOTS")
print("=" * 80)

# Select interesting cases
high_risk_idx = y_proba_sample.nlargest(3).index
low_risk_idx = y_proba_sample.nsmallest(3).index

sample_cases = list(high_risk_idx) + list(low_risk_idx)

print(f"\n📊 Creating waterfall plots for {len(sample_cases)} sample customers...")

for original_idx in sample_cases:
    # Find position in sample
    position = list(X_shap_sample.index).index(original_idx)
    
    risk_level = "HIGH" if y_proba_sample.loc[original_idx] > 0.5 else "LOW"
    churn_prob = y_proba_sample.loc[original_idx]
    
    # Create SHAP Explanation object for waterfall plot
    explanation = shap.Explanation(
        values=shap_values[position],
        base_values=explainer.expected_value,
        data=X_shap_sample.iloc[position].values,
        feature_names=feature_names
    )
    
    # Create waterfall plot (shows top features automatically)
    plt.figure(figsize=(10, 8))
    shap.waterfall_plot(explanation, max_display=10, show=False)  # Top 10 features
    
    # Add custom title
    plt.title(f'Customer {original_idx} | Churn Risk: {risk_level} ({churn_prob:.1%})', 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f'shap_force_plots/shap_waterfall_customer_{original_idx}.png', dpi=300, bbox_inches='tight')
    plt.close()

print(f"  Saved {len(sample_cases)} waterfall plot images")

print("\n💡 Waterfall plots show:")
print("   - Red bars: Features pushing TOWARD churn")
print("   - Blue bars: Features pushing AWAY from churn")
print("   - Only top 10 most impactful features (less clutter!)")

# ============================================================================
# ADDITIONAL: CREATE DATA PACKAGE FOR DASHBOARD
# ============================================================================

print("\n[BONUS] Creating additional data package for dashboard...")

# Create the package with all needed data
shap_data_package_for_dashboard = {
    'shap_values': shap_values,
    'X_data': X_shap_sample,
    'y_proba': y_proba_sample,
    'customer_indices': X_shap_sample.index.tolist(),
    'feature_names': feature_names,
    'expected_value': float(explainer.expected_value)
}

joblib.dump(shap_data_package_for_dashboard, 'shap_data_package.pkl')
print("✓ Saved: shap_data_package.pkl (for dashboard)")

# ============================================================================
# 6. LLM-POWERED INSIGHTS GENERATION
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING LLM-POWERED INSIGHTS")
print("=" * 80)

def generate_llm_insights(customer_index, shap_data, global_context):
    """
    Generate human-readable insights using local LLM (Qwen via LM Studio)
    """
    
    # Prepare context
    top_3_features = []
    for i in range(1, 4):
        feature = shap_data[f'Top_Driver_{i}']
        shap_val = shap_data[f'Top_Driver_{i}_SHAP']
        feat_val = shap_data[f'Top_Driver_{i}_Value']
        direction = "increasing" if shap_val > 0 else "decreasing"
        top_3_features.append(f"{feature} (value: {feat_val:.2f}, impact: {direction} churn risk)")
    
    churn_prob = shap_data['Churn_Probability']
    
    # Create detailed prompt
    prompt = f"""You are an expert insurance retention strategist analyzing customer churn risk.

CUSTOMER PROFILE:
- Customer ID: {customer_index}
- Churn Risk Probability: {churn_prob:.1%}
- Risk Level: {"HIGH RISK" if churn_prob > 0.7 else "MODERATE RISK" if churn_prob > 0.4 else "LOW RISK"}

TOP 3 CHURN DRIVERS FOR THIS CUSTOMER:
1. {top_3_features[0]}
2. {top_3_features[1]}
3. {top_3_features[2]}

INDUSTRY CONTEXT:
Our model has identified that across all customers, the most important churn factors are:
{', '.join(global_context.head(5)['Feature'].tolist())}

YOUR TASK:
Generate a clear, actionable analysis with these sections:

1. CUSTOMER PERSONA (1 sentence):
   Describe this customer's profile in simple business terms.

2. RISK ASSESSMENT (2-3 sentences):
   Explain WHY this customer is at their current risk level based on the top drivers.

3. RETENTION STRATEGY (3-4 bullet points):
   Provide specific, actionable recommendations the retention team can implement immediately.
   Focus on addressing the top drivers you identified.

Keep your language non-technical, concise, and action-oriented. This will be read by customer service agents, not data scientists."""

    if not LM_STUDIO_AVAILABLE:
        return {
            'persona': f"Customer with {churn_prob:.1%} churn probability",
            'risk_assessment': f"Key drivers: {', '.join([f.split('(')[0].strip() for f in top_3_features])}",
            'retention_strategy': [
                "Review customer account details",
                "Contact customer to understand concerns",
                "Offer retention incentives"
            ],
            'raw_analysis': "LM Studio not available - using fallback analysis"
        }
    
    try:
        response = requests.post(
            LM_STUDIO_URL,
            json={
                "model": "qwen/qwen3-4b-thinking-2507",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert insurance retention strategist. Provide clear, actionable insights in a structured format."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 6400
            },
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            analysis = result['choices'][0]['message']['content']
            
            # Parse the response (simple extraction)
            return {
                'raw_analysis': analysis,
                'customer_index': customer_index,
                'churn_probability': churn_prob,
                'top_drivers': top_3_features
            }
        else:
            print(f"   ⚠️  LLM request failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"   ⚠️  LLM generation error: {str(e)}")
        return None

# Generate insights for high-risk customers
print("\n🤖 Generating LLM insights for high-risk customers...")

llm_insights = []

# Select top 10 high-risk customers for LLM analysis
high_risk_customers = individual_explanations_df.nlargest(10, 'Churn_Probability')

for idx, row in high_risk_customers.iterrows():
    print(f"   Analyzing customer {row['Customer_Index']}... ", end='')
    
    insights = generate_llm_insights(
        customer_index=row['Customer_Index'],
        shap_data=row,
        global_context=global_importance_df
    )
    
    if insights:
        llm_insights.append(insights)
        print("✓")
    else:
        print("⚠️ Failed")

# Save LLM insights
if llm_insights:
    llm_insights_df = pd.DataFrame(llm_insights)
    llm_insights_df.to_csv('llm_retention_insights.csv', index=False)
    print(f"\nSaved: llm_retention_insights.csv ({len(llm_insights)} insights)")
    
    # Display sample
    print("\n📝 Sample LLM-Generated Insight:")
    print("=" * 80)
    if len(llm_insights) > 0:
        sample = llm_insights[0]
        print(f"Customer: {sample['customer_index']}")
        print(f"Churn Risk: {sample['churn_probability']:.1%}")
        print(f"\nAnalysis:\n{sample['raw_analysis']}")
else:
    print("\n⚠️  No LLM insights generated (LM Studio may not be running)")

# ============================================================================
# 7. CREATE SHAP SUMMARY FOR DASHBOARD
# ============================================================================

print("\n" + "=" * 80)
print("CREATING DASHBOARD-READY SUMMARIES")
print("=" * 80)

# Save complete SHAP data for dashboard
dashboard_data = {
    'global_importance': global_importance_df,
    'individual_explanations': individual_explanations_df,
    'sample_indices': X_shap_sample.index.tolist(),
    'feature_names': feature_names,
    'expected_value': float(explainer.expected_value)
}

joblib.dump(dashboard_data, 'shap_dashboard_data.pkl')
print("Saved: shap_dashboard_data.pkl")

# ============================================================================
# 8. SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("SHAP + LLM ANALYSIS COMPLETE!")
print("=" * 80)

print("\n📊 Analysis Summary:")
print(f"  - Customers analyzed: {len(X_shap_sample):,}")
print(f"  - Features: {len(feature_names)}")
print(f"  - Top churn driver: {global_importance_df.iloc[0]['Feature']}")
print(f"  - LLM insights generated: {len(llm_insights)}")

print("\n📁 Files created:")
print("  1. shap_explainer.pkl")
print("  2. shap_values.npy")
print("  3. shap_sample_indices.npy")
print("  4. shap_summary_plot.png")
print("  5. shap_feature_importance_bar.png")
print("  6. shap_global_importance.csv")
print("  7. shap_individual_explanations.csv")
print("  8. shap_force_plot_customer_*.png (multiple files)")
print("  9. llm_retention_insights.csv")
print(" 10. shap_dashboard_data.pkl")

print("\n✅ Ready to build the Streamlit dashboard!")
print("\n💡 Dashboard will show:")
print("   - Global SHAP summary (for strategists)")
print("   - Individual force plots (for retention agents)")
print("   - LLM-powered retention recommendations")