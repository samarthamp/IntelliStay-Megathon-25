"""
AI-Powered Churn Retention Dashboard
Combines SHAP explainability, DiCE counterfactuals, and LLM insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import requests
from io import BytesIO
import base64
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Churn Retention Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 600;
        color: #0f766e;
        text-align: center;
        padding: 1.5rem 0;
        border-bottom: 3px solid #14b8a6;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #0f766e;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    .metric-card {
        background-color: #f8fafc;
        padding: 1.2rem;
        border-radius: 0.5rem;
        border-left: 4px solid #14b8a6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .high-risk {
        color: #dc2626;
        font-weight: 600;
    }
    .medium-risk {
        color: #ea580c;
        font-weight: 600;
    }
    .low-risk {
        color: #059669;
        font-weight: 600;
    }
    .stButton>button {
        width: 100%;
        background-color: #0d9488;
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 0.5rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #0f766e;
        box-shadow: 0 4px 6px rgba(0,0,0,0.15);
    }
    .insight-box {
        background-color: #f0fdfa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #14b8a6;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 0.25rem;
        font-weight: 600;
        font-size: 0.875rem;
    }
    .status-urgent {
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #fca5a5;
    }
    .status-warning {
        background-color: #ffedd5;
        color: #9a3412;
        border: 1px solid #fdba74;
    }
    .status-normal {
        background-color: #d1fae5;
        color: #065f46;
        border: 1px solid #6ee7b7;
    }
    
    /* 3D SIDEBAR STYLING */
    section[data-testid="stSidebar"] {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        box-shadow: 
            inset -10px 0 20px rgba(15, 118, 110, 0.05),
            5px 0 20px rgba(0, 0, 0, 0.1);
    }
    
    section[data-testid="stSidebar"] > div {
        background: transparent;
        padding: 2rem 1rem;
    }
    
    /* SIDEBAR WIDTH */
    section[data-testid="stSidebar"] {
        width: 320px !important;  /* Default is 244px */
        min-width: 320px !important;
    }
    
    section[data-testid="stSidebar"] > div {
        width: 320px !important;
        min-width: 320px !important;
    }
    
    /* Adjust main content to compensate */
    .main .block-container {
        max-width: calc(100% - 320px);
    }
    
    /* Navigation Title */
    section[data-testid="stSidebar"] h1 {
        background: linear-gradient(135deg, #0f766e 0%, #14b8a6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        text-shadow: 0 2px 10px rgba(15, 118, 110, 0.2);
    }
    
    /* Radio buttons - 3D cards */
    .stRadio > div {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 
            0 4px 6px rgba(0, 0, 0, 0.07),
            0 1px 3px rgba(0, 0, 0, 0.06),
            inset 0 1px 0 rgba(255, 255, 255, 0.8);
        border: 1px solid rgba(226, 232, 240, 0.8);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .stRadio > div:hover {
        transform: translateY(-2px);
        box-shadow: 
            0 10px 20px rgba(15, 118, 110, 0.15),
            0 3px 6px rgba(0, 0, 0, 0.08),
            inset 0 1px 0 rgba(255, 255, 255, 1);
    }
    
    /* Radio button options */
    .stRadio label {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 2px solid transparent;
        transition: all 0.3s ease;
        cursor: pointer;
        box-shadow: 
            0 2px 4px rgba(0, 0, 0, 0.05),
            inset 0 1px 0 rgba(255, 255, 255, 0.9);
    }
    
    .stRadio label:hover {
        border-color: #14b8a6;
        background: linear-gradient(135deg, #f0fdfa 0%, #ccfbf1 100%);
        transform: translateX(4px);
        box-shadow: 
            0 4px 8px rgba(20, 184, 166, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 1);
    }
    
    /* Selected radio button */
    .stRadio input:checked + label {
        background: linear-gradient(135deg, #14b8a6 0%, #0d9488 100%);
        color: white;
        font-weight: 600;
        border-color: #0f766e;
        box-shadow: 
            0 6px 12px rgba(20, 184, 166, 0.3),
            inset 0 1px 3px rgba(255, 255, 255, 0.3),
            inset 0 -1px 3px rgba(0, 0, 0, 0.2);
        transform: translateX(6px);
    }
            
    .stRadio label > div {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
    
    /* Quick Stats Section - 3D Cards */
    section[data-testid="stSidebar"] .element-container {
        background: white;
        padding: 1.25rem;
        margin: 0.75rem 0;
        border-radius: 12px;
        box-shadow: 
            0 4px 6px rgba(0, 0, 0, 0.07),
            0 1px 3px rgba(0, 0, 0, 0.06),
            inset 0 1px 0 rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(226, 232, 240, 0.6);
        position: relative;
        overflow: hidden;
    }
    
    section[data-testid="stSidebar"] .element-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(180deg, #14b8a6 0%, #0d9488 100%);
        box-shadow: 0 0 10px rgba(20, 184, 166, 0.5);
    }
    
    section[data-testid="stSidebar"] .element-container:hover {
        transform: translateX(4px);
        box-shadow: 
            0 8px 16px rgba(15, 118, 110, 0.15),
            0 2px 6px rgba(0, 0, 0, 0.08),
            inset 0 1px 0 rgba(255, 255, 255, 1);
        transition: all 0.3s ease;
    }
    
    /* Metrics in sidebar */
    section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
        font-size: 1.75rem;
        font-weight: 700;
        background: linear-gradient(135deg, #0f766e 0%, #14b8a6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 2px 4px rgba(15, 118, 110, 0.1);
    }
    
    section[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        color: #64748b;
        font-weight: 600;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    section[data-testid="stSidebar"] [data-testid="stMetricDelta"] {
        font-weight: 600;
    }
    
    /* Divider */
    section[data-testid="stSidebar"] hr {
        margin: 1.5rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, #14b8a6 50%, transparent 100%);
        box-shadow: 0 1px 2px rgba(20, 184, 166, 0.3);
    }
    
    /* Quick Stats Title */
    section[data-testid="stSidebar"] h3 {
        color: #0f766e;
        font-weight: 700;
        font-size: 1.1rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        position: relative;
        padding-left: 1rem;
    }
    
    section[data-testid="stSidebar"] h3::before {
        content: '';
        position: absolute;
        left: 0;
        top: 50%;
        transform: translateY(-50%);
        width: 4px;
        height: 80%;
        background: linear-gradient(180deg, #14b8a6 0%, #0d9488 100%);
        border-radius: 2px;
        box-shadow: 0 0 8px rgba(20, 184, 166, 0.4);
    }
    
    /* Glassmorphism overlay effect */
    section[data-testid="stSidebar"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 200px;
        background: linear-gradient(180deg, 
            rgba(240, 253, 250, 0.8) 0%, 
            rgba(240, 253, 250, 0) 100%);
        pointer-events: none;
        z-index: 1;
    }
    
    section[data-testid="stSidebar"] > div {
        position: relative;
        z-index: 2;
    }
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 600;
        color: #0f766e;
    }
    
    /* Info box styling */
    .stAlert {
        background-color: #ecfdf5;
        border-left: 4px solid #14b8a6;
    }
    /* REFINED IMAGE STYLING */
img {
    border: 1px solid #cbd5e1;
    border-radius: 10px;
    padding: 1.25rem;
    background: #ffffff;
    box-shadow: 
        0 2px 8px rgba(0, 0, 0, 0.04),
        0 1px 2px rgba(0, 0, 0, 0.03);
    position: relative;
    transition: all 0.3s ease;
}

img::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 4px;
    background: linear-gradient(90deg, #14b8a6 0%, #0d9488 100%);
    border-radius: 10px 10px 0 0;
}

img:hover {
    transform: translateY(-3px);
    box-shadow: 
        0 8px 16px rgba(15, 118, 110, 0.12),
        0 2px 4px rgba(0, 0, 0, 0.06);
    border-color: #14b8a6;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD ALL MODELS AND DATA
# ============================================================================

@st.cache_resource
def load_models_and_data():
    """Load all required models and data"""
    try:
        # Core model
        model = joblib.load('final_xgboost_model.pkl')
        model.set_params(device='cpu')
        print("Model set to CPU mode")
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        
        # SHAP data
        shap_data_package = joblib.load('shap_data_package.pkl')
        shap_explainer = joblib.load('shap_explainer.pkl')
        global_importance = pd.read_csv('shap_global_importance.csv')
        
        # Test data
        test_data = pd.read_csv('test_data_with_predictions.csv')
        
        # DiCE (if available)
        try:
            dice_package = joblib.load('dice_explainer.pkl')
            dice_available = True
        except:
            dice_package = None
            dice_available = False
        
        return {
            'model': model,
            'scaler': scaler,
            'feature_names': feature_names,
            'shap_data': shap_data_package,
            'shap_explainer': shap_explainer,
            'global_importance': global_importance,
            'test_data': test_data,
            'dice_package': dice_package,
            'dice_available': dice_available
        }
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

# Load everything
with st.spinner("🔄 Loading AI models and data..."):
    data = load_models_and_data()

if data is None:
    st.stop()

# Extract for convenience
model = data['model']
scaler = data['scaler']
feature_names = data['feature_names']
shap_data = data['shap_data']
shap_explainer = data['shap_explainer']
global_importance = data['global_importance']
test_data = data['test_data']
dice_package = data['dice_package']
dice_available = data['dice_available']

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def predict_safely(model, scaler, feature_data):
    """
    Stable CPU prediction for dashboard.
    'feature_data' should be a 1D or 2D NumPy array.
    """
    if feature_data.ndim == 1:
        feature_data = feature_data.reshape(1, -1)
    
    X_scaled = scaler.transform(feature_data)
    probability = model.predict_proba(X_scaled)[0][1]
    
    return float(probability)

def generate_waterfall_plot(customer_idx):
    """Generate SHAP waterfall plot for a customer"""
    try:
        position = shap_data['customer_indices'].index(customer_idx)
        
        shap_values = shap_data['shap_values'][position]
        expected_value = shap_data['expected_value']
        customer_data = shap_data['X_data'].iloc[position].values
        
        explanation = shap.Explanation(
            values=shap_values,
            base_values=expected_value,
            data=customer_data,
            feature_names=feature_names
        )
        
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.waterfall_plot(explanation, max_display=10, show=False)
        plt.title(f'SHAP Analysis - Customer {customer_idx}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    except:
        return None

def get_customer_shap_features(customer_idx):
    """Get top SHAP features for a customer"""
    try:
        position = shap_data['customer_indices'].index(customer_idx)
        
        shap_values = shap_data['shap_values'][position]
        customer_data = shap_data['X_data'].iloc[position]
        churn_prob = shap_data['y_proba'].iloc[position]
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'shap_value': shap_values,
            'value': customer_data.values,
            'abs_shap': np.abs(shap_values)
        }).sort_values('abs_shap', ascending=False)
        
        top_features = []
        for idx, row in feature_importance.head(5).iterrows():
            top_features.append({
                'feature': row['feature'],
                'shap_value': row['shap_value'],
                'value': row['value'],
                'churn_prob': churn_prob
            })
        
        return top_features
    except:
        return None

def generate_dice_counterfactuals(customer_idx):
    """Generate DiCE counterfactuals for a customer"""
    if not dice_available or dice_package is None:
        return None
    
    try:
        dice_explainer = dice_package['dice_explainer']
        actionable_features = dice_package['actionable_features']
        wrapped_model = dice_package['wrapped_model']
        
        # Get customer data from the dice_data
        dice_data = dice_package['dice_data']
        
        if customer_idx not in dice_data.index:
            return None
        
        customer_data = dice_data.loc[customer_idx:customer_idx].drop('Churn', axis=1, errors='ignore')
        
        # Generate counterfactuals
        dice_cf = dice_explainer.generate_counterfactuals(
            query_instances=customer_data,
            total_CFs=3,
            desired_class=0,
            features_to_vary=actionable_features,
            permitted_range={
                'curr_ann_amt': [
                    customer_data['curr_ann_amt'].values[0] * 0.5,
                    customer_data['curr_ann_amt'].values[0]
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
            }
        )
        
        cf_examples = dice_cf.cf_examples_list[0]
        if cf_examples.final_cfs_df is None or len(cf_examples.final_cfs_df) == 0:
            return None
        
        return cf_examples.final_cfs_df
        
    except Exception as e:
        st.warning(f"⚠️ DiCE unavailable: {str(e)}")
        return None

def get_unscaled_customer_data(customer_idx):
    """Get original unscaled data for a customer"""
    try:
        # Try to load from dice_data (unscaled)
        if dice_available and dice_package is not None:
            dice_data = dice_package['dice_data']
            if customer_idx in dice_data.index:
                return dice_data.loc[customer_idx].drop('Churn', errors='ignore')
        
        # Fallback: inverse transform scaled data
        position = shap_data['customer_indices'].index(customer_idx)
        scaled_data = shap_data['X_data'].iloc[position]
        
        # Inverse transform
        unscaled = scaler.inverse_transform(scaled_data.values.reshape(1, -1))
        unscaled_series = pd.Series(unscaled[0], index=feature_names)
        
        return unscaled_series
    except:
        return None

# ============================================================================
# LLM HELPER FUNCTIONS
# ============================================================================

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"

def check_llm_available():
    """Check if LM Studio is running"""
    try:
        response = requests.get("http://localhost:1234/v1/models", timeout=2)
        return response.status_code == 200
    except:
        return False

def generate_retention_insights(customer_idx, shap_features, global_context, dice_results=None):
    """Generate LLM-powered retention insights with behavioral nudges"""

    # Format SHAP features
    top_drivers_text = []
    for i, feat in enumerate(shap_features[:3], 1):
        direction = "increasing" if feat['shap_value'] > 0 else "decreasing"
        top_drivers_text.append(
            f"{i}. **{feat['feature']}** (value: {feat['value']:.2f}) - "
            f"{direction} churn risk by {abs(feat['shap_value']):.3f}"
        )

    # Add DiCE recommendations if available
    dice_context = ""
    if dice_results:
        dice_context = f"\n\nRECOMMENDED ACTIONS (from DiCE analysis):\n{dice_results}"

    # Determine primary churn driver category for nudge selection
    primary_driver = shap_features[0]['feature']

    prompt = f"""You are an expert auto insurance retention strategist with expertise in behavioral economics.

CUSTOMER RISK PROFILE:
- Churn Probability: {shap_features[0]['churn_prob']:.1%}
- Risk Level: {"HIGH" if shap_features[0]['churn_prob'] > 0.7 else "MODERATE" if shap_features[0]['churn_prob'] > 0.4 else "LOW"}

TOP 3 CHURN DRIVERS:
{chr(10).join(top_drivers_text)}

INDUSTRY CONTEXT:
Key churn factors across all customers: {', '.join(global_context.head(5)['Feature'].tolist())}
{dice_context}

PRIMARY CHURN DRIVER: {primary_driver}

Provide a clear, actionable analysis with behavioral science principles:

**CUSTOMER PERSONA:** (1 sentence describing this customer)

**WHY THEY MIGHT LEAVE:** (2-3 sentences explaining the risk)

**RETENTION ACTIONS:**
- [Immediate action 1 - financial/operational]
- [Immediate action 2 - service/relationship]
- [Immediate action 3 - value-add/benefit]

**BEHAVIORAL NUDGE STRATEGY:**
Based on the primary churn driver, recommend ONE behavioral economics nudge from these categories:

1. **Loss Aversion**: Emphasize what customer loses by leaving (e.g., "You'll lose your 3-year accident-free bonus worth $XXX")
2. **Commitment Device**: Lock in benefits (e.g., "Lock in your safe driver score for next year at current rate")
3. **Social Proof**: Show peer behavior (e.g., "87% of customers in your area renewed this plan")
4. **Reciprocity**: Offer unexpected value (e.g., "Complimentary vehicle health check-up as a thank you")
5. **Scarcity**: Limited-time opportunity (e.g., "This renewal rate expires in 48 hours")
6. **Anchoring**: Frame savings positively (e.g., "You've saved $XXX with us over 3 years")

**NUDGE RECOMMENDATION:**
[Select the most appropriate nudge based on churn driver and provide a ready-to-send message template]

**MESSAGE TEMPLATE:**
[Write a 2-3 sentence message the agent can send directly to the customer]

Keep it concise, non-technical, and action-oriented."""

    if not check_llm_available():
        # Enhanced fallback with nudge
        nudge_map = {
            'days_tenure': ('Loss Aversion', 'Remind customer of their loyalty history and benefits earned'),
            'curr_ann_amt': ('Anchoring', 'Frame total savings over their customer lifetime'),
            'premium_to_income_ratio': ('Reciprocity', 'Offer complimentary policy review'),
            'age_in_years': ('Social Proof', 'Show renewal rates for similar age group'),
            'income': ('Commitment Device', 'Lock in rate for multi-year term')
        }

        nudge_category, nudge_action = nudge_map.get(
            primary_driver,
            ('Reciprocity', 'Offer value-added service')
        )

        return f"""**CUSTOMER PERSONA:** Customer with {shap_features[0]['churn_prob']:.1%} churn risk

**WHY THEY MIGHT LEAVE:** Key risk factors include {shap_features[0]['feature']}, {shap_features[1]['feature']}, and {shap_features[2]['feature']}.

**RETENTION ACTIONS:**
- Review and optimize premium structure
- Proactive outreach to address concerns
- Offer loyalty incentives

**BEHAVIORAL NUDGE STRATEGY:**
Category: {nudge_category}

**NUDGE RECOMMENDATION:**
{nudge_action}

**MESSAGE TEMPLATE:**
"Hi [Customer Name], we noticed you've been with us for [tenure]. As a valued customer, we'd like to discuss how we can better serve you. Add nudge message here pls. Can we schedule a quick call this week?"

*Note: LM Studio unavailable - using fallback analysis with behavioral nudges*"""

    try:
        response = requests.post(
            LM_STUDIO_URL,
            json={
                "messages": [
                    {"role": "system", "content": "You are an expert insurance retention strategist with deep knowledge of behavioral economics and customer psychology."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 800
            },
            timeout=120
        )

        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return "Error: Could not generate insights"
    except:
        return "Error: LLM request failed"

    try:
        response = requests.post(
            LM_STUDIO_URL,
            json={
                "messages": [
                    {"role": "system", "content": "You are an expert insurance retention strategist."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 6400
            },
            timeout=120
        )

        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return "Error: Could not generate insights"
    except:
        return "Error: LLM request failed"

# ============================================================================
# SIDEBAR - NAVIGATION
# ============================================================================

st.sidebar.title("🛡️ Navigation")
page = st.sidebar.radio(
    "Select View:",
    ["🌐 Churn Dashboard", "👤 Customer Deep Dive", "🎮 Retention Simulator"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Quick Stats")
total_customers = len(test_data)
high_risk = len(test_data[test_data['Churn_Probability'] > 0.7])
medium_risk = len(test_data[(test_data['Churn_Probability'] > 0.4) & (test_data['Churn_Probability'] <= 0.7)])
low_risk = len(test_data[test_data['Churn_Probability'] <= 0.4])

st.sidebar.metric("Total Customers", f"{total_customers:,}")
st.sidebar.metric("High Risk", f"{high_risk:,}", delta=f"{high_risk/total_customers:.1%}")
st.sidebar.metric("Medium Risk", f"{medium_risk:,}")
st.sidebar.metric("Low Risk", f"{low_risk:,}")

# ============================================================================
# PAGE 1: GLOBAL DASHBOARD
# ============================================================================

if page == "🌐 Churn Dashboard":
    st.markdown('<h1 class="main-header">IntelliStay Churn Analysis Dashboard </h1>', unsafe_allow_html=True)
    
    st.markdown("### 📈 Portfolio Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_churn = test_data['Churn_Probability'].mean()
        st.metric(
            "Average Churn Risk",
            f"{avg_churn:.1%}",
            delta=f"{(avg_churn - 0.115):.1%} vs baseline"
        )
    
    with col2:
        st.metric(
            "High Risk Customers",
            f"{high_risk:,}",
            delta=f"-{high_risk}" if high_risk < 1000 else f"+{high_risk}",
            delta_color="inverse"
        )
    
    with col3:
        at_risk_value = high_risk * 950  # Avg premium
        st.metric(
            "Revenue at Risk",
            f"${at_risk_value:,.0f}",
            delta="Annual"
        )
    
    with col4:
        llm_status = "🟢 Online" if check_llm_available() else "🔴 Offline"
        st.metric(
            "AI Strategist",
            llm_status,
            delta="LM Studio" if check_llm_available() else "Fallback"
        )
    
    st.markdown("---")
    
    # Global SHAP Summary
    st.markdown("### Top Churn Drivers (Global Analysis)")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Feature Importance")
        st.image('results/shap_feature_importance_bar.png', width='stretch')
    
    with col2:
        st.markdown("#### Impact Direction")
        st.image('results/shap_summary_plot.png', width='stretch')

    st.markdown("---")
    
    # Top drivers table
    st.markdown("### 📊 Strategic Insights")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### Top 10 Churn Drivers")
        top_10 = global_importance.head(10)
        st.dataframe(
            top_10.style.background_gradient(cmap='Reds', subset=['Mean_Abs_SHAP']),
            hide_index=True,
            use_container_width=True
        )
    
    with col2:
        st.markdown("#### Risk Distribution")
        risk_dist = pd.DataFrame({
            'Risk Level': ['Low (0-40%)', 'Medium (40-70%)', 'High (70-100%)'],
            'Count': [low_risk, medium_risk, high_risk],
            'Percentage': [
                f"{low_risk/total_customers:.1%}",
                f"{medium_risk/total_customers:.1%}",
                f"{high_risk/total_customers:.1%}"
            ]
        })
        st.dataframe(risk_dist, hide_index=True, use_container_width=True)
    
    st.markdown("---")
    
    # Vulnerable customers list
    st.markdown("### 🚨 Immediate Action Required")
    
    vulnerable = test_data.nlargest(20, 'Churn_Probability')[
        ['Churn_Probability', 'Churn_Predicted']
    ].reset_index()
    vulnerable.columns = ['Customer ID', 'Churn Risk', 'Predicted Churn']
    vulnerable['Churn Risk'] = vulnerable['Churn Risk'].apply(lambda x: f"{x:.1%}")
    vulnerable['Status'] = '🔴 URGENT'
    
    st.dataframe(
        vulnerable.style.apply(
            lambda x: ['background-color: #ffcccc' if i % 2 == 0 else 'background-color: #ffe6e6' 
                         for i in range(len(x))],
            axis=0
        ),
        hide_index=True,
        use_container_width=True
    )

# ============================================================================
# PAGE 2: CUSTOMER DEEP DIVE
# ============================================================================

elif page == "👤 Customer Deep Dive":
    st.markdown('<h1 class="main-header">👤 Customer Deep Dive Analysis</h1>', unsafe_allow_html=True)
    
    # Search interface
    st.markdown("### Customer Search")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Get available customer IDs
        available_customers = shap_data['customer_indices']
        
        # Search options
        search_option = st.selectbox(
            "Search by:",
            ["Customer ID", "Risk Level"]
        )
        
        if search_option == "Customer ID":
            customer_idx = st.selectbox(
                "Select Customer ID:",
                options=available_customers,
                index=0
            )
        else:
            risk_level = st.select_slider(
                "Select Risk Level:",
                options=["Low", "Medium", "High"]
            )
            
            if risk_level == "High":
                filtered = test_data[test_data['Churn_Probability'] > 0.7]
            elif risk_level == "Medium":
                filtered = test_data[(test_data['Churn_Probability'] > 0.4) & 
                                     (test_data['Churn_Probability'] <= 0.7)]
            else:
                filtered = test_data[test_data['Churn_Probability'] <= 0.4]
            
            available_filtered = [idx for idx in filtered.index if idx in available_customers]
            
            if available_filtered:
                customer_idx = st.selectbox(
                    "Select Customer:",
                    options=available_filtered,
                    index=0
                )
            else:
                st.warning("No customers found in this risk category with SHAP data")
                st.stop()
    
    with col2:
        st.metric("Customer ID", customer_idx)
    
    with col3:
        churn_prob = test_data.loc[customer_idx, 'Churn_Probability']
        risk_label = "HIGH" if churn_prob > 0.7 else "MEDIUM" if churn_prob > 0.4 else "LOW"
        risk_color = "high-risk" if churn_prob > 0.7 else "low-risk"
        st.markdown(f'<p class="{risk_color}">Risk: {risk_label}</p>', unsafe_allow_html=True)
        st.metric("Churn Probability", f"{churn_prob:.1%}")
    
    st.markdown("---")
    
    # Analyze button
    analyze_button = st.button("🔬 ANALYZE CUSTOMER", type="primary", width='stretch')
    
    if analyze_button or 'last_analyzed' in st.session_state and st.session_state.last_analyzed == customer_idx:
        st.session_state.last_analyzed = customer_idx
        
        with st.spinner("🤖 Running AI analysis..."):
            # Get SHAP features
            shap_features = get_customer_shap_features(customer_idx)
            
            # Generate DiCE counterfactuals
            dice_cf = generate_dice_counterfactuals(customer_idx) if dice_available else None
            
            # Format DiCE for LLM
            dice_summary = ""
            if dice_cf is not None and len(dice_cf) > 0:
                dice_summary = "Reduce annual premium by 20-30%"
            
            # Generate LLM insights
            llm_insights = generate_retention_insights(
                customer_idx,
                shap_features,
                global_importance,
                dice_summary
            )
        
        st.success("✅ Analysis complete!")
        
        # Display results
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="section-header">Risk Factor Analysis (SHAP)</div>', unsafe_allow_html=True)
            st.markdown("*Why is this customer at risk?*")
            
            waterfall_fig = generate_waterfall_plot(customer_idx)
            if waterfall_fig:
                st.pyplot(waterfall_fig)
            else:
                st.warning("Could not generate SHAP plot")
            
            # Top drivers
            st.markdown("#### Top Risk Factors")
            if shap_features:
                for i, feat in enumerate(shap_features[:5], 1):
                    direction = "🔴" if feat['shap_value'] > 0 else "🔵"
                    st.markdown(
                        f"{direction} **{feat['feature']}**: {feat['value']:.2f} "
                        f"(Impact: {feat['shap_value']:+.3f})"
                    )
        
        with col2:
            st.markdown('<div class="section-header">AI-Powered Strategic Recommendations</div>', unsafe_allow_html=True)
            st.markdown("*Powered by Local LLM*")
            
            st.markdown(f'<div class="insight-box">{llm_insights}</div>', unsafe_allow_html=True)
    
        st.markdown("---")
        
        # DiCE Counterfactuals
        if dice_available and dice_cf is not None:
            st.markdown('<div class="section-header">Counterfactual Retention Strategies</div>', unsafe_allow_html=True)
            st.markdown("*Minimum changes needed to reduce churn risk*")
            
            # Get original values
            position = shap_data['customer_indices'].index(customer_idx)
            original_data = shap_data['X_data'].iloc[position]
            
            # Display options
            for cf_idx in range(min(3, len(dice_cf))):
                cf_row = dice_cf.iloc[cf_idx]
                
                # Calculate new probability using the safe GPU function
                cf_features = cf_row[feature_names].values
                cf_prob = predict_safely(model, scaler, cf_features)
                
                with st.expander(f"Option {cf_idx + 1}: Reduce risk to {cf_prob:.1%}", expanded=(cf_idx==0)):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "New Churn Risk",
                            f"{cf_prob:.1%}",
                            delta=f"{(cf_prob - churn_prob):.1%}",
                            delta_color="inverse"
                        )
                    
                    with col2:
                        risk_reduction = churn_prob - cf_prob
                        st.metric(
                            "Risk Reduction",
                            f"{risk_reduction:.1%}",
                            delta="Improvement"
                        )
                    
                    with col3:
                        if cf_prob < 0.4:
                            new_status = "✅ LOW RISK"
                        elif cf_prob < 0.7:
                            new_status = "⚠️ MEDIUM RISK"
                        else:
                            new_status = "🔴 HIGH RISK"
                        st.metric("New Status", new_status)
                    
                    st.markdown("#### Required Actions:")
                    
                    # Show changes
                    actionable = dice_package['actionable_features']
                    for feat in actionable:
                        orig_val = original_data[feat]
                        new_val = cf_row[feat]
                        
                        if abs(new_val - orig_val) / (abs(orig_val) + 1e-10) > 0.001:
                            pct_change = ((new_val - orig_val) / (abs(orig_val) + 1e-10)) * 100
                            
                            st.markdown(
                                f"💰 **{feat}**: ${orig_val:.2f} → ${new_val:.2f} "
                                f"*({pct_change:+.1f}%)*"
                            )

# ============================================================================
# PAGE 3: RETENTION SIMULATOR
# ============================================================================

elif page == "🎮 Retention Simulator":
    st.markdown('<h1 class="main-header">🎮 Interactive Retention Simulator</h1>', unsafe_allow_html=True)
    
    st.markdown("### 🔧 Select Customer & Adjust Parameters")
    
    # Customer selection
    available_customers = shap_data['customer_indices']
    
    # Filter to only show customers in dice_data if DiCE available
    if dice_available and dice_package is not None:
        dice_data = dice_package['dice_data']
        available_customers = [idx for idx in available_customers if idx in dice_data.index]
    
    if len(available_customers) == 0:
        st.error("❌ No customers available for simulation")
        st.stop()
    
    customer_idx = st.selectbox(
        "Select Customer ID:",
        options=available_customers,
        index=0,
        key='sim_customer_select'
    )
    
    # Get original data (UNSCALED)
    try:
        original_data = get_unscaled_customer_data(customer_idx)
        
        if original_data is None:
            st.error("❌ Could not load customer data")
            st.stop()
        
        # Get probability from test_data
        original_prob = test_data.loc[customer_idx, 'Churn_Probability']
        
    except Exception as e:
        st.error(f"Error loading customer data: {str(e)}")
        st.stop()
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="section-header">Premium Adjustment Parameters</div>', unsafe_allow_html=True)
        
        st.info("💡 Use sliders to simulate discount scenarios")
        
        # Get original values safely
        orig_annual = float(original_data['curr_ann_amt'])
        orig_monthly = float(original_data['monthly_premium'])
        orig_income = float(original_data.get('income', 50000))
        
        # Sliders for actionable features
        curr_ann_amt = st.slider(
            "Annual Premium ($)",
            min_value=orig_annual * 0.5,
            max_value=orig_annual * 1.0,  # Can only decrease (discount)
            value=orig_annual,
            step=10.0,
            help="Adjust the annual premium amount (max 50% discount)",
            key='slider_annual'
        )
        
        monthly_premium =curr_ann_amt / 12.0
        # Calculate derived values
        premium_to_income = curr_ann_amt / (orig_income + 1)
        premium_affordability = premium_to_income * 100
        
        st.markdown("---")
        st.markdown("#### 📊 Calculated Metrics:")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric(
                "Premium/Income Ratio", 
                f"{premium_to_income:.4f}",
                delta=f"{(premium_to_income - original_data['premium_to_income_ratio']):.4f}"
            )
        with col_b:
            st.metric(
                "Affordability %", 
                f"{premium_affordability:.2f}%",
                delta=f"{(premium_affordability - original_data['premium_affordability']):.2f}%"
            )
        
        # Simulate button
        simulate_clicked = st.button(
            "🔮 SIMULATE IMPACT", 
            type="primary", 
            width='stretch',
            key='simulate_button'
        )
        
        if simulate_clicked:
            with st.spinner("🔄 Running simulation..."):
                try:
                    # Create modified customer data
                    modified_data = original_data.copy()
                    modified_data['curr_ann_amt'] = curr_ann_amt
                    modified_data['monthly_premium'] = monthly_premium
                    modified_data['premium_to_income_ratio'] = premium_to_income
                    modified_data['premium_affordability'] = premium_affordability
                    
                    # Use the safe GPU prediction function
                    X_modified = modified_data[feature_names].values
                    new_prob = predict_safely(model, scaler, X_modified)
                    
                    # Save to session state
                    st.session_state.sim_result = {
                        'original_prob': float(original_prob),
                        'new_prob': float(new_prob),
                        'curr_ann_amt': curr_ann_amt,
                        'monthly_premium': monthly_premium,
                        'orig_annual': orig_annual,
                        'orig_monthly': orig_monthly
                    }
                    
                    st.success("✅ Simulation complete!")
                    
                except Exception as e:
                    st.error(f"❌ Simulation failed: {str(e)}")
    
    with col2:
        st.markdown('<div class="section-header">Simulation Results</div>', unsafe_allow_html=True)
        
        if 'sim_result' in st.session_state:
            result = st.session_state.sim_result
            
            # Metrics
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric(
                    "Original Risk",
                    f"{result['original_prob']:.1%}"
                )
            
            with col_b:
                st.metric(
                    "New Risk",
                    f"{result['new_prob']:.1%}",
                    delta=f"{(result['new_prob'] - result['original_prob']):.1%}",
                    delta_color="inverse"
                )
            
            with col_c:
                reduction = result['original_prob'] - result['new_prob']
                st.metric(
                    "Risk Reduction",
                    f"{reduction:.1%}"
                )
            
            # Visual comparison
            st.markdown("#### 📈 Risk Comparison")
            
            fig, ax = plt.subplots(figsize=(10, 4))
            
            categories = ['Original', 'Simulated']
            values = [result['original_prob'] * 100, result['new_prob'] * 100]
            colors = ['#d62728' if v > 70 else '#ff7f0e' if v > 40 else '#2ca02c' for v in values]
            
            bars = ax.barh(categories, values, color=colors, alpha=0.7)
            ax.set_xlabel('Churn Probability (%)', fontweight='bold')
            ax.set_title('Churn Risk: Before & After', fontsize=14, fontweight='bold')
            ax.axvline(x=70, color='red', linestyle='--', alpha=0.3, linewidth=2, label='High Risk')
            ax.axvline(x=40, color='orange', linestyle='--', alpha=0.3, linewidth=2, label='Medium Risk')
            ax.legend(loc='lower right')
            ax.set_xlim([0, 100])
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, values)):
                ax.text(val + 2, i, f'{val:.1f}%', va='center', fontweight='bold', fontsize=12)
            
            st.pyplot(fig)
            plt.close()
            
            st.markdown("---")
            
            # Recommendations
            if result['new_prob'] < result['original_prob']:
                st.success("✅ **Positive Impact!** These changes reduce churn risk.")
                
                discount_annual = result['orig_annual'] - result['curr_ann_amt']
                discount_pct = (discount_annual / (result['orig_annual'] + 1e-10)) * 100
                
                st.markdown("#### 💡 Recommended Action Plan:")
                st.markdown(f"""
                **Premium Adjustment:**
                - Original annual premium: **${result['orig_annual']:.2f}**
                - New annual premium: **${result['curr_ann_amt']:.2f}**
                - Discount offered: **${discount_annual:.2f}** ({discount_pct:.1f}%)
                
                **Expected Impact:**
                - Churn risk reduction: **{(result['original_prob'] - result['new_prob']):.1%}**
                - New monthly payment: **${result['monthly_premium']:.2f}**
                
                **Next Steps:**
                1. 📞 Contact customer within 24-48 hours
                2. 💬 Present personalized retention offer
                3. 📋 Document interaction in CRM
                4. 📊 Schedule follow-up in 30 days
                """)
                
                # Calculate ROI
                annual_revenue = result['orig_annual']
                retention_cost = discount_annual
                expected_value = annual_revenue * (1 - result['new_prob'])
                
                st.markdown("#### 💰 Business Impact:")
                col_x, col_y, col_z = st.columns(3)
                with col_x:
                    st.metric("Annual Revenue", f"${annual_revenue:.2f}")
                with col_y:
                    st.metric("Retention Cost", f"${retention_cost:.2f}")
                with col_z:
                    st.metric("Expected Value", f"${expected_value:.2f}")
                
            else:
                st.warning("⚠️ **Limited impact detected.**")
                st.markdown("""
                **Current premium adjustments may not be sufficient.**
                
                **Consider alternative strategies:**
                - 🎁 Non-monetary incentives (better coverage, loyalty perks)
                - 🤝 Personal account review with agent
                - 📞 Proactive customer service outreach
                - 💳 Payment plan flexibility
                - 🏆 Loyalty program enrollment
                """)
        
        else:
            st.info("👆 **How to use:**\n\n1. Adjust the sliders on the left\n2. Click 'SIMULATE IMPACT'\n3. View predicted results here")
            
            # Show current profile
            st.markdown("#### 📋 Current Customer Profile")
            
            profile_data = pd.DataFrame({
                'Metric': [
                    'Annual Premium',
                    'Monthly Premium',
                    'Current Churn Risk',
                    'Risk Category'
                ],
                'Value': [
                    f"${orig_annual:.2f}",
                    f"${orig_monthly:.2f}",
                    f"{original_prob:.1%}",
                    "🔴 HIGH" if original_prob > 0.7 else "🟡 MEDIUM" if original_prob > 0.4 else "🟢 LOW"
                ]
            })
            
            st.dataframe(profile_data, hide_index=True, use_container_width=True)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>AI-Powered Churn Retention Intelligence</strong></p>
    <p>Made by Red Devils</p>
    <p>Built for Megathon'25 🏆</p>
</div>
""", unsafe_allow_html=True)