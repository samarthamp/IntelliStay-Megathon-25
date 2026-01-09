"""
LLM Utility Functions for Dashboard
Generates on-demand insights for selected customers
"""

import requests
import pandas as pd

# LM Studio configuration
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"

def test_lm_studio_connection():
    """Test if LM Studio is available"""
    try:
        response = requests.get("http://localhost:1234/v1/models", timeout=3)
        return response.status_code == 200
    except:
        return False
    
def generate_waterfall_plot(customer_idx, shap_data_package, output_path=None):
    """
    Generate a waterfall plot for a specific customer
    
    Args:
        customer_idx: Customer index from test set
        shap_data_package: Loaded shap_data_package.pkl
        output_path: Where to save (optional, returns fig if None)
    
    Returns:
        matplotlib figure or saves to file
    """
    import matplotlib.pyplot as plt
    import shap
    
    # Find customer in SHAP sample
    try:
        position = shap_data_package['customer_indices'].index(customer_idx)
    except ValueError:
        return None  # Customer not in SHAP sample
    
    # Get SHAP values for this customer
    shap_values = shap_data_package['shap_values'][position]
    expected_value = shap_data_package['expected_value']
    feature_names = shap_data_package['feature_names']
    customer_data = shap_data_package['X_data'].iloc[position].values
    
    # Create explanation object
    explanation = shap.Explanation(
        values=shap_values,
        base_values=expected_value,
        data=customer_data,
        feature_names=feature_names
    )
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.waterfall_plot(explanation, max_display=10, show=False)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        return fig

def generate_customer_insights(customer_data, shap_features, global_context):
    """
    Generate LLM-powered insights for a specific customer
    
    Args:
        customer_data: Dict with customer info (index, probability, etc.)
        shap_features: List of dicts with top SHAP features
        global_context: DataFrame with global feature importance
    
    Returns:
        Dict with structured insights or None if failed
    """
    
    # Format SHAP features for prompt
    top_drivers_text = []
    for i, feat in enumerate(shap_features[:3], 1):
        direction = "increasing" if feat['shap_value'] > 0 else "decreasing"
        top_drivers_text.append(
            f"{i}. {feat['feature']} (value: {feat['value']:.2f}, "
            f"impact: {direction} churn risk by {abs(feat['shap_value']):.3f})"
        )
    
    churn_prob = customer_data['churn_probability']
    risk_level = "HIGH" if churn_prob > 0.7 else "MODERATE" if churn_prob > 0.4 else "LOW"
    
    # Create prompt
    prompt = f"""You are an expert auto insurance retention strategist analyzing customer churn risk.

CUSTOMER PROFILE:
- Customer ID: {customer_data['customer_id']}
- Churn Risk Score: {churn_prob:.1%}
- Risk Level: {risk_level} RISK

TOP 3 CHURN DRIVERS FOR THIS CUSTOMER:
{chr(10).join(top_drivers_text)}

COMPANY-WIDE CONTEXT:
Across all customers, the most important churn factors are: {', '.join(global_context.head(5)['Feature'].tolist())}

YOUR TASK:
Provide a clear, actionable analysis in this exact format:

**CUSTOMER PERSONA:** (One sentence describing this customer)

**WHY THEY MIGHT LEAVE:** (2-3 sentences explaining the churn risk based on the top drivers)

**RETENTION ACTIONS:**
- [Action 1: Specific, immediate action]
- [Action 2: Specific, immediate action]  
- [Action 3: Specific, immediate action]

Keep it concise, non-technical, and action-oriented. Focus on what a retention agent can DO right now."""

    try:
        response = requests.post(
            LM_STUDIO_URL,
            json={
                "model": "qwen/qwen3-4b-thinking-2507",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert insurance retention strategist. Provide clear, actionable insights."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 400,
                "stream": False
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            analysis = result['choices'][0]['message']['content']
            return {
                'success': True,
                'analysis': analysis,
                'customer_id': customer_data['customer_id'],
                'churn_probability': churn_prob,
                'risk_level': risk_level
            }
        else:
            return {
                'success': False,
                'error': f"LLM API returned status {response.status_code}"
            }
            
    except requests.exceptions.Timeout:
        return {
            'success': False,
            'error': "LLM request timed out (30s). Model may be loading."
        }
    except Exception as e:
        return {
            'success': False,
            'error': f"LLM generation failed: {str(e)}"
        }

def generate_fallback_insights(customer_data, shap_features):
    """
    Generate basic insights without LLM (fallback)
    """
    churn_prob = customer_data['churn_probability']
    risk_level = "HIGH" if churn_prob > 0.7 else "MODERATE" if churn_prob > 0.4 else "LOW"
    
    top_feature = shap_features[0]['feature']
    
    return {
        'success': True,
        'analysis': f"""**CUSTOMER PERSONA:** {risk_level} risk customer with churn probability of {churn_prob:.1%}

**WHY THEY MIGHT LEAVE:** Primary risk factor is {top_feature}. This customer shows patterns similar to others who have churned in the past.

**RETENTION ACTIONS:**
- Review account history and recent interactions
- Contact customer to understand concerns  
- Offer personalized retention incentive
- Schedule follow-up within 7 days

*Note: LM Studio unavailable - using basic analysis*""",
        'customer_id': customer_data['customer_id'],
        'churn_probability': churn_prob,
        'risk_level': risk_level
    }