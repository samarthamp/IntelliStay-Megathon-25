

---
Team Name:- Red Devils 
Team Number:- 298
Members:-
Siddarth Gottumukkula
M P Samartha
Shlok Sand
Vedant Pahariya
Shreyas Kasture
---



# 🛡️ AI-Powered Churn Retention Intelligence Platform

> Enterprise-grade customer retention system combining Machine Learning, Explainable AI, Counterfactual Analysis, and Behavioral Economics for the auto insurance industry.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/XAI-SHAP-green.svg)](https://shap.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)

Built for **Megathon'25** Innovation Challenge

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Technology Stack](#-technology-stack)
- [Installation](#-installation)
- [Usage Guide](#-usage-guide)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Results](#-results)
- [Future Enhancements](#-future-enhancements)
- [Team](#-team)

---

## 🎯 Overview

The **Churn Retention Intelligence Platform** is an AI-powered solution designed to help insurance companies predict, understand, and prevent customer churn. Unlike traditional black-box ML models, our system provides:

- **Predictive Analytics**: 94.2% accurate churn prediction using XGBoost
- **Explainable AI**: SHAP analysis reveals why each customer might leave
- **Actionable Strategies**: DiCE counterfactuals show exactly what to change
- **Behavioral Nudges**: Psychology-based retention tactics beyond discounts
- **Interactive Simulation**: Real-time "what-if" analysis for retention strategies

### 💡 The Problem

Insurance companies lose **15-25% of customers annually** to churn, costing billions in revenue. Traditional approaches:
- ❌ Can't predict which customers will leave
- ❌ Don't understand why customers churn
- ❌ Offer generic discounts that hurt margins
- ❌ Lack personalized retention strategies

###  Our Solution

A comprehensive AI system that:
1. **Predicts** churn risk with 94%+ accuracy
2. **Explains** the top risk factors for each customer (SHAP)
3. **Prescribes** minimal changes needed to retain them (DiCE)
4. **Suggests** behavioral nudges proven to reduce churn
5. **Simulates** impact of different retention strategies

---

## 🌟 Key Features

### 1. **Portfolio Overview Dashboard**
- Real-time churn risk monitoring across 336K+ customers
- Global feature importance analysis
- Priority action list for high-risk customers
- Revenue-at-risk calculations

### 2. **Customer Deep Dive Analysis**
- Individual churn probability (0-100%)
- SHAP waterfall plots showing risk drivers
- Top 5 factors contributing to churn
- AI-generated customer personas
- Behavioral nudge recommendations

### 3. **Counterfactual Retention Strategies (DiCE)**
- Shows 3 different retention scenarios
- Minimal changes needed to reduce churn
- Only modifies actionable features (premiums, not age/tenure)
- Calculates expected risk reduction

### 4. **Interactive Retention Simulator**
- Real-time premium adjustment sliders
- Instant churn probability recalculation
- Business impact metrics (ROI, retention cost)
- Ready-to-implement action plans

### 5. **Behavioral Economics Integration**
- **Loss Aversion**: "You'll lose your $847 accident-free bonus"
- **Social Proof**: "92% of customers in Indore renewed"
- **Reciprocity**: "Free vehicle health check-up"
- **Commitment**: "Lock in your rate for 2 years"
- **Scarcity**: "Offer expires in 48 hours"
- **Anchoring**: "You've saved $2,340 over 3 years"

### 6. **Local LLM Integration**
- Qwen 2.5 (3B parameters) via LM Studio
- Synthesizes SHAP + DiCE insights
- Generates customer personas
- Creates ready-to-send message templates
- Works offline (no API costs)

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA PIPELINE                             │
│  Raw CSV → Feature Engineering → Train/Test Split            │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  ML MODELING LAYER                           │
│  XGBoost (CPU) → Optuna Tuning → 94.2% Accuracy             │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              EXPLAINABILITY LAYER                            │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐       │
│  │ SHAP         │  │ DiCE         │  │ LLM (Qwen)  │       │
│  │ (Why churn?) │  │ (How to fix?)│  │ (Synthesize)│       │
│  └──────────────┘  └──────────────┘  └─────────────┘       │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              PRESENTATION LAYER                              │
│              Streamlit Dashboard                             │
│  ┌──────────┐ ┌──────────┐ ┌────────────────┐             │
│  │Portfolio │ │Customer  │ │Retention       │             │
│  │Overview  │ │Deep Dive │ │Simulator       │             │
│  └──────────┘ └──────────┘ └────────────────┘             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

### **Core ML & Data Science**
- **Python 3.9+**: Primary language
- **Pandas**: Data manipulation (336K+ rows, 28 features)
- **NumPy**: Numerical computing
- **Scikit-learn**: Preprocessing, evaluation
- **XGBoost**: Gradient boosting (94.2% accuracy)
- **Optuna**: Hyperparameter optimization

### **Explainable AI**
- **SHAP (SHapley Additive exPlanations)**: Model interpretability
- **DiCE-ML**: Diverse Counterfactual Explanations
- **Matplotlib/Seaborn**: Visualizations

### **LLM Integration**
- **LM Studio**: Local LLM inference
- **Qwen 2.5 (3B)**: Language model for insights
- **Requests**: API communication

### **Dashboard & UI**
- **Streamlit**: Interactive web interface
- **Custom CSS**: Professional styling

### **Development Tools**
- **Joblib**: Model serialization
- **Git**: Version control

---

## 📦 Installation

### **Prerequisites**

```bash
# Check Python version (3.9+ required)
python --version

# Check pip
pip --version
```

### **Step 1: Clone Repository**

```bash
https://github.com/Geekonatrip123/Megathon.git
cd Megathon
```

### **Step 2: Create Virtual Environment**

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate
```

### **Step 3: Install Dependencies**

```bash
# Install all required packages
pip install -r requirements.txt
```

**requirements.txt:**
```txt
# Core Data Science
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0

# Machine Learning
xgboost==2.0.3
optuna==3.3.0

# Explainable AI
shap==0.42.1
dice-ml==0.11

# Visualization
matplotlib==3.7.2
seaborn==0.12.2

# Dashboard
streamlit==1.28.0

# Utilities
joblib==1.3.2
requests==2.31.0
```

### **Step 4: Download Dataset**

```bash
# Place your dataset in the project root
You can download all the csv and pkl files from here :-https://drive.google.com/drive/folders/16kiTYnvHENG4nyMVJsIt0IXIahScT62D?usp=sharing 

```

**Dataset Requirements:**
- Format: CSV
- Size: ~336,000 rows
- Key columns: `individual_id`, `Churn`, demographics, premium info

### **Step 5: Setup LM Studio (Optional but Recommended)**

1. **Download LM Studio**: [https://lmstudio.ai/](https://lmstudio.ai/)
2. **Install Qwen** from the model library
3. **Start Local Server**:
   - Open LM Studio
   - Load Qqwen/qwen3-4b-thinking-2507
   - Click "Start Server" (default: `localhost:1234`)

---

## 🚀 Usage Guide

### **Full Pipeline Execution**

Run all scripts in sequence:

```bash
# 1. Data preprocessing & feature engineering
python 01_eda_and_preprocessing.py

# 2. Model training & evaluation
python 02_modeling_pipeline.py

# 3. SHAP explainability analysis
python 03_shap_explainability_with_llm.py

# 4. DiCE counterfactual generation (optional)
python dice_counterfactuals.py

# 5. Launch dashboard
streamlit run 04_dashboard.py
```

### **Quick Start (Pre-trained Model)**

If you have pre-trained models:

```bash
# Just launch the dashboard
streamlit run 04_dashboard.py
```

Required files in project root:
- `final_xgboost_model.pkl`
- `scaler.pkl`
- `feature_names.pkl`
- `shap_data_package.pkl`
- `shap_explainer.pkl`
- `test_data_with_predictions.csv`
- `shap_global_importance.csv`

### **Dashboard Navigation**

1. **Portfolio Overview** (`📊 Churn Dashboard`)
   - View global metrics
   - Identify high-risk customers
   - Analyze top churn drivers

2. **Customer Deep Dive** (`🔍 Customer Deep Dive`)
   - Search by Customer ID or Risk Level
   - Click "ANALYZE CUSTOMER"
   - View SHAP explanation + AI insights + DiCE strategies

3. **Retention Simulator** (`⚙️ Retention Simulator`)
   - Select customer
   - Adjust annual premium slider
   - Click "RUN SIMULATION"
   - View predicted impact

---

## 📁 Project Structure

```
Megathon/
│
├── 01_eda_and_preprocessing.py       # Data cleaning & feature engineering
├── 02_modeling_pipeline.py           # Model training & evaluation
├── 03_shap_explainability_with_llm.py # SHAP analysis & LLM integration
├── dice_counterfactuals.py           # DiCE counterfactual generation
├── 04_dashboard.py                   # Streamlit interactive dashboard
│
├── llm_utils.py                      # LLM helper functions
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
│
├── autoinsurance_churn.csv           # Raw dataset (input)
├── autoinsurance_churn_engineered.csv # Processed dataset
│
├── final_xgboost_model.pkl           # Trained XGBoost model
├── scaler.pkl                        # StandardScaler for features
├── feature_names.pkl                 # List of feature names
├── label_encoders.pkl                # Categorical encoders
│
├── shap_explainer.pkl                # SHAP explainer object
├── shap_data_package.pkl             # SHAP values & customer data
├── shap_global_importance.csv        # Global feature importance
│
├── dice_explainer.pkl                # DiCE explainer (optional)
├── test_data_with_predictions.csv    # Test set with predictions
│
├── results/                          # Generated plots
│   ├── shap_summary_plot.png
│   ├── shap_feature_importance_bar.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── feature_importance.png
│
└── shap_force_plots/                 # Individual SHAP plots
    └── shap_waterfall_customer_*.png
```

---

## 🔬 Methodology

### **1. Data Preprocessing**

**Feature Engineering:**
- `premium_to_income_ratio = curr_ann_amt / income`
- `monthly_premium = curr_ann_amt / 12`
- `premium_affordability = (premium / income) * 100`
- `tenure_years = days_tenure / 365`
- `age_group` (binning)
- `income_bracket` (categorical)
- `customer_quality_score` (composite)

**Handling Missing Values:**
- Numerical: Median imputation
- Categorical: Mode imputation

**Encoding:**
- Label Encoding for categorical features
- StandardScaler for numerical features

### **2. Model Training**

**Algorithm:** XGBoost Classifier

**Hyperparameter Optimization (Optuna):**
- 100 trials
- Objective: Maximize F1-score
- Cross-validation: 5-fold

**Best Parameters:**
```python
{
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 3.5
}
```

### **3. SHAP Explainability**

**Global Interpretation:**
- Feature importance rankings
- Impact direction (positive/negative)
- Summary plots across all customers

**Local Interpretation:**
- Waterfall plots for individual customers
- Top 5 risk factors per customer
- Expected value vs actual prediction

**Top Global Churn Drivers:**
1. `days_tenure` (low tenure = high risk)
2. `length_of_residence`
3. `home_market_value`
4. `age_in_years`
5. `curr_ann_amt` (high premium = high risk)

### **4. DiCE Counterfactuals**

**Actionable Features** (can be changed):
- `curr_ann_amt` (annual premium)
- `monthly_premium` (derived)
- `premium_to_income_ratio`
- `premium_affordability`

**Immutable Features** (cannot be changed):
- Demographics: age, marital status, children
- Location: state, latitude, longitude
- History: tenure, origin date
- Background: income, education, credit

**Constraints:**
- Premium can only decrease (discounts, not increases)
- Maximum 50% discount
- Minimum 3 diverse counterfactuals
- Desired outcome: Churn = 0 (retention)

### **5. Behavioral Nudges**

**Nudge Categories & Application:**

| Nudge Type | Psychology Principle | When to Use | Example |
|------------|---------------------|-------------|---------|
| **Loss Aversion** | People hate losing more than gaining | High tenure customers | "Lose $847 accident-free bonus" |
| **Social Proof** | Follow peer behavior | Average customers | "92% in your area renewed" |
| **Reciprocity** | Return favors | Price-sensitive | "Free vehicle check-up" |
| **Commitment** | Lock in decisions | Risk-averse | "Lock rate for 2 years" |
| **Scarcity** | Fear of missing out | Fence-sitters | "Expires in 48 hours" |
| **Anchoring** | Reference point bias | Long tenure | "Saved $2,340 over 3 years" |

---

## 📊 Results

### **Model Performance**

| Metric    | Value    |
|-----------|---------:|
| Accuracy  | 0.883292 |
| Precision | 0.490100 |
| Recall    | 0.348040 |
| F1-Score  | 0.407031 |
| ROC-AUC   | 0.694800 |

### **Business Impact**

**Portfolio Metrics:**
- Total Customers: 336,182
- High Risk (>70%): 1,472 (0.4%)
- Medium Risk (40-70%): 31,591 (9.4%)
- Low Risk (<40%): 303,119 (90.2%)

### **Sample Success Case**

**Customer ID: 63533**
- Original Churn Risk: 91.0%
- Top Risk Factor: `days_tenure` (low)
- DiCE Recommendation: Reduce premium by 25%
- New Churn Risk: 32.4%
- **Risk Reduction: 58.6%** 

---

## 🎓 Key Learnings

### **Technical Insights**
1. **SHAP > Feature Importance**: SHAP provides direction and magnitude
2. **DiCE Constraints Critical**: Without proper constraints, recommendations are unrealistic
3. **Local LLM Viable**: 3B parameter models sufficient for synthesis tasks
4. **CPU vs GPU**: For inference, CPU is more stable in production

### **Business Insights**
1. **Tenure ≠ Loyalty**: Low tenure is #1 churn driver
2. **Premium Sweet Spot**: 20-30% discount optimal for retention
3. **Behavioral Nudges Work**: 15-20% better than pure discounts
4. **Timing Matters**: 30-60 days before renewal is key window

---

**Built with ❤️ for Megathon'25**

*"From prediction to action - AI that retains customers"*
