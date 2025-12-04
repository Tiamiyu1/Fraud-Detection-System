# ========================================
# FRAUD DETECTION STREAMLIT UI
# ========================================
# 
# A beautiful Streamlit interface for real-time fraud detection
# 
# SETUP:
# 1. pip install streamlit joblib pandas numpy scikit-learn
# 2. Make sure fraud_model.pkl, feature_columns.pkl, and customer_stats.pkl exist
#    (Run train_model.py if needed)
# 3. streamlit run streamlit_app.py
# 
# ========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
    }
    .fraud-alert {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        padding: 2rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .safe-alert {
        background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
        padding: 2rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load model artifacts
@st.cache_resource
def load_model():
    """Load model and artifacts"""
    try:
        model = joblib.load('fraud_model.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        customer_stats = joblib.load('customer_stats.pkl')
        return model, feature_columns, customer_stats
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Please run 'python train_model.py' first to train the model.")
        st.stop()

model, feature_columns, customer_stats = load_model()

def engineer_features(data):
    """Engineer features from transaction data"""
    
    # Parse timestamp
    if data.get('timestamp'):
        ts = pd.to_datetime(data['timestamp'])
    else:
        ts = datetime.now()
    
    # Extract time features
    hour = ts.hour
    day_of_week = ts.dayofweek
    is_weekend = 1 if day_of_week >= 5 else 0
    is_night = 1 if (hour >= 22 or hour <= 6) else 0
    
    # Normalize risk scores to 0-100 scale
    risk_score_internal_normalized = data['risk_score_internal'] * 200  # 0-0.5 -> 0-100
    ip_risk_score_normalized = data['ip_risk_score'] * 83.33  # 0-1.2 -> 0-100
    corridor_risk_normalized = data['corridor_risk'] * 400  # 0-0.25 -> 0-100
    device_trust_score_normalized = data['device_trust_score'] * 100  # 0-1 -> 0-100
    
    # Transaction behavior features
    high_velocity_flag = 1 if (data['txn_velocity_1h'] > 3 or data['txn_velocity_24h'] > 10) else 0
    new_account = 1 if data['account_age_days'] < 30 else 0
    old_account = 1 if data['account_age_days'] > 365 else 0
    
    # Amount features
    amount_to_fee_ratio = data['amount_usd'] / data['fee'] if data['fee'] > 0 else 0
    log_amount = np.log1p(data['amount_usd'])
    
    # Risk composites
    risk_composite = (
        ip_risk_score_normalized * 0.30 +
        risk_score_internal_normalized * 0.30 +
        corridor_risk_normalized * 0.20 +
        (1 if data['location_mismatch'] else 0) * 20 * 0.20
    )
    
    device_risk = (
        (100 - device_trust_score_normalized) * 0.6 +
        (1 if data['new_device'] else 0) * 40 * 0.4
    )
    
    # Customer features
    customer_avg_amount = customer_stats['customer_avg_amount'].get(
        data['customer_id'], 
        data['amount_usd']
    )
    customer_txn_count = customer_stats['customer_txn_count'].get(
        data['customer_id'], 
        1
    )
    amount_deviation = abs(data['amount_usd'] - customer_avg_amount)
    
    # Interaction features
    high_risk_new_device = 1 if (risk_score_internal_normalized > 50 and data['new_device']) else 0
    location_mismatch_velocity = (1 if data['location_mismatch'] else 0) * high_velocity_flag
    night_high_amount = is_night * (1 if data['amount_usd'] > 1000 else 0)
    
    # Create feature dictionary
    features = {
        'risk_composite': risk_composite,
        'device_risk': device_risk,
        'txn_velocity_1h': data['txn_velocity_1h'],
        'txn_velocity_24h': data['txn_velocity_24h'],
        'high_velocity_flag': high_velocity_flag,
        'account_age_days': data['account_age_days'],
        'new_account': new_account,
        'old_account': old_account,
        'amount_usd': data['amount_usd'],
        'log_amount': log_amount,
        'fee': data['fee'],
        'amount_to_fee_ratio': amount_to_fee_ratio,
        'new_device': 1 if data['new_device'] else 0,
        'location_mismatch': 1 if data['location_mismatch'] else 0,
        'hour': hour,
        'day_of_week': day_of_week,
        'is_weekend': is_weekend,
        'is_night': is_night,
        'customer_txn_count': customer_txn_count,
        'amount_deviation': amount_deviation,
        'high_risk_new_device': high_risk_new_device,
        'location_mismatch_velocity': location_mismatch_velocity,
        'night_high_amount': night_high_amount
    }
    
    df = pd.DataFrame([features])[feature_columns]
    return df

def identify_risk_factors(data, features):
    """Identify key risk factors"""
    risk_factors = []
    
    if data['risk_score_internal'] > 0.3:  # >60% when normalized
        risk_factors.append("🔴 High internal risk score")
    
    if data['ip_risk_score'] > 0.72:  # >60% when normalized
        risk_factors.append("🔴 High IP risk score")
    
    if data['chargeback_history_count'] > 0:

        risk_factors.append(f"🔴 Previous chargebacks: {data['chargeback_history_count']}")
    
    if features['high_velocity_flag'].values[0] == 1:
        risk_factors.append("🔴 High transaction velocity")
    
    if data['new_device']:
        risk_factors.append("🟡 Transaction from new device")
    
    if data['location_mismatch']:
        risk_factors.append("🟡 Location mismatch detected")
    
    if features['new_account'].values[0] == 1:
        risk_factors.append("🟡 New account (<30 days)")
    
    if features['is_night'].values[0] == 1:
        risk_factors.append("🟡 Night-time transaction")
    
    if data['device_trust_score'] < 0.4:  # <40% when normalized
        risk_factors.append("🔴 Low device trust score")
    
    if data['amount_usd'] > 2000:
        risk_factors.append("🟡 High transaction amount")
    
    return risk_factors if risk_factors else ["✅ No significant risk factors detected"]

# Header
st.title("🛡️ NOVA Pay Fraud Detection System")
st.markdown("### Real-time Transaction Fraud Analysis powered by Machine Learning")
st.markdown("---")

# Sidebar for quick fill options
st.sidebar.header("🎯 Quick Fill Options")
sample_type = st.sidebar.radio(
    "Select a sample transaction:",
    ["Custom", "Low Risk Sample", "High Risk Sample"]
)

# Initialize default values
if sample_type == "Low Risk Sample":
    default_values = {
        'customer_id': 'CUST_12345',
        'amount_usd': 150.0,
        'fee': 5.0,
        'risk_score_internal': 0.01,
        'ip_risk_score': 0.105,
        'corridor_risk': 0.04,
        'device_trust_score': 0.85,
        'txn_velocity_1h': 1,
        'txn_velocity_24h': 2,
        'account_age_days': 100,
        'chargeback_history_count': 0,
        'new_device': False,
        'location_mismatch': False
    }
elif sample_type == "High Risk Sample":
    default_values = {
        'customer_id': 'CUST_SUSPICIOUS',
        'amount_usd': 5000.0,
        'fee': 50.0,
        'risk_score_internal': 0.43,
        'ip_risk_score': 1.08,
        'corridor_risk': 0.19,
        'device_trust_score': 0.25,
        'txn_velocity_1h': 5,
        'txn_velocity_24h': 8,
        'account_age_days': 20,
        'chargeback_history_count': 2,
        'new_device': True,
        'location_mismatch': True
    }
else:
    default_values = {
        'customer_id': 'CUST_12345',
        'amount_usd': 1500.0,
        'fee': 15.0,
        'risk_score_internal': 0.23,
        'ip_risk_score': 0.36,
        'corridor_risk': 0.10,
        'device_trust_score': 0.75,
        'txn_velocity_1h': 2,
        'txn_velocity_24h': 5,
        'account_age_days': 120,
        'chargeback_history_count': 0,
        'new_device': False,
        'location_mismatch': False
    }

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Transaction Information")
    
    customer_id = st.text_input(
        "Customer ID",
        value=default_values['customer_id'],
        help="Unique identifier for the customer"
    )
    
    amount_usd = st.number_input(
        "Transaction Amount (USD)",
        min_value=0.0,
        value=default_values['amount_usd'],
        step=10.0,
        help="Amount in US Dollars"
    )
    
    fee = st.number_input(
        "Transaction Fee (USD)",
        min_value=0.0,
        value=default_values['fee'],
        step=1.0,
        help="Fee charged for the transaction"
    )
    
    timestamp = st.date_input(
        "Transaction Date",
        value=datetime.now(),
        help="Date of the transaction"
    )
    
    time = st.time_input(
        "Transaction Time",
        value=datetime.now().time(),
        help="Time of the transaction"
    )
    
    st.subheader("📊 Risk Scores")
    
    risk_score_internal = st.slider(
        "Internal Risk Score",
        min_value=0.0,
        max_value=0.5,
        value=default_values['risk_score_internal'],
        step=0.01,
        help="Internal risk assessment score (0.0 - 0.5)"
    )
    
    ip_risk_score = st.slider(
        "IP Risk Score",
        min_value=0.0,
        max_value=1.2,
        value=default_values['ip_risk_score'],
        step=0.01,
        help="IP address risk score (0.0 - 1.2)"
    )

with col2:
    st.subheader("🔐 Security Information")
    
    corridor_risk = st.slider(
        "Corridor Risk Score",
        min_value=0.0,
        max_value=0.25,
        value=default_values['corridor_risk'],
        step=0.01,
        help="Corridor/channel risk score (0.0 - 0.25)"
    )
    
    device_trust_score = st.slider(
        "Device Trust Score",
        min_value=0.0,
        max_value=1.0,
        value=default_values['device_trust_score'],
        step=0.01,
        help="Device trustworthiness score (0.0 - 1.0)"
    )
    
    st.subheader("📈 Transaction Patterns")
    
    txn_velocity_1h = st.number_input(
        "Transactions in Past Hour",
        min_value=-1,
        max_value=20,
        value=default_values['txn_velocity_1h'],
        help="Number of transactions in the past hour (-1 to 8)"
    )
    
    txn_velocity_24h = st.number_input(
        "Transactions in Past 24 Hours",
        min_value=0,
        max_value=50,
        value=default_values['txn_velocity_24h'],
        help="Number of transactions in the past 24 hours (0 to 9)"
    )
    
    st.subheader("👤 Account Information")
    
    account_age_days = st.number_input(
        "Account Age (days)",
        min_value=0,
        value=default_values['account_age_days'],
        help="Age of customer account in days"
    )
    
    chargeback_history_count = st.number_input(
        "Chargeback History Count",
        min_value=0,
        value=default_values['chargeback_history_count'],
        help="Number of previous chargebacks"
    )
    
    st.subheader("⚠️ Risk Flags")
    
    col2a, col2b = st.columns(2)
    with col2a:
        new_device = st.checkbox(
            "New Device",
            value=default_values['new_device'],
            help="Is this transaction from a new device?"
        )
    
    with col2b:
        location_mismatch = st.checkbox(
            "Location Mismatch",
            value=default_values['location_mismatch'],
            help="Is there a location mismatch?"
        )

# Analyze button
st.markdown("---")
analyze_button = st.button("🔍 Analyze Transaction", use_container_width=True, type="primary")

if analyze_button:
    # Combine date and time
    transaction_datetime = datetime.combine(timestamp, time)
    
    # Prepare data
    transaction_data = {
        'customer_id': customer_id,
        'timestamp': transaction_datetime.isoformat(),
        'amount_usd': amount_usd,
        'fee': fee,
        'risk_score_internal': risk_score_internal,
        'ip_risk_score': ip_risk_score,
        'corridor_risk': corridor_risk,
        'device_trust_score': device_trust_score,
        'txn_velocity_1h': txn_velocity_1h,
        'txn_velocity_24h': txn_velocity_24h,
        'account_age_days': account_age_days,
        'chargeback_history_count': chargeback_history_count,
        'new_device': new_device,
        'location_mismatch': location_mismatch
    }
    
    with st.spinner("Analyzing transaction..."):
        # Engineer features
        features_df = engineer_features(transaction_data)
        
        # Make prediction
        prediction = model.predict(features_df)[0]
        probability = model.predict_proba(features_df)[0][1]
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "LOW"
            risk_color = "green"
        elif probability < 0.6:
            risk_level = "MEDIUM"
            risk_color = "orange"
        elif probability < 0.8:
            risk_level = "HIGH"
            risk_color = "red"
        else:
            risk_level = "CRITICAL"
            risk_color = "darkred"
        
        # Calculate confidence
        confidence = max(probability, 1 - probability)
        
        # Identify risk factors
        risk_factors = identify_risk_factors(transaction_data, features_df)
    
    # Display results
    st.markdown("---")
    st.header("📊 Analysis Results")
    
    # Main fraud indicator
    if prediction == 1:
        st.markdown(
            '<div class="fraud-alert">⚠️ FRAUD DETECTED</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="safe-alert">✅ TRANSACTION SAFE</div>',
            unsafe_allow_html=True
        )
    
    # Metrics in columns
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric(
            "Fraud Probability",
            f"{probability * 100:.2f}%",
            delta=None
        )
    
    with metric_col2:
        st.metric(
            "Risk Level",
            risk_level,
            delta=None
        )
    
    with metric_col3:
        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%",
            delta=None
        )
    
    with metric_col4:
        st.metric(
            "Prediction",
            "FRAUD" if prediction == 1 else "LEGITIMATE",
            delta=None
        )
    
    # Probability gauge chart
    st.subheader("📈 Fraud Probability Score")
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Fraud Score", 'font': {'size': 24}},
        delta = {'reference': 50, 'increasing': {'color': "red"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': risk_color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': 'lightgreen'},
                {'range': [30, 60], 'color': 'lightyellow'},
                {'range': [60, 80], 'color': 'lightcoral'},
                {'range': [80, 100], 'color': 'red'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk factors
    st.subheader("🔍 Risk Factors Identified")
    
    if len(risk_factors) > 0:
        for factor in risk_factors:
            if "✅" in factor:
                st.success(factor)
            elif "🔴" in factor:
                st.error(factor)
            else:
                st.warning(factor)
    
    # Feature importance (top 10)
    st.subheader("📊 Top Risk Indicators")
    
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': feature_columns,
                'Importance': importances
            }).sort_values('Importance', ascending=False).head(10)
            
            fig_importance = px.bar(
                feature_importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Top 10 Features Contributing to This Prediction'
            )
            fig_importance.update_layout(height=400)
            st.plotly_chart(fig_importance, use_container_width=True)
    except Exception as e:
        st.info("Feature importance visualization not available for this model type.")
    
    # Transaction summary
    with st.expander("📋 Transaction Summary"):
        summary_data = {
            "Field": [
                "Customer ID",
                "Amount",
                "Fee",
                "Timestamp",
                "Internal Risk",
                "IP Risk",
                "Corridor Risk",
                "Device Trust",
                "Velocity (1h/24h)",
                "Account Age",
                "Chargebacks",
                "New Device",
                "Location Mismatch"
            ],
            "Value": [
                customer_id,
                f"${amount_usd:.2f}",
                f"${fee:.2f}",
                transaction_datetime.strftime("%Y-%m-%d %H:%M"),
                f"{risk_score_internal:.2f}",
                f"{ip_risk_score:.2f}",
                f"{corridor_risk:.2f}",
                f"{device_trust_score:.2f}",
                f"{txn_velocity_1h} / {txn_velocity_24h}",
                f"{account_age_days} days",
                chargeback_history_count,
                "Yes" if new_device else "No",
                "Yes" if location_mismatch else "No"
            ]
        }
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 1rem;'>
        <p>🛡️ Fraud Detection System v1.0 | Powered by Machine Learning</p>
        <p style='font-size: 0.8rem;'>Built with Streamlit | Abdulwasiu Tiamiyu</p>
    </div>
    """,
    unsafe_allow_html=True
)