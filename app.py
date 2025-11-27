import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Set page config
st.set_page_config(
    page_title="Fortebank Anti-Fraud MVP",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load models
@st.cache_resource
def load_models():
    with open('models.pkl', 'rb') as f:
        return pickle.load(f)

try:
    models_dict = load_models()
    xgb_model = models_dict['xgb']
    lgb_model = models_dict['lgb']
    rf_model = models_dict['rf']
    meta_learner = models_dict['meta_learner']
    scaler = models_dict['scaler']
    best_threshold = models_dict['best_threshold']
    st.success("✓ Models loaded successfully")
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()

def prepare_features(transaction_dict, amount_median=1000, amount_p75=3000):
    """Transform raw transaction into 44 features (matches trained model)"""
    amount = transaction_dict.get('amount', 0)
    hour = transaction_dict.get('hour_of_day', 0)
    day_of_week = transaction_dict.get('day_of_week', 0)
    month = transaction_dict.get('month', 1)

    monthly_os_changes = transaction_dict.get('monthly_os_changes', 0)
    monthly_phone_model_changes = transaction_dict.get('monthly_phone_model_changes', 0)
    logins_7d = transaction_dict.get('logins_7d', 0)
    logins_30d = transaction_dict.get('logins_30d', 0)
    freq_7d = transaction_dict.get('freq_7d', 0)
    freq_30d = transaction_dict.get('freq_30d', 1)
    freq_change_ratio = transaction_dict.get('freq_change_ratio', 0)
    avg_interval_30d = transaction_dict.get('avg_interval_30d', 0)
    std_interval_30d = transaction_dict.get('std_interval_30d', 0)
    burstiness = transaction_dict.get('burstiness', 0)
    fano_factor = transaction_dict.get('fano_factor', 0)
    zscore_7d = transaction_dict.get('zscore_7d', 0)
    recipient_frequency = transaction_dict.get('recipient_frequency', 1)
    client_tx_count_7d = transaction_dict.get('client_tx_count_7d', 0)
    client_tx_count_30d = transaction_dict.get('client_tx_count_30d', 0)

    amount_log = np.log1p(max(amount, 0))
    is_weekend = 1 if day_of_week >= 5 else 0
    device_instability = (monthly_os_changes + monthly_phone_model_changes) / 2
    login_surge_ratio = freq_7d / (freq_30d + 1e-5)
    is_inactive_user = 1 if logins_30d < 5 else 0
    session_interval_anomaly = 1 if abs(zscore_7d) > 2 else 0
    login_burstiness_high = 1 if burstiness > 0.3 else 0
    recent_device_change = 1 if monthly_phone_model_changes > 0 else 0
    device_instability_x_amount = device_instability * (1 if amount > amount_median else 0)
    inactive_x_large_tx = is_inactive_user * (1 if amount > amount_p75 else 0)
    anomaly_timing = ((1 if (hour < 8 or hour >= 17) else 0) * session_interval_anomaly)

    features = [
        amount_log, hour, is_weekend, recipient_frequency, client_tx_count_7d, client_tx_count_30d,
        device_instability, login_surge_ratio, is_inactive_user, session_interval_anomaly,
        login_burstiness_high, recent_device_change, monthly_os_changes, monthly_phone_model_changes,
        logins_7d, logins_30d, freq_7d, freq_change_ratio, avg_interval_30d, std_interval_30d,
        burstiness, fano_factor, zscore_7d, device_instability_x_amount, inactive_x_large_tx, anomaly_timing
    ]

    # CRITICAL: Match exactly with training data encoding
    # DOW: 6 categories, drop first = 5 features
    dow_encoded = [1 if day_of_week == i else 0 for i in range(2, 7)]
    
    # Month: 12 categories, drop first = 11 features
    month_encoded = [1 if month == i else 0 for i in range(2, 12)]
    
    # Amount: 4 categories, drop first = 3 features
    if amount <= 500:
        amount_encoded = [1, 0, 0]
    elif amount <= 1500:
        amount_encoded = [0, 1, 0]
    elif amount <= 3000:
        amount_encoded = [0, 0, 1]
    else:
        amount_encoded = [0, 0, 0]

    features.extend(dow_encoded)      # 26 + 5 = 31
    features.extend(month_encoded)    # 31 + 11 = 42
    features.extend(amount_encoded)   # 42 + 2 = 44 ✅

    return np.array(features).reshape(1, -1)

def predict_fraud(transaction_dict):
    """Make fraud prediction"""
    features = prepare_features(transaction_dict)
    features_scaled = scaler.transform(features)
    
    pred_xgb = xgb_model.predict_proba(features_scaled)[:, 1][0]
    pred_lgb = lgb_model.predict_proba(features_scaled)[:, 1][0]
    pred_rf = rf_model.predict_proba(features_scaled)[:, 1][0]
    
    meta_input = np.array([[pred_xgb, pred_lgb, pred_rf]])
    fraud_prob = meta_learner.predict_proba(meta_input)[:, 1][0]
    
    decision = "BLOCK" if fraud_prob >= best_threshold else "ALLOW"
    risk_level = 'HIGH' if fraud_prob >= 0.7 else 'MEDIUM' if fraud_prob >= best_threshold else 'LOW'
    
    return {
        'fraud_probability': fraud_prob,
        'decision': decision,
        'risk_level': risk_level,
        'confidence': max(fraud_prob, 1 - fraud_prob),
        'xgb': pred_xgb,
        'lgb': pred_lgb,
        'rf': pred_rf
    }

# UI
st.markdown("# 🛡️ Fortebank Anti-Fraud MVP")
st.markdown("**Real-time transaction fraud detection system**")

tab1, tab2, tab3 = st.tabs(["🧪 Test Transaction", "📊 Batch Upload", "ℹ️ About"])

with tab1:
    st.subheader("Test Single Transaction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Transaction Details**")
        amount = st.number_input("Amount (USD)", value=2500.0, min_value=0.0, step=100.0)
        hour_of_day = st.slider("Hour of Day (0-23)", 0, 23, 14)
        day_of_week = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)
        month = st.slider("Month (1-12)", 1, 12, 6)
    
    with col2:
        st.markdown("**Behavioral - Logins**")
        logins_7d = st.number_input("Logins (7 days)", value=5, min_value=0)
        logins_30d = st.number_input("Logins (30 days)", value=15, min_value=0)
        freq_7d = st.number_input("Frequency (7 days)", value=10, min_value=0)
        freq_30d = st.number_input("Frequency (30 days)", value=25, min_value=1)
    
    with col3:
        st.markdown("**Device & Activity**")
        monthly_os_changes = st.number_input("OS Changes (monthly)", value=0, min_value=0)
        monthly_phone_model_changes = st.number_input("Phone Model Changes", value=0, min_value=0)
        burstiness = st.slider("Burstiness Score", 0.0, 1.0, 0.2)
        zscore_7d = st.slider("Z-Score (7 days)", -3.0, 3.0, 0.5)
    
    with st.expander("⚙️ Advanced Parameters"):
        col1, col2, col3 = st.columns(3)
        with col1:
            freq_change_ratio = st.number_input("Freq Change Ratio", value=0.4, step=0.1)
            avg_interval_30d = st.number_input("Avg Interval (30d)", value=48.0, step=1.0)
        with col2:
            std_interval_30d = st.number_input("Std Interval (30d)", value=12.0, step=1.0)
            fano_factor = st.number_input("Fano Factor", value=1.5, step=0.1)
        with col3:
            recipient_frequency = st.number_input("Recipient Frequency", value=5, min_value=1)
            client_tx_count_7d = st.number_input("TX Count (7d)", value=3, min_value=0)
            client_tx_count_30d = st.number_input("TX Count (30d)", value=10, min_value=0)
    
    if st.button("🔍 Predict Fraud Risk", use_container_width=True):
        transaction = {
            'amount': amount, 'hour_of_day': hour_of_day, 'day_of_week': day_of_week,
            'month': month, 'logins_7d': logins_7d, 'logins_30d': logins_30d,
            'freq_7d': freq_7d, 'freq_30d': freq_30d, 'freq_change_ratio': freq_change_ratio,
            'monthly_os_changes': monthly_os_changes, 'monthly_phone_model_changes': monthly_phone_model_changes,
            'avg_interval_30d': avg_interval_30d, 'std_interval_30d': std_interval_30d,
            'burstiness': burstiness, 'fano_factor': fano_factor, 'zscore_7d': zscore_7d,
            'recipient_frequency': recipient_frequency, 'client_tx_count_7d': client_tx_count_7d,
            'client_tx_count_30d': client_tx_count_30d
        }
        
        result = predict_fraud(transaction)
        st.markdown("---")
        
        if result['decision'] == 'BLOCK':
            st.error(f"### 🚫 {result['decision']}")
        else:
            st.success(f"### ✅ {result['decision']}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Fraud Probability", f"{result['fraud_probability']*100:.1f}%")
        with col2:
            st.metric("Risk Level", result['risk_level'])
        with col3:
            st.metric("Confidence", f"{result['confidence']*100:.1f}%")
        with col4:
            st.metric("Threshold", f"{best_threshold:.2f}")
        
        st.markdown("**Individual Model Scores:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**XGBoost**\n{result['xgb']*100:.1f}%")
        with col2:
            st.info(f"**LightGBM**\n{result['lgb']*100:.1f}%")
        with col3:
            st.info(f"**Random Forest**\n{result['rf']*100:.1f}%")

with tab2:
    st.subheader("Batch Transaction Upload")
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    if uploaded_file and st.button("Predict All Transactions"):
        df = pd.read_csv(uploaded_file)
        results = []
        for idx, row in df.iterrows():
            result = predict_fraud(row.to_dict())
            result['transaction_id'] = idx
            results.append(result)
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)
        csv = results_df.to_csv(index=False)
        st.download_button("📥 Download Results", csv, "fraud_predictions.csv", "text/csv")

with tab3:
    st.markdown("""
    ## About This MVP
    **Fortebank Anti-Fraud Detection System**
    - 3 Base Models: XGBoost, LightGBM, Random Forest
    - Meta-Learner: Logistic Regression
    - 42 features from transaction + behavioral data
    """)

with st.sidebar:
    st.markdown("### 📊 Model Info")
    st.info(f"Best Threshold: {best_threshold:.2f}")
