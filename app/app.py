import streamlit as st
import sys
import os

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Predict import predict
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Premium Styling
st.set_page_config(
    page_title="ChurnGuard AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Styling Upgrade
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    
    .stApp {
        background: transparent;
    }
    
    .prediction-card {
        padding: 30px;
        border-radius: 20px;
        backdrop-filter: blur(10px);
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .prediction-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-container {
        display: flex;
        justify-content: space-around;
        padding: 15px;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        margin-bottom: 20px;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
        border: none;
        color: white;
        font-weight: 600;
        letter-spacing: 0.5px;
        height: 3.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://img.icons8.com/fluency/100/000000/shield.png", width=100)
    st.title("ChurnGuard AI")
    st.caption("v2.1.0 Premium Pipeline")
    
    st.divider()
    
    # Model Status Metrics
    st.write("📡 **Engine Status**")
    st.success("Operational - Logistic Engine")
    
    st.write("📊 **Model Accuracy**")
    st.progress(0.75, text="75% Accuracy")
    
    st.write("🎯 **Model Precision**")
    st.progress(0.83, text="0.83 ROC-AUC")
    
    st.divider()
    st.info("💡 **Pro Tip**: Long-term contracts and Online Security features are the strongest inhibitors of churn.")

# Header
st.markdown("<h1 style='text-align: center; color: white;'>🚀 Intelligence-Driven Churn Analytics</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8;'>Analyze customer data with high-precision machine learning to predict retention risk.</p>", unsafe_allow_html=True)
st.markdown("---")

# Main Analysis Form
st.markdown("### 🎯 Primary Risk Factors")
col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (Months)", 0, 72, 12, help="How many months has the customer been with the company?")
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

with col2:
    internet_service = st.selectbox("Internet Service Type", ["DSL", "Fiber optic", "No"])
    feedback = st.text_area("Customer Feedback Input (NLP Layer)", height=128, placeholder="Example: Very bad experience, poor support...")

st.markdown("<br>", unsafe_allow_html=True)

# Advanced Section
with st.expander("🛠️ Advanced Customer Profiling (Optional Defaults Applied)"):
    adv_col1, adv_col2, adv_col3 = st.columns(3)
    
    with adv_col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen Status", [0, 1])
        partner = st.selectbox("Partnered?", ["No", "Yes"])
        dependents = st.selectbox("Dependents?", ["No", "Yes"])

    with adv_col2:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        online_security = st.selectbox("Online Security Add-on", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup Add-on", ["No", "Yes", "No internet service"])

    with adv_col3:
        tech_support = st.selectbox("Tech Support Add-on", ["No", "Yes", "No internet service"])
        paperless = st.selectbox("Paperless Billing?", ["Yes", "No"])
        payment_method = st.selectbox("Payment Gateway", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=tenure * monthly_charges)

    # Pre-set some low-impact features to avoid clutter
    streaming_tv = "No" 
    streaming_movies = "No"
    device_protection = "No"

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🚀 EXECUTE PREDICTIVE ANALYSIS"):
    with st.spinner("Synchronizing with ML Core..."):
        try:
            input_data = {
                "gender": gender,
                "SeniorCitizen": senior_citizen,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
                "Customer_Feedback": feedback
            }

            prediction, probability = predict(input_data)

            st.markdown("---")
            
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                st.markdown("### Analysis Result")
                if prediction == 1:
                    st.error("🚨 AT-RISK")
                    st.markdown(f"#### Probability: {probability*100:.1f}%")
                else:
                    st.success("✅ STABLE")
                    st.markdown(f"#### Probability: {probability*100:.1f}%")

            with res_col2:
                st.markdown("### Churn Risk Meter")
                # Visual gauge using progress bar (customized)
                color = "red" if probability > 0.5 else "green"
                st.progress(probability)
                
                if prediction == 1:
                    st.warning("Customer shows high churn intent signals. Immediate intervention recommended.")
                else:
                    st.info("Customer shows strong retention signals. Potential for upselling.")

        except Exception as e:
            st.error(f"Prediction Error: {e}")
            logger.error(f"Streamlit UI error: {e}")