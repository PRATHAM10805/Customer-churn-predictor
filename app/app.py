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

# Custom CSS for glassmorphism and modern feel
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        border: 2px solid white;
    }
    .reportview-container .main .block-container{
        padding-top: 2rem;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://img.icons8.com/plasticine/100/000000/analytics.png", width=100)
    st.title("Settings & Info")
    st.info("ChurnGuard AI uses advanced Random Forest models to predict customer behavior.")
    st.divider()
    st.subheader("Industry-Ready Pipeline")
    st.write("✓ Pydantic Validation")
    st.write("✓ Structured Logging")
    st.write("✓ FastAPI Layer")

st.title("📊 Customer Churn Analysis")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📅 Customer Tenure")
    tenure = st.slider("Select Tenure (Months)", 0, 72, 12)
    
    st.subheader("💳 Financial Info")
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0, step=0.1)

with col2:
    st.subheader("📄 Contract Details")
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

st.subheader("💬 Customer Feedback Sentiment")
feedback = st.text_area("Analyze Feedback:", placeholder="Example: Very bad experience, poor support...")

st.markdown("---")
if st.button("🚀 Analyze Churn Risk"):
    with st.spinner("Analyzing data through ML model..."):
        try:
            result = predict(
                tenure,
                monthly_charges,
                contract,
                internet_service,
                feedback
            )

            if result == 1:
                st.markdown('<div class="prediction-card" style="background-color: rgba(255, 75, 75, 0.2); border: 2px solid #ff4b4b;">', unsafe_allow_html=True)
                st.error("⚠️ CRITICAL: High likelihood of churn detected.")
                st.write("Recommended Action: Send an immediate retention offer.")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="prediction-card" style="background-color: rgba(76, 175, 80, 0.2); border: 2px solid #4CAF50;">', unsafe_allow_html=True)
                st.success("✅ SAFE: Customer is likely to remain loyal.")
                st.write("Recommended Action: Upsell additional services.")
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.toast("Analysis complete!")

        except Exception as e:
            st.error(f"Prediction Error: {e}")
            logger.error(f"Streamlit UI error: {e}")