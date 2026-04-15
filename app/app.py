import streamlit as st
import sys
import os

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Predict import predict   # ✅ fixed

st.set_page_config(page_title="Churn Prediction", layout="centered")

st.title("📊 Customer Churn Prediction")

tenure = st.slider("Tenure (months)", 0, 72)
monthly_charges = st.number_input("Monthly Charges")

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

feedback = st.text_area("Customer Feedback")

if st.button("Predict"):
    try:
        result = predict(
            tenure,
            monthly_charges,
            contract,
            internet_service,
            feedback
        )

        if result == 1:
            st.error("⚠️ High chance of churn")
        else:
            st.success("✅ Customer likely to stay")

    except Exception as e:
        st.error(f"Error: {e}")