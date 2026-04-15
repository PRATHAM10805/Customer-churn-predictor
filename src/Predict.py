import pickle
import pandas as pd
import os

# Load model once (important for performance)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model.pkl")

model = None


def load_model():
    global model

    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

    return model


def predict(tenure, monthly_charges, contract, internet_service, feedback):

    model = load_model()

    input_df = pd.DataFrame([{
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "Contract": contract,
        "InternetService": internet_service,
        "Customer_Feedback": feedback
    }])

    try:
        prediction = model.predict(input_df)[0]
        return int(prediction)

    except Exception as e:
        raise ValueError(f"Prediction error: {e}")