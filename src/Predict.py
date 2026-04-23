import pickle
import pandas as pd
import os
from pydantic import BaseModel, Field
from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger(__name__)
config = load_config()

class PredictionInput(BaseModel):
    gender: str = Field(...)
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: str = Field(...)
    Dependents: str = Field(...)
    tenure: int = Field(..., ge=0, le=100)
    PhoneService: str = Field(...)
    MultipleLines: str = Field(...)
    InternetService: str = Field(...)
    OnlineSecurity: str = Field(...)
    OnlineBackup: str = Field(...)
    DeviceProtection: str = Field(...)
    TechSupport: str = Field(...)
    StreamingTV: str = Field(...)
    StreamingMovies: str = Field(...)
    Contract: str = Field(...)
    PaperlessBilling: str = Field(...)
    PaymentMethod: str = Field(...)
    MonthlyCharges: float = Field(...)
    TotalCharges: float = Field(...)
    Customer_Feedback: str = Field(...)

# Load model once (important for performance)
MODEL_PATH = config['model']['path']

model = None


def load_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model not found at {MODEL_PATH}")
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Model loaded successfully from {MODEL_PATH}")

    return model


DEFAULT_VALUES = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "TotalCharges": 0.0  # Will be calculated if missing
}

def predict(input_data: dict):
    try:
        # Merge input with defaults
        full_data = DEFAULT_VALUES.copy()
        full_data.update(input_data)
        
        # Smart calculation for TotalCharges if not provided
        if full_data["TotalCharges"] == 0.0 and "tenure" in full_data and "MonthlyCharges" in full_data:
            full_data["TotalCharges"] = full_data["tenure"] * full_data["MonthlyCharges"]

        # Validate inputs
        data_in = PredictionInput(**full_data)
        
        model = load_model()
        input_df = pd.DataFrame([data_in.dict()])

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        logger.info(f"Prediction: {prediction}, Probability: {probability:.4f}")
        return int(prediction), float(probability)

    except Exception as e:
        logger.exception("Prediction failed")
        raise ValueError(f"Prediction error: {e}")