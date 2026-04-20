import pickle
import pandas as pd
import os
from pydantic import BaseModel, Field
from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger(__name__)
config = load_config()

class PredictionInput(BaseModel):
    tenure: int = Field(..., ge=0, le=100)
    monthly_charges: float = Field(..., alias="MonthlyCharges")
    contract: str = Field(..., alias="Contract")
    internet_service: str = Field(..., alias="InternetService")
    feedback: str = Field(..., alias="Customer_Feedback")

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


def predict(tenure, monthly_charges, contract, internet_service, feedback):
    try:
        # Validate inputs
        data_in = PredictionInput(
            tenure=tenure,
            MonthlyCharges=monthly_charges,
            Contract=contract,
            InternetService=internet_service,
            Customer_Feedback=feedback
        )
        
        model = load_model()
        input_df = pd.DataFrame([data_in.dict(by_alias=True)])

        prediction = model.predict(input_df)[0]
        logger.info(f"Prediction successful: {prediction}")
        return int(prediction)

    except Exception as e:
        logger.exception("Prediction failed")
        raise ValueError(f"Prediction error: {e}")