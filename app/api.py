from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.Predict import predict
from src.utils.logger import get_logger

logger = get_logger(__name__)
app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")

class ChurnRequest(BaseModel):
    tenure: int
    MonthlyCharges: float
    Contract: str
    InternetService: str
    Customer_Feedback: str
    
    # Optional fields with defaults
    gender: str = "Female"
    SeniorCitizen: int = 0
    Partner: str = "No"
    Dependents: str = "No"
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    TotalCharges: float = 0.0

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
def get_prediction(request: ChurnRequest):
    try:
        prediction, probability = predict(request.dict())
        return {
            "churn_prediction": prediction, 
            "churn_probability": round(probability, 4),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"API Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
