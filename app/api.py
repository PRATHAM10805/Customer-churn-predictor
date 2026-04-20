from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.Predict import predict
from src.utils.logger import get_logger

logger = get_logger(__name__)
app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")

class ChurnRequest(BaseModel):
    tenure: int = Field(..., ge=0, le=100)
    monthly_charges: float = Field(..., alias="MonthlyCharges")
    contract: str = Field(..., alias="Contract")
    internet_service: str = Field(..., alias="InternetService")
    feedback: str = Field(..., alias="Customer_Feedback")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
def get_prediction(request: ChurnRequest):
    try:
        result = predict(
            request.tenure,
            request.monthly_charges,
            request.contract,
            request.internet_service,
            request.feedback
        )
        return {"churn_prediction": result, "status": "success"}
    except Exception as e:
        logger.error(f"API Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
