# 📊 Customer Churn Prediction System

🚀 An industry-ready Machine Learning system to predict customer churn using **Structured Data + NLP-based Customer Feedback**, featuring high-performance models and a production-style deployment architecture.

---

## 🔗 Project Overview

This system identifies customers at risk of leaving by analyzing a comprehensive set of features:
*   **Demographics**: Gender, Seniority, Partners, Dependents.
*   **Service Details**: Internet type, Online Security, Tech Support, Streaming services.
*   **Billing Information**: Tenure, Contract type, Monthly and Total charges.
*   **Customer Sentiment**: Synthetic feedback processed using **NLP (TF-IDF Vectorization)**.

---

## 🧠 Key Features

*   ✅ **Multi-Model Pipeline**: Automated comparison between **Random Forest** and **Logistic Regression**.
*   ✅ **Advanced Feature Engineering**: Combines categorical, numerical, and text-based signals.
*   ✅ **Imbalance Mitigation**: Uses class-weighting to accurately detect churners in imbalanced datasets.
*   ✅ **Production-Grade API**: Built with **FastAPI** for low-latency, high-throughput predictions.
*   ✅ **Interactive Dashboard**: **Streamlit** interface for manual testing and visualization.
*   ✅ **Config-Driven Architecture**: Easily tune hyperparameters via `config.yaml`.

---

## 🏗️ Project Structure

```bash
Customer-churn-predictor/
├── app/
│   ├── api.py           # FastAPI REST Backend
│   └── app.py           # Streamlit Web UI
├── src/
│   ├── Train.py         # Multi-model training pipeline
│   ├── Predict.py       # Centralized prediction logic
│   ├── Data_preprocessing.py # Data cleaning & NLP generation
│   └── utils/           # Logging & Config loaders
├── config.yaml          # Central configuration file
├── Telco.csv            # Raw dataset
├── requirements.txt     # Dependency list
└── Main.py              # Command Center (CLI)
```

---

## ⚙️ Tech Stack

*   **Logic**: Python 3.10+
*   **ML Framework**: Scikit-learn
*   **API**: FastAPI + Uvicorn
*   **UI**: Streamlit
*   **Data**: Pandas, NumPy
*   **Serialization**: Pickle

---

## 🔄 Model Comparison (Latest Results)

| Algorithm | F1-Score | ROC-AUC |
| :--- | :--- | :--- |
| **Logistic Regression** | 0.6073 | **0.8293** |
| **Random Forest** | **0.6109** | 0.8283 |

*The system currently saves the **Logistic Regression** model by default as it provides the best ROC-AUC (ranking ability).*

---

## ▶️ Getting Started

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Training & Evaluation
Train both models and save the best one:
```bash
python Main.py train
```

### 3. Launch the API
```bash
python Main.py api
```

### 4. Launch the UI Dashboard
```bash
python Main.py ui
```

---

## 🔍 API Usage

**POST** `/predict`
```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 1,
  "PhoneService": "No",
  "MultipleLines": "No phone service",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 29.85,
  "TotalCharges": 29.85,
  "Customer_Feedback": "Average service, nothing special"
}
```

---

## 📊 Performance Highlights
*   **ROC-AUC**: ~0.83 (Excellent ranking ability)
*   **Recall (Churn)**: ~73% (Identifies the majority of at-risk customers)
