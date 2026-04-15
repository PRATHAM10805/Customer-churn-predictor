# 📊 Customer Churn Prediction System

🚀 An end-to-end Machine Learning project to predict customer churn using **structured data + NLP-based customer feedback**, deployed with an interactive web interface.

---

## 🔗 Project Overview

Customer churn prediction helps businesses identify customers who are likely to leave.
This project builds a scalable ML pipeline that combines:

* 📈 Customer behavior data (tenure, charges)
* 🧾 Service-related data (contract, internet service)
* 💬 Customer feedback (NLP using TF-IDF)

---

## 🧠 Key Features

* ✅ End-to-end ML pipeline using Scikit-learn
* ✅ NLP integration for customer sentiment signals
* ✅ Hyperparameter tuning with GridSearchCV
* ✅ Modular and production-style code structure
* ✅ Real-time predictions via Streamlit UI
* ✅ Ready for API deployment (FastAPI + Uvicorn)

---

## 🏗️ Project Structure

```bash
customer-churn-prediction/
│
├── data/
│   └── telco.csv
│
├── models/
│   └── model.pkl
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│
├── app/
│   └── app.py
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

* **Python**
* **Scikit-learn**
* **NLP (TF-IDF Vectorization)**
* **Streamlit**
* *(Optional)* FastAPI & Uvicorn for backend deployment

---

## 🔄 Workflow

### 1. Data Preprocessing

* Cleaned missing values
* Encoded categorical variables using OneHotEncoder
* Generated NLP feature from customer feedback

### 2. Feature Engineering

* Numerical features → scaling
* Categorical features → one-hot encoding
* Text features → TF-IDF vectorization

### 3. Model Training

* Random Forest Classifier
* Hyperparameter tuning using GridSearchCV
* Evaluated using F1-score, precision, recall

### 4. Deployment

* Streamlit app for real-time prediction

---

## ▶️ How to Run

### 1. Clone repository

```bash
git clone <your-repo-link>
cd customer-churn-prediction
```

---

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Train the model

```bash
python -m src.train
```

---

### 4. Run the app

```bash
streamlit run app/app.py
```

---

## 🧪 Example Input

| Feature          | Value               |
| ---------------- | ------------------- |
| Tenure           | 5                   |
| Monthly Charges  | 95                  |
| Contract         | Month-to-month      |
| Internet Service | Fiber optic         |
| Feedback         | Very bad experience |

---

## 📊 Output

* ⚠️ High chance of churn
* ✅ Customer likely to stay

---

## 🔍 Highlights

* Combines **structured + unstructured (text) data**
* Uses **Pipeline + ColumnTransformer** (industry standard)
* Avoids data leakage
* Scalable and reusable design

---


