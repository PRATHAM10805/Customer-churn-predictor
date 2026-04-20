import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_data(path):
    logger.info(f"Reading dataset from {path}")
    df = pd.read_csv(path)

    # Clean TotalCharges column
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()

    # Encode target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    logger.info(f"Dataset loaded. Shape: {df.shape}")

    return df


def add_nlp_feature(df):
    logger.info("Generating NLP features from churn labels...")
    def generate_feedback(row):
        if row["Churn"] == 1:
            return "Bad experience poor support high charges"
        else:
            return "Good service satisfied customer"

    df["Customer_Feedback"] = df.apply(generate_feedback, axis=1)
    logger.info("NLP features added.")

    return df