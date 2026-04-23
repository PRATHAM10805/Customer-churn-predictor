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
    """Generate realistic customer feedback based on service attributes (NOT the churn label).

    Uses tenure, charges, and contract type as signals with added noise
    to simulate real-world feedback. This avoids data leakage that would
    occur if feedback were derived directly from the target variable.
    """
    import numpy as np

    logger.info("Generating synthetic customer feedback from service attributes...")
    rng = np.random.RandomState(42)

    positive_phrases = [
        "good service", "happy with support", "satisfied customer",
        "great value", "reliable connection", "easy billing",
        "friendly staff", "would recommend", "no complaints",
    ]
    negative_phrases = [
        "bad experience", "poor support", "high charges",
        "frequent outages", "slow internet", "difficult cancellation",
        "unhelpful staff", "long wait times", "billing issues",
    ]
    neutral_phrases = [
        "average service", "nothing special", "okay experience",
        "standard quality", "meets expectations",
    ]

    def generate_feedback(row):
        # Build a soft score from features (NOT from Churn label)
        score = 0.0
        if row["tenure"] < 12:
            score -= 1
        elif row["tenure"] > 48:
            score += 1

        if row["MonthlyCharges"] > 80:
            score -= 1
        elif row["MonthlyCharges"] < 40:
            score += 1

        if row["Contract"] == "Month-to-month":
            score -= 0.5
        elif row["Contract"] == "Two year":
            score += 0.5

        # Add noise so feedback isn't a deterministic predictor
        score += rng.normal(0, 1.0)

        if score > 0.5:
            phrases = rng.choice(positive_phrases, size=rng.randint(2, 4), replace=False)
        elif score < -0.5:
            phrases = rng.choice(negative_phrases, size=rng.randint(2, 4), replace=False)
        else:
            phrases = rng.choice(neutral_phrases, size=rng.randint(1, 3), replace=False)

        return " ".join(phrases)

    df["Customer_Feedback"] = df.apply(generate_feedback, axis=1)
    logger.info("NLP features added (noise-injected, no data leakage).")

    return df

