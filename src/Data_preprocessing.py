import pandas as pd

def load_data(path):
    df = pd.read_csv(path)

    # Clean TotalCharges column
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()

    # Encode target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    return df


def add_nlp_feature(df):
    def generate_feedback(row):
        if row["Churn"] == 1:
            return "Bad experience poor support high charges"
        else:
            return "Good service satisfied customer"

    df["Customer_Feedback"] = df.apply(generate_feedback, axis=1)

    return df