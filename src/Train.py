import os
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from Data_preprocessing import load_data, add_nlp_feature


def save_model(model, path="models/model.pkl"):
    # Create folder automatically
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(model, f)

    print(f"✅ Model saved at {path}")


def train():

    print("🔹 Loading data...")
    df = load_data("C:/Users/mehta/OneDrive/Desktop/Customer-churn-prediction/data/Telco.csv")   # ✅ relative path (better)

    print("🔹 Adding NLP feature...")
    df = add_nlp_feature(df)

    numeric_features = ['tenure', 'MonthlyCharges']
    categorical_features = ['Contract', 'InternetService']
    text_feature = 'Customer_Feedback'

    X = df[numeric_features + categorical_features + [text_feature]]
    y = df['Churn']

    print("🔹 Building pipeline...")

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
            ('text', TfidfVectorizer(max_features=500), text_feature)
        ]
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        'model__n_estimators': [100, 200],
        'model__max_depth': [5, 10]
    }

    print("🔹 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("🔹 Training model (this may take time)...")
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1)

    grid.fit(X_train, y_train)

    print("🔹 Evaluating model...")
    y_pred = grid.predict(X_test)

    print("Best Params:", grid.best_params_)
    print(classification_report(y_test, y_pred))

    print("🔹 Saving model...")
    save_model(grid.best_estimator_)


if __name__ == "__main__":
    train()