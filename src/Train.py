import os
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from src.Data_preprocessing import load_data, add_nlp_feature
from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger(__name__)
config = load_config()

def save_model(model, path=config['model']['path']):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved successfully at {path}")

def train():
    logger.info("Starting model training pipeline...")
    
    data_path = config['data']['raw_path']
    if not os.path.exists(data_path):
        logger.error(f"Dataset not found at {data_path}")
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    logger.info(f"Loading data from {data_path}...")
    df = load_data(data_path)

    logger.info("Adding NLP features...")
    df = add_nlp_feature(df)

    numeric_features = ['tenure', 'MonthlyCharges']
    categorical_features = ['Contract', 'InternetService']
    text_feature = 'Customer_Feedback'

    X = df[numeric_features + categorical_features + [text_feature]]
    y = df['Churn']

    logger.info("Building preprocessor and pipeline...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
            ('text', TfidfVectorizer(max_features=500), text_feature)
        ]
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(random_state=config['data']['random_state']))
    ])

    param_grid = {
        'model__n_estimators': config['training']['n_estimators'],
        'model__max_depth': config['training']['max_depth']
    }

    logger.info("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['data']['test_size'], random_state=config['data']['random_state']
    )

    logger.info(f"Commencing GridSearchCV with param_grid: {param_grid}")
    grid = GridSearchCV(pipeline, param_grid, cv=config['training']['cv'], scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)

    logger.info("Evaluating model performance...")
    y_pred = grid.predict(X_test)
    report = classification_report(y_test, y_pred)
    logger.info(f"Best Params: {grid.best_params_}")
    logger.info(f"\n{report}")

    logger.info("Saving best estimator...")
    save_model(grid.best_estimator_)

if __name__ == "__main__":
    train()