import os
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, f1_score

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

    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
    text_feature = 'Customer_Feedback'

    X = df[numeric_features + categorical_features + [text_feature]]
    y = df['Churn']

    logger.info("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['data']['test_size'], random_state=config['data']['random_state']
    )

    logger.info("Building preprocessor...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
            ('text', TfidfVectorizer(max_features=500), text_feature)
        ]
    )

    models_to_train = [
        {
            'name': 'Random Forest',
            'estimator': RandomForestClassifier(random_state=config['data']['random_state'], class_weight='balanced'),
            'params': {
                'model__n_estimators': config['training']['rf']['n_estimators'],
                'model__max_depth': config['training']['rf']['max_depth'],
                'model__min_samples_split': config['training']['rf']['min_samples_split'],
                'model__min_samples_leaf': config['training']['rf']['min_samples_leaf'],
            }
        },
        {
            'name': 'Logistic Regression',
            'estimator': LogisticRegression(random_state=config['data']['random_state'], class_weight='balanced', max_iter=1000),
            'params': {
                'model__C': config['training']['lr']['C'],
                'model__solver': config['training']['lr']['solver'],
            }
        }
    ]

    best_overall_score = -1
    best_overall_model = None
    results = []

    for model_info in models_to_train:
        logger.info(f"Training {model_info['name']}...")
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model_info['estimator'])
        ])

        grid = GridSearchCV(pipeline, model_info['params'], cv=config['training']['cv'], scoring='f1', n_jobs=-1)
        grid.fit(X_train, y_train)

        y_pred = grid.predict(X_test)
        y_prob = grid.predict_proba(X_test)[:, 1]
        
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        logger.info(f"{model_info['name']} - Best Params: {grid.best_params_}")
        logger.info(f"{model_info['name']} - F1 Score: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
        
        results.append({
            'name': model_info['name'],
            'f1': f1,
            'roc_auc': roc_auc,
            'estimator': grid.best_estimator_
        })

        if roc_auc > best_overall_score:
            best_overall_score = roc_auc
            best_overall_model = grid.best_estimator_

    logger.info("\n" + "="*30 + "\nMODEL COMPARISON\n" + "="*30)
    for res in results:
        logger.info(f"{res['name']}: F1={res['f1']:.4f}, ROC-AUC={res['roc_auc']:.4f}")
    
    logger.info(f"Saving best overall model (based on ROC-AUC)...")
    save_model(best_overall_model)

if __name__ == "__main__":
    train()