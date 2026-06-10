import joblib
import time
import os
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from data_prep import prepare_churn_data

def train_and_save_model(csv_path, model_path=None, preprocessor_path=None):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if model_path is None:
        model_path = os.path.join(base_dir, 'models', 'model.joblib')
    if preprocessor_path is None:
        preprocessor_path = os.path.join(base_dir, 'models', 'preprocessor.joblib')
        
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    script_start_time = time.time()
    print("1. Preparing data...")
    # We assume 'Churn' is the target column. Update if your CSV uses a different name.
    X_train, X_test, y_train, y_test, preprocessor = prepare_churn_data(csv_path)

    print("\n2. Training and Tuning XGBoost model (this may take a minute)...")
    # Calculate ratio of negative to positive class for scale_pos_weight (handles imbalance)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss')
    
    # Define the Grid Search to test different hyperparameter combinations
    param_grid = {
        'n_estimators': [50, 100, 200, 300, 500],
        'max_depth': [3, 4, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }
    
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
    
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    end_time = time.time()
    
    model = grid_search.best_estimator_
    print(f"\nBest parameters found: {grid_search.best_params_}")
    print(f"Hyperparameter tuning took {end_time - start_time:.2f} seconds.")

    print("\n3. Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\n4. Saving model and preprocessor...")
    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Done! Saved to {model_path} and {preprocessor_path}")
    script_end_time = time.time()
    print(f"\nTotal script execution time: {script_end_time - script_start_time:.2f} seconds.")

if __name__ == "__main__":
    # Dynamically resolve the path to the dataset
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_file_path = os.path.join(base_dir, 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    train_and_save_model(csv_file_path)