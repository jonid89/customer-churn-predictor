import joblib
import time
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from data_prep import prepare_churn_data

def train_and_save_model(csv_path, model_path='model.joblib', preprocessor_path='preprocessor.joblib'):
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

if __name__ == "__main__":
    # Run the training script using the Telco Customer Churn database
    train_and_save_model('./data/WA_Fn-UseC_-Telco-Customer-Churn.csv')