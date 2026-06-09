import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from data_prep import prepare_churn_data

def train_and_save_model(csv_path, model_path='model.joblib', preprocessor_path='preprocessor.joblib'):
    print("1. Preparing data...")
    # We assume 'Churn' is the target column. Update if your CSV uses a different name.
    X_train, X_test, y_train, y_test, preprocessor = prepare_churn_data(csv_path)

    print("\n2. Training HistGradientBoosting model...")
    model = HistGradientBoostingClassifier(class_weight='balanced', max_iter=100, random_state=42)
    model.fit(X_train, y_train)

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