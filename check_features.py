import joblib
import pandas as pd

# 1. Load the trained model and preprocessor
model = joblib.load('model.joblib')
preprocessor = joblib.load('preprocessor.joblib')

# 2. Extract the feature names from the pipeline
# Get numeric column names
numeric_features = preprocessor.transformers_[0][2]
# Get one-hot encoded categorical column names
categorical_features = preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out(preprocessor.transformers_[1][2])

all_feature_names = list(numeric_features) + list(categorical_features)

# 3. Map to the XGBoost feature importances
importances = model.feature_importances_
df_importances = pd.DataFrame({'Feature': all_feature_names, 'Importance (%)': importances * 100})
df_importances = df_importances.sort_values(by='Importance (%)', ascending=False).reset_index(drop=True)

print("Top 10 Most Impactful Features:")
print(df_importances.head(10))