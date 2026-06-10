import streamlit as st
import pandas as pd
import joblib
import os
import tab_action
import tab_dashboard

# 1. Set up the page
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

st.title("Customer Churn Predictor 🔮")
st.write("Analyze customer profiles using our XGBoost model to predict churn risk, understand key risk factors, and generate AI-driven retention emails.")

# 2. Load the model and preprocessor
@st.cache_resource
def load_components():
    # Dynamically resolve absolute paths so it works from anywhere
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'models', 'model.joblib')
    preprocessor_path = os.path.join(base_dir, 'models', 'preprocessor.joblib')
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor

try:
    model, preprocessor = load_components()
except Exception as e:
    st.error(f"Failed to load the model or preprocessor. Ensure 'model.joblib' and 'preprocessor.joblib' exist. Error: {e}")
    st.stop()

# 3. Load the Database
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    return pd.read_csv(file_path)

try:
    df = load_data()
except Exception as e:
    st.error(f"Could not load the database. Error: {e}")
    st.stop()

# Extract feature names for explainability and charts
numeric_features = preprocessor.transformers_[0][2]
categorical_features = preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out(preprocessor.transformers_[1][2])
all_feature_names = list(numeric_features) + list(categorical_features)

importances = model.feature_importances_
df_importances = pd.DataFrame({'Feature': all_feature_names, 'Importance (%)': importances * 100})
df_importances = df_importances.sort_values(by='Importance (%)', ascending=False).reset_index(drop=True)

# 4. Create UI Tabs
tab1, tab2 = st.tabs(["🎯 Action Center", "📊 Dashboard & Model"])

with tab1:
    tab_action.render(df, model, preprocessor, importances, all_feature_names)

with tab2:
    tab_dashboard.render(df, df_importances)