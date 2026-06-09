import streamlit as st
import pandas as pd
import joblib
import os

# 1. Set up the page
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.title("Customer Churn Predictor 🔮")
st.write("Upload a customer dataset to automatically predict who is at risk of churning.")

# 2. Load the model and preprocessor
# @st.cache_resource ensures we only load these files once, saving time on reruns
@st.cache_resource
def load_components():
    # Assuming the app is run from the project root where the joblib files were saved
    model = joblib.load('model.joblib')
    preprocessor = joblib.load('preprocessor.joblib')
    return model, preprocessor

try:
    model, preprocessor = load_components()
except Exception as e:
    st.error(f"Failed to load the model or preprocessor. Ensure 'model.joblib' and 'preprocessor.joblib' exist. Error: {e}")
    st.stop()

# 3. File uploader
uploaded_file = st.file_uploader("Upload your customer CSV file", type=["csv"])

if uploaded_file is not None:
    # Read data
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Data")
    st.dataframe(df.head())

    # 4. Process the data exactly as we did in training
    # Save customerID for the final output if it exists, then drop it from the features
    customer_ids = None
    if 'customerID' in df.columns:
        customer_ids = df['customerID']
        df = df.drop(columns=['customerID'])
        
    # Ensure TotalCharges is numeric
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # If the user accidentally uploaded the target column ('Churn'), drop it
    if 'Churn' in df.columns:
        df = df.drop(columns=['Churn'])

    if st.button("Predict Churn"):
        with st.spinner("Processing data and making predictions..."):
            # Apply the exact same transformations used in training
            X_processed = preprocessor.transform(df)
            
            # 5. Make predictions
            predictions = model.predict(X_processed)
            probabilities = model.predict_proba(X_processed)[:, 1] # Probability of Class 1 (Yes)

            # 6. Format and display results
            results_df = pd.DataFrame()
            if customer_ids is not None:
                results_df['customerID'] = customer_ids
            results_df['Churn Prediction'] = ['Yes' if p == 1 else 'No' for p in predictions]
            results_df['Churn Risk (%)'] = (probabilities * 100).round(2)
            
            st.write("### Prediction Results")
            # Highlight the highest risk customers in red
            st.dataframe(results_df.style.highlight_max(subset=['Churn Risk (%)'], axis=0, color='lightcoral'))