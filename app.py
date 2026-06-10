import streamlit as st
import pandas as pd
import joblib
import os
import altair as alt
import numpy as np
import random

# 1. Set up the page
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.title("Customer Churn Predictor 🔮")
st.write("Analyze customer profiles using our XGBoost model to predict churn risk, understand key risk factors, and generate AI-driven retention emails.")

# 2. Load the model and preprocessor
@st.cache_resource
def load_components():
    model = joblib.load('model.joblib')
    preprocessor = joblib.load('preprocessor.joblib')
    return model, preprocessor

try:
    model, preprocessor = load_components()
except Exception as e:
    st.error(f"Failed to load the model or preprocessor. Ensure 'model.joblib' and 'preprocessor.joblib' exist. Error: {e}")
    st.stop()

# 3. Load the Database
@st.cache_data
def load_data():
    # Dynamically find the data file whether running from root or src
    file_path = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    if not os.path.exists(file_path):
        file_path = '../data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
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
    st.write("### 1. Customer Profile")
    
    # Initialize session state for randomizer
    if 'rand_trigger' not in st.session_state:
        st.session_state['rand_trigger'] = False

    def randomize_inputs():
        st.session_state['contract'] = random.choice(df['Contract'].dropna().unique())
        st.session_state['internet'] = random.choice(df['InternetService'].dropna().unique())
        st.session_state['security'] = random.choice(df['OnlineSecurity'].dropna().unique())
        st.session_state['support'] = random.choice(df['TechSupport'].dropna().unique())
        st.session_state['movies'] = random.choice(df['StreamingMovies'].dropna().unique())
        st.session_state['payment'] = random.choice(df['PaymentMethod'].dropna().unique())
        st.session_state['tenure'] = random.randint(1, 72)
        st.session_state['monthly'] = round(random.uniform(20.0, 110.0), 2)

    # Defaults
    for key, default in [('contract', 'Month-to-month'), ('internet', 'Fiber optic'), 
                         ('security', 'No'), ('support', 'No'), ('movies', 'No'), 
                         ('payment', 'Electronic check'), ('tenure', 12), ('monthly', 75.0)]:
        if key not in st.session_state:
            st.session_state[key] = default

    st.button("🎲 Randomize Customer", on_click=randomize_inputs)

    input_col, result_col = st.columns([1, 1], gap="large")

    with input_col:
        st.write("#### Manual Inputs")
        contract = st.selectbox("Contract Type", df['Contract'].dropna().unique(), key='contract')
        internet = st.selectbox("Internet Service", df['InternetService'].dropna().unique(), key='internet')
        security = st.selectbox("Online Security", df['OnlineSecurity'].dropna().unique(), key='security')
        support = st.selectbox("Tech Support", df['TechSupport'].dropna().unique(), key='support')
        movies = st.selectbox("Streaming Movies", df['StreamingMovies'].dropna().unique(), key='movies')
        payment = st.selectbox("Payment Method", df['PaymentMethod'].dropna().unique(), key='payment')
        tenure = st.slider("Tenure (Months)", 0, 72, key='tenure')
        monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, key='monthly')

    with result_col:
        st.write("#### 2. Risk Analysis")
        
        # Construct full dictionary for pipeline
        input_dict = {
            'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
            'tenure': tenure, 'PhoneService': 'Yes', 'MultipleLines': 'No', 
            'InternetService': internet, 'OnlineSecurity': security, 
            'OnlineBackup': 'No', 'DeviceProtection': 'No', 'TechSupport': support, 
            'StreamingTV': 'No', 'StreamingMovies': movies, 'Contract': contract, 
            'PaperlessBilling': 'Yes', 'PaymentMethod': payment, 
            'MonthlyCharges': monthly, 'TotalCharges': monthly * tenure if tenure > 0 else monthly
        }

        input_df = pd.DataFrame([input_dict])
        X_processed = preprocessor.transform(input_df)
        prob = model.predict_proba(X_processed)[0][1] * 100

        # Explainability Calculation
        contributions = X_processed[0] * importances
        top_indices = np.argsort(contributions)[::-1][:3]
        top_reasons = [all_feature_names[i] for i in top_indices if contributions[i] > 0]

        if prob > 70:
            st.metric(label="Predicted Churn Risk", value=f"{prob:.1f}%", delta="High Risk", delta_color="inverse")
        elif prob > 40:
            st.metric(label="Predicted Churn Risk", value=f"{prob:.1f}%", delta="Medium Risk", delta_color="off")
        else:
            st.metric(label="Predicted Churn Risk", value=f"{prob:.1f}%", delta="Low Risk", delta_color="normal")

        st.write("##### Top Risk Factors:")
        if prob > 40 and len(top_reasons) > 0:
            for reason in top_reasons:
                st.write(f"- {reason.replace('_', ': ')}")
        elif prob <= 40:
            st.success("Customer profile looks stable.")

    st.write("---")
    st.write("### 3. Action Center")
    if prob > 70:
        st.warning("⚠️ High Churn Risk Detected. Immediate retention action recommended.")
        if st.button("✉️ Generate Retention Email with Claude", type="primary"):
            with st.spinner("Drafting personalized email..."):
                st.info(f"Subject: Let's upgrade your experience!\n\nHi there,\n\nWe noticed you've been with us for {tenure} months on a {contract} plan. We value your business and would love to offer you a complimentary upgrade to our premium support.\n\n*[Anthropic API Integration coming next!]*")
    else:
        st.success("✅ Customer is at a safe retention level. No proactive retention email required.")

with tab2:
    st.write("### Telco Customer Database")
    st.dataframe(df.head(100), height=150)

    st.write("---")
    st.write("### Model Performance (XGBoost)")
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Overall Accuracy", value="76%")
    col2.metric(label="Churn Recall (Caught)", value="78%") 
    col3.metric(label="F1-Score", value="77%")

    st.write("---")
    st.write("### Top Impactful Features")
    chart = alt.Chart(df_importances.head(10)).mark_bar().encode(
        x=alt.X('Importance (%):Q', title='Importance (%)'),
        y=alt.Y('Feature:N', sort='-x', title='Feature'),
        color=alt.Color('Importance (%):Q', scale=alt.Scale(scheme='blues'), legend=None),
        tooltip=['Feature', 'Importance (%)']
    ).interactive()
    st.altair_chart(chart, use_container_width=True)