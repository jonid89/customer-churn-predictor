import streamlit as st
import pandas as pd
import numpy as np
import random
import os
from anthropic import Anthropic

def get_sanitized_api_key():
    """Fetch API key from Streamlit secrets or environment and sanitize it."""
    # Streamlit secrets usually take precedence in production
    key = None
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        pass
    
    # Fallback to .env for local development
    if not key:
        key = os.getenv("ANTHROPIC_API_KEY")
        
    if not key:
        raise ValueError("ANTHROPIC_API_KEY is missing. Please set it in .env or Streamlit Secrets.")
    
    # Remove whitespace, quotes, and hidden carriage returns
    return str(key).strip().replace('"', '').replace("'", "").replace('\r', '')

def render(df, model, preprocessor, importances, all_feature_names):
    # Initialize session state for randomizer
    if 'rand_trigger' not in st.session_state:
        st.session_state['rand_trigger'] = False

    high_threshold = 70
    medium_threshold = 50

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

    input_col, result_col = st.columns([0.55, 0.45], gap="xxlarge")

    with input_col:
        header_col, btn_col = st.columns([0.6, 0.4])
        with header_col:
            st.write("#### Manual Inputs")
        with btn_col:
            st.button("🎲 Randomize Customer", on_click=randomize_inputs, use_container_width=True)
            
        # Helper function for side-by-side form inputs
        def input_row(label, widget_type, *args, **kwargs):
            c1, c2 = st.columns([0.4, 0.6])
            c1.markdown(f"<div style='margin-top: 5px;'>{label}</div>", unsafe_allow_html=True)
            kwargs['label_visibility'] = "collapsed"
            return getattr(c2, widget_type)(label, *args, **kwargs)

        contract = input_row("Contract Type", "selectbox", df['Contract'].dropna().unique(), key='contract')
        internet = input_row("Internet Service", "selectbox", df['InternetService'].dropna().unique(), key='internet')
        security = input_row("Online Security", "selectbox", df['OnlineSecurity'].dropna().unique(), key='security')
        support = input_row("Tech Support", "selectbox", df['TechSupport'].dropna().unique(), key='support')
        movies = input_row("Streaming Movies", "selectbox", df['StreamingMovies'].dropna().unique(), key='movies')
        payment = input_row("Payment Method", "selectbox", df['PaymentMethod'].dropna().unique(), key='payment')
        tenure = input_row("Tenure (Months)", "slider", 0, 72, key='tenure')
        monthly = input_row("Monthly Charges ($)", "number_input", 0.0, 200.0, key='monthly')

    with result_col:
        
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

        st.write("#### Predicted Churn Risk:")
        if prob > high_threshold:
            st.metric(label="High Risk", value=f"{prob:.1f}%", delta="High Risk", delta_color="inverse", label_visibility="collapsed")
        elif prob > medium_threshold:
            st.metric(label="Medium Risk", value=f"{prob:.1f}%", delta="Medium Risk", delta_color="off", label_visibility="collapsed")
        else:
            st.metric(label="Low Risk", value=f"{prob:.1f}%", delta="Low Risk", delta_color="normal", label_visibility="collapsed")

        st.write("###### Top Risk Factors:")
        if prob > medium_threshold and len(top_reasons) > 0:
            for reason in top_reasons:
                st.write(f"- {reason.replace('_', ': ')}")
        elif prob <= medium_threshold:
            st.success("Customer profile looks stable.")

    st.write("---")
    st.write("### 🤖 AI Action")
    if prob > high_threshold:
        st.warning("⚠️ High Churn Risk Detected. Immediate retention action recommended.")
        if st.button("✉️ Generate Retention Email with Claude"):
            with st.spinner("Drafting personalized email using Claude..."):
                try:
                    api_key = get_sanitized_api_key()
                    client = Anthropic(api_key=api_key)
                    top_reasons_str = ', '.join([r.replace('_', ': ') for r in top_reasons])
                    
                    prompt = f"""
                    You are an expert customer retention specialist for a telecommunications company.
                    Write a highly personalized retention email to a customer who is at high risk of churning.
                    
                    Customer Profile:
                    - Tenure: {tenure} months
                    - Contract Type: {contract}
                    - Monthly Charges: ${monthly:.2f}
                    - Main reasons for churn risk: {top_reasons_str}
                    
                    Instructions:
                    1. Write an engaging subject line.
                    2. Be warm and empathetic.
                    3. Address their specific profile subtly (e.g., if they are month-to-month, offer an incentive to switch to a 1-year plan).
                    4. Offer a compelling retention incentive (discount, free upgrade, etc.).
                    5. Keep the email concise and professional. Sign off as "The Retention Team".
                    """
                    
                    response = client.messages.create(
                        model="claude-haiku-4-5-20251001",
                        max_tokens=400,
                        temperature=0.7,
                        system="You are a world-class telecom retention expert.",
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )
                    
                    st.info(response.content[0].text)
                except ValueError as e:
                    st.error(f"API Key Error: {e}")
                except Exception as e:
                    st.error(f"Error generating email from Anthropic: {e}")
    else:
        st.success("✅ Customer is at a safe retention level. No proactive retention email required.")