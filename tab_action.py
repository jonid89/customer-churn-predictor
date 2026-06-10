import streamlit as st
import pandas as pd
import numpy as np
import random

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

    input_col, result_col = st.columns([0.6, 0.4], gap="xlarge")

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
    st.write("### Action Center")
    if prob > 70:
        st.warning("⚠️ High Churn Risk Detected. Immediate retention action recommended.")
        if st.button("✉️ Generate Retention Email with Claude", type="primary"):
            with st.spinner("Drafting personalized email..."):
                st.info(f"Subject: Let's upgrade your experience!\n\nHi there,\n\nWe noticed you've been with us for {tenure} months on a {contract} plan. We value your business and would love to offer you a complimentary upgrade to our premium support.\n\n*[Anthropic API Integration coming next!]*")
    else:
        st.success("✅ Customer is at a safe retention level. No proactive retention email required.")