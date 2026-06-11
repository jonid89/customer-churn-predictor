# Customer Churn Predictor 🔮

An interactive web application built with [![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B.svg?style=flat&logo=Streamlit&logoColor=white)](https://customer-churn-predictor-email-generator.streamlit.app/) that predicts telecom customer churn risk and helps you take proactive action. 

## 🌟 What the App Can Do
- **Predict Churn Risk:** Manually input customer details or generate random profiles to see the likelihood of a customer leaving.
- **Explainability:** Identifies the top risk factors specific to a customer's profile so you know *why* they are at risk.
- **AI Action Center:** If a high churn risk is detected, the app integrates with Claude (Anthropic AI) to automatically draft a highly personalized retention email offering targeted incentives.
- **Dashboard & Analytics:** Explore the underlying Telco Customer dataset and visualize the top features driving churn.

## 🧠 About the Model
The core of the predictor is a hyperparameter-tuned **XGBoost Classifier** coupled with a scikit-learn preprocessing pipeline (handling missing values, scaling numeric inputs, and one-hot encoding categorical features). 

**Model Performance:**
- **Overall Accuracy:** ~76%
- **Churn Recall (Caught):** ~78%
- **F1-Score:** ~77%

The model is trained on the classic Telco Customer Churn dataset and accounts for class imbalances dynamically using `scale_pos_weight`.

## 🚀 How to Use

### 1. Setup and Installation
Ensure you have Python installed, then install the required dependencies (it is recommended to use a virtual environment):
```bash
pip install streamlit pandas numpy scikit-learn xgboost anthropic altair joblib
```

### 2. Configure the API Key
To use the AI email generation feature, you need an Anthropic API key. Add it to a `.env` file in your project root or to Streamlit's secrets file (`.streamlit/secrets.toml`):
```toml
# .streamlit/secrets.toml
ANTHROPIC_API_KEY = "your-anthropic-api-key-here"
```

### 3. Run the Streamlit App
Launch the web app locally from the root of your project:
```bash
streamlit run streamlit/app.py
```

### 4. Retrain the Model (Optional)
If you want to experiment with different hyperparameters or update the dataset, you can trigger the training script. 

It will automatically process the data, perform a Grid Search for the best hyperparameters, and save the updated `model.joblib` and `preprocessor.joblib` into the `models/` directory:
```bash
python src/train_model.py
```