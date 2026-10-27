import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import xgboost  # registers XGBClassifier so joblib can load models/model.joblib

# Same gates as streamlit/tab_action.py
HIGH_THRESHOLD = 70
MEDIUM_THRESHOLD = 50


def project_paths():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return {
        "csv": os.path.join(base_dir, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv"),
        "model": os.path.join(base_dir, "models", "model.joblib"),
        "preprocessor": os.path.join(base_dir, "models", "preprocessor.joblib"),
    }


def risk_band(probability):
    if probability > HIGH_THRESHOLD:
        return "high"
    if probability > MEDIUM_THRESHOLD:
        return "medium"
    return "low"


def score_customers(csv_path, model_path, preprocessor_path):
    for path in (csv_path, model_path, preprocessor_path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    df = pd.read_csv(csv_path)
    if "customerID" in df.columns:
        customer_ids = df["customerID"]
    else:
        customer_ids = pd.Series([None] * len(df))

    if "Churn" in df.columns:
        already_churned = df["Churn"].astype(str).str.strip().str.lower().eq("yes")
    else:
        already_churned = pd.Series([False] * len(df))

    features = df.drop(columns=[c for c in ("customerID", "Churn") if c in df.columns])
    if "TotalCharges" in features.columns:
        features["TotalCharges"] = pd.to_numeric(features["TotalCharges"], errors="coerce")

    processed = preprocessor.transform(features)
    if hasattr(processed, "toarray"):
        processed = processed.toarray()
    processed = np.asarray(processed)

    numeric_features = list(preprocessor.transformers_[0][2])
    categorical_features = list(
        preprocessor.transformers_[1][1]
        .named_steps["onehot"]
        .get_feature_names_out(preprocessor.transformers_[1][2])
    )
    feature_names = numeric_features + categorical_features
    importances = np.asarray(model.feature_importances_)
    if processed.shape[1] != len(feature_names) or processed.shape[1] != len(importances):
        raise RuntimeError(
            "Transformed columns do not match the saved XGBoost model. Retrain with src/train_model.py."
        )

    probabilities = model.predict_proba(processed)[:, 1] * 100
    contributions = processed * importances
    top_index = np.argsort(contributions, axis=1)[:, -3:][:, ::-1]

    records = []
    for i in range(len(features)):
        probability = float(probabilities[i])
        reasons = []
        for feature_index in top_index[i]:
            if contributions[i, feature_index] > 0:
                reasons.append(str(feature_names[feature_index]).replace("_", ": "))

        tenure = features.iloc[i]["tenure"] if "tenure" in features.columns else None
        contract = features.iloc[i]["Contract"] if "Contract" in features.columns else None
        monthly = features.iloc[i]["MonthlyCharges"] if "MonthlyCharges" in features.columns else None

        records.append(
            {
                "customerID": None if pd.isna(customer_ids.iloc[i]) else str(customer_ids.iloc[i]),
                "already_churned": bool(already_churned.iloc[i]),
                "tenure": None if pd.isna(tenure) else int(tenure),
                "contract": None if pd.isna(contract) else str(contract),
                "monthly_charges": None if pd.isna(monthly) else float(monthly),
                "churn_probability": probability,
                "risk_band": risk_band(probability),
                "top_reasons": reasons,
                "top_reasons_str": ", ".join(reasons),
            }
        )
    return records


def main():
    paths = project_paths()
    try:
        records = score_customers(paths["csv"], paths["model"], paths["preprocessor"])
    except FileNotFoundError as exc:
        print(f"Missing file: {exc}", file=sys.stderr)
        print("Train the artifacts with: python src/train_model.py", file=sys.stderr)
        sys.exit(1)

    outreach = sum(
        1
        for row in records
        if row["churn_probability"] > HIGH_THRESHOLD and not row["already_churned"]
    )
    print(
        f"Scored {len(records)} customers. High-risk outreach candidates: {outreach}.",
        file=sys.stderr,
    )
    json.dump(records, sys.stdout)
    sys.stdout.write("\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
