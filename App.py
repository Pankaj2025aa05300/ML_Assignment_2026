import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef
)

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(page_title="Dry Bean Model Comparison", layout="wide")
st.title("🌱 Dry Bean Classification - Model Comparison")

# -------------------------------
# Base Directory
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------------
# Load Scaler
# -------------------------------
scaler_path = os.path.join(BASE_DIR, "model", "saved_models", "scaler.pkl")

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

# -------------------------------
# Model Dictionary
# -------------------------------
model_options = {
    "Logistic Regression": os.path.join(BASE_DIR, "model", "saved_models", "Logistic_Regression.pkl"),
    "Decision Tree": os.path.join(BASE_DIR, "model", "saved_models", "Decision_Tree.pkl"),
    "KNN": os.path.join(BASE_DIR, "model", "saved_models", "KNN.pkl"),
    "Naive Bayes": os.path.join(BASE_DIR, "model", "saved_models", "Naive_Bayes.pkl"),
    "Random Forest": os.path.join(BASE_DIR, "model", "saved_models", "Random_Forest.pkl"),
    "XGBoost": os.path.join(BASE_DIR, "model", "saved_models", "XGBoost.pkl")
}

# -------------------------------
# Sidebar - Model Selection
# -------------------------------
st.sidebar.header("Select Model")
selected_model_name = st.sidebar.selectbox(
    "Choose a model:",
    list(model_options.keys())
)

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload Dry Bean CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    if "Class" not in df.columns:
        st.error("❌ CSV must contain 'Class' column.")
        st.stop()

    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Encode target consistently
    y_encoded, class_names = pd.factorize(y)

    # Scale features
    X_scaled = scaler.transform(X)

    # Load Selected Model
    model_path = model_options[selected_model_name]

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Predictions
    y_pred = model.predict(X_scaled)

    # Some models may not support predict_proba
    try:
        y_proba = model.predict_proba(X_scaled)
        auc = roc_auc_score(y_encoded, y_proba, multi_class='ovr')
    except:
        auc = np.nan

    # Metrics
    accuracy = accuracy_score(y_encoded, y_pred)
    precision = precision_score(y_encoded, y_pred, average='weighted')
    recall = recall_score(y_encoded, y_pred, average='weighted')
    f1 = f1_score(y_encoded, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_encoded, y_pred)

    # -------------------------------
    # Display Results
    # -------------------------------
    st.subheader(f"📊 Performance of {selected_model_name}")

    evaluation_dict = {
        "AUC Score": auc,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC Score": mcc
    }

    results_df = pd.DataFrame(evaluation_dict, index=[selected_model_name])
    st.dataframe(results_df)

    st.success("✅ Model evaluation completed successfully!")

else:
    st.info("Please upload a CSV file to evaluate the model.")
