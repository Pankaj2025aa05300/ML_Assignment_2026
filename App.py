import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# -------------------------------------------------
# Page Setup
# -------------------------------------------------
st.set_page_config(page_title="Dry Bean ML App", layout="wide")
st.title("🌱 Dry Bean Classification - Model Evaluation")

# -------------------------------------------------
# Base Directory
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model", "saved_models")

# -------------------------------------------------
# Load Scaler
# -------------------------------------------------
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

if not os.path.exists(scaler_path):
    st.error("❌ scaler.pkl not found inside model/saved_models/")
    st.stop()

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

# -------------------------------------------------
# Load Label Encoder
# -------------------------------------------------
label_encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")

if not os.path.exists(label_encoder_path):
    st.error("❌ label_encoder.pkl not found inside model/saved_models/")
    st.stop()

with open(label_encoder_path, "rb") as f:
    label_encoder = pickle.load(f)

# -------------------------------------------------
# Model Paths
# -------------------------------------------------
model_options = {
    "Logistic Regression": os.path.join(MODEL_DIR, "Logistic_Regression.pkl"),
    "Decision Tree": os.path.join(MODEL_DIR, "Decision_Tree.pkl"),
    "KNN": os.path.join(MODEL_DIR, "KNN.pkl"),
    "Naive Bayes": os.path.join(MODEL_DIR, "Naive_Bayes.pkl"),
    "Random Forest": os.path.join(MODEL_DIR, "Random_Forest.pkl"),
    "XGBoost": os.path.join(MODEL_DIR, "XGBoost.pkl")
}

# -------------------------------------------------
# Sidebar - Model Selection
# -------------------------------------------------
st.sidebar.header("🔍 Model Selection")
selected_model_name = st.sidebar.selectbox(
    "Choose a Model",
    list(model_options.keys())
)

# -------------------------------------------------
# File Upload
# -------------------------------------------------
uploaded_file = st.file_uploader("📂 Upload Test Dataset (CSV only)", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    if "Class" not in df.columns:
        st.error("❌ Dataset must contain 'Class' column.")
        st.stop()

    # Separate features and target
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Ensure feature order matches training
    try:
        X = X[scaler.feature_names_in_]
    except:
        pass

    # Encode labels using SAME encoder as training
    try:
        y_encoded = label_encoder.transform(y)
    except Exception:
        st.error("❌ Label mismatch between training and uploaded dataset.")
        st.stop()

    # Apply saved scaler
    try:
        X_scaled = scaler.transform(X)
    except Exception:
        st.error("❌ Feature mismatch between training and uploaded dataset.")
        st.stop()

    # -------------------------------------------------
    # Load Selected Model
    # -------------------------------------------------
    model_path = model_options[selected_model_name]

    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {selected_model_name}")
        st.stop()

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Predictions
    y_pred = model.predict(X_scaled)

    # AUC Score (if available)
    try:
        y_proba = model.predict_proba(X_scaled)
        auc = roc_auc_score(y_encoded, y_proba, multi_class="ovr")
    except:
        auc = np.nan

    # -------------------------------------------------
    # Evaluation Metrics
    # -------------------------------------------------
    evaluation_dict = {
        "Accuracy": accuracy_score(y_encoded, y_pred),
        "Precision": precision_score(y_encoded, y_pred, average="weighted"),
        "Recall": recall_score(y_encoded, y_pred, average="weighted"),
        "F1 Score": f1_score(y_encoded, y_pred, average="weighted"),
        "AUC Score": auc,
        "MCC Score": matthews_corrcoef(y_encoded, y_pred)
    }

    st.subheader("📊 Evaluation Metrics")
    metrics_df = pd.DataFrame(evaluation_dict, index=[selected_model_name])
    st.dataframe(metrics_df)

    # -------------------------------------------------
    # Confusion Matrix
    # -------------------------------------------------
    st.subheader("📌 Confusion Matrix")

    cm = confusion_matrix(y_encoded, y_pred)

    fig, ax = plt.subplots()
    im = ax.imshow(cm)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    plt.colorbar(im)

    st.pyplot(fig)

    # -------------------------------------------------
    # Classification Report
    # -------------------------------------------------
    st.subheader("📝 Classification Report")

    report = classification_report(
        y_encoded,
        y_pred,
        target_names=label_encoder.classes_,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    st.success("✅ Model evaluation completed successfully!")

else:
    st.info("Please upload the test dataset CSV to evaluate the model.")
