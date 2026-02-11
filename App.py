import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -----------------------------
# Load models safely (NO caching, NO top-level loading)
# -----------------------------
def load_models():
    models = {}

    with open("model/random_forest.pkl", "rb") as f:
        models["Random Forest"] = pickle.load(f)

    with open("model/logistic_regression.pkl", "rb") as f:
        models["Logistic Regression"] = pickle.load(f)

    with open("model/decision_tree.pkl", "rb") as f:
        models["Decision Tree"] = pickle.load(f)

    with open("model/knn.pkl", "rb") as f:
        models["KNN"] = pickle.load(f)

    with open("model/naive_bayes.pkl", "rb") as f:
        models["Naive Bayes"] = pickle.load(f)

    with open("model/xgboost.pkl", "rb") as f:
        models["XGBoost"] = pickle.load(f)

    with open("model/label_encoder.pkl", "rb") as f:
        models["Label Encoder"] = pickle.load(f)

    return models


# -----------------------------
# Main Streamlit App
# -----------------------------
def main():
    st.set_page_config(page_title="Dry Bean Classification", layout="wide")
    st.title("🌱 Dry Bean Classification App")

    st.write(
        """
        This application predicts the **class of Dry Beans**
        using multiple Machine Learn
