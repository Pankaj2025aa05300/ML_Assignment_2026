import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

st.title("Dry Bean Classification App")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    if 'Class' in df.columns:
        X = df.drop('Class', axis=1)
        y = df['Class']
        le = LabelEncoder()
        y = le.fit_transform(y)
    else:
        X = df
        y = None

    model = RandomForestClassifier()
    model.fit(X, y) if y is not None else model.fit(X, np.zeros(len(X)))

    y_pred = model.predict(X)

    if y is not None:
        st.subheader("Classification Report")
        st.text(classification_report(y, y_pred))
    else:
        st.subheader("Predictions")
        st.write(pd
