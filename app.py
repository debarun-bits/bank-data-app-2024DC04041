import streamlit as st
import pandas as pd
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.special import expit

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
)


st.set_page_config(
    page_title="Bank Data ML App",
    layout="wide"
)

st.title("Bank Data Classification App")


models = joblib.load("model/saved_models_2024DC04041.pkl")


uploaded_file = st.file_uploader(
    "Upload test CSV file",
    type=["csv"]
)

selected_model_name = st.selectbox(
    "Select Model",
    list(models.keys())
)


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=";")

    if "y" not in df.columns:
        st.error("Uploaded file must contain target column 'y'")
        st.stop()

    y_true = df["y"].map({"no": 0, "yes": 1})
    X_test = df.drop("y", axis=1)

    model = models[selected_model_name]


    y_pred = model.predict(X_test)


    try:
       
        y_prob = model.predict_proba(X_test)[:, 1]

    except Exception:
       
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X_test)
            y_prob = expit(scores)
        else:
            st.error("Selected model does not support probability prediction.")
            st.stop()


    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", round(accuracy_score(y_true, y_pred), 3))
    col1.metric("AUC", round(roc_auc_score(y_true, y_prob), 3))

    col2.metric("Precision", round(precision_score(y_true, y_pred), 3))
    col2.metric("Recall", round(recall_score(y_true, y_pred), 3))

    col3.metric("F1 Score", round(f1_score(y_true, y_pred), 3))
    col3.metric("MCC", round(matthews_corrcoef(y_true, y_pred), 3))


    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)
