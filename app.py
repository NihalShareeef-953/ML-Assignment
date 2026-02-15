import streamlit as st
import pandas as pd
import joblib
import os
import io

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)


# Page config and styling
st.set_page_config(page_title="Credit Card Approval", layout="wide")

st.markdown(
    """
    <style>
    /* Background gradient */
    .stApp {
      background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
      color: #0f172a;
    }
    /* Card style for metrics */
    .metric-card {
      background: linear-gradient(180deg, rgba(255,255,255,0.9), rgba(255,255,255,0.75));
      padding: 14px;
      border-radius: 12px;
      box-shadow: 0 4px 18px rgba(12, 44, 97, 0.08);
      text-align: center;
      margin-bottom: 10px;
    }
    .large-title { font-size:28px; font-weight:700; color:#072146; }
    .subtitle { color:#334155; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown("<div class='large-title'>💳 Credit Card Approval</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload test data, choose a model, and view rich evaluation metrics.</div>", unsafe_allow_html=True)

# Sidebar for controls & info
st.sidebar.header("Controls")
st.sidebar.write("Upload your CSV and select a trained model to evaluate.")
st.sidebar.markdown("---")
with open("test_data.csv", "rb") as file:
    st.sidebar.download_button("Download sample test_data.csv", data=file, file_name="test_data.csv", mime="text/csv")


model_pkl_map = {
    "Logistic Regression":"Logistic_Regression", 
    "Decision Tree Classifier":"Decision_Tree", 
    "K-Nearest Neighbor Classifier":"KNN", 
    "Naive Bayes Classifier":"Naive_Bayes", 
    "Ensemble Model - Random Forest":"Random_Forest", 
    "Ensemble Model - XGBoost":"XGBoost"}


def compute(uploaded_file, model_name):
    df = pd.read_csv(uploaded_file)
    X_test_new = df.drop("Approved", axis=1)
    y_test = df["Approved"]
    loaded_model = joblib.load(os.path.join("models",f"{model_name}.pkl"))

    y_pred = loaded_model.predict(X_test_new)
    # Some classifiers may not support predict_proba; handle gracefully
    try:
        y_proba = loaded_model.predict_proba(X_test_new)[:, 1]
    except Exception:
        y_proba = np.zeros_like(y_pred, dtype=float)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.0
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    metrics = {
        "Metric": ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"],
        "Value": [
            round(accuracy, 4),
            round(auc, 4),
            round(precision, 4),
            round(recall, 4),
            round(f1, 4),
            round(mcc, 4),
        ]
    }

    metrics_df = pd.DataFrame(metrics)

    df["Predictions"] = y_pred

    # Confusion matrix figure (seaborn heatmap)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')

    return df, metrics_df, fig


# Create form
with st.form("predict"):
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    model_choice = st.selectbox(
        "Select Model",
        list(model_pkl_map.keys())
    )
    submit = st.form_submit_button("Run Evaluation")


if submit:
    if uploaded_file is not None:
        st.success("Processing file...")
        st.write("**Selected Model:**", model_choice)
        try:
            df_res, metrics_df, cm_fig = compute(uploaded_file, model_pkl_map[model_choice])

            # Top-level layout: metrics cards + charts + table
            col1, col2 = st.columns([1, 2])

            # Metric cards
            with col1:
                for i, row in metrics_df.iterrows():
                    st.markdown(f"<div class='metric-card'><strong>{row['Metric']}</strong><div style='font-size:20px; color:#0b69ff'>{row['Value']}</div></div>", unsafe_allow_html=True)

            # Confusion matrix and table
            with col2:
                st.pyplot(cm_fig)
                st.subheader("Predictions")
                st.dataframe(df_res, use_container_width=True)

            # Provide download for predictions
            csv_buf = io.StringIO()
            df_res.to_csv(csv_buf, index=False)
            csv_buf.seek(0)
            st.download_button("Download predictions CSV", data=csv_buf.getvalue(), file_name="predictions.csv", mime="text/csv")

        except FileNotFoundError as e:
            st.error(f"Model file not found: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.error("Please upload a CSV file.")
