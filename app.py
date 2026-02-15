import streamlit as st
import pandas as pd
import joblib
import os

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)


def calculate_metrics(y_test,y_pred):
    pass

st.title("Credit Card approval")


with open("test_data.csv", "rb") as file:
    st.download_button(
        label="Download CSV",
        data=file,
        file_name="test_data.csv",
        mime="text/csv"
    )



# Create form
with st.form("predict"):
    
    # Upload CSV
    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )
    
    # Model Dropdown
    model_choice = st.selectbox(
        "Select Model",
        ["Logistic Regression", "Decision Tree Classifier", "K-Nearest Neighbor Classifier", 
        "Naive Bayes Classifier", "Ensemble Model - Random Forest", "Ensemble Model - XGBoost"]

    )
    
    # Submit button
    submit = st.form_submit_button("Submit")

def compute(uploaded_file, model_name):
    df = pd.read_csv(uploaded_file)
    X_test_new = df.drop("Approved", axis=1)
    y_test = df["Approved"]
    loaded_model = joblib.load(os.path.join("models",f"{model_name}.pkl"))
    
    y_pred = loaded_model.predict(X_test_new)
    y_proba = loaded_model.predict_proba(X_test_new)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    metrics = {
    "Metric": ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"],
    "Value": [
        round(accuracy, 4),
        round(auc, 4),
        round(precision, 4),
        round(recall, 4),
        round(f1, 4),
        round(mcc, 4)
        ]
        }

    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics)
    
    df["Predictions"] = y_pred
    st.subheader("Predictions")
    st.dataframe(df)

    st.subheader("Model Evaluation Metrics")
    st.dataframe(metrics_df, use_container_width=True)
    
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()

    im = ax.imshow(cm)

    # Add labels
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")

    # Show numbers inside matrix
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    # Tick labels
    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"])

    st.pyplot(fig)



model_pkl_map = {
    "Logistic Regression":"Logistic_Regression", 
    "Decision Tree Classifier":"Decision_Tree", 
    "K-Nearest Neighbor Classifier":"KNN", 
    "Naive Bayes Classifier":"Naive_Bayes", 
    "Ensemble Model - Random Forest":"Random_Forest", 
    "Ensemble Model - XGBoost":"XGBoost"}

# After form submission
if submit:
    if uploaded_file is not None:
        st.success("File uploaded successfully!")
        st.write("Selected Model:", model_choice)
        compute(uploaded_file,model_pkl_map[model_choice])
    else:
        st.error("Please upload a CSV file.")