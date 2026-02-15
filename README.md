# ML-Assignment

## Problem Statement
build a multiple machine learning classification model that predicts whether a credit card application will be approved or not based on the given applicant and application features.

## Dataset description
We are using a credit card approval dataset.
It has 690 instances and 16 features, target variable Approved (Approval status).<br>
Data distribution ~ 0.445 (This shows that data is fairly distributed)

It contains a mix of demographic, financial, employment, and credit-related attributes used to predict whether an application will be approved. The dataset includes both numerical and categorical features, making it suitable for evaluating multiple machine learning classification algorithms.

## Models used
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. aive Bayes Classifier - Gaussian
5. Ensemble Model - Random Forest
6. Ensemble Model - XGBoost

## Model Performance

| Model Name          | Accuracy | AUC  | Precision | Recall | F1 Score | MCC  |
|--------------------|----------|------|-----------|--------|----------|------|
| Logistic Regression | 0.83     | 0.90 | 0.84      | 0.83   | 0.83     | 0.67 |
| Decision Tree       | 0.75     | 0.75 | 0.77      | 0.71   | 0.74     | 0.50 |
| KNN                 | 0.86     | 0.90 | 0.90      | 0.83   | 0.86     | 0.73 |
| Naive Bayes         | 0.75     | 0.79 | 0.77      | 0.73   | 0.75     | 0.50 |
| Random Forest       | 0.86     | 0.90 | 0.88      | 0.84   | 0.86     | 0.73 |
| XGBoost             | 0.81     | 0.88 | 0.82      | 0.80   | 0.81     | 0.62 |

## Model observations

| Model Name          | Observation |
|---------------------|-------------|
| Logistic Regression | Balanced performance with strong AUC (0.90) and stable precision–recall. A reliable and interpretable baseline model. |
| Decision Tree       | Lower recall and MCC indicate weaker generalization. Likely overfitting compared to ensemble methods. |
| KNN                 | High accuracy and the best precision (0.90) show excellent classification with minimal false positives. One of the top-performing models. |
| Naive Bayes         | Moderate and consistent across metrics but limited by feature independence assumption. Performance is comparable to Decision Tree. |
| Random Forest       | Top performer with the best overall balance of accuracy, recall, F1, and MCC. Most robust and well-generalized model. |
| XGBoost             | Strong AUC and balanced metrics with good predictive power. Performance can improve further with hyperparameter tuning. |
