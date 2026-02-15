# ML-Assignment

### Problem Statment
build a multiple machine learning classification model that predicts whether a credit card application will be approved or not based on the given applicant and application features.

### Dataset description
We are using a credit card approval dataset.
It has 690 intances and 16 features, target variable Approved (Approval status) - 0 or 1
Target variable is binary and distribution ~ 0.445

It contains a mix of demographic, financial, employment, and credit-related attributes used to predict whether an application will be approved. The dataset includes both numerical and categorical features, making it suitable for evaluating multiple machine learning classification algorithms.


### Models used
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier N
4. aive Bayes Classifier - Gaussian
5. Ensemble Model - Random Forest
6. Ensemble Model - XGBoost

## 📊 Model Performance

| Model               | Accuracy | AUC  | Precision | Recall | F1 Score | MCC  |
|--------------------|----------|------|-----------|--------|----------|------|
| Logistic Regression | 0.83     | 0.90 | 0.84      | 0.83   | 0.83     | 0.67 |
| Decision Tree       | 0.75     | 0.75 | 0.77      | 0.71   | 0.74     | 0.50 |
| KNN                 | 0.86     | 0.90 | 0.90      | 0.83   | 0.86     | 0.73 |
| Naive Bayes         | 0.75     | 0.79 | 0.77      | 0.73   | 0.75     | 0.50 |
| Random Forest       | 0.86     | 0.90 | 0.88      | 0.84   | 0.86     | 0.73 |
| XGBoost             | 0.81     | 0.88 | 0.82      | 0.80   | 0.81     | 0.62 |

