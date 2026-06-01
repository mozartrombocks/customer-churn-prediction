# Customer Churn Prediction & Retention Analytics Platform

## Overview

This project develops an end-to-end machine learning pipeline for predicting customer churn in a telecommunications company. The objective is to identify customers who are likely to discontinue service and provide actionable insights that can help improve customer retention strategies.

The project covers the complete data science workflow, including:

* Data cleaning and preprocessing
* Exploratory Data Analysis (EDA)
* Feature engineering
* Machine learning model development
* Model evaluation and comparison
* Feature importance analysis
* Interactive Streamlit deployment
* Business recommendations

The dataset contains customer demographic information, service subscriptions, billing behavior, contract details, and churn outcomes.

---

## Business Problem

Customer churn is one of the most important challenges faced by subscription-based businesses. Losing existing customers directly impacts revenue and profitability, while acquiring new customers is significantly more expensive than retaining current ones.

The goal of this project is to:

1. Predict which customers are at risk of churning.
2. Identify the most influential factors driving churn.
3. Provide data-driven recommendations for improving retention.

---

## Dataset

The project uses the IBM Telco Customer Churn dataset containing information for 7,043 customers.

### Target Variable

* **Churn Label**

  * Yes = Customer churned
  * No = Customer retained

### Features Include

* Customer demographics
* Contract information
* Billing information
* Internet services
* Payment methods
* Customer Lifetime Value (CLTV)
* Tenure history

Several columns that introduced data leakage were removed:

* Churn Value
* Churn Score
* Churn Reason

---

## Exploratory Data Analysis

Exploratory Data Analysis was performed to understand customer behavior and identify potential churn patterns.

### Analyses Included

* Churn distribution
* Contract type versus churn rate
* Tenure versus churn rate
* Monthly charges versus churn status

Generated figures are available in:

```text
reports/figures/
```

Key observations:

* Month-to-month contracts experience substantially higher churn rates.
* Customers with shorter tenure are significantly more likely to churn.
* Higher monthly charges are associated with elevated churn risk.
* Long-term contracts correlate with improved retention.

---

## Project Structure

```text
customer-churn-prediction/

├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── xgboost.pkl
│
├── notebooks/
│   └── churn_eda.ipynb
│
├── reports/
│   ├── customer_churn_report.pdf
│   ├── figures/
│   └── screenshots/
│
├── src/
│   ├── config.py
│   ├── data_prep.py
│   ├── eda.py
│   ├── train.py
│   ├── evaluate.py
│   ├── feature_importance.py
│   ├── predict.py
│   └── utils.py
│
├── README.md
├── requirements.txt
└── main.py
```

---

## Data Preprocessing

The preprocessing pipeline performs:

* Missing value handling
* Numerical scaling
* Categorical encoding using One-Hot Encoding
* Train / Validation / Test splitting
* Removal of leakage variables

The final datasets are stored in:

```text
data/processed/
```

---

## Machine Learning Models

Three classification models were implemented and compared:

### Logistic Regression

Serves as the baseline model.

Advantages:

* Interpretable
* Fast training
* Strong baseline performance

### Random Forest

Ensemble tree-based classifier.

Advantages:

* Captures nonlinear relationships
* Handles mixed feature types

### XGBoost

Gradient boosted decision trees.

Advantages:

* High predictive performance
* Robust handling of feature interactions
* Industry-standard solution for structured tabular data

---

## Validation Results

| Model               | Validation ROC-AUC |
| ------------------- | ------------------ |
| Logistic Regression | 0.8436             |
| Random Forest       | 0.8360             |
| XGBoost             | 0.8560             |

XGBoost achieved the strongest validation performance and was selected as the primary model.

---

## Test Performance

### XGBoost

* Test ROC-AUC ≈ 0.85

The model demonstrates strong discrimination between churned and retained customers and provides meaningful probability estimates for business decision-making.

---

## Feature Importance Analysis

Feature importance was extracted from the trained ensemble model.

The most influential variables include:

* Contract type
* Tenure Months
* Monthly Charges
* Total Charges
* Customer Lifetime Value (CLTV)
* Internet Service

Only the Top 15 features are displayed to improve interpretability.

Generated outputs:

```text
reports/figures/feature_importance.png
reports/figures/top_features.csv
```

---

## Streamlit Dashboard

An interactive dashboard was developed using Streamlit to allow users to generate real-time churn predictions.

Features:

* Customer input interface
* Churn probability estimation
* Risk categorization
* Retention recommendations

Launch locally:

```bash
streamlit run app/streamlit_app.py
```

---

## Business Insights

The analysis suggests that:

### High-Risk Customers

* Month-to-month contracts
* Short tenure
* High monthly charges

### Lower-Risk Customers

* Long-term contracts
* Established customer relationships
* Lower billing volatility

### Recommended Actions

1. Promote long-term contract adoption.
2. Focus retention campaigns on new customers.
3. Provide incentives for customers exhibiting elevated churn risk.
4. Use churn probability scores to prioritize customer outreach efforts.

---

## Technical Report

A complete technical report describing the methodology, experimentation, evaluation, deployment, and business implications is available at:

```text
reports/customer_churn_report.pdf
```

---

## Technologies Used

### Programming

* Python

### Data Science

* Pandas
* NumPy
* Scikit-Learn
* XGBoost

### Visualization

* Matplotlib

### Deployment

* Streamlit

### Model Persistence

* Joblib

---

## Future Improvements

Potential future enhancements include:

* Hyperparameter optimization using RandomizedSearchCV
* SHAP explainability analysis
* Cloud deployment (AWS)
* Automated retraining pipeline
* Model monitoring
* CI/CD integration using GitHub Actions

##
