
# Anti Money Laundering With Machine Learning

## Project Overview

This project implements and evaluates a suite of machine learning models to detect fraudulent financial transactions. Using the **PaySim dataset**, which simulates mobile money transactions, we explore both *supervised* and *unsupervised* learning techniques to identify potentially fraudulent activities.

The primary goal is to compare the effectiveness of different algorithms, from traditional classifiers like **Logistic Regression** and **Random Forest** to specialized anomaly detection methods like **Isolation Forest** and **One-Class SVM**.

## Workflow & Methodology

The project follows a structured machine learning workflow:

1. **Data Loading & EDA**: The script begins by unzipping and loading the dataset. Initial analysis reveals that fraudulent activities are exclusively linked to `CASH_OUT` and `TRANSFER` transaction types, so the project focuses on these.

2. **Feature Engineering**: To improve model performance, new features are engineered to capture suspicious patterns:

   * `**errorBalanceOrg**`: Captures the discrepancy in the *sender's* account balance post-transaction. This turned out to be a highly predictive feature.

   * `**errorBalanceDest**`: Captures the discrepancy in the *recipient's* account balance.

   * `**hourOfDay**`: The transaction hour, extracted to identify potential temporal patterns in fraud.

3. **Data Preprocessing**: Numerical features are standardized using `StandardScaler`, and categorical features are one-hot encoded to prepare the data for modeling.

4. **Model Training & Evaluation**: The core of the project involves comparing several models. The **Area Under the Precision-Recall Curve (AUPRC)** is used as the primary metric due to the severe class imbalance.

   * ***Supervised Models***: To handle the class imbalance, the training data is resampled using **SMOTE (Synthetic Minority Over-sampling Technique)**.

     * Logistic Regression

     * Random Forest Classifier

     * XGBoost Classifier

     * Decision Tree Classifier

   * ***Unsupervised Anomaly Detection***: Two distinct approaches are evaluated:

     * **Method 1 (Outlier Detection)**: Models are trained on the full, imbalanced training dataset.

     * **Method 2 (Novelty Detection)**: Models are trained **exclusively on non-fraudulent** transactions to build a model of "normal" behavior.

     * *Models Tested*: Isolation Forest & One-Class SVM.

## Results & Key Findings

The evaluation clearly demonstrates the superiority of supervised learning for this specific problem.

### Model Performance Summary (AUPRC)

| **Model** | **Training Method** | **AUPRC Score** |
|---|---|---|
| **Random Forest** | **Supervised (SMOTE)** | **`0.9978`** |
| **XGBoost** | **Supervised (SMOTE)** | **`0.9918`** |
| Decision Tree | Supervised (SMOTE) | `0.9440` |
| Logistic Regression | Supervised (SMOTE) | `0.6011` |
| One-Class SVM | Novelty Detection | `0.2973` |
| One-Class SVM | Outlier Detection | `0.0730` |
| Isolation Forest | Outlier Detection | `0.0434` |
| Isolation Forest | Novelty Detection | `0.0351` |

### Conclusions

1. ***Supervised Models Excel***: Supervised models, particularly **Random Forest and XGBoost**, are highly effective at identifying the specific patterns that define fraud in this dataset, achieving near-perfect AUPRC scores. The engineered `errorBalanceOrg` feature was a critical predictor.

2. ***Anomaly Detection is Less Effective***: Unsupervised methods performed poorly in comparison because the fraudulent transactions, while malicious, were not statistically "different" enough from normal transactions to be easily isolated as generic outliers. They blended in too well.

3. ***Novelty Detection Improves One-Class SVM***: Training the One-Class SVM *exclusively on normal data* (Novelty Detection) significantly improved its performance, though it still fell far short of the top supervised models.

## How to Run

### Dependencies

Make sure you have the following Python libraries installed:

```
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn
```

### Execution

1. Place the dataset `PS_20174392719_1491204439457_log.csv.zip` in a `/content/` directory relative to the script.

2. Run the `fraud_detection_workflow.py` script.

   ```
   python fraud_detection_workflow.py
   ```

The script will execute the entire workflow from data loading to the final model evaluations and print the results to the console.
"""
