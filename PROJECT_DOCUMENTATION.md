# AI Sales Lead Prediction System

## 📌 Project Overview

This project aims to build a Machine Learning system that predicts whether a sales lead will convert into a customer based on historical CRM data.

The system helps businesses prioritize high-quality leads and optimize sales efforts.

---

## 🎯 Problem Statement

Sales teams receive a large number of leads, but only a small percentage converts into customers. Identifying high-potential leads manually is inefficient and error-prone.

This project uses Machine Learning to:

* Predict lead conversion
* Improve sales efficiency
* Reduce wasted effort

---

## 🧠 Type of ML Problem

* **Type:** Binary Classification
* **Target Variable:** `Converted`

  * 1 → Converted
  * 0 → Not Converted

---

## 📂 Dataset Description

* Total Rows: 9240
* Total Features: 37 (before preprocessing)

### Feature Categories:

* Lead Source & Origin
* Website Interaction Metrics
* Customer Demographics
* Marketing Channels
* CRM Scoring Metrics

---

## ⚙️ Project Workflow

1. Data Understanding (EDA)
2. Data Preparation
3. Data Cleaning & Preprocessing
4. Stratified Train-Test Split
5. Model Training
6. Model Evaluation
7. Model Selection

---

# 🟦 STEP 1: DATA UNDERSTANDING

* Loaded dataset using pandas
* Identified structure using:

  * `.head()`
  * `.info()`
  * `.shape()`
* Understood business meaning of each feature

---

# 🟦 STEP 2: EXPLORATORY DATA ANALYSIS (EDA)

### Key Actions:

* Checked missing values
* Analyzed target distribution
* Visualized categorical relationships
* Studied numerical feature behavior
* Generated correlation heatmap

### Key Insights:

* Dataset is imbalanced
* Website engagement impacts conversion
* Lead source affects conversion rate

---

# 🟦 STEP 3: DATA PREPARATION

### Actions Taken:

* Dropped identifier columns:

  * Prospect ID
  * Lead Number

* Separated:

  * Features (X)
  * Target (y)

### Missing Value Strategy:

* > 40% missing → Drop
* 10–40% → Impute
* <10% → Keep

### Feature Classification:

* Numerical Features
* Categorical Features

---

# 🟦 STEP 4: DATA CLEANING & PREPROCESSING

## 🧹 Column Removal

Dropped columns with high missing values:

* Lead Quality
* Asymmetrique Index & Score columns

---

## 🔧 Preprocessing Pipelines

### Numerical Pipeline:

* Median Imputation
* Standard Scaling

### Categorical Pipeline:

* Most Frequent Imputation
* One-Hot Encoding

---

## 🔗 ColumnTransformer

Used to apply:

* Numerical pipeline → numerical features
* Categorical pipeline → categorical features

---

## 🔄 Pipeline Execution

* Used `fit_transform()` to preprocess data
* Output converted from NumPy array to DataFrame
* Preserved column names for interpretability

---

## ✅ Validation Results

* Final Shape: (9240, 200)
* Missing Values: 0
* Data Types: All numeric

---

## ⚠️ Important Considerations

* Avoided manual preprocessing
* Used pipelines to prevent data leakage
* Maintained reproducibility

---

# 🟦 STEP 5: STRATIFIED TRAIN-TEST SPLIT

Used:

* `StratifiedShuffleSplit`

### Why?

* Dataset is imbalanced
* Maintains class distribution

### Split:

* 80% Training
* 20% Testing

---

# 🟦 MODELING STRATEGY (UPCOMING)

We will train multiple models:

* Logistic Regression
* Decision Tree
* Random Forest
* (Optional) XGBoost

### Model Selection Criteria:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

---

# 🧠 Key ML Concepts Used

* Data Imputation
* Feature Encoding
* Feature Scaling
* Pipeline Architecture
* Stratified Sampling
* Avoiding Data Leakage

---

# 🧑‍💻 Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

---

# 🚀 Future Improvements

* Hyperparameter tuning
* Feature selection
* Model deployment (Streamlit / Flask)
* Explainability (SHAP)

---

# 🎤 Interview Explanation (Short)

“I built a machine learning system to predict sales lead conversion. I performed EDA, handled missing values using pipelines, encoded categorical variables, and used stratified sampling to maintain class balance. Multiple models were evaluated to select the best-performing one.”

---

# ✅ Conclusion

This project demonstrates a complete end-to-end ML workflow, from data understanding to preprocessing and preparation for model training, following industry best practices.
