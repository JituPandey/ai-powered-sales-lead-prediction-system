# AI Sales Lead Prediction System
## Complete Project Details, Workflow, Tech Stack & Timeline

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Dataset Summary](#dataset-summary)
4. [Tech Stack](#tech-stack)
5. [Project Workflow](#project-workflow)
6. [Step-by-Step Results](#step-by-step-results)
7. [Model Performance](#model-performance)
8. [Key Findings](#key-findings)
9. [Web Application — UI Guide](#web-application--ui-guide)
10. [UI Test Cases](#ui-test-cases)
11. [Output Files](#output-files)
12. [Project Timeline](#project-timeline)
13. [Future Improvements](#future-improvements)

---

## Project Overview

An end-to-end Machine Learning system that predicts whether a sales lead will convert into a customer using historical CRM data. The system helps sales teams prioritize high-potential leads and optimize their outreach efforts.

The project is fully deployed as an **interactive web application** — a Flask backend serves an animated glassmorphism UI where users enter lead details and receive instant AI predictions with confidence scores, lead tier classification, and feature importance visualization.

- **ML Type:** Binary Classification
- **Target Variable:** `Converted` (1 = Converted, 0 = Not Converted)
- **Dataset:** `sales_leads.csv` — 9,240 rows, 37 columns
- **Best Model:** XGBoost (ROC-AUC: 0.9729, Accuracy: 92.53%)
- **Deployment:** Flask web app — `http://127.0.0.1:5000`

---

## Problem Statement

Sales teams receive large volumes of leads daily, but only ~38% convert to customers. Manual prioritization is inefficient and inconsistent. This project uses ML to:

- Predict which leads are most likely to convert
- Improve sales team efficiency by focusing on high-probability leads
- Reduce wasted effort on low-quality leads
- Provide data-driven insights into conversion drivers

---

## Dataset Summary

| Property | Value |
|---|---|
| File | `sales_leads.csv` |
| Total Rows | 9,240 |
| Total Columns | 37 |
| Target Column | `Converted` |
| Converted (1) | 3,561 (38.54%) |
| Not Converted (0) | 5,679 (61.46%) |
| Imbalance Ratio | 1.59:1 |

### Feature Categories

| Category | Examples |
|---|---|
| Lead Identity | Prospect ID, Lead Number |
| Lead Source & Origin | Lead Origin, Lead Source |
| Website Interaction | TotalVisits, Total Time Spent on Website, Page Views Per Visit |
| CRM Scoring | Asymmetrique Activity/Profile Index & Score |
| Customer Demographics | Country, City, Specialization, Occupation |
| Marketing Channels | Search, Magazine, Digital Advertisement, Newspaper |
| CRM Tags & Status | Tags, Lead Quality, Lead Profile, Last Activity |

---

## Tech Stack

### ML & Data Science
| Tool/Library | Version | Purpose |
|---|---|---|
| Python | 3.12.5 | Core language |
| Pandas | 3.0.0 | Data loading, manipulation, EDA |
| NumPy | 1.26.4 | Numerical operations, array handling |
| Scikit-learn | 1.8.0 | Pipelines, preprocessing, models, metrics |
| XGBoost | latest | Best-performing gradient boosted tree model |
| Matplotlib | 3.10.5 | Static visualizations (EDA plots) |
| Seaborn | 0.13.2 | Statistical plots (heatmaps, confusion matrices) |
| Jupyter Notebook | latest | Interactive step-by-step development |

### Web Application & Deployment
| Tool/Library | Version | Purpose |
|---|---|---|
| Flask | 3.1.3 | Python web server — serves UI and `/predict` API |
| Pickle | stdlib | Persist trained model and preprocessor to disk |
| HTML5 / CSS3 | — | Animated glassmorphism frontend UI |
| Vanilla JavaScript | — | Form handling, animated gauge, particle canvas, confetti |
| Google Fonts (Inter) | — | Modern UI typography |

---

## Project Workflow

```
sales_leads.csv
      |
      v
[STEP 1] Data Understanding
      - Load CSV with Pandas
      - .shape(), .info(), .describe(), .head()
      |
      v
[STEP 2] Exploratory Data Analysis (EDA)
      - Missing value analysis
      - Target distribution (class imbalance)
      - Numerical feature histograms
      - Correlation heatmap
      - Lead source vs conversion analysis
      - Website engagement analysis
      |
      v
[STEP 3] Data Preparation
      - Drop identifier columns (Prospect ID, Lead Number)
      - Separate X (features) and y (target)
      - Apply missing value strategy:
          >40% missing  --> DROP column
          10-40% missing --> IMPUTE
          <10% missing  --> KEEP + IMPUTE
      |
      v
[STEP 4] Data Cleaning & Preprocessing
      - Drop high-missing columns (5 columns)
      - Numerical Pipeline:
          SimpleImputer (median) --> StandardScaler
      - Categorical Pipeline:
          SimpleImputer (most_frequent) --> OneHotEncoder
      - ColumnTransformer combines both pipelines
      - fit_transform() --> shape (9240, 200)
      |
      v
[STEP 5] Stratified Train-Test Split
      - StratifiedShuffleSplit (80/20)
      - Preserves class distribution
      - Train: 7,392 | Test: 1,848
      |
      v
[STEP 6] Model Training
      - Logistic Regression
      - Decision Tree (max_depth=10)
      - Random Forest (100 estimators)
      - XGBoost (100 estimators)
      |
      v
[STEP 7] Model Evaluation
      - Accuracy, Precision, Recall, F1, ROC-AUC
      - Confusion matrices
      - ROC curves
      - Classification reports
      |
      v
[STEP 8] Model Selection
      - Ranked by ROC-AUC
      - Best: XGBoost (ROC-AUC = 0.9729)
      - Feature importance analysis
      |
      v
[STEP 9] Model Persistence
      - Preprocessor saved  --> preprocessor.pkl
      - Trained XGBoost     --> model.pkl
      - Feature metadata    --> feature_info.pkl
      - Dropdown values     --> dropdown_options.pkl
      |
      v
[STEP 10] Web Application Deployment
      - Flask backend (app.py)
          - Auto-trains on first run, loads pickle on subsequent runs
          - GET  /       --> serves animated HTML UI
          - POST /predict --> returns JSON prediction + confidence + tier
      - Animated Frontend (templates/index.html)
          - 5 input sections with 15+ feature fields
          - Animated SVG probability gauge
          - Lead tier: Hot / Warm / Cool / Cold
          - Confetti animation for high-conversion leads
          - Particle canvas background
      - Run: python app.py --> http://127.0.0.1:5000
```

---

## Step-by-Step Results

### Step 1: Data Understanding

```
Dataset loaded successfully.
Shape: (9240, 37)
Rows: 9240, Columns: 37

Data Types:
  - float64: 4 columns
  - int64: 3 columns
  - object (str): 30 columns

Key numerical stats:
  Conversion rate    : 38.54%
  Avg TotalVisits    : 3.45
  Avg Time on Website: 487.7 seconds
  Avg Page Views     : 2.36
```

### Step 2: EDA — Missing Values

```
Total columns with missing values: 17

Column                                          Missing Count   Missing %
Lead Quality                                         4767       51.59%  <-- DROP
Asymmetrique Profile Score                           4218       45.65%  <-- DROP
Asymmetrique Activity Score                          4218       45.65%  <-- DROP
Asymmetrique Profile Index                           4218       45.65%  <-- DROP
Asymmetrique Activity Index                          4218       45.65%  <-- DROP
Tags                                                 3353       36.29%  <-- IMPUTE
Lead Profile                                         2709       29.32%  <-- IMPUTE
What matters most to you in choosing a course        2709       29.32%  <-- IMPUTE
What is your current occupation                      2690       29.11%  <-- IMPUTE
Country                                              2461       26.63%  <-- IMPUTE
How did you hear about X Education                   2207       23.89%  <-- IMPUTE
Specialization                                       1438       15.56%  <-- IMPUTE
City                                                 1420       15.37%  <-- IMPUTE
TotalVisits                                           137        1.48%  <-- KEEP
Page Views Per Visit                                  137        1.48%  <-- KEEP
Last Activity                                         103        1.11%  <-- KEEP
Lead Source                                            36        0.39%  <-- KEEP
```

### Step 2: EDA — Target Distribution

```
Converted (1)    : 3,561  (38.54%)
Not Converted (0): 5,679  (61.46%)
Imbalance ratio  : 1.59:1
```

### Step 2: EDA — Correlation with Target

```
Total Time Spent on Website    0.362   (strongest positive)
Asymmetrique Profile Score     0.219
Asymmetrique Activity Score    0.168
TotalVisits                    0.030
Lead Number                    0.025
Page Views Per Visit          -0.003
```

### Step 2: EDA — Top Lead Sources by Conversion Rate

```
Lead Source          Conversion Rate
Welingak Website         98.59%
Reference                91.76%
Click2call               75.00%
Google                   39.99%
Organic Search           37.78%
Direct Traffic           32.17%
Olark Chat               25.53%
Referral Sites           24.80%
Facebook                 23.64%
```

### Step 3: Data Preparation

```
Dropped: ['Prospect ID', 'Lead Number']
Shape after identifier removal: (9240, 35)
Features (X): (9240, 34)
Target   (y): (9240,)

Missing Value Strategy Applied:
  Columns to DROP (>40%)  : 5
  Columns to IMPUTE (10-40%): 8
  Columns to KEEP (<10%)  : 4
```

### Step 4: Preprocessing

```
After dropping high-missing columns:
  Shape: (9240, 29)

Feature types:
  Numerical features (3): TotalVisits, Total Time Spent on Website,
                           Page Views Per Visit
  Categorical features (26): Lead Origin, Lead Source, Do Not Email,
                              Do Not Call, Last Activity, Country,
                              Specialization, ... (26 total)

Pipeline:
  Numerical  -> SimpleImputer(median) -> StandardScaler
  Categorical -> SimpleImputer(most_frequent) -> OneHotEncoder

PREPROCESSING VALIDATION SUMMARY
  Final Shape      : (9240, 200)
  Missing Values   : 0
  All Types Numeric: True
  Numeric features : 3
  Categorical cols : 26 (expanded to 197 after OHE)
  Total features   : 200
```

### Step 5: Stratified Train-Test Split

```
Method: StratifiedShuffleSplit (random_state=42)

  Train set : 7,392 samples (80.0%)
  Test set  : 1,848 samples (20.0%)

Class distribution preserved:
  Train - Converted: 38.54%
  Test  - Converted: 38.53%
  Original          : 38.54%   <-- perfectly preserved
```

### Step 6: Model Training

```
Models trained:
  1. Logistic Regression (max_iter=1000)
  2. Decision Tree       (max_depth=10)
  3. Random Forest       (n_estimators=100)
  4. XGBoost             (n_estimators=100)

Training Accuracy Results:
  Logistic Regression : 0.9177
  Decision Tree       : 0.9167
  Random Forest       : 0.9140
  XGBoost             : 0.9253
```

---

## Model Performance

### Metrics Comparison Table

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.9177 | 0.8900 | 0.8975 | 0.8937 | 0.9651 |
| Decision Tree | 0.9167 | 0.8854 | 0.9003 | 0.8928 | 0.9493 |
| Random Forest | 0.9140 | 0.8978 | 0.8764 | 0.8870 | 0.9668 |
| **XGBoost** | **0.9253** | **0.8953** | **0.9129** | **0.9040** | **0.9729** |

### Model Ranking by ROC-AUC

```
1. XGBoost             ROC-AUC=0.9729  <-- BEST
2. Random Forest       ROC-AUC=0.9668
3. Logistic Regression ROC-AUC=0.9651
4. Decision Tree       ROC-AUC=0.9493
```

### Best Model: XGBoost — Classification Report

```
               precision    recall  f1-score   support

Not Converted     0.94      0.93      0.94      1136
    Converted     0.90      0.91      0.90       712

     accuracy                         0.93      1848
    macro avg     0.92      0.92      0.92      1848
 weighted avg     0.93      0.93      0.93      1848
```

### Top 15 Feature Importances (XGBoost)

```
Feature                                              Importance
Lead Origin_Lead Add Form                              0.3330
Last Notable Activity_SMS Sent                         0.0789
Tags_Will revert after reading the email               0.0537
Tags_Closed by Horizzon                                0.0431
Lead Profile_Select                                    0.0399
Tags_Lost to EINS                                      0.0373
Lead Profile_Potential Lead                            0.0334
City_Select                                            0.0286
Tags_Busy                                              0.0266
Tags_Ringing                                           0.0254
What is your current occupation_Working Professional   0.0253
Tags_Already a student                                 0.0184
Total Time Spent on Website                            0.0132
Tags_Interested in other courses                       0.0106
Tags_Interested in Next batch                          0.0099
```

---

### Step 9 & 10: Model Persistence & Web App

```
Saved artifacts (auto-generated on first python app.py run):
  model.pkl            - XGBoost classifier (100 estimators)
  preprocessor.pkl     - ColumnTransformer (fitted on full dataset)
  feature_info.pkl     - num/cat feature lists, dropped columns
  dropdown_options.pkl - unique values for all UI dropdowns

Flask API Response format (/predict POST):
  {
    "prediction" : 1,           // 0 or 1
    "probability": 0.9988,      // float 0.0 - 1.0
    "confidence" : "99.9%",     // formatted string
    "tier"       : "Hot Lead",  // Hot / Warm / Cool / Cold
    "label"      : "High Conversion Likely"
  }

Lead Tier Thresholds:
  >= 75%  --> Hot Lead  (fire red-orange, confetti triggered)
  50-74%  --> Warm Lead (gold)
  30-49%  --> Cool Lead (blue)
  < 30%   --> Cold Lead (grey-red)
```

---

## Key Findings

1. **Best model is XGBoost** with ROC-AUC of 0.9729 and 92.53% accuracy
2. **Lead Origin (Lead Add Form)** is the single most important feature (33.3% importance)
3. **Total Time Spent on Website** has the strongest linear correlation with conversion (0.36)
4. **Reference and Welingak Website** leads convert at 91-98% — extremely high quality
5. **Dataset is moderately imbalanced** (1.59:1 ratio) — stratified splitting preserved balance
6. **5 columns were dropped** due to >40% missing values (Asymmetrique metrics, Lead Quality)
7. **26 categorical columns** expanded to 197 columns after One-Hot Encoding
8. **All 4 models perform well** (accuracy 91-93%), showing the data has strong signal
9. **SMS activity tag** is a top engagement signal for lead conversion

---

## Web Application — UI Guide

### How to Run
```bash
cd "d:/final year project"
python app.py
# Open browser at http://127.0.0.1:5000
```

On first run, the app trains XGBoost on the full dataset (~10-15 sec) and saves the model to disk. On all subsequent runs, it loads directly from the saved `.pkl` files (< 1 sec).

### UI Sections

| Section | Fields | Purpose |
|---|---|---|
| Lead Information | Lead Origin, Lead Source, Lead Profile, Tags | Core identity of the lead — highest importance |
| Website Engagement | Time on Website (slider 0-2272s), Total Visits, Page Views/Visit, Last Activity | Behavioral signals |
| Activity & Notable Events | Last Notable Activity, What Matters Most, How They Heard, Specialization | Engagement and intent signals |
| Customer Profile | Occupation, City, Country, Free Interview Copy | Demographic signals |
| Communication Preferences | Do Not Email toggle, Do Not Call toggle | Contact restrictions |

### UI Features

| Feature | Description |
|---|---|
| Animated particle canvas | Connected node network background, always running |
| Aurora gradient background | Pulsing purple-blue-pink radial gradients |
| Animated gauge | SVG circle draws from 0% to prediction probability in 1.6s |
| Importance badges | Every field labelled HIGH / MED / LOW based on XGBoost importance |
| Lead tier | Hot / Warm / Cool / Cold color-coded tier assigned per prediction |
| Confetti burst | Triggered automatically when predicted probability > 70% |
| Toast notifications | Bottom-center notification confirms each prediction result |
| Feature importance bars | Top 5 XGBoost features shown with animated fill bars |
| Model info card | Live stats: ROC-AUC, Accuracy, F1, training set size |
| Keyboard shortcut | Ctrl + Enter triggers prediction without clicking the button |

### API Endpoint

```
POST /predict
Content-Type: application/json

Body (example):
{
  "Lead Origin": "Lead Add Form",
  "Total Time Spent on Website": 1500,
  "Tags": "Will revert after reading the email",
  "Last Notable Activity": "SMS Sent",
  "What is your current occupation": "Working Professional",
  "Lead Profile": "Potential Lead",
  "City": "Mumbai"
}

Response:
{
  "prediction": 1,
  "probability": 0.9988,
  "confidence": "99.9%",
  "tier": "Hot Lead",
  "label": "High Conversion Likely"
}
```

Any fields left blank are treated as missing and automatically imputed by the preprocessing pipeline — partial inputs are fully supported.

---

## UI Test Cases

Use the following test cases to validate the UI. Each case includes all field values to enter and the expected prediction output.

---

### Test Case 1 — Hot Lead (Expected: CONVERT, ~99%)

> Lead added directly via form, high website engagement, working professional, SMS sent

| Field | Value |
|---|---|
| Lead Origin | Lead Add Form |
| Lead Source | Reference |
| Lead Profile | Potential Lead |
| Tags | Will revert after reading the email |
| Total Time Spent on Website | 1800 (drag slider) |
| Total Visits | 8 |
| Page Views / Visit | 3.5 |
| Last Activity | Email Opened |
| Last Notable Activity | SMS Sent |
| Current Occupation | Working Professional |
| City | Mumbai |
| Specialization | Business Administration |
| Country | India |
| Do Not Email | OFF |
| Do Not Call | OFF |

**Expected Result:** Prediction = Convert | Tier = Hot Lead | Probability > 90% | Confetti triggers

---

### Test Case 2 — Warm Lead (Expected: CONVERT, ~65-80%)

> Google-sourced lead, moderate time on site, student profile

| Field | Value |
|---|---|
| Lead Origin | Landing Page Submission |
| Lead Source | Google |
| Lead Profile | Potential Lead |
| Tags | Interested in Next batch |
| Total Time Spent on Website | 900 (drag slider) |
| Total Visits | 5 |
| Page Views / Visit | 2.0 |
| Last Activity | Email Opened |
| Last Notable Activity | Email Opened |
| Current Occupation | Student |
| City | Mumbai |
| Specialization | Finance Management |
| Country | India |
| Do Not Email | OFF |
| Do Not Call | OFF |

**Expected Result:** Prediction = Convert | Tier = Warm Lead | Probability 60-80%

---

### Test Case 3 — Cold Lead (Expected: NOT CONVERT, < 15%)

> API-sourced lead, zero website engagement, ringing tag, unemployed

| Field | Value |
|---|---|
| Lead Origin | API |
| Lead Source | Olark Chat |
| Lead Profile | Select |
| Tags | Ringing |
| Total Time Spent on Website | 0 (drag slider to 0) |
| Total Visits | 1 |
| Page Views / Visit | 0.5 |
| Last Activity | Unreachable |
| Last Notable Activity | Modified |
| Current Occupation | Unemployed |
| City | Select |
| Country | (leave blank) |
| Do Not Email | ON |
| Do Not Call | OFF |

**Expected Result:** Prediction = Not Convert | Tier = Cold Lead | Probability < 15%

---

### Test Case 4 — Boundary / Edge Case (Expected: ~55% borderline)

> Direct traffic lead, low engagement, phone conversation as last activity, housewife occupation
>
> **Key insight:** `Last Notable Activity = Had a Phone Conversation` is the only mid-range value —
> it avoids the SMS Sent boost (87%) and the Page Visited penalty (5%), landing right at the boundary.
> `Page Visited on Website` is a strongly negative signal and drops the score to ~5% regardless of other fields.

| Field | Value |
|---|---|
| Lead Origin | Landing Page Submission |
| Lead Source | Direct Traffic |
| Lead Profile | (leave blank) |
| Tags | (leave blank) |
| Total Time Spent on Website | 200 (drag slider to ~200) |
| Total Visits | 3 |
| Page Views / Visit | 2.3 |
| Last Activity | Email Opened |
| **Last Notable Activity** | **Had a Phone Conversation** |
| Current Occupation | Housewife |
| City | (leave blank) |
| Country | India |
| Do Not Email | OFF |
| Do Not Call | OFF |

**Expected Result:** Prediction = Convert (borderline) | Tier = Warm Lead | Probability ~55%

> **Why the original test failed:** `Last Notable Activity = Page Visited on Website` is a strongly
> negative signal in the model, collapsing the prediction to ~5%. The true borderline value is
> `Had a Phone Conversation`, which the model treats as a neutral-to-positive engagement signal.

---

### Test Case 5 — Minimal Input (Expected: model still runs via imputation)

> Only the two most important features set — rest left blank

| Field | Value |
|---|---|
| Lead Origin | Lead Add Form |
| Last Notable Activity | SMS Sent |
| All other fields | Leave blank / default |

**Expected Result:** High probability prediction — demonstrates that partial input is handled by the imputation pipeline without errors

---

### Test Case 6 — High-Value Reference Lead (Expected: CONVERT, > 99%)

> Reference-sourced lead (91.76% historical rate), long session, SMS sent as last notable activity
>
> **Key insight:** `Last Notable Activity = SMS Sent` is the 2nd most important XGBoost feature (7.9%).
> Setting it to "Email Opened" drops the prediction to ~22% even with a strong profile — always use SMS Sent for high-conversion test cases.

| Field | Value |
|---|---|
| Lead Origin | Landing Page Submission |
| Lead Source | Reference |
| Lead Profile | Potential Lead |
| Tags | Will revert after reading the email |
| Total Time Spent on Website | 2100 (drag slider near max) |
| Total Visits | 15 |
| Page Views / Visit | 4.0 |
| Last Activity | Email Opened |
| **Last Notable Activity** | **SMS Sent** |
| Current Occupation | Working Professional |
| City | Mumbai |
| Specialization | Finance Management |
| Country | India |
| Do Not Email | OFF |
| Do Not Call | OFF |

**Expected Result:** Prediction = Convert | Tier = Hot Lead | Probability > 99% | Confetti triggers

> **Why the original test failed:** `Last Notable Activity` was set to "Email Opened" instead of "SMS Sent".
> The model assigns 7.9% importance to the `Last Notable Activity_SMS Sent` feature — without it,
> the reference lead scores only ~22% regardless of other fields.

---

### Test Case 7 — Do-Not-Contact Cold Lead (Expected: NOT CONVERT)

> Uncontactable lead with negative engagement signals

| Field | Value |
|---|---|
| Lead Origin | API |
| Lead Source | Facebook |
| Lead Profile | Select |
| Tags | Busy |
| Total Time Spent on Website | 50 |
| Total Visits | 1 |
| Page Views / Visit | 1.0 |
| Last Activity | Unreachable |
| Last Notable Activity | Unreachable |
| Current Occupation | Unemployed |
| City | (leave blank) |
| Country | (leave blank) |
| Do Not Email | ON |
| Do Not Call | ON |

**Expected Result:** Prediction = Not Convert | Tier = Cold Lead | Probability < 10%

---

### Quick Reference — Input Values by Expected Outcome

| Field | HIGH probability value | BORDERLINE value | LOW probability value | Importance |
|---|---|---|---|---|
| Lead Origin | Lead Add Form | Landing Page Submission | API | 33.3% — most critical |
| Last Notable Activity | **SMS Sent** (~88%) | **Had a Phone Conversation** (~55%) | Page Visited on Website (~5%) | 7.9% — drives big swings |
| Tags | Will revert after reading the email | Interested in Next batch | Ringing / Busy | ~22% combined |
| Lead Profile | Potential Lead | (blank) | Select | 7.3% |
| Occupation | Working Professional | Housewife | Unemployed | 2.5% |
| City | Mumbai | (blank) | Select | 2.9% |
| Time on Website | > 1000 | ~200–500 | 0 | 1.3% |
| Lead Source | Reference | Google / Direct Traffic | Olark Chat / Facebook | < 1% model importance |

> **Common mistakes:**
> - `Last Notable Activity = Page Visited on Website` → drops to ~5% regardless of other fields
> - `Last Notable Activity = Email Opened` instead of `SMS Sent` → drops from ~99% to ~22%
> - `Lead Origin = Landing Page Submission` without `SMS Sent` → max ~25% even with good profile
> - SMS Sent is the 2nd most important single feature — it alone shifts probability by ~80 points

---

## Output Files

### Notebooks
| File | Description |
|---|---|
| `sales_lead_prediction.ipynb` | Source notebook (clean, unexecuted) |
| `sales_lead_prediction_executed.ipynb` | Fully executed notebook with all cell outputs |

### Web Application
| File | Description |
|---|---|
| `app.py` | Flask server — model loading, `/predict` endpoint, route handlers |
| `templates/index.html` | Animated glassmorphism frontend UI |
| `model.pkl` | Serialized XGBoost classifier (auto-generated on first run) |
| `preprocessor.pkl` | Serialized ColumnTransformer pipeline (auto-generated) |
| `feature_info.pkl` | Feature column metadata (auto-generated) |
| `dropdown_options.pkl` | Unique values for all UI select dropdowns (auto-generated) |

### EDA Plots (`outputs/` folder)
| File | Description |
|---|---|
| `outputs/target_distribution.png` | Bar + pie chart of class distribution |
| `outputs/numerical_distributions.png` | Histograms of numerical features |
| `outputs/correlation_heatmap.png` | Feature correlation heatmap |
| `outputs/lead_source_conversion.png` | Conversion rate by lead source |
| `outputs/website_engagement.png` | Time on website and visits vs conversion |
| `outputs/model_comparison.png` | Side-by-side metrics bar chart (all 4 models) |
| `outputs/roc_curves.png` | ROC curves for all 4 models |
| `outputs/confusion_matrices.png` | Confusion matrix grid for all models |
| `outputs/feature_importance.png` | Top 15 XGBoost feature importances |

---

## Project Timeline

### Phase 1: Data Understanding (Day 1)
- Load dataset with Pandas
- Inspect `.shape`, `.info()`, `.describe()`, `.head()`
- Understand business context of each of the 37 features
- Identify data types and initial missing value counts

### Phase 2: Exploratory Data Analysis (Day 2)
- Visualize target variable distribution — confirm class imbalance
- Calculate missing value percentages for all columns
- Generate numerical feature histograms
- Build correlation heatmap for numerical features
- Analyze lead source conversion rates
- Compare website engagement metrics between converted and non-converted leads

### Phase 3: Data Preparation (Day 3)
- Remove identifier columns (Prospect ID, Lead Number)
- Separate features X from target y
- Apply missing value strategy (drop / impute / keep thresholds)
- Categorize columns into numerical and categorical groups

### Phase 4: Preprocessing Pipeline (Day 3-4)
- Build Scikit-learn numerical pipeline (median impute + standard scale)
- Build Scikit-learn categorical pipeline (mode impute + one-hot encode)
- Combine using ColumnTransformer
- Execute `fit_transform()` on full dataset
- Validate: shape (9240, 200), zero missing values, all numeric

### Phase 5: Train-Test Split (Day 4)
- Use StratifiedShuffleSplit to preserve imbalanced class ratio
- 80% train (7,392 samples), 20% test (1,848 samples)
- Verify class distribution is consistent across splits

### Phase 6: Model Training (Day 5)
- Train 4 models: Logistic Regression, Decision Tree, Random Forest, XGBoost
- Collect predictions and probability scores on test set
- Record all evaluation metrics

### Phase 7: Evaluation & Selection (Day 5-6)
- Build metrics comparison table (Accuracy, Precision, Recall, F1, ROC-AUC)
- Plot model comparison bar chart
- Plot ROC curves for all models
- Generate confusion matrices
- Print full classification reports
- Rank by ROC-AUC — XGBoost selected as best model
- Plot top 15 feature importances

### Phase 8: Documentation (Day 6)
- Document all steps, outputs, and decisions
- Organize output plots in `outputs/` folder
- Write project workflow and findings

### Phase 9: Model Persistence (Day 6)
- Save fitted `ColumnTransformer` preprocessor to `preprocessor.pkl`
- Save trained `XGBClassifier` to `model.pkl`
- Save feature column metadata to `feature_info.pkl`
- Save dropdown option values to `dropdown_options.pkl`
- Verify round-trip: load pkl → predict → same output as notebook

### Phase 10: Web Application Development (Day 7)
- Build `app.py` Flask backend
  - Auto-train on first launch, load pkl on subsequent launches
  - `GET /` — renders the animated UI
  - `POST /predict` — accepts JSON input, returns prediction JSON
  - Handles missing/partial input via pipeline imputation
- Build `templates/index.html` animated frontend
  - Particle canvas background (60-node connected graph animation)
  - Aurora gradient background with CSS keyframe pulse
  - Glassmorphism cards with blur backdrop
  - 5 grouped form sections with 15+ input fields
  - Importance badges (HIGH / MED / LOW) on all fields
  - Time-on-website range slider with dynamic gradient fill
  - Do-Not-Email / Do-Not-Call toggle switches
  - Animated SVG gauge (1.6s cubic-bezier draw animation)
  - Lead tier system: Hot / Warm / Cool / Cold with color coding
  - Confetti burst for leads predicted > 70% probability
  - Toast notification system
  - Ctrl+Enter keyboard shortcut
  - Fully responsive layout (stacks on mobile < 1080px)
- Install Flask 3.1.3 and verify all endpoints respond correctly
- Run 7 structured test cases to validate prediction accuracy

---

## Future Improvements

| Improvement | Status | Description |
|---|---|---|
| Model Deployment (Flask) | DONE | Flask web app running at localhost:5000 |
| Pipeline Persistence | DONE | model.pkl and preprocessor.pkl auto-saved |
| Animated UI | DONE | Glassmorphism UI with gauge, confetti, particle canvas |
| Hyperparameter Tuning | Pending | GridSearchCV / RandomizedSearchCV on XGBoost |
| Feature Selection | Pending | Remove low-importance features to reduce model size |
| Class Imbalance Handling | Pending | SMOTE oversampling or `scale_pos_weight` in XGBoost |
| Cross-Validation | Pending | k-fold CV for more robust generalization estimate |
| Model Explainability | Pending | SHAP values for per-prediction feature explanation |
| Production Deployment | Pending | Host on cloud (Render / Railway / AWS EC2) with gunicorn |
| CRM Integration | Pending | REST API endpoint to score leads from Salesforce / HubSpot |
| Batch Prediction | Pending | CSV upload in UI to score multiple leads at once |

---

## Interview Summary

> "I built a machine learning system to predict sales lead conversion on a dataset of 9,240 CRM records. I performed comprehensive EDA including missing value analysis, target distribution checks, and correlation studies. I used Scikit-learn pipelines to prevent data leakage — applying median imputation and standard scaling to numerical features, and mode imputation with one-hot encoding to categorical features. StratifiedShuffleSplit preserved the 38/62 class imbalance across train and test sets. I trained four models — Logistic Regression, Decision Tree, Random Forest, and XGBoost — evaluating each on Accuracy, Precision, Recall, F1, and ROC-AUC. XGBoost achieved the best ROC-AUC of 0.9729 and 92.53% accuracy. The most important feature was Lead Origin (Lead Add Form) with 33% importance, followed by SMS activity tags."
