# ev-buyer-analysis
Predicting Electric Vehicle Purchase Intent based on Demographics and Infrastructure

# Electric Vehicle (EV) Buyer Analysis

**Exploratory Data Analysis, EV Purchase Prediction and Range Anxiety Prediction**

| | |
|---|---|
| **Project type** | Penalty assignment (internal marks upgrade) |
| **Author** | _[Your Name]_ |
| **Roll No. / Course** | _[Roll No.]_ / _[Course Name]_ |
| **Submission deadline** | 16 October |
| **Main file** | `EV_Buyer_Analysis.ipynb` |
| **Language / tools** | Python 3, Jupyter Notebook, pandas, scikit-learn, seaborn |

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Introduction and Objectives](#2-introduction-and-objectives)
3. [Dataset Description](#3-dataset-description)
4. [Methodology](#4-methodology)
5. [Results](#5-results)
6. [Key Findings and Discussion](#6-key-findings-and-discussion)
7. [Limitations](#7-limitations)
8. [Future Work](#8-future-work)
9. [Project Structure](#9-project-structure)
10. [How to Run](#10-how-to-run)
11. [Conclusion](#11-conclusion)

---

## 1. Abstract

This project analyses a dataset of 10,000 potential electric vehicle (EV) buyers to understand what influences the decision to buy an EV and how worried buyers are about driving range ("range anxiety"). The work has three parts. First, an exploratory data analysis (EDA) cleans the data (about 1.8% missing values in each of three columns) and examines how city type affects home-charging availability and whether environmental concern is related to income. Second, a binary classification model predicts whether a person will buy an EV (`Will_Buy_EV`). Third, a multiclass classification model predicts the level of range anxiety (`Low`, `Medium`, `High`).

The EDA shows that home charging depends strongly on city type (90.8% of rural, 79.7% of suburban and only 39.8% of urban buyers have it), while environmental concern is unrelated to income. For purchase prediction, a Random Forest reached an accuracy of 0.85, an F1-score of 0.64 and an ROC-AUC of 0.90, and a Logistic Regression found 87% of actual buyers; the most influential factors were **subsidy availability, environmental concern and income**, with range anxiety acting as a strong deterrent. For range-anxiety prediction, a Random Forest reached a macro-F1 of 0.60, and **home-charging availability** was by far the dominant factor, followed by daily commute distance and the number of public charging stations near home and work. Both targets are heavily imbalanced, so evaluation focuses on precision, recall, F1 and macro-averaged metrics rather than accuracy alone.

**Keywords:** electric vehicles, EV adoption, range anxiety, exploratory data analysis, classification, imbalanced data, Random Forest, Logistic Regression

---

## 2. Introduction and Objectives

Whether people buy electric vehicles depends on more than price. Charging infrastructure, driving habits, subsidies and attitudes toward the environment all play a part. Understanding these factors helps manufacturers, governments and charging-network planners decide where to focus.

**Objectives (as per the assignment):**

| Task | Objective | Marks |
|---|---|---|
| 1 | Exploratory Data Analysis: inspect, clean, summarise and visualise the data; study *City Type vs. home charging* and *Environmental Concern vs. Income* | 5 |
| 2 | Build and evaluate a binary classifier for `Will_Buy_EV` | 5 |
| 3 | Build and evaluate a multiclass classifier for `Range_Anxiety_Level` and explain the factors behind it | 5 |

---

## 3. Dataset Description

* **Source:** `https://raw.githubusercontent.com/bikrantj/ev-buyer-analysis/refs/heads/main/dataset.csv`
* **Size:** 10,000 rows x 15 columns, no duplicate rows
* **Missing values:** about 1.8% in each of 3 columns (178 in income, 181 in commute distance, 184 in environmental concern)

### Data dictionary

| Column | Type | Description |
|---|---|---|
| `Buyer_ID` | Identifier | Unique ID per buyer (not used as a feature) |
| `Age` | Numeric | Age in years (25 to 69) |
| `Gender` | Categorical | Male, Female, Other |
| `Annual_Income_USD` | Numeric | Annual income in US dollars (about $30k to $223k); has missing values |
| `City_Type` | Categorical | Urban, Suburban, Rural |
| `Daily_Commute_km` | Numeric | Daily commute distance in km; has missing values |
| `Number_of_Cars_Owned` | Numeric | Number of cars in the household (1 to 4) |
| `Current_Car_Type` | Categorical | Sedan, SUV, Hatchback, Truck |
| `Charging_Stations_Near_Home` | Numeric | Number of charging stations near home (0 to 14) |
| `Charging_Stations_Near_Work` | Numeric | Number of charging stations near work (0 to 19) |
| `Home_Charging_Possible` | Categorical | Whether the buyer can charge at home (Yes/No) |
| `Environmental_Concern_Level` | Ordinal | Concern for the environment, 1 (low) to 5 (high); has missing values |
| `Subsidy_Available` | Categorical | Whether a purchase subsidy is available (Yes/No) |
| `Range_Anxiety_Level` | Categorical (**Target 2**) | Low, Medium, High |
| `Will_Buy_EV` | Categorical (**Target 1**) | Yes/No |

_Column descriptions are interpreted from the column names and the assignment brief._

### Class balance (both targets are imbalanced)

| Target | Distribution |
|---|---|
| `Will_Buy_EV` | No 82.5% / Yes 17.5% |
| `Range_Anxiety_Level` | Low 83.1% / Medium 15.2% / High 1.7% (167 people) |

---

## 4. Methodology

### 4.1 Task 1: Exploratory Data Analysis

1. **Inspection:** shape, data types, unique values, missing values, duplicates.
2. **Cleaning:** median imputation for `Annual_Income_USD`, `Daily_Commute_km` and `Environmental_Concern_Level` (median is robust to the few extreme values; for the 1-5 score it keeps a valid whole number). Outliers were checked with boxplots and the IQR rule (55 income and 39 commute outliers) and **kept**, because they are plausible values rather than errors.
3. **Descriptive statistics and visualisations:** summary tables, histograms, count plots, boxplots of buyers vs. non-buyers, and a correlation heatmap.
4. **City Type vs. home charging:** cross-tabulation, stacked bar chart, chi-square test of independence and Cramer's V.
5. **Environmental Concern vs. Income:** boxplots, bar charts, Pearson and Spearman correlation, and the Kruskal-Wallis test.

### 4.2 Task 2: EV Purchase Prediction (binary classification)

* **Target:** `Will_Buy_EV` (Yes = 1, No = 0).
* **Features:** all attributes except `Buyer_ID`, including the required income, commute distance and charging availability. `Range_Anxiety_Level` is used as an ordinal feature (Low = 0, Medium = 1, High = 2).
* **Split:** 80% train / 20% test, **stratified** to keep the buyer rate the same in both sets (`random_state=42`).
* **Preprocessing (inside a scikit-learn `Pipeline`, so no data leakage):** median imputation and standard scaling for numeric features; most-frequent imputation and one-hot encoding for categorical features.
* **Models:** Logistic Regression and Random Forest (300 trees), both with `class_weight="balanced"` to handle the imbalance.
* **Evaluation:** accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, classification report, 5-fold stratified cross-validation, and feature importance / coefficients.

### 4.3 Task 3: Range Anxiety Prediction (multiclass classification)

* **Target:** `Range_Anxiety_Level` (Low / Medium / High).
* **Features:** all attributes except `Buyer_ID` and `Will_Buy_EV`. The purchase decision was excluded on purpose because it is a *consequence* of anxiety, so using it would be circular.
* **Split:** 80/20 stratified (test set: 1,662 Low, 305 Medium, 33 High).
* **Preprocessing:** the same imputation, scaling and encoding pipeline as Task 2.
* **Feature selection:** (1) mutual information between each feature and the target; (2) `SelectFromModel` with a Random Forest, keeping features with above-average importance.
* **Models compared:** Logistic Regression (all features), Random Forest (all features), Random Forest (selected features); all with balanced class weights.
* **Evaluation:** accuracy, balanced accuracy, macro precision/recall/F1, weighted F1, per-class classification report, confusion matrices (counts and row-normalised), 5-fold cross-validation, and permutation importance on the test set.

---

## 5. Results

### 5.1 Task 1: key EDA results

| Question | Result |
|---|---|
| Missing values | Income 178, commute 181, environmental concern 184 (about 1.8% each); all imputed |
| Home charging by city type | Rural **90.8%**, Suburban **79.7%**, Urban **39.8%** |
| Statistical test (city type vs. home charging) | Chi-square = 2029.3, p < 0.001, Cramer's V = 0.45 (moderate-to-strong association) |
| Environmental concern vs. income | Pearson r = +0.005, Spearman rho = +0.008, Kruskal-Wallis p = 0.92 (**no relationship**) |
| Buyers vs. non-buyers | Buyers have higher income (about $97.6k vs. $82.8k) and higher environmental concern (4.09 vs. 2.75) |
| Strongest correlations with `Will_Buy_EV` | Environmental concern (0.36), subsidy (0.32), income (0.17) |

### 5.2 Task 2: EV purchase prediction (test set, 2,000 rows)

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | CV F1 (5-fold) |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.816 | 0.486 | **0.866** | 0.622 | **0.914** | 0.610 +/- 0.013 |
| Random Forest | **0.852** | **0.559** | 0.734 | **0.635** | 0.902 | 0.621 +/- 0.019 |

Majority-class baseline accuracy (always predict "No") = 0.825, so accuracy alone is misleading here.

**Confusion matrices**

| | Random Forest | | Logistic Regression | |
|---|---|---|---|---|
| | Pred. No | Pred. Yes | Pred. No | Pred. Yes |
| **Actual No** | 1447 | 203 | 1329 | 321 |
| **Actual Yes** | 93 | 257 | 47 | 303 |

### 5.3 Task 3: range anxiety prediction (test set, 2,000 rows)

| Model | Accuracy | Balanced accuracy | Macro F1 |
|---|---|---|---|
| Logistic Regression (all features) | 0.797 | 0.712 | 0.553 |
| **Random Forest (all features)** | **0.848** | 0.616 | **0.600** |
| Random Forest (6 selected features) | 0.851 | 0.608 | 0.592 |

Cross-validated macro-F1 of the best model: 0.595 +/- 0.015.

**Best model: per-class report**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Low | 0.957 | 0.876 | 0.915 | 1662 |
| Medium | 0.503 | 0.761 | 0.606 | 305 |
| High | 0.412 | 0.212 | 0.280 | 33 |

**Confusion matrix (rows = actual, columns = predicted)**

| | Pred. Low | Pred. Medium | Pred. High |
|---|---|---|---|
| **Actual Low** | 1456 | 204 | 2 |
| **Actual Medium** | 65 | 232 | 8 |
| **Actual High** | 1 | 25 | 7 |

---

## 6. Key Findings and Discussion

**What drives the decision to buy an EV**
1. **Subsidy availability** is the strongest factor: 27.4% of people with a subsidy would buy versus only 2.5% without.
2. **Environmental concern** is the most important feature in the Random Forest; more concerned buyers are much more likely to buy.
3. **Income** has a clear positive effect. It is independent of environmental concern, so the two act as separate influences.
4. **Range anxiety** is a strong deterrent: 0.6% of High-anxiety people would buy versus 19.9% of Low-anxiety people.
5. Commute distance and charging availability have a smaller *direct* effect; they act mainly through range anxiety.

**What drives range anxiety**
1. **Home charging** is the dominant factor. Among buyers with home charging, 98.6% have Low anxiety and none has High anxiety; 166 of the 167 High-anxiety buyers have no home charging.
2. **Daily commute distance:** average commute is about 39 km (Low), 48 km (Medium) and 60 km (High).
3. **Public charging nearby:** High-anxiety buyers have fewer stations near work (about 4.5 vs. 7.6) and near home (3.6 vs. 5.4).
4. **City type** matters mostly indirectly, through its effect on home charging (urban buyers lack it more often).
5. Income, age, gender, environmental concern and cars owned add almost nothing to the prediction.

**Model behaviour**
* Both models beat the "always No" baseline by a wide margin (ROC-AUC about 0.90).
* There is a precision/recall trade-off: Logistic Regression finds more real buyers, Random Forest raises fewer false alarms. The better choice depends on the cost of each type of error.
* In Task 3 the errors are mostly between *neighbouring* levels (25 of 33 High cases were labelled Medium), so the model captures the ordering of anxiety even when it misses the exact class.

**Practical implication:** reducing range anxiety by helping buyers get home or workplace charging and by expanding public charging near homes and workplaces, especially for long-distance urban commuters, is likely to raise EV adoption.

---

## 7. Limitations

* **Class imbalance:** only 17.5% of people are buyers and only 1.7% have High anxiety (33 test cases), so the High-class results are noisy and should be interpreted carefully.
* **Moderate precision:** about half of the people predicted to buy are false positives.
* **Associations, not causation:** findings describe patterns in this dataset, not proven cause and effect.
* **Stated intention:** `Will_Buy_EV` records whether a person says they will buy, not actual purchases.
* **No hyper-parameter tuning:** models use sensible defaults with balanced class weights.
* The very clean patterns (for example, concern being completely unrelated to income) suggest the data may be simulated, so results may not transfer directly to real populations.

---

## 8. Future Work

* Tune hyper-parameters with grid or random search.
* Try gradient boosting (e.g. XGBoost, LightGBM) and compare with the current models.
* Adjust the decision threshold to favour recall or precision, depending on the use case.
* Use resampling (e.g. SMOTE) or collect more High-anxiety examples.
* Treat range anxiety as an ordinal target with an ordinal classification method.
* Validate the findings on real survey data.

---

## 9. Project Structure

```
.
├── EV_Buyer_Analysis.ipynb   # Full analysis: code, outputs, charts and explanations
└── README.md                 # Project documentation (this file)
```

The dataset is **not** stored locally: the notebook downloads it directly from the URL given in Section 3.

**Notebook outline**

| Section | Content |
|---|---|
| 0 | Setup and data loading |
| 1 (Task 1) | 1.1 Inspection, 1.2 Cleaning, 1.3 Descriptive statistics and visualisations, 1.4 City Type vs. home charging, 1.5 Environmental concern vs. income |
| 2 (Task 2) | 2.1 Features and preprocessing, 2.2 Train/test split, 2.3 Training and evaluation, 2.4 Results and findings |
| 3 (Task 3) | 3.1 Preprocessing and feature selection, 3.2 Training and evaluation, 3.3 Factors influencing range anxiety, 3.4 Results and discussion |
| Conclusion | Summary of results and possible improvements |

---

## 10. How to Run

### Option A: Google Colab (easiest)
1. Open [colab.research.google.com](https://colab.research.google.com), choose **File > Upload notebook**, and select `EV_Buyer_Analysis.ipynb`.
2. Click **Runtime > Run all**. All libraries used are already installed in Colab.

### Option B: Locally

**Requirements:** Python 3.9 or newer (tested on Python 3.12) and an internet connection (to download the dataset).

```bash
# 1. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install the dependencies
pip install numpy pandas scipy matplotlib seaborn scikit-learn jupyter

# 3. Start Jupyter and open the notebook
jupyter notebook EV_Buyer_Analysis.ipynb
```

Then choose **Kernel > Restart & Run All**.

**Libraries used:** `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `scikit-learn` (tested with scikit-learn 1.8.0).

**Reproducibility:** a fixed random seed (`RANDOM_STATE = 42`) is used for all splits and models, so results are repeatable. Small differences may appear with other library versions.

---

## 11. Conclusion

The analysis shows that EV adoption in this dataset is shaped mainly by **subsidies, environmental attitude and income**, and that **range anxiety**, which in turn depends mostly on **home charging availability, commute distance and nearby public charging**, is a major barrier. City type influences adoption indirectly because urban buyers far less often have home charging. Random Forest models achieved the best overall balance of performance on both prediction tasks (purchase: F1 0.64; anxiety: macro-F1 0.60), while accounting for the strong class imbalance.

---

_Dataset: `bikrantj/ev-buyer-analysis` on GitHub. Prepared as part of a penalty assignment for internal marks._
