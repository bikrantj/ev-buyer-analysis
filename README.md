# EV Buyer Analysis

Predicting electric vehicle purchase intent and range anxiety from demographics and charging infrastructure.

| | |
|---|---|
| Project type | Data mining and machine learning |
| Author | Bikrant Jung Budhathoki |
| Main file | `EV_Buyer_Analysis.ipynb` |
| Tools | Python 3, Jupyter, pandas, scikit-learn, seaborn |

## Contents

1. [Overview](#1-overview)
2. [Project structure](#2-project-structure)
3. [How to run](#3-how-to-run)
4. [Dataset](#4-dataset)
5. [Methodology](#5-methodology)
6. [Results](#6-results)
7. [Discussion](#7-discussion)
8. [Limitations and future work](#8-limitations-and-future-work)
9. [Conclusion](#9-conclusion)

---

## 1. Overview

This project looks at a dataset of 10,000 potential EV buyers and asks two questions: what makes someone say they will buy an EV, and what makes them worry about driving range ("range anxiety")?

The work has three tasks:

1. **Exploratory data analysis.** Clean the data, then check how city type relates to home charging and whether environmental concern relates to income.
2. **Purchase prediction.** A binary classifier for `Will_Buy_EV`.
3. **Range anxiety prediction.** A three-class classifier for `Range_Anxiety_Level` (Low, Medium, High).

Short version of the findings:

- Home charging depends heavily on city type: 90.8% of rural, 79.7% of suburban and only 39.8% of urban people have it.
- Environmental concern is unrelated to income.
- For purchase intent, the Random Forest reached accuracy 0.85, F1 0.64 and ROC-AUC 0.90. Logistic Regression caught 87% of actual buyers but with more false alarms.
- Subsidy availability, environmental concern and income matter most for buying. High range anxiety strongly discourages it.
- For range anxiety, home charging is by far the biggest factor, followed by commute distance and nearby public charging. The best model reached a macro-F1 of 0.60.

Both targets are imbalanced, so I judge models on precision, recall, F1 and macro averages rather than accuracy alone.

## 2. Project structure

```
.
├── EV_Buyer_Analysis.ipynb   # Code, outputs, charts and explanations
├── dataset.csv               # Dataset
└── README.md                 # This file
```

Notebook outline:

| Section | Content |
|---|---|
| 0 | Setup and data loading |
| 1 (Task 1) | Inspection, cleaning, descriptive statistics, city type vs. home charging, environmental concern vs. income |
| 2 (Task 2) | Features and preprocessing, train/test split, training and evaluation, findings |
| 3 (Task 3) | Preprocessing and feature selection, training and evaluation, factors behind range anxiety, discussion |
| Conclusion | Summary and possible improvements |

## 3. How to run

### Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com), choose **File > Upload notebook** and select `EV_Buyer_Analysis.ipynb`.
2. Click **Runtime > Run all**. Every library used is already installed in Colab.

### Locally

You need Python 3.9 or newer (tested on 3.12) and an internet connection, since the notebook downloads the dataset.

```bash
# Optional: create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install numpy pandas scipy matplotlib seaborn scikit-learn jupyter

# Start Jupyter
jupyter notebook EV_Buyer_Analysis.ipynb
```

Then choose **Kernel > Restart & Run All**.

Libraries: `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `scikit-learn` (tested with 1.8.0).

All splits and models use `RANDOM_STATE = 42`, so results should repeat. Other library versions may give small differences.

## 4. Dataset

- **Source:** <https://raw.githubusercontent.com/bikrantj/ev-buyer-analysis/refs/heads/main/dataset.csv>
- **Size:** 10,000 rows, 15 columns, no duplicate rows
- **Missing values:** about 1.8% in three columns (178 in income, 181 in commute distance, 184 in environmental concern)

### Data dictionary

| Column | Type | Description |
|---|---|---|
| `Buyer_ID` | Identifier | Unique ID (not used as a feature) |
| `Age` | Numeric | Age in years (25 to 69) |
| `Gender` | Categorical | Male, Female, Other |
| `Annual_Income_USD` | Numeric | Annual income, about $30k to $223k; has missing values |
| `City_Type` | Categorical | Urban, Suburban, Rural |
| `Daily_Commute_km` | Numeric | Daily commute distance; has missing values |
| `Number_of_Cars_Owned` | Numeric | Cars in the household (1 to 4) |
| `Current_Car_Type` | Categorical | Sedan, SUV, Hatchback, Truck |
| `Charging_Stations_Near_Home` | Numeric | Stations near home (0 to 14) |
| `Charging_Stations_Near_Work` | Numeric | Stations near work (0 to 19) |
| `Home_Charging_Possible` | Categorical | Can charge at home (Yes/No) |
| `Environmental_Concern_Level` | Ordinal | 1 (low) to 5 (high); has missing values |
| `Subsidy_Available` | Categorical | Purchase subsidy available (Yes/No) |
| `Range_Anxiety_Level` | Categorical (target 2) | Low, Medium, High |
| `Will_Buy_EV` | Categorical (target 1) | Yes/No |

Column descriptions are interpreted from the column names and the assignment brief.

### Class balance

| Target | Distribution |
|---|---|
| `Will_Buy_EV` | No 82.5%, Yes 17.5% |
| `Range_Anxiety_Level` | Low 83.1%, Medium 15.2%, High 1.7% (167 people) |

## 5. Methodology

### Task 1: Exploratory data analysis

1. **Inspection:** shape, data types, unique values, missing values, duplicates.
2. **Cleaning:** I filled missing income, commute distance and environmental concern with the median. The median is not thrown off by extreme values, and for the 1 to 5 concern score it keeps a valid whole number. Boxplots and the IQR rule flagged 55 income and 39 commute outliers. I kept them because they are plausible values, not errors.
3. **Descriptive statistics:** summary tables, histograms, count plots, boxplots of buyers vs. non-buyers, and a correlation heatmap.
4. **City type vs. home charging:** cross-tabulation, stacked bar chart, chi-square test and Cramér's V.
5. **Environmental concern vs. income:** boxplots, bar charts, Pearson and Spearman correlation, and a Kruskal-Wallis test.

### Task 2: Purchase prediction (binary)

- **Target:** `Will_Buy_EV` (Yes = 1, No = 0).
- **Features:** everything except `Buyer_ID`. `Range_Anxiety_Level` enters as an ordinal feature (Low = 0, Medium = 1, High = 2).
- **Split:** 80/20, stratified so both sets keep the same buyer rate (`random_state=42`).
- **Preprocessing:** done inside a scikit-learn `Pipeline` to avoid data leakage. Numeric features get median imputation and standard scaling; categorical features get most-frequent imputation and one-hot encoding.
- **Models:** Logistic Regression and Random Forest (300 trees), both with `class_weight="balanced"`.
- **Evaluation:** accuracy, precision, recall, F1, ROC-AUC, confusion matrix, classification report, 5-fold stratified cross-validation, and feature importance or coefficients.

### Task 3: Range anxiety prediction (multiclass)

- **Target:** `Range_Anxiety_Level` (Low, Medium, High).
- **Features:** everything except `Buyer_ID` and `Will_Buy_EV`. I left out the purchase decision on purpose: it is partly a result of anxiety, so using it as a predictor would be circular.
- **Split:** 80/20, stratified (test set: 1,662 Low, 305 Medium, 33 High).
- **Preprocessing:** same pipeline as Task 2.
- **Feature selection:** mutual information, and `SelectFromModel` with a Random Forest (keeping features with above-average importance).
- **Models compared:** Logistic Regression (all features), Random Forest (all features), Random Forest (selected features), all with balanced class weights.
- **Evaluation:** accuracy, balanced accuracy, macro precision/recall/F1, weighted F1, per-class report, confusion matrices, 5-fold cross-validation, and permutation importance on the test set.

## 6. Results

### Task 1: EDA

| Question | Result |
|---|---|
| Missing values | Income 178, commute 181, environmental concern 184; all imputed |
| Home charging by city type | Rural 90.8%, Suburban 79.7%, Urban 39.8% |
| City type vs. home charging (test) | Chi-square = 2029.3, p < 0.001, Cramér's V = 0.45 (moderate to strong) |
| Environmental concern vs. income | Pearson r = 0.005, Spearman rho = 0.008, Kruskal-Wallis p = 0.92 (no relationship) |
| Buyers vs. non-buyers | Higher income (about $97.6k vs. $82.8k) and higher concern (4.09 vs. 2.75) for buyers |
| Strongest correlations with `Will_Buy_EV` | Environmental concern (0.36), subsidy (0.32), income (0.17) |

### Task 2: Purchase prediction (test set, 2,000 rows)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV F1 (5-fold) |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.816 | 0.486 | 0.866 | 0.622 | 0.914 | 0.610 ± 0.013 |
| Random Forest | 0.852 | 0.559 | 0.734 | 0.635 | 0.902 | 0.621 ± 0.019 |

Always predicting "No" scores 0.825 accuracy, so accuracy alone says little here.

Confusion matrices:

| | RF: Pred. No | RF: Pred. Yes | LR: Pred. No | LR: Pred. Yes |
|---|---|---|---|---|
| **Actual No** | 1447 | 203 | 1329 | 321 |
| **Actual Yes** | 93 | 257 | 47 | 303 |

### Task 3: Range anxiety prediction (test set, 2,000 rows)

| Model | Accuracy | Balanced accuracy | Macro F1 |
|---|---|---|---|
| Logistic Regression (all features) | 0.797 | 0.712 | 0.553 |
| Random Forest (all features) | 0.848 | 0.616 | 0.600 |
| Random Forest (6 selected features) | 0.851 | 0.608 | 0.592 |

The best model (Random Forest, all features) has a cross-validated macro-F1 of 0.595 ± 0.015.

Per-class report:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Low | 0.957 | 0.876 | 0.915 | 1662 |
| Medium | 0.503 | 0.761 | 0.606 | 305 |
| High | 0.412 | 0.212 | 0.280 | 33 |

Confusion matrix (rows = actual, columns = predicted):

| | Pred. Low | Pred. Medium | Pred. High |
|---|---|---|---|
| **Actual Low** | 1456 | 204 | 2 |
| **Actual Medium** | 65 | 232 | 8 |
| **Actual High** | 1 | 25 | 7 |

## 7. Discussion

### What drives the decision to buy

- **Subsidies** matter most: 27.4% of people with a subsidy would buy, against 2.5% without.
- **Environmental concern** is the top feature in the Random Forest. More concerned people are much more likely to buy.
- **Income** has a clear positive effect. Since it is unrelated to environmental concern, the two seem to act separately.
- **Range anxiety** deters buying: 0.6% of High-anxiety people would buy, versus 19.9% of Low-anxiety people.
- Commute distance and charging access have little direct effect. They seem to work mainly through range anxiety.

### What drives range anxiety

- **Home charging** dominates. Among people with home charging, 98.6% have Low anxiety and none has High. Of the 167 High-anxiety people, 166 have no home charging.
- **Commute distance** rises with anxiety: about 39 km for Low, 48 km for Medium and 60 km for High.
- **Public charging** is thinner for High-anxiety people: about 4.5 stations near work (vs. 7.6) and 3.6 near home (vs. 5.4).
- **City type** matters mostly indirectly, because urban people less often have home charging.
- Income, age, gender, environmental concern and number of cars add almost nothing.

### Model behaviour

- Both purchase models clearly beat the "always No" baseline (ROC-AUC around 0.90).
- Logistic Regression finds more real buyers, while the Random Forest raises fewer false alarms. Which is better depends on what a missed buyer costs compared with a wasted pitch.
- In Task 3, most errors are between neighbouring levels (25 of the 33 High cases were labelled Medium). With only 33 High cases this is suggestive, not conclusive.

### Practical takeaway

Helping people get home or workplace charging, and adding public chargers near homes and workplaces (especially for long-distance urban commuters), would likely do the most to reduce range anxiety and raise EV adoption.

## 8. Limitations and future work

### Limitations

- **Class imbalance.** Only 17.5% are buyers and only 1.7% have High anxiety, so High-class results are noisy.
- **Moderate precision.** About half of the people predicted to buy are false positives.
- **Association, not causation.** The findings describe patterns in this dataset. Also, `Will_Buy_EV` is stated intention, not actual purchases.
- **Likely synthetic data.** Patterns are very clean (for example, concern is completely unrelated to income), so results may not carry over to real populations.

### Future work

- Tune hyper-parameters with grid or random search (the models currently use defaults with balanced class weights).
- Try gradient boosting such as XGBoost or LightGBM.
- Adjust the decision threshold to favour recall or precision.
- Use resampling (e.g. SMOTE) or collect more High-anxiety examples.
- Treat range anxiety as an ordinal target.
- Validate on real survey data.

## 9. Conclusion

In this dataset, EV purchase intent is driven mainly by subsidies, environmental concern and income, while range anxiety is a major barrier. Range anxiety itself depends mostly on home charging, commute distance and nearby public charging. City type affects adoption indirectly, since urban residents far less often have home charging. Random Forest gave the best overall balance on both tasks (purchase F1 0.64, anxiety macro-F1 0.60) despite the heavy class imbalance.
