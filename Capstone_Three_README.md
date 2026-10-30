# Optimizing Term Deposit Marketing Campaigns for a European Bank

![cover](reports/figures/10_feature_importance.png)

A Portuguese retail bank runs phone campaigns to convert existing customers into term deposit holders. The baseline conversion rate is roughly 11 to 16 percent, which means for every subscription the bank secures, five or more calls to non converting customers still cost agent time, telephony, and operational overhead. This project builds a county level ranked list of customers by subscription probability so the call center can prioritize the highest propensity prospects first. The goal is to lift realized campaign ROI without reducing total subscriptions.

## 1. Data

The project uses the UCI Bank Marketing dataset published alongside the Moro, Cortez and Rita (2014) paper on bank telemarketing. It covers 41,188 customer records across 45 months of campaign activity from May 2008 through November 2010.

> * [UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
> * [Kaggle mirror (easier CSV download)](https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing)

Features include client attributes (age, occupation, marital status, education, credit and loan status), campaign attributes (contact channel, month and day, number of prior contacts), and macroeconomic attributes for the period (employment variation rate, consumer price index, consumer confidence index, Euribor 3 month rate). The binary target records whether the customer subscribed to the term deposit.

The repository ships with a schema faithful synthetic sample so the pipeline runs end to end without a download. The code runs unchanged on the real UCI file; swap the file in and rerun.

## 2. Method

This is a binary classification problem with moderate class imbalance (about 16 percent positive rate). Rather than build one model and hope it works, I compared three ML approaches against two baselines.

Baselines:

1. Majority class predictor. Always says no. ROC AUC is 0.500 by construction and defines the mathematical floor.
2. Rule based targeting from EDA Section 9. Contact equals cellular, month in {mar, sep, oct, dec}, job in {student, retired, unemployed, admin, management}. This is what a marketing analyst could implement in an afternoon with no ML at all.

ML candidates:

1. Logistic Regression with L2 regularization, as an interpretable baseline. Hyperparameter C tuned via GridSearchCV.
2. Random Forest, a bagged tree ensemble that handles non linear patterns and feature interactions natively.
3. Gradient Boosting Regressor, a sequential tree ensemble that is typically the strongest performer on tabular business data.

Selection metric was cross validated ROC AUC on the training set, with precision at top k and expected profit at optimal threshold as the business facing metrics on the held out test set.

## 3. Data Wrangling

[Data Wrangling Notebook](01_data_wrangling.ipynb)

The wrangling step reduced 41,211 raw rows to 41,188 cleaned rows across 23 columns ready for modeling. Key cleaning decisions:

* The dataset uses the string "unknown" as its placeholder for missing categorical data across six columns. I converted these to true NaN. For the two columns where the missing rate is meaningful (default at 21 percent and education at 4 percent), I added binary indicator flags, because the fact that a value is missing may itself carry signal.
* The pdays column uses 999 to encode "never previously contacted". Left as a numeric value this would corrupt every mean and median downstream, so I recoded it to NaN and added a was_contacted_before indicator that preserves the information.
* I dropped the call duration feature entirely. The UCI documentation itself notes that a call of zero seconds implies the outcome is no by definition. Any model trained with duration will look artificially strong because it has learned the tautology that long calls convert; it has no predictive value before the call is placed.
* 23 exact duplicate rows from source system re-exports were removed.
* The yes/no target was encoded to 0/1 for downstream modeling.
* Month and day_of_week were converted to ordered categoricals so plots and models render them chronologically rather than alphabetically.

## 4. EDA

[Exploratory Data Analysis Notebook](02_eda.ipynb)

The EDA surfaced the segments that drive subscription and the segments that waste calls.

Target distribution. The positive class rate is 15.8 percent, giving a 5.3 to 1 imbalance that rules out plain accuracy as an evaluation metric.

![target distribution](reports/figures/01_target_distribution.png)

Subscription rate by categorical feature. This is the single most informative chart in the EDA. Green bars convert above the overall baseline; blue bars convert below.

![subscription by categorical](reports/figures/04_subscription_by_categorical.png)

The headlines:

* Contact type is the single most actionable lever. Cellular converts at 20.5 percent versus telephone at 7.6 percent, a 2.7 times lift.
* Month shows a striking seasonal pattern. December (53 percent), March (49 percent), September and October (46 percent) convert 3 to 4 times higher than May (12 percent). The high volume campaign months are the low conversion months, which suggests the bank is spending calls in exactly the wrong months.
* Job segments split cleanly. Students (35 percent), retired (31 percent), and unemployed (25 percent) convert well above baseline. Blue collar (10 percent) and services (12 percent) convert below.
* Prior campaign success is the strongest single predictor for the small subset of customers who had any prior contact at all. Poutcome success converts at 53 percent versus 15 percent for customers with no prior history.

Correlation matrix. Numeric features show modest correlations with the target. The strongest numeric signals are age and the socio economic cluster (euribor3m, emp.var.rate), but the real signal in this dataset lives in the categorical features.

![correlation matrix](reports/figures/06_correlation_matrix.png)

## 5. Pre-processing and Modeling

[Pre-processing and Modeling Notebook](03_preprocessing_modeling.ipynb)

Pre-processing produced the model ready train and test splits:

* Imputed remaining missing values (categoricals filled with "missing" category, pdays median imputed).
* Dropped three redundant socio economic columns (cons.price.idx, cons.conf.idx, nr.employed) that were highly correlated with euribor3m.
* One hot encoded the categorical columns with drop_first=True, expanding 19 features to 52.
* 80/20 stratified train/test split with random_state=42 (32,950 training rows, 8,238 testing rows).
* StandardScaler fit on the training set only, then applied to both partitions.

All five models were evaluated with the same 3-fold stratified cross validation harness on the training set. ROC AUC was the primary metric.

![model comparison](reports/figures/07_model_comparison.png)

| Model | CV ROC AUC | Notes |
|---|---|---|
| Baseline: Majority Class | 0.500 | Mathematical floor |
| Baseline: Rule-Based | 0.524 | Single threshold rule from EDA |
| Logistic Regression (C=0.01, balanced) | 0.739 | Interpretable baseline |
| Random Forest (max_depth=8, n_est=200) | 0.738 | Captures non linear patterns |
| Gradient Boosting (max_depth=3, n_est=200, lr=0.1) | 0.745 | Selected as final model |

WINNER: Gradient Boosting Classifier
Hyperparameters: n_estimators=200, max_depth=3, learning_rate=0.1

> All three ML approaches land within 0.7 points of ROC AUC of each other. That closeness is informative: once categoricals are one hot encoded, the signal is mostly linear and additive. The tree models' edge comes almost entirely from the U shaped age effect and a small number of feature interactions. A production deployment could reasonably choose Logistic Regression for its clearer interpretability.

## 6. Test Set Performance

The test set was touched exactly once at the final evaluation step.

Test ROC AUC: 0.740
Test PR AUC: 0.381

![ROC and PR curves](reports/figures/08_roc_pr_curves.png)

Precision at top k is the business facing metric. If the call center calls the top 10 percent of the ranked list rather than a random 10 percent, those calls convert 2.8 times better.

| Top k | Precision | Recall | Lift over random |
|---|---|---|---|
| Top 5% | 54% | 17% | 3.4x |
| Top 10% | 45% | 29% | 2.8x |
| Top 20% | 36% | 46% | 2.3x |

## 7. Business Impact

Expected profit depends on cost per call and revenue per subscription. Under illustrative assumptions of $8 per outbound call and $80 per subscription:

![threshold vs profit](reports/figures/09_threshold_profit.png)

* Optimal probability threshold: 0.09
* Expected profit on test set: $49,280
* Call everyone baseline profit: $38,416
* Lift over call everyone: +$10,864 (+28 percent)

The real numbers come from the bank's finance team. In production I would sweep across a range of cost to revenue ratios before committing to a specific threshold.

## 8. Feature Importance

![feature importance](reports/figures/10_feature_importance.png)

The Gradient Boosting model's native feature importance confirms the EDA findings almost perfectly:

1. contact_telephone (negative, so cellular wins)
2. age (U shape captured by tree splits)
3. month_oct, month_mar, month_dec, month_sep (the seasonal spikes)
4. poutcome_success (strongest small segment signal)
5. euribor3m and emp.var.rate (macroeconomic context)

That consistency across two independent analyses adds confidence that the model is picking up real signal rather than fitting noise.

## 9. Recommendations

The three concrete recommendations the modeling supports:

1. For marketing analytics: use the model's ranked prospect list as the primary targeting input for the next campaign. At a 10 percent cutoff the call center converts 2.8 times better than random, which translates to materially higher ROI at the same call volume.

2. For call center operations: shift campaign timing into the high conversion months (March, September, October, December) wherever possible. The current pattern of concentrating volume in May is operationally backwards given the data.

3. For the term deposit product team: the student, retired, and unemployed segments subscribe at rates 2 to 3 times above baseline. These segments may be worth a dedicated product tier or communication plan rather than being folded into the general campaign.

## 10. Limitations and Future Improvements

* Sample size and vintage. The data covers 2008 through 2010, which spans the acute phase of the financial crisis but is now more than fifteen years old. Conclusions about specific customer segments should be treated as methodological demonstrations rather than current market truth.
* Missing features. The dataset has no account balance, no transaction history, no full CRM view. A real bank model would combine this campaign data with internal customer features and almost certainly do better.
* Production stack. I used scikit-learn's GradientBoostingClassifier for portability. XGBoost or LightGBM would be drop in replacements with meaningful speed and regularization improvements on the full dataset.
* Interpretation. Native feature importances give a useful ranking but do not explain individual predictions. SHAP values would answer "why is this specific customer on today's call list" and should be the next addition.
* Calibration. Tree ensemble raw probability outputs are notoriously miscalibrated. CalibratedClassifierCV would fix this, which matters for the threshold versus profit analysis to hold up in production.

## 11. Reports and Reference

* [Final Project Report (PDF)](reports/Capstone_Three_Final_Report.pdf)
* [Slide Deck (PPTX)](reports/Capstone_Three_Slide_Deck.pptx)
* [Model Metrics File](reports/Model_Metrics.txt)

## 12. Repository Structure

```
capstone_three_bank_marketing/
├── README.md                            <- this file
├── 01_data_wrangling.ipynb              <- Step 2: cleaning and recoding
├── 02_eda.ipynb                         <- Step 3: exploratory analysis
├── 03_preprocessing_modeling.ipynb      <- Steps 4 and 5: pipeline + modeling
├── bank-additional-full.csv.gz          <- raw input (synthetic sample)
├── bank_marketing_cleaned.csv.gz        <- Step 2 output
├── final_model.joblib                   <- Step 5 output: serialized final model
├── generate_sample_data.py              <- reproduces the sample data
└── reports/
    ├── Capstone_Three_Final_Report.pdf
    ├── Capstone_Three_Slide_Deck.pptx
    ├── Model_Metrics.txt
    └── figures/                         <- charts used in the report and README
```

## 13. How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy joblib jupyter
jupyter notebook
```

Then run the notebooks in order (01 → 03). Every notebook expects the data files in the same directory as the notebook itself, so no path configuration is needed. Pandas handles the .csv.gz files natively.

## 14. Credits

Thanks to my Springboard mentor for guidance across the six step Data Science Method, and to Moro, Cortez and Rita for publishing the original UCI Bank Marketing dataset that this project builds on.
