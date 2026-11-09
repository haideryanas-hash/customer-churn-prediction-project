# Model Card - Customer Churn Prediction (BRD-01)

## 1. Model details
* **Model:** logistic regression inside an sklearn Pipeline (scaling + one-hot encoding) with Platt-scaling calibration. Tuned parameters: `{'C': 0.0315}`.
* **Saved as:** `models/churn_model.joblib` (a `ChurnModel` object: pipeline + calibrator + thresholds).
* **Why this model:** all three candidates scored within about 0.002 ROC-AUC in cross-validation (table below). The project picks the simplest model within 0.003 of the best, because the dataset is small and a linear model is easier to explain and less likely to overfit.
* **Data:** IBM Telco Customer Churn sample (7,043 customers, fictional company, single snapshot). Target = churned in the last month (26.5%).
* **Inputs:** 24 engineered/raw features (tenure, charges, contract, payment, services, add-on count, price-change gap, ...). `gender` and `SeniorCitizen` are deliberately **not** inputs. `TotalCharges` blanks (11 customers, tenure 0) were set to 0.

## 2. Intended use
Rank customers by churn risk so the retention team contacts the highest-value-to-save customers first, with a reason for each. **Not** intended for automated decisions without human review, or for pricing, credit or any non-retention use. Not validated for other companies or time periods.

## 3. Training procedure
Stratified 80/20 split (5,634 train / 1,409 hold-out). 5-fold stratified CV on the training part. Class imbalance is handled with class weights. The study below shows weights barely change ranking quality, so probability calibration + threshold selection is what actually controls the precision/recall trade-off.

| model | class_weights | cv_roc_auc | cv_pr_auc |
|---|---|---|---|
| logistic_regression | False | 0.847 | 0.663 |
| logistic_regression | True | 0.847 | 0.661 |
| random_forest | False | 0.846 | 0.661 |
| random_forest | True | 0.846 | 0.661 |
| xgboost | False | 0.844 | 0.659 |
| xgboost | True | 0.846 | 0.661 |

Tuned model comparison (5-fold CV, training data):

| model | cv_roc_auc | cv_roc_auc_std | cv_pr_auc |
|---|---|---|---|
| logistic_regression | 0.847 | 0.012 | 0.662 |
| random_forest | 0.847 | 0.010 | 0.662 |
| xgboost | 0.849 | 0.011 | 0.666 |

## 4. Performance on the untouched hold-out set
| Metric | Hold-out value | BRD target | 95% bootstrap CI | Result |
|---|---|---|---|---|
| roc_auc | 0.844 | >= 0.83 | [0.821, 0.865] | PASS |
| top_decile_lift | 2.752 | >= 2.5 | [2.505, 3.077] | PASS |
| recall | 0.797 | >= 0.7 | [0.756, 0.835] | PASS |
| precision | 0.510 | >= 0.5 | [0.470, 0.551] | PASS |
| net_benefit_vs_random | 40624.247 | > 0 | - | PASS |

Threshold used: **0.26** (chosen on training out-of-fold predictions; see `reports/cost_benefit_analysis.md`).

**Read this honestly:** every target is met on the point estimate, but with only 1,409 hold-out customers the uncertainty is real. The lower bounds of the precision interval (0.470) and ROC-AUC interval (0.821) are below their targets, so a different random split could land just under. Precision (0.510) passes by a thin margin.

## 5. Risk tiers (BR-02)
High = probability >= 0.26; Medium = between 0.08 and 0.26; Low = below 0.08 (below the cost break-even, so an offer is not worth its cost). Hold-out validation (actual churn rate should rise from Low to High):

| risk_tier | customers | actual_churn_rate | avg_predicted |
|---|---|---|---|
| High | 584 | 0.510 | 0.526 |
| Medium | 358 | 0.176 | 0.157 |
| Low | 467 | 0.028 | 0.033 |

## 6. Fairness check (BR-06) - hold-out, at the operating threshold
| attribute | group | customers | actual_churn_rate | roc_auc | recall_tpr | precision | false_positive_rate | flag_rate |
|---|---|---|---|---|---|---|---|---|
| gender | Female | 687 | 0.281 | 0.840 | 0.793 | 0.539 | 0.265 | 0.413 |
| gender | Male | 722 | 0.251 | 0.848 | 0.801 | 0.483 | 0.287 | 0.416 |
| SeniorCitizen | 0 | 1187 | 0.233 | 0.846 | 0.757 | 0.479 | 0.249 | 0.367 |
| SeniorCitizen | 1 | 222 | 0.441 | 0.779 | 0.908 | 0.601 | 0.476 | 0.667 |

**Findings**
* **Gender:** small differences (recall 0.79 vs 0.80; precision 0.54 vs 0.48). With about 700 customers per group these gaps are within normal noise.
* **Senior citizens:** actual churn is higher (44% vs 23%) and the model flags 67% of them vs 37% of non-seniors. Recall is higher for seniors (0.91 vs 0.76) but so is the false-positive rate (0.48 vs 0.25), and ROC-AUC is lower (0.78 vs 0.85). Only 222 seniors are in the hold-out, so this is indicative, not conclusive.
* `gender` and `SeniorCitizen` are **not model inputs**; the gap comes from correlated features (e.g. fiber optic, high monthly charges). A false positive only costs an extra retention offer, so the harm is low, but **Compliance should review** this before launch and decide whether to monitor flag rates by group.

## 7. Limitations
* Single snapshot, no dates: no time-based validation, possible hidden leakage (see `reports/leakage_review.md`).
* Business numbers (offer cost, success rate, customer value) are assumptions. Under them, contacting everyone is nearly as profitable as targeting the High tier (see the cost-benefit report), so confirm the real costs.
* The model predicts who is likely to leave, **not who will respond to an offer** (uplift modelling is out of scope).
* Calibration was fitted on cross-validated predictions; re-check it whenever the model is retrained or the customer base changes.

## 8. Retraining and monitoring
`python run_pipeline.py` retrains everything and regenerates all reports. Monitor monthly: ROC-AUC, actual churn rate per tier, and calibration.
