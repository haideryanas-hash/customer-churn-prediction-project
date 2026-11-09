# Leakage Review (BRD acceptance criterion)

| Check | Result |
|---|---|
| Hold-out is split first (stratified 80/20) and evaluated **once**, after all choices are frozen | Yes - `src/train.py` |
| Scaling, one-hot encoding and model fitting happen **inside one sklearn Pipeline**, refitted per CV fold | Yes - `src/features.py`, `src/modeling.py` |
| Feature engineering is **row-level only** (no group means, no target encoding, no use of other rows) | Yes - `add_features()` |
| Target `Churn` never used to build a feature; `customerID` dropped | Yes |
| Probability calibrator and thresholds are fitted on **training out-of-fold** predictions, not on the hold-out | Yes |
| Model choice made with CV on the training part only | Yes |
| `TotalCharges` blanks (11 rows, all tenure = 0) set to 0 - deterministic rule, nothing learned from data | Yes |
| Suspicious single features (single-feature AUC > 0.85) | None - see table below |
| Sensitive attributes (`gender`, `SeniorCitizen`) kept out of the model inputs | Yes - used only for the fairness audit |

**Single-feature AUC scan** (a depth-3 tree on one feature at a time, 5-fold CV on training data). A leaked column would score near 1.0.

| feature | single_feature_auc | flag |
|---|---|---|
| Contract | 0.740 | ok |
| tenure | 0.730 | ok |
| tenure_bucket | 0.716 | ok |
| OnlineSecurity | 0.708 | ok |
| TechSupport | 0.705 | ok |
| InternetService | 0.698 | ok |
| OnlineBackup | 0.679 | ok |
| PaymentMethod | 0.676 | ok |

**Remaining caveat:** the dataset is one snapshot with no dates, so "customer churned within the last month" is described with features measured at the same moment. We cannot prove that every feature was known *before* the churn happened. Before production use, confirm with the data owner that all features are available at scoring time.
