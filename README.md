# Customer Churn Prediction and Retention Targeting (BRD-01)

Predicts which ConnectPlus Telecom (fictional) customers are likely to churn, explains why, and tells the retention team
**whom to contact first** so the budget is spent where it returns the most. Built on the IBM Telco Customer Churn dataset
(7,043 customers).

## Results (included run, untouched hold-out set of 1,409 customers)

| BRD target | Result | Status |
|---|---|---|
| ROC-AUC >= 0.83 | 0.844 (95% CI 0.821-0.865) | PASS |
| Top-decile lift >= 2.5x | 2.75x (CI 2.51-3.08) | PASS |
| Recall >= 0.70 | 0.797 | PASS |
| Precision >= 0.50 | 0.510 (CI 0.470-0.551) | PASS (thin margin) |
| Net benefit vs random targeting | +40,624 USD (simulated) | PASS |
| Top-3 drivers for every customer | 7,043 / 7,043 | PASS |

Read `MODEL_CARD.md` and `reports/cost_benefit_analysis.md` for the caveats (thin precision margin, senior-citizen fairness gap,
and the fact that "contact everyone" is almost as profitable as targeting under the BRD's assumed costs).

## Folder map

```
churn_project/
├── README.md                  <- you are here
├── MODEL_CARD.md              <- model card (generated)
├── requirements.txt
├── run_pipeline.py            <- ONE command that runs everything
├── data/                      <- Telco CSV (check Kaggle licence before redistributing)
├── src/                       <- modular code
│   ├── config.py              <- all settings + business assumptions (edit here)
│   ├── data.py                <- load, clean, split
│   ├── features.py            <- feature engineering + preprocessing pipeline
│   ├── modeling.py            <- models, tuning, calibration, ChurnModel
│   ├── evaluation.py          <- metrics, lift, cost-benefit, threshold choice
│   ├── train.py               <- stage 1: train + evaluate + save model
│   ├── explain.py             <- stage 2: SHAP + top-3 drivers
│   ├── fairness.py            <- stage 3: subgroup check
│   ├── plots.py               <- all figures
│   └── reporting.py           <- writes model card + reports
├── notebooks/                 <- 01 EDA, 02 modelling/evaluation (already executed)
├── .vscode/                   <- F5 run configs for VS Code
├── models/churn_model.joblib  <- trained model (pipeline + calibrator + thresholds)
├── reports/                   <- every output (see below)
└── app/streamlit_app.py       <- retention dashboard
```

## Setup and run in VS Code

You need **Python 3.10 or newer** and VS Code with the **Python** and **Jupyter** extensions (both by Microsoft).

1. **Open the folder:** VS Code -> File -> Open Folder -> select `churn_project`. Open the folder itself, not a single file
   inside it, otherwise `import src` fails.
2. **Open a terminal inside VS Code:** Terminal -> New Terminal.
3. **Create the virtual environment and install packages** (one time):

   Windows (Command Prompt is easiest)
   ```
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
   Mac / Linux
   ```
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
   If Windows PowerShell blocks `activate`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or switch the
   terminal to Command Prompt. When VS Code asks "We noticed a new environment, do you want to select it?" click **Yes**.
4. **Select the interpreter (important):** press Ctrl+Shift+P -> `Python: Select Interpreter` -> choose the one inside `venv`
   (`.\venv\Scripts\python.exe` on Windows, `./venv/bin/python` on Mac/Linux). The bottom-right of VS Code should show it.
5. **Run the whole project (a few minutes; most of the time is model tuning):**
   ```
   python run_pipeline.py
   ```
   or press F5 and pick **"1) Run full pipeline"**. It trains and tunes the models, picks thresholds, evaluates on the hold-out
   set, explains every customer with SHAP, scores all 7,043 customers, runs the fairness check, and writes all figures and
   reports. It prints a PASS/FAIL scorecard at the end.
6. **Open the dashboard:**
   ```
   streamlit run app/streamlit_app.py
   ```
   or F5 -> **"2) Streamlit dashboard"**. Ctrl+click the `http://localhost:8501` link in the terminal (a browser tab opens).
   Use the sidebar to change cost assumptions and filters, then download the contact list. Stop it with Ctrl+C in the terminal.
7. **Notebooks:** double-click `notebooks/01_EDA_and_Cleaning.ipynb`, then `02_...`. At the top right click **Select Kernel** ->
   Python Environments -> the `venv` one. They are already executed, so you can read the outputs immediately; use "Run All"
   to re-run. Notebook 02 reads the files created in step 5, so run step 5 first if you changed anything.
8. **Read the reports:** open `MODEL_CARD.md` or any `.md` file in `reports/` and press Ctrl+Shift+V for a formatted preview.
   Figures are in `reports/figures/` (click a `.png` to view it).

The random seed is fixed, so re-running gives the same numbers.

**Tested:** full pipeline, dashboard (headless) and notebooks in a fresh virtual environment built from `requirements.txt` (Python 3.12). The notebooks were executed without errors.

## Where is each BRD deliverable?

| BRD item | File |
|---|---|
| BR-01 churn probability for every customer | `reports/scored_customers.csv` -> `churn_probability` |
| BR-02 High / Medium / Low tier | `reports/scored_customers.csv` -> `risk_tier` |
| BR-03 top-3 drivers per customer | `reports/scored_customers.csv` -> `driver_1..3`; `reports/shap_report.md` |
| BR-04 cost-based threshold | `reports/cost_benefit_analysis.md`, `reports/threshold_analysis_*.csv` |
| BR-05 dashboard + export | `app/streamlit_app.py` |
| BR-06 subgroup comparison | `reports/fairness_by_group.csv`, `MODEL_CARD.md` section 6 |
| Leakage review | `reports/leakage_review.md`, `reports/leakage_audit.csv` |
| EDA / modelling notebooks | `notebooks/` |
| Trained model | `models/churn_model.joblib` |

## Using the saved model in your own code

```python
import joblib, pandas as pd
from src.modeling import ChurnModel, PlattCalibrator      # needed so joblib can unpickle
model = joblib.load("models/churn_model.joblib")
p = model.predict_proba(new_customers_df)   # df has the same raw columns as the Telco CSV
tier = model.risk_tier(p)                   # "High" / "Medium" / "Low"
```

## Changing assumptions
Edit `src/config.py` (offer cost, success rate, value months, targets, seed) and re-run `python run_pipeline.py`.
The dashboard sliders change the assumptions for the contact list only (the model and tiers stay the same).

## Troubleshooting
* `ModuleNotFoundError: src` -> you opened a sub-folder or a single file; use File -> Open Folder on `churn_project`.
* Notebook shows `ModuleNotFoundError` (pandas, shap...) or kernel not found -> top right Select Kernel -> pick the `venv` interpreter.
* Dashboard says "No results found" -> run `python run_pipeline.py` first.
* `python` not found on Mac/Linux -> use `python3`.
* Slow install -> `shap` and `xgboost` are the big packages; just wait.

## Data note
Dataset: IBM Telco Customer Churn (Kaggle, `blastchar/telco-customer-churn`). It is a sample for a fictional company; cost parameters
are assumptions, not observed values. Review the Kaggle licence before redistributing the data.
