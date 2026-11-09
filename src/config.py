"""Central settings. Change numbers here (or in the dashboard sliders), not inside the code."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
for _d in (MODELS_DIR, REPORTS_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20           # stratified 80/20 hold-out (BRD section 7)
CV_FOLDS = 5               # stratified 5-fold CV on the training part
N_SEARCH_ITER = int(os.environ.get("CHURN_SEARCH_ITER", 20))  # random-search trials per model (set CHURN_SEARCH_ITER=2 for a quick test)

# Sensitive attributes: used ONLY for the fairness audit, not as model inputs (see MODEL_CARD.md).
SENSITIVE_COLS = ["gender", "SeniorCitizen"]
USE_SENSITIVE_AS_INPUT = False

# Business assumptions from BRD section 8 (illustrative, configurable)
OFFER_COST = 20.0          # USD per contacted customer
OFFER_SUCCESS_RATE = 0.30  # share of contacted churners who accept the offer
VALUE_MONTHS = 12          # customer value = 12 x MonthlyCharges

# Acceptance targets from BRD section 3
TARGETS = {"roc_auc": 0.83, "top_decile_lift": 2.5, "recall": 0.70, "precision": 0.50}
SAFETY_MARGIN = 0.02       # buffer used when picking the threshold, so hold-out stays above target
