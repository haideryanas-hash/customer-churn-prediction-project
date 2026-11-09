"""Feature engineering + preprocessing (all fitting happens inside the sklearn Pipeline -> no leakage)."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import USE_SENSITIVE_AS_INPUT

ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "n_addon_services", "avg_monthly_spend",
            "charge_gap", "is_autopay", "has_protection", "has_family"]
CAT_COLS = ["PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
            "PaperlessBilling", "PaymentMethod", "Partner", "Dependents", "tenure_bucket"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level features only (never uses other rows or the target)."""
    d = df.copy()
    d["n_addon_services"] = sum((d[c] == "Yes").astype(int) for c in ADDONS)
    d["avg_monthly_spend"] = np.where(d["tenure"] > 0, d["TotalCharges"] / d["tenure"].clip(lower=1),
                                      d["MonthlyCharges"])
    d["charge_gap"] = d["MonthlyCharges"] - d["avg_monthly_spend"]   # > 0 means bill rose vs history
    d["is_autopay"] = d["PaymentMethod"].str.contains("automatic").astype(int)
    d["has_protection"] = ((d["OnlineSecurity"] == "Yes") | (d["TechSupport"] == "Yes")).astype(int)
    d["has_family"] = ((d["Partner"] == "Yes") | (d["Dependents"] == "Yes")).astype(int)
    d["tenure_bucket"] = pd.cut(d["tenure"], [-1, 6, 12, 24, 48, 100],
                                labels=["0-6m", "7-12m", "13-24m", "25-48m", "49m+"]).astype(str)
    return d


def feature_lists():
    num, cat = list(NUM_COLS), list(CAT_COLS)
    if USE_SENSITIVE_AS_INPUT:
        cat += ["gender"]
        num += ["SeniorCitizen"]
    return num, cat


def build_preprocessor() -> ColumnTransformer:
    num, cat = feature_lists()
    return ColumnTransformer(
        [("num", StandardScaler(), num),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)],
        verbose_feature_names_out=False)


def make_X(df: pd.DataFrame) -> pd.DataFrame:
    num, cat = feature_lists()
    return add_features(df)[num + cat]


def original_column(feature_name: str) -> str:
    """'Contract_Month-to-month' -> 'Contract' (numeric names are returned unchanged)."""
    _, cat = feature_lists()
    for c in sorted(cat, key=len, reverse=True):
        if feature_name.startswith(c + "_"):
            return c
    return feature_name


def readable(feature_name: str) -> str:
    col = original_column(feature_name)
    return f"{col} = {feature_name[len(col) + 1:]}" if col != feature_name else feature_name
