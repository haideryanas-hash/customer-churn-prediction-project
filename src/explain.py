"""Stage 2 - SHAP explainability (global + per-customer top-3 drivers)."""
import numpy as np
import pandas as pd
import shap

from .features import make_X, original_column, readable


def _explainer(model, background_t):
    est = model.pipeline.named_steps["model"]
    if model.name == "logistic_regression":
        return shap.LinearExplainer(est, background_t)
    return shap.TreeExplainer(est)


def shap_values(model, df, background_df):
    """Returns (shap matrix [n x features], transformed X, feature names). Positive SHAP = pushes churn risk UP."""
    prep = model.pipeline.named_steps["prep"]
    Xt = prep.transform(make_X(df))
    bg = prep.transform(make_X(background_df))
    bg = bg[np.random.default_rng(0).choice(len(bg), min(100, len(bg)), replace=False)]
    sv = _explainer(model, bg).shap_values(Xt)
    if isinstance(sv, list):          # older shap: list per class
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:                  # newer shap: (n, features, classes)
        sv = sv[:, :, 1]
    return sv, Xt, list(prep.get_feature_names_out())


# Columns that describe the same business idea are merged, so the "top 3 drivers" are not three
# near-duplicates (e.g. tenure, tenure bucket and total charges are all "how long have they been with us").
GROUPS = {"tenure_bucket": "tenure", "TotalCharges": "tenure",
          "avg_monthly_spend": "MonthlyCharges", "charge_gap": "MonthlyCharges",
          "has_family": "Partner/Dependents", "Partner": "Partner/Dependents", "Dependents": "Partner/Dependents",
          "is_autopay": "PaymentMethod",
          "has_protection": "Security/TechSupport", "OnlineSecurity": "Security/TechSupport", "TechSupport": "Security/TechSupport"}


def describe(group, row):
    """Human-readable 'what is true for THIS customer' text for a driver group."""
    if group == "tenure":
        return f"tenure = {row['tenure']:g} months"
    if group == "MonthlyCharges":
        return f"MonthlyCharges = ${row['MonthlyCharges']:.2f}"
    if group == "Partner/Dependents":
        return f"Partner = {row['Partner']}, Dependents = {row['Dependents']}"
    if group == "Security/TechSupport":
        return f"OnlineSecurity = {row['OnlineSecurity']}, TechSupport = {row['TechSupport']}"
    v = row[group]
    return f"{group} = {v:g}" if isinstance(v, (int, float, np.number)) else f"{group} = {v}"


def by_original_column(sv, names):
    """Sum SHAP values of one-hot columns and of same-idea columns into one number per driver group."""
    groups = pd.Series([GROUPS.get(original_column(n), original_column(n)) for n in names])
    return pd.DataFrame({g: sv[:, np.where(groups == g)[0]].sum(axis=1) for g in groups.unique()})


def top_drivers(model, df, background_df, k=3):
    """Top-k risk-increasing drivers for every customer, e.g. 'Contract = Month-to-month (+0.82)'."""
    sv, _, names = shap_values(model, df, background_df)
    col_sv = by_original_column(sv, names)
    X = make_X(df).reset_index(drop=True)
    out = {}
    arr = col_sv.values
    order = np.argsort(-arr, axis=1)[:, :k]          # most risk-increasing first
    for j in range(k):
        cols = col_sv.columns[order[:, j]]
        effect = arr[np.arange(len(arr)), order[:, j]]
        out[f"driver_{j + 1}"] = [describe(c, X.iloc[i]) for i, c in enumerate(cols)]
        out[f"driver_{j + 1}_effect"] = np.round(effect, 3)
    return pd.DataFrame(out), col_sv, sv, names


def global_importance(sv, names):
    col_sv = by_original_column(sv, names)
    return (col_sv.abs().mean().sort_values(ascending=False).rename("mean_abs_shap").reset_index()
            .rename(columns={"index": "feature"}))


def level_effects(sv, Xt, names, min_support=100):
    """Average SHAP of each category level among customers who have it (what raises / lowers risk)."""
    rows = []
    for j, n in enumerate(names):
        if original_column(n) != n:  # categorical level
            mask = Xt[:, j] == 1
            if mask.sum() >= min_support:
                rows.append({"level": readable(n), "customers": int(mask.sum()), "avg_shap": float(sv[mask, j].mean())})
    return pd.DataFrame(rows).sort_values("avg_shap", ascending=False)
