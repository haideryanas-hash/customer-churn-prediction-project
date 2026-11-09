"""Model zoo, tuning, probability calibration and the saved ChurnModel wrapper."""
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold, cross_val_predict,
                                     cross_val_score, cross_validate)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .config import CV_FOLDS, N_SEARCH_ITER, RANDOM_STATE
from .features import build_preprocessor, make_X

MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost"]   # order = simplicity (used for tie-break)
TREE_MODELS = {"random_forest", "xgboost"}


def cv_splitter():
    return StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def make_pipeline(name: str, weighted: bool = True, pos_weight: float = 1.0, n_jobs: int = 1) -> Pipeline:
    """Class-imbalance handling = class weights (weighted=True)."""
    if name == "logistic_regression":
        m = LogisticRegression(max_iter=3000, C=0.1, class_weight="balanced" if weighted else None)
    elif name == "random_forest":
        m = RandomForestClassifier(n_estimators=300, min_samples_leaf=10, max_features="sqrt", n_jobs=n_jobs,
                                   class_weight="balanced_subsample" if weighted else None,
                                   random_state=RANDOM_STATE)
    elif name == "xgboost":
        m = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8,
                          eval_metric="logloss", n_jobs=n_jobs, random_state=RANDOM_STATE,
                          scale_pos_weight=pos_weight if weighted else 1.0)
    else:
        raise ValueError(name)
    return Pipeline([("prep", build_preprocessor()), ("model", m)])


SEARCH_SPACES = {
    "logistic_regression": {"model__C": loguniform(1e-3, 10)},
    "random_forest": {"model__n_estimators": randint(200, 500), "model__max_depth": [4, 6, 8, 12, None],
                      "model__min_samples_leaf": randint(3, 30), "model__max_features": ["sqrt", 0.3, 0.5]},
    "xgboost": {"model__n_estimators": randint(150, 600), "model__learning_rate": loguniform(0.01, 0.15),
                "model__max_depth": randint(2, 6), "model__min_child_weight": randint(1, 10),
                "model__subsample": uniform(0.6, 0.4), "model__colsample_bytree": uniform(0.5, 0.5),
                "model__reg_lambda": loguniform(0.5, 20)},
}


# ---------- study 1: does imbalance handling help? ----------
def imbalance_study(X, y, pos_weight):
    rows = []
    for name in MODEL_NAMES:
        for weighted in (False, True):
            r = cross_validate(make_pipeline(name, weighted, pos_weight), X, y, cv=cv_splitter(),
                               scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision"}, n_jobs=-1)
            rows.append({"model": name, "class_weights": weighted, "cv_roc_auc": r["test_roc_auc"].mean(),
                         "cv_pr_auc": r["test_pr_auc"].mean()})
    return pd.DataFrame(rows)


# ---------- study 2: tune every model with random search ----------
def tune_all(X, y, pos_weight):
    results, best = [], {}
    for name in MODEL_NAMES:
        base = make_pipeline(name, True, pos_weight)
        n_iter = N_SEARCH_ITER
        if name == "logistic_regression":
            n_iter = 12
        s = RandomizedSearchCV(base, SEARCH_SPACES[name], n_iter=n_iter, scoring="roc_auc", cv=cv_splitter(),
                               random_state=RANDOM_STATE, n_jobs=-1, refit=True)
        s.fit(X, y)
        cv = cross_validate(s.best_estimator_, X, y, cv=cv_splitter(), scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision"},
                            n_jobs=-1)
        results.append({"model": name, "cv_roc_auc": cv["test_roc_auc"].mean(), "cv_roc_auc_std": cv["test_roc_auc"].std(),
                        "cv_pr_auc": cv["test_pr_auc"].mean(), "best_params": {k.replace("model__", ""): (round(v, 4) if isinstance(v, float) else v)
                                                                               for k, v in s.best_params_.items()}})
        best[name] = s.best_estimator_
    return pd.DataFrame(results), best


def pick_model(cv_table: pd.DataFrame, tolerance=0.003):
    """Parsimony rule: the simplest model whose CV ROC-AUC is within `tolerance` of the best.
    Reason: small dataset -> simpler model = lower overfitting risk and easier to explain."""
    top = cv_table["cv_roc_auc"].max()
    ok = cv_table[cv_table["cv_roc_auc"] >= top - tolerance]
    return min(ok["model"], key=MODEL_NAMES.index)


# ---------- calibration ----------
class PlattCalibrator:
    """Maps raw model scores to real-world churn probabilities (monotonic, so ranking is unchanged).
    Needed because class weights inflate raw scores, and the cost model needs honest probabilities."""

    def fit(self, raw, y):
        self.lr = LogisticRegression(C=1e6, max_iter=1000).fit(self._logit(raw), y)
        return self

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def transform(self, raw):
        return self.lr.predict_proba(self._logit(raw))[:, 1]


class ChurnModel:
    """The one object saved to disk: preprocessing + model + calibrator + thresholds."""

    def __init__(self, name, pipeline, calibrator, thresholds, params=None):
        self.name, self.pipeline, self.calibrator = name, pipeline, calibrator
        self.t_op, self.t_floor = thresholds["t_op"], min(thresholds["t_cost"], thresholds["t_op"])
        self.params = params or {}

    def predict_proba(self, df_raw: pd.DataFrame) -> np.ndarray:
        return self.calibrator.transform(self.pipeline.predict_proba(make_X(df_raw))[:, 1])

    def risk_tier(self, p) -> np.ndarray:
        p = np.asarray(p)
        return np.where(p >= self.t_op, "High", np.where(p >= self.t_floor, "Medium", "Low"))


def oof_scores(pipeline, X, y):
    """Out-of-fold raw scores on the training set (every row is scored by a model that never saw it)."""
    return cross_val_predict(pipeline, X, y, cv=cv_splitter(), method="predict_proba", n_jobs=-1)[:, 1]


# ---------- leakage audit ----------
def leakage_audit(X: pd.DataFrame, y):
    """Fit a tiny depth-3 tree on each single feature. A feature with a suspiciously high AUC
    (> 0.85) would indicate it secretly contains the answer (target leakage)."""
    rows = []
    for col in X.columns:
        col_x = pd.get_dummies(X[[col]]) if X[col].dtype == object or str(X[col].dtype) == "str" else X[[col]]
        auc = cross_val_score(DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE), col_x, y,
                              cv=cv_splitter(), scoring="roc_auc").mean()
        rows.append({"feature": col, "single_feature_auc": round(float(auc), 4), "flag": "CHECK" if auc > 0.85 else "ok"})
    return pd.DataFrame(rows).sort_values("single_feature_auc", ascending=False)
