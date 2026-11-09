"""Metrics, lift/gain tables, cost-benefit model and threshold selection."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import OFFER_COST, OFFER_SUCCESS_RATE, SAFETY_MARGIN, TARGETS, VALUE_MONTHS


# ---------- core metrics ----------
def customer_value(monthly_charges, months=VALUE_MONTHS):
    return np.asarray(monthly_charges, dtype=float) * months


def top_decile_lift(y, p, frac=0.10):
    y, p = np.asarray(y), np.asarray(p)
    n = int(np.ceil(len(y) * frac))
    idx = np.argsort(-p, kind="stable")[:n]
    return float(y[idx].mean() / y.mean())


def metrics_at(y, p, t):
    y, flag = np.asarray(y), np.asarray(p) >= t
    tp = int((flag & (y == 1)).sum())
    prec = tp / flag.sum() if flag.sum() else 0.0
    rec = tp / (y == 1).sum()
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"threshold": float(t), "precision": float(prec), "recall": float(rec), "f1": float(f1),
            "n_flagged": int(flag.sum()), "share_flagged": float(flag.mean())}


def summary_metrics(y, p, t):
    return {"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
            "top_decile_lift": top_decile_lift(y, p), **metrics_at(y, p, t)}


def lift_table(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(-p, kind="stable")
    d = pd.DataFrame({"y": y[order], "p": p[order]})
    d["decile"] = np.arange(len(d)) * bins // len(d) + 1
    g = d.groupby("decile").agg(customers=("y", "size"), churners=("y", "sum"), avg_prob=("p", "mean"))
    g["churn_rate"] = g["churners"] / g["customers"]
    g["lift"] = g["churn_rate"] / y.mean()
    g["cum_churners_captured"] = g["churners"].cumsum() / y.sum()
    return g.reset_index()


# ---------- cost-benefit model (BR-04) ----------
def net_benefit(y, flag, value, offer_cost=OFFER_COST, success=OFFER_SUCCESS_RATE):
    """Realised net benefit of contacting the flagged customers:
    success_rate x value of the churners we reached  -  offer cost x number contacted."""
    y, flag, value = np.asarray(y), np.asarray(flag).astype(bool), np.asarray(value)
    return float(success * value[flag & (y == 1)].sum() - offer_cost * flag.sum())


def random_net_benefit(y, value, n_contact, offer_cost=OFFER_COST, success=OFFER_SUCCESS_RATE):
    """Expected net benefit if the same number of customers were picked at random."""
    y, value = np.asarray(y), np.asarray(value)
    return float(n_contact * (success * np.mean(value * y) - offer_cost))


def threshold_sweep(y, p, value, grid=None, offer_cost=OFFER_COST, success=OFFER_SUCCESS_RATE):
    grid = np.round(np.arange(0.02, 0.96, 0.01), 2) if grid is None else grid
    rows = []
    for t in grid:
        m = metrics_at(y, p, t)
        flag = np.asarray(p) >= t
        m["net_benefit"] = net_benefit(y, flag, value, offer_cost, success)
        m["random_net_benefit"] = random_net_benefit(y, value, m["n_flagged"], offer_cost, success)
        rows.append(m)
    return pd.DataFrame(rows)


def select_thresholds(sweep: pd.DataFrame):
    """t_cost : threshold with the highest net benefit (pure cost-benefit answer).
    t_op   : threshold with the highest net benefit among those that also satisfy the BRD
             recall/precision targets (with a small safety margin). This is the operating threshold."""
    t_cost = float(sweep.loc[sweep["net_benefit"].idxmax(), "threshold"])
    note = ""
    for margin in (SAFETY_MARGIN, 0.0):
        ok = sweep[(sweep["recall"] >= TARGETS["recall"] + margin) & (sweep["precision"] >= TARGETS["precision"] + margin)]
        if len(ok):
            t_op = float(ok.loc[ok["net_benefit"].idxmax(), "threshold"])
            note = f"recall/precision targets met with margin {margin:.2f}"
            break
    else:
        t_op = t_cost
        note = "WARNING: no threshold meets both recall and precision targets; using cost-optimal threshold"
    return {"t_cost": t_cost, "t_op": t_op, "note": note,
            "feasible_range": [float(ok["threshold"].min()), float(ok["threshold"].max())] if len(ok) else None}


def bootstrap_ci(y, p, t, n_boot=1000, seed=42):
    """95% bootstrap confidence interval for the headline hold-out metrics.
    Shows how much a number could move just by luck of which customers landed in the hold-out."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        m = summary_metrics(y[i], p[i], t)
        rows.append({k: m[k] for k in ("roc_auc", "top_decile_lift", "recall", "precision")})
    d = pd.DataFrame(rows)
    return {c: [float(d[c].quantile(0.025)), float(d[c].quantile(0.975))] for c in d}
