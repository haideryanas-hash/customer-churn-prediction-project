"""Stage 3 - subgroup fairness check on the hold-out set (BR-06)."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import SENSITIVE_COLS


def fairness_report(test: pd.DataFrame, p, threshold):
    """Compare model quality across gender and senior-citizen groups at the operating threshold."""
    p = np.asarray(p)
    rows = []
    for attr in SENSITIVE_COLS:
        for g in sorted(test[attr].unique()):
            idx = np.where(test[attr].values == g)[0]
            y, pp = test["Churn"].values[idx], p[idx]
            flag = pp >= threshold
            tp = (flag & (y == 1)).sum()
            rows.append({"attribute": attr, "group": str(g), "customers": len(idx),
                         "actual_churn_rate": y.mean(), "avg_predicted_prob": pp.mean(),
                         "roc_auc": roc_auc_score(y, pp), "recall_tpr": tp / max((y == 1).sum(), 1),
                         "precision": tp / max(flag.sum(), 1), "false_positive_rate": (flag & (y == 0)).sum() / max((y == 0).sum(), 1),
                         "flag_rate": flag.mean()})
    df = pd.DataFrame(rows)
    gaps = (df.groupby("attribute")
              .agg(recall_gap=("recall_tpr", lambda s: s.max() - s.min()),
                   fpr_gap=("false_positive_rate", lambda s: s.max() - s.min()),
                   auc_gap=("roc_auc", lambda s: s.max() - s.min()),
                   flag_rate_ratio_min_over_max=("flag_rate", lambda s: s.min() / s.max()),
                   smallest_group=("customers", "min")).reset_index())
    return df, gaps
