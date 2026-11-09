"""All figures are saved to reports/figures/ (and returned so notebooks can show them)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from .config import FIG_DIR
from .evaluation import lift_table
from .features import add_features, readable

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 9})
BLUE, RED, GREY = "#2b6cb0", "#c0392b", "#8a8f98"


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, bbox_inches="tight")
    return fig


def eda_figures(df: pd.DataFrame):
    d = add_features(df)
    fig, ax = plt.subplots(2, 3, figsize=(14, 7.5))
    for a, col in zip(ax.flat[:4], ["Contract", "InternetService", "PaymentMethod", "tenure_bucket"]):
        order = d.groupby(col)["Churn"].mean().sort_values(ascending=False)
        if col == "tenure_bucket":
            order = order.reindex(["0-6m", "7-12m", "13-24m", "25-48m", "49m+"])
        a.bar(order.index, order.values * 100, color=BLUE)
        a.axhline(d["Churn"].mean() * 100, color=RED, ls="--", lw=1, label="overall 26.5%")
        a.set_title(f"Churn rate by {col}"); a.set_ylabel("churn %"); a.tick_params(axis="x", rotation=20)
    ax[0, 0].legend()
    for churn, c, lab in ((0, BLUE, "stayed"), (1, RED, "churned")):
        ax[1, 1].hist(d.loc[d.Churn == churn, "MonthlyCharges"], bins=30, alpha=0.6, color=c, label=lab, density=True)
    ax[1, 1].set_title("MonthlyCharges distribution"); ax[1, 1].legend()
    g = d.groupby("n_addon_services")["Churn"].agg(["mean", "size"])
    ax[1, 2].bar(g.index, g["mean"] * 100, color=BLUE)
    ax[1, 2].set_title("Churn rate by number of add-on services"); ax[1, 2].set_xlabel("add-ons subscribed")
    ax[1, 2].set_ylabel("churn %")
    fig.suptitle("EDA: what separates churners from stayers", fontsize=13, y=1.0)
    return _save(fig, "eda_overview.png")


def model_figures(y, p, oof_y, oof_p, sweep_holdout, th):
    y, p = np.asarray(y), np.asarray(p)
    fig, ax = plt.subplots(2, 3, figsize=(14, 7.5))
    fpr, tpr, _ = roc_curve(y, p)
    ax[0, 0].plot(fpr, tpr, color=BLUE, label=f"AUC {roc_auc_score(y, p):.3f}"); ax[0, 0].plot([0, 1], [0, 1], "--", color=GREY)
    ax[0, 0].set(title="ROC curve (hold-out)", xlabel="False positive rate", ylabel="True positive rate"); ax[0, 0].legend()
    pr, rc, _ = precision_recall_curve(y, p)
    ax[0, 1].plot(rc, pr, color=BLUE, label=f"PR-AUC {average_precision_score(y, p):.3f}")
    ax[0, 1].axhline(y.mean(), ls="--", color=GREY, label="random")
    ax[0, 1].set(title="Precision-Recall (hold-out)", xlabel="Recall", ylabel="Precision"); ax[0, 1].legend()
    lt = lift_table(y, p)
    ax[0, 2].bar(lt["decile"], lt["lift"], color=BLUE); ax[0, 2].axhline(1, ls="--", color=GREY)
    ax[0, 2].set(title=f"Lift by decile (top decile = {lt['lift'].iloc[0]:.2f}x)", xlabel="decile (1 = highest risk)", ylabel="lift vs random")
    ax[1, 0].plot(np.r_[0, np.arange(1, 11) / 10], np.r_[0, lt["cum_churners_captured"]], marker="o", color=BLUE, label="model")
    ax[1, 0].plot([0, 1], [0, 1], "--", color=GREY, label="random")
    ax[1, 0].set(title="Cumulative gains", xlabel="share of customers contacted", ylabel="share of churners reached"); ax[1, 0].legend()
    fp, mp = calibration_curve(y, p, n_bins=8, strategy="quantile")
    ax[1, 1].plot(mp, fp, marker="o", color=BLUE, label="model"); ax[1, 1].plot([0, 1], [0, 1], "--", color=GREY, label="perfect")
    ax[1, 1].set(title="Calibration (hold-out)", xlabel="predicted probability", ylabel="actual churn rate"); ax[1, 1].legend()
    ax[1, 2].plot(sweep_holdout["threshold"], sweep_holdout["net_benefit"] / 1000, color=BLUE, label="model targeting")
    ax[1, 2].plot(sweep_holdout["threshold"], sweep_holdout["random_net_benefit"] / 1000, color=GREY, ls="--", label="random, same #contacted")
    ax[1, 2].axvline(th["t_op"], color=RED, ls=":", label=f"operating t={th['t_op']:.2f}")
    ax[1, 2].axvline(th["t_cost"], color="green", ls=":", label=f"cost-optimal t={th['t_cost']:.2f}")
    ax[1, 2].set(title="Simulated campaign net benefit (hold-out)", xlabel="probability threshold", ylabel="net benefit (USD thousand)"); ax[1, 2].legend(fontsize=7)
    return _save(fig, "model_performance.png")


def threshold_tradeoff(sweep, th):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(sweep["threshold"], sweep["recall"], label="recall", color=BLUE)
    ax.plot(sweep["threshold"], sweep["precision"], label="precision", color=RED)
    ax.axhline(0.70, color=BLUE, ls=":", lw=1); ax.axhline(0.50, color=RED, ls=":", lw=1)
    ax.axvline(th["t_op"], color="k", ls="--", label=f"chosen t={th['t_op']:.2f}")
    ax.set(title="Precision / recall vs threshold (training out-of-fold)", xlabel="threshold", ylabel="score"); ax.legend()
    return _save(fig, "threshold_tradeoff.png")


def shap_figures(sv, Xt, names, importance):
    top = importance.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top["feature"], top["mean_abs_shap"], color=BLUE)
    ax.set(title="Global churn drivers (mean |SHAP| per feature)", xlabel="average impact on churn log-odds")
    _save(fig, "shap_global_importance.png")
    plt.figure(figsize=(8, 6))
    shap.summary_plot(sv, Xt, feature_names=[readable(n) for n in names], max_display=15, show=False)
    plt.title("SHAP summary: red = high feature value, right = pushes churn risk up")
    fig2 = plt.gcf(); fig2.tight_layout(); fig2.savefig(FIG_DIR / "shap_beeswarm.png", bbox_inches="tight")
    return fig, fig2


def fairness_figure(fdf):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    labels = fdf["attribute"] + "=" + fdf["group"]
    x = np.arange(len(fdf)); w = 0.38
    ax[0].bar(x - w / 2, fdf["recall_tpr"], w, label="recall", color=BLUE); ax[0].bar(x + w / 2, fdf["precision"], w, label="precision", color=RED)
    ax[0].set_xticks(x, labels, rotation=20); ax[0].set_title("Recall / precision by group"); ax[0].legend()
    ax[1].bar(x - w / 2, fdf["actual_churn_rate"], w, label="actual churn rate", color=GREY); ax[1].bar(x + w / 2, fdf["avg_predicted_prob"], w, label="avg predicted prob", color=BLUE)
    ax[1].set_xticks(x, labels, rotation=20); ax[1].set_title("Calibration by group"); ax[1].legend()
    return _save(fig, "fairness_by_group.png")
