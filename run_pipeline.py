"""Run the WHOLE project end to end:   python run_pipeline.py

Stages: train -> SHAP -> score every customer -> fairness -> figures -> written reports.
All outputs land in models/ and reports/.
"""
import json

import numpy as np
import pandas as pd

from src import explain, fairness, plots, reporting, train
from src.config import OFFER_COST, OFFER_SUCCESS_RATE, REPORTS_DIR
from src.data import load_clean
from src.evaluation import customer_value
from src.features import make_X


def main():
    model, tr, te, res = train.run()
    full = load_clean()
    t = res["thresholds"]

    print("\n[SHAP] Explaining every customer ...")
    drivers, col_sv, sv_all, names = explain.top_drivers(model, full, tr)
    sv_te, Xt_te, names = explain.shap_values(model, te, tr)
    imp = explain.global_importance(sv_te, names)
    imp.to_csv(REPORTS_DIR / "shap_global_importance.csv", index=False)
    lev = explain.level_effects(sv_te, Xt_te, names)
    lev.to_csv(REPORTS_DIR / "shap_level_effects.csv", index=False)
    plots.shap_figures(sv_te, Xt_te, names, imp)

    print("[SCORE] Scoring all customers (BR-01, BR-02, BR-03) ...")
    p = model.predict_proba(full)
    scored = pd.DataFrame({
        "customerID": full["customerID"],
        "split": np.where(full["customerID"].isin(te["customerID"]), "holdout", "train"),
        "churn_probability": np.round(p, 4),
        "risk_tier": model.risk_tier(p),
    }).join(drivers)
    scored["expected_net_value"] = np.round(p * OFFER_SUCCESS_RATE * customer_value(full["MonthlyCharges"]) - OFFER_COST, 2)
    keep = ["gender", "SeniorCitizen", "tenure", "Contract", "InternetService", "PaymentMethod", "MonthlyCharges", "Churn"]
    scored = scored.join(full[keep].rename(columns={"Churn": "actual_churn"}))
    scored.sort_values("churn_probability", ascending=False).to_csv(REPORTS_DIR / "scored_customers.csv", index=False)
    assert scored[["driver_1", "driver_2", "driver_3"]].notna().all().all(), "every customer must have 3 drivers"
    tier_check = (scored[scored.split == "holdout"].groupby("risk_tier")
                  .agg(customers=("actual_churn", "size"), actual_churn_rate=("actual_churn", "mean"),
                       avg_predicted=("churn_probability", "mean")).reindex(["High", "Medium", "Low"]))
    tier_check.to_csv(REPORTS_DIR / "risk_tier_validation_holdout.csv")
    print(tier_check.round(3).to_string())

    print("[FAIRNESS] Subgroup check on hold-out (BR-06) ...")
    p_te = model.predict_proba(te)
    fdf, gaps = fairness.fairness_report(te, p_te, t["t_op"])
    fdf.to_csv(REPORTS_DIR / "fairness_by_group.csv", index=False)
    gaps.to_csv(REPORTS_DIR / "fairness_gaps.csv", index=False)
    print(fdf.round(3).to_string(index=False))

    print("[PLOTS] Saving figures ...")
    plots.eda_figures(full)
    oof = pd.read_csv(REPORTS_DIR / "train_oof_predictions.csv")
    sweep_te = pd.read_csv(REPORTS_DIR / "threshold_analysis_holdout.csv")
    plots.model_figures(te["Churn"], p_te, oof["y"], oof["p"], sweep_te, t)
    plots.threshold_tradeoff(pd.read_csv(REPORTS_DIR / "threshold_analysis_train_oof.csv"), t)
    plots.fairness_figure(fdf)

    print("[REPORTS] Writing SHAP report, cost-benefit analysis, leakage review and model card ...")
    reporting.write_all()
    print("\nDONE. Open the dashboard with:  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
