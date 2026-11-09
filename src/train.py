"""Stage 1 - train, tune, calibrate, pick thresholds, evaluate ONCE on the hold-out set, save model."""
import json

import joblib
import numpy as np
import pandas as pd

from .config import MODELS_DIR, OFFER_COST, OFFER_SUCCESS_RATE, REPORTS_DIR, TARGETS
from .data import load_clean, split
from .evaluation import (bootstrap_ci, customer_value, lift_table, net_benefit, random_net_benefit, select_thresholds,
                         summary_metrics, threshold_sweep)
from .features import make_X
from .modeling import (ChurnModel, PlattCalibrator, imbalance_study, leakage_audit, oof_scores, pick_model,
                       tune_all)


def run():
    print("[1/7] Loading and splitting data ...")
    df = load_clean()
    train, test = split(df)
    Xtr, ytr = make_X(train), train["Churn"].values
    yte = test["Churn"].values
    pos_weight = float((ytr == 0).sum() / (ytr == 1).sum())
    print(f"      train={len(train)}  hold-out={len(test)}  churn rate train={ytr.mean():.3f} hold-out={yte.mean():.3f}")

    print("[2/7] Leakage audit (single-feature AUC scan) ...")
    leak = leakage_audit(Xtr, ytr)
    leak.to_csv(REPORTS_DIR / "leakage_audit.csv", index=False)
    print(f"      max single-feature AUC = {leak.single_feature_auc.max():.3f}  -> {(leak.flag == 'CHECK').sum()} features flagged")

    print("[3/7] Class-imbalance study (with vs without class weights, 5-fold CV) ...")
    imb = imbalance_study(Xtr, ytr, pos_weight)
    imb.to_csv(REPORTS_DIR / "imbalance_study.csv", index=False)
    print(imb.round(4).to_string(index=False))

    print("[4/7] Tuning Logistic Regression, Random Forest, XGBoost (random search, 5-fold CV) ...")
    cv_table, best = tune_all(Xtr, ytr, pos_weight)
    cv_table.to_csv(REPORTS_DIR / "model_comparison_cv.csv", index=False)
    print(cv_table.drop(columns="best_params").round(4).to_string(index=False))
    chosen = pick_model(cv_table)
    pipe = best[chosen]
    print(f"      -> chosen model: {chosen}")

    print("[5/7] Calibrating probabilities and selecting thresholds on out-of-fold training predictions ...")
    oof_raw = oof_scores(pipe, Xtr, ytr)
    cal = PlattCalibrator().fit(oof_raw, ytr)
    oof_p = cal.transform(oof_raw)
    sweep = threshold_sweep(ytr, oof_p, customer_value(train["MonthlyCharges"]))
    sweep.to_csv(REPORTS_DIR / "threshold_analysis_train_oof.csv", index=False)
    th = select_thresholds(sweep)
    print(f"      cost-optimal threshold={th['t_cost']:.2f} | operating threshold={th['t_op']:.2f} ({th['note']})")

    model = ChurnModel(chosen, pipe, cal, th, params=cv_table.set_index("model").loc[chosen, "best_params"])

    print("[6/7] Evaluating ONCE on the untouched hold-out set ...")
    p_te = model.predict_proba(test)
    val_te = customer_value(test["MonthlyCharges"])
    res = {"chosen_model": chosen, "best_params": model.params, "thresholds": th,
           "holdout_at_operating_threshold": summary_metrics(yte, p_te, th["t_op"]),
           "holdout_at_cost_optimal_threshold": summary_metrics(yte, p_te, th["t_cost"]),
           "holdout_bootstrap_95ci": bootstrap_ci(yte, p_te, th["t_op"])}

    # --- business value simulation on hold-out ---
    cb = {}
    for label, t in (("operating", th["t_op"]), ("cost_optimal", th["t_cost"])):
        flag = p_te >= t
        n = int(flag.sum())
        model_nb = net_benefit(yte, flag, val_te)
        rand_nb = random_net_benefit(yte, val_te, n)
        cb[label] = {"threshold": t, "customers_contacted": n, "campaign_cost": OFFER_COST * n,
                     "net_benefit_model": model_nb, "net_benefit_random_same_budget": rand_nb,
                     "uplift_vs_random": model_nb - rand_nb}
    everyone = np.ones(len(yte), bool)
    cb["contact_everyone"] = {"net_benefit": net_benefit(yte, everyone, val_te)}
    cb["assumptions"] = {"offer_cost": OFFER_COST, "success_rate": OFFER_SUCCESS_RATE, "value": "12 x MonthlyCharges"}
    res["cost_benefit_holdout"] = cb

    # --- target scorecard ---
    m = res["holdout_at_operating_threshold"]
    res["scorecard"] = {
        "roc_auc": {"value": m["roc_auc"], "target": f">= {TARGETS['roc_auc']}", "pass": m["roc_auc"] >= TARGETS["roc_auc"]},
        "top_decile_lift": {"value": m["top_decile_lift"], "target": f">= {TARGETS['top_decile_lift']}", "pass": m["top_decile_lift"] >= TARGETS["top_decile_lift"]},
        "recall": {"value": m["recall"], "target": f">= {TARGETS['recall']}", "pass": m["recall"] >= TARGETS["recall"]},
        "precision": {"value": m["precision"], "target": f">= {TARGETS['precision']}", "pass": m["precision"] >= TARGETS["precision"]},
        "net_benefit_vs_random": {"value": cb["operating"]["uplift_vs_random"], "target": "> 0", "pass": cb["operating"]["uplift_vs_random"] > 0},
    }
    pd.DataFrame(lift_table(yte, p_te)).to_csv(REPORTS_DIR / "lift_table_holdout.csv", index=False)
    threshold_sweep(yte, p_te, val_te).to_csv(REPORTS_DIR / "threshold_analysis_holdout.csv", index=False)
    # arrays used later by plots
    pd.DataFrame({"customerID": test["customerID"], "y": yte, "p": p_te, "monthly": test["MonthlyCharges"]}).to_csv(
        REPORTS_DIR / "holdout_predictions.csv", index=False)
    pd.DataFrame({"y": ytr, "p": oof_p}).to_csv(REPORTS_DIR / "train_oof_predictions.csv", index=False)

    print("[7/7] Saving model ...")
    joblib.dump(model, MODELS_DIR / "churn_model.joblib")
    with open(REPORTS_DIR / "metrics.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    print("\n=== TARGET SCORECARD (hold-out) ===")
    for k, v in res["scorecard"].items():
        print(f"  {'PASS' if v['pass'] else 'FAIL'}  {k:<24} value={v['value']:.3f}  target {v['target']}")
    return model, train, test, res


if __name__ == "__main__":
    run()
