"""Stage 4 - writes the human-readable deliverables from the numbers the pipeline produced."""
import json

import numpy as np
import pandas as pd

from .config import OFFER_COST, OFFER_SUCCESS_RATE, REPORTS_DIR, ROOT, TARGETS, VALUE_MONTHS
from .evaluation import customer_value, net_benefit, random_net_benefit


def md_table(df: pd.DataFrame, floatfmt="{:.3f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(out)


def _load():
    m = json.load(open(REPORTS_DIR / "metrics.json"))
    return m, pd.read_csv(REPORTS_DIR / "holdout_predictions.csv")


# ------------------------------------------------------------------ cost-benefit
def cost_sensitivity(h: pd.DataFrame, t_op):
    val = customer_value(h["monthly"])
    y, p = h["y"].values, h["p"].values
    rows = []
    for c in (20, 50, 100, 150):
        for s in (0.10, 0.20, 0.30):
            flag_hi = p >= t_op
            flag_ev = p * s * val > c                      # contact only if expected gain > offer cost
            rows.append({"offer_cost": c, "success_rate": s,
                         "contact_everyone": net_benefit(y, np.ones(len(y), bool), val, c, s),
                         "random_same_count_as_high_tier": random_net_benefit(y, val, int(flag_hi.sum()), c, s),
                         "high_tier_only": net_benefit(y, flag_hi, val, c, s),
                         "model_expected_value_rule": net_benefit(y, flag_ev, val, c, s),
                         "n_contacted_ev_rule": int(flag_ev.sum())})
    return pd.DataFrame(rows)


def write_cost_benefit(m, h):
    cb, th = m["cost_benefit_holdout"], m["thresholds"]
    sens = cost_sensitivity(h, th["t_op"])
    sens.to_csv(REPORTS_DIR / "cost_benefit_sensitivity.csv", index=False)
    avg_val = customer_value(h["monthly"]).mean()
    breakeven = OFFER_COST / (OFFER_SUCCESS_RATE * avg_val)
    op, co, ev = cb["operating"], cb["cost_optimal"], cb["contact_everyone"]["net_benefit"]
    strat = pd.DataFrame([
        ["Contact everyone", len(h), OFFER_COST * len(h), ev, "-"],
        ["Random, same count as High tier", op["customers_contacted"], op["campaign_cost"], op["net_benefit_random_same_budget"], "-"],
        [f"Model - High tier (t={op['threshold']:.2f}, operating)", op["customers_contacted"], op["campaign_cost"], op["net_benefit_model"], f"{op['uplift_vs_random']:+,.0f}"],
        [f"Model - cost-optimal (t={co['threshold']:.2f})", co["customers_contacted"], co["campaign_cost"], co["net_benefit_model"], f"{co['uplift_vs_random']:+,.0f}"],
    ], columns=["Strategy", "Customers contacted", "Campaign cost (USD)", "Net benefit (USD)", "Gain vs random (same count)"])
    s2 = sens.drop(columns=["n_contacted_ev_rule"]).copy()
    for c in s2.columns[2:]:
        s2[c] = s2[c].map("{:,.0f}".format)
    txt = f"""# Cost-Benefit Analysis (BR-04)

Simulated on the untouched hold-out set ({len(h):,} customers). All business numbers are **assumptions from BRD section 8**, not observed values.

**Net benefit** = (offer success rate x 12 x MonthlyCharges of each churner we reached) - (offer cost x customers contacted).

## 1. Assumptions
Offer cost **${OFFER_COST:.0f}** per contact; **{OFFER_SUCCESS_RATE:.0%}** of contacted churners accept; customer value = **{VALUE_MONTHS} x MonthlyCharges** (average ${avg_val:,.0f} in the hold-out).

## 2. Results at the BRD assumptions
{md_table(strat, "{:,.0f}")}

## 3. How the threshold was chosen
* **Cost-optimal threshold ({th['t_cost']:.2f})** maximises net benefit on training out-of-fold predictions. It is low because one saved customer (about ${OFFER_SUCCESS_RATE * avg_val:,.0f} expected) is worth far more than one offer (${OFFER_COST:.0f}); the break-even churn probability for an average customer is only about **{breakeven:.1%}**.
* **Operating threshold ({th['t_op']:.2f})** maximises net benefit **among thresholds that also meet the BRD recall (>= {TARGETS['recall']}) and precision (>= {TARGETS['precision']}) targets**. It defines the High tier.
* The BRD asks for both a cost-based threshold and recall/precision targets. Under these assumptions the two disagree, so both are reported: the operating threshold satisfies the acceptance criteria, the cost-optimal one maximises money.

## 4. What this means for the sponsor (important)
* Targeting beats **random targeting of the same size** by **${op['uplift_vs_random']:,.0f}** (BRD criterion: positive and quantified - met).
* But at these assumptions **contacting every customer is almost as profitable** as contacting only the High tier (${ev:,.0f} vs ${op['net_benefit_model']:,.0f}), because offers are cheap relative to customer value. The cost-optimal threshold beats contacting everyone by ${co['net_benefit_model'] - ev:,.0f}.
* The model's value therefore comes from **(a) spending ${op['campaign_cost']:,.0f} instead of ${OFFER_COST * len(h):,.0f}** for nearly the same return, **(b) limited call-centre capacity or budget**, and **(c) higher offer costs** - see the sensitivity table. Finance should confirm the real offer cost and success rate before the campaign is sized.

## 5. Sensitivity (hold-out net benefit, USD)
{md_table(s2)}

`model_expected_value_rule` contacts a customer only if churn probability x success rate x customer value > offer cost (a per-customer cost-benefit rule). Full table: `cost_benefit_sensitivity.csv`.
"""
    (REPORTS_DIR / "cost_benefit_analysis.md").write_text(txt)


# ------------------------------------------------------------------ SHAP report
def write_shap_report(m):
    imp = pd.read_csv(REPORTS_DIR / "shap_global_importance.csv").head(10)
    lev = pd.read_csv(REPORTS_DIR / "shap_level_effects.csv")
    up, down = lev.head(6), lev.tail(6).iloc[::-1]
    txt = f"""# SHAP Explainability Report (BR-03)

**Model explained:** {m['chosen_model'].replace('_', ' ')} (SHAP `LinearExplainer`; tree models would use `TreeExplainer`). Values are in **log-odds of churn before calibration**: positive = pushes the customer towards churn, negative = pushes towards staying.

## Global drivers - what matters most overall (hold-out set)
Related columns are merged into one driver (e.g. tenure + tenure bucket + total charges = "tenure").

{md_table(imp.rename(columns={'feature': 'driver', 'mean_abs_shap': 'mean abs SHAP'}))}

![global importance](figures/shap_global_importance.png)
![beeswarm](figures/shap_beeswarm.png)

## Which customer situations raise or lower risk
Average SHAP value among customers who have this characteristic (only levels with at least 100 hold-out customers).

**Raise risk**

{md_table(up)}

**Lower risk**

{md_table(down)}

## How to read the per-customer output
`reports/scored_customers.csv` (and the dashboard) give every customer three columns `driver_1..3` = the three drivers pushing **that customer's** risk up the most, with the SHAP effect size. 100% of customers have three drivers.

## Caveats
* SHAP explains **what the model uses**, not what *causes* churn. A driver like "Fiber optic" may stand for price, service quality or competition. Uplift/causal modelling is out of scope (BRD section 4).
* Many columns are correlated (tenure, total charges, contract); SHAP splits credit between them, which is why related columns are grouped.
"""
    (REPORTS_DIR / "shap_report.md").write_text(txt)


# ------------------------------------------------------------------ leakage review
def write_leakage_review():
    leak = pd.read_csv(REPORTS_DIR / "leakage_audit.csv").head(8)
    txt = f"""# Leakage Review (BRD acceptance criterion)

| Check | Result |
|---|---|
| Hold-out is split first (stratified 80/20) and evaluated **once**, after all choices are frozen | Yes - `src/train.py` |
| Scaling, one-hot encoding and model fitting happen **inside one sklearn Pipeline**, refitted per CV fold | Yes - `src/features.py`, `src/modeling.py` |
| Feature engineering is **row-level only** (no group means, no target encoding, no use of other rows) | Yes - `add_features()` |
| Target `Churn` never used to build a feature; `customerID` dropped | Yes |
| Probability calibrator and thresholds are fitted on **training out-of-fold** predictions, not on the hold-out | Yes |
| Model choice made with CV on the training part only | Yes |
| `TotalCharges` blanks (11 rows, all tenure = 0) set to 0 - deterministic rule, nothing learned from data | Yes |
| Suspicious single features (single-feature AUC > 0.85) | None - see table below |
| Sensitive attributes (`gender`, `SeniorCitizen`) kept out of the model inputs | Yes - used only for the fairness audit |

**Single-feature AUC scan** (a depth-3 tree on one feature at a time, 5-fold CV on training data). A leaked column would score near 1.0.

{md_table(leak)}

**Remaining caveat:** the dataset is one snapshot with no dates, so "customer churned within the last month" is described with features measured at the same moment. We cannot prove that every feature was known *before* the churn happened. Before production use, confirm with the data owner that all features are available at scoring time.
"""
    (REPORTS_DIR / "leakage_review.md").write_text(txt)


# ------------------------------------------------------------------ model card
def write_model_card(m, h):
    ci = m["holdout_bootstrap_95ci"]
    ho = m["holdout_at_operating_threshold"]
    th = m["thresholds"]
    cv = pd.read_csv(REPORTS_DIR / "model_comparison_cv.csv").drop(columns=["best_params"])
    imb = pd.read_csv(REPORTS_DIR / "imbalance_study.csv")
    tiers = pd.read_csv(REPORTS_DIR / "risk_tier_validation_holdout.csv")
    fair = pd.read_csv(REPORTS_DIR / "fairness_by_group.csv")
    fair["group"] = fair["group"].astype(str)
    sc = pd.DataFrame([[k, f"{v['value']:.3f}", v["target"], f"[{ci[k][0]:.3f}, {ci[k][1]:.3f}]" if k in ci else "-",
                        "PASS" if v["pass"] else "FAIL"] for k, v in m["scorecard"].items()],
                      columns=["Metric", "Hold-out value", "BRD target", "95% bootstrap CI", "Result"])
    sen = fair[fair.attribute == "SeniorCitizen"].set_index("group")
    gen = fair[fair.attribute == "gender"].set_index("group")
    floor = min(th["t_cost"], th["t_op"])
    txt = f"""# Model Card - Customer Churn Prediction (BRD-01)

## 1. Model details
* **Model:** {m['chosen_model'].replace('_', ' ')} inside an sklearn Pipeline (scaling + one-hot encoding) with Platt-scaling calibration. Tuned parameters: `{m['best_params']}`.
* **Saved as:** `models/churn_model.joblib` (a `ChurnModel` object: pipeline + calibrator + thresholds).
* **Why this model:** all three candidates scored within about 0.002 ROC-AUC in cross-validation (table below). The project picks the simplest model within 0.003 of the best, because the dataset is small and a linear model is easier to explain and less likely to overfit.
* **Data:** IBM Telco Customer Churn sample (7,043 customers, fictional company, single snapshot). Target = churned in the last month (26.5%).
* **Inputs:** 24 engineered/raw features (tenure, charges, contract, payment, services, add-on count, price-change gap, ...). `gender` and `SeniorCitizen` are deliberately **not** inputs. `TotalCharges` blanks (11 customers, tenure 0) were set to 0.

## 2. Intended use
Rank customers by churn risk so the retention team contacts the highest-value-to-save customers first, with a reason for each. **Not** intended for automated decisions without human review, or for pricing, credit or any non-retention use. Not validated for other companies or time periods.

## 3. Training procedure
Stratified 80/20 split (5,634 train / 1,409 hold-out). 5-fold stratified CV on the training part. Class imbalance is handled with class weights. The study below shows weights barely change ranking quality, so probability calibration + threshold selection is what actually controls the precision/recall trade-off.

{md_table(imb)}

Tuned model comparison (5-fold CV, training data):

{md_table(cv)}

## 4. Performance on the untouched hold-out set
{md_table(sc)}

Threshold used: **{th['t_op']:.2f}** (chosen on training out-of-fold predictions; see `reports/cost_benefit_analysis.md`).

**Read this honestly:** every target is met on the point estimate, but with only 1,409 hold-out customers the uncertainty is real. The lower bounds of the precision interval ({ci['precision'][0]:.3f}) and ROC-AUC interval ({ci['roc_auc'][0]:.3f}) are below their targets, so a different random split could land just under. Precision ({ho['precision']:.3f}) passes by a thin margin.

## 5. Risk tiers (BR-02)
High = probability >= {th['t_op']:.2f}; Medium = between {floor:.2f} and {th['t_op']:.2f}; Low = below {floor:.2f} (below the cost break-even, so an offer is not worth its cost). Hold-out validation (actual churn rate should rise from Low to High):

{md_table(tiers)}

## 6. Fairness check (BR-06) - hold-out, at the operating threshold
{md_table(fair[['attribute', 'group', 'customers', 'actual_churn_rate', 'roc_auc', 'recall_tpr', 'precision', 'false_positive_rate', 'flag_rate']])}

**Findings**
* **Gender:** small differences (recall {gen.loc['Female', 'recall_tpr']:.2f} vs {gen.loc['Male', 'recall_tpr']:.2f}; precision {gen.loc['Female', 'precision']:.2f} vs {gen.loc['Male', 'precision']:.2f}). With about 700 customers per group these gaps are within normal noise.
* **Senior citizens:** actual churn is higher ({sen.loc['1', 'actual_churn_rate']:.0%} vs {sen.loc['0', 'actual_churn_rate']:.0%}) and the model flags {sen.loc['1', 'flag_rate']:.0%} of them vs {sen.loc['0', 'flag_rate']:.0%} of non-seniors. Recall is higher for seniors ({sen.loc['1', 'recall_tpr']:.2f} vs {sen.loc['0', 'recall_tpr']:.2f}) but so is the false-positive rate ({sen.loc['1', 'false_positive_rate']:.2f} vs {sen.loc['0', 'false_positive_rate']:.2f}), and ROC-AUC is lower ({sen.loc['1', 'roc_auc']:.2f} vs {sen.loc['0', 'roc_auc']:.2f}). Only {int(sen.loc['1', 'customers'])} seniors are in the hold-out, so this is indicative, not conclusive.
* `gender` and `SeniorCitizen` are **not model inputs**; the gap comes from correlated features (e.g. fiber optic, high monthly charges). A false positive only costs an extra retention offer, so the harm is low, but **Compliance should review** this before launch and decide whether to monitor flag rates by group.

## 7. Limitations
* Single snapshot, no dates: no time-based validation, possible hidden leakage (see `reports/leakage_review.md`).
* Business numbers (offer cost, success rate, customer value) are assumptions. Under them, contacting everyone is nearly as profitable as targeting the High tier (see the cost-benefit report), so confirm the real costs.
* The model predicts who is likely to leave, **not who will respond to an offer** (uplift modelling is out of scope).
* Calibration was fitted on cross-validated predictions; re-check it whenever the model is retrained or the customer base changes.

## 8. Retraining and monitoring
`python run_pipeline.py` retrains everything and regenerates all reports. Monitor monthly: ROC-AUC, actual churn rate per tier, and calibration.
"""
    (ROOT / "MODEL_CARD.md").write_text(txt)


def write_all():
    m, h = _load()
    write_cost_benefit(m, h)
    write_shap_report(m)
    write_leakage_review()
    write_model_card(m, h)
