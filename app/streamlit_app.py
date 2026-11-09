"""Retention dashboard (BR-05).   Run:  streamlit run app/streamlit_app.py
It reads the files written by `python run_pipeline.py` (reports/ folder), so run the pipeline first."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
st.set_page_config(page_title="Churn Retention Dashboard", page_icon="📞", layout="wide")


@st.cache_data
def load():
    return (pd.read_csv(REP / "scored_customers.csv"), json.load(open(REP / "metrics.json")),
            pd.read_csv(REP / "risk_tier_validation_holdout.csv"), pd.read_csv(REP / "fairness_by_group.csv"))


if not (REP / "scored_customers.csv").exists():
    st.error("No results found. Run `python run_pipeline.py` first, then refresh this page.")
    st.stop()

df, metrics, tiers_val, fair = load()
th = metrics["thresholds"]

st.title("Customer Churn - Retention Targeting")
st.caption(f"Model: {metrics['chosen_model'].replace('_', ' ')} | High tier = churn probability >= {th['t_op']:.2f}, "
           f"Medium >= {min(th['t_cost'], th['t_op']):.2f}, Low below that. Data: IBM Telco sample (fictional company).")

# ---------------- sidebar: assumptions + filters ----------------
with st.sidebar:
    st.header("Cost assumptions")
    cost = st.number_input("Offer cost per contact (USD)", 0.0, 500.0, 20.0, 5.0)
    succ = st.slider("Share of contacted churners who accept", 0.05, 0.90, 0.30, 0.05)
    months = st.slider("Customer value = N months of MonthlyCharges", 1, 36, 12)
    st.header("Filters")
    tiers = st.multiselect("Risk tier", ["High", "Medium", "Low"], default=["High", "Medium"])
    contracts = st.multiselect("Contract", sorted(df["Contract"].unique()), default=sorted(df["Contract"].unique()))
    internet = st.multiselect("Internet service", sorted(df["InternetService"].unique()), default=sorted(df["InternetService"].unique()))
    t_lo, t_hi = st.slider("Tenure (months)", 0, int(df["tenure"].max()), (0, int(df["tenure"].max())))
    only_holdout = st.checkbox("Only customers the model never saw (hold-out)", value=False)

df = df.copy()
df["expected_net_value"] = (df["churn_probability"] * succ * months * df["MonthlyCharges"] - cost).round(2)
mask = (df["risk_tier"].isin(tiers) & df["Contract"].isin(contracts) & df["InternetService"].isin(internet)
        & df["tenure"].between(t_lo, t_hi))
if only_holdout:
    mask &= df["split"] == "holdout"
view = df[mask].sort_values("expected_net_value", ascending=False)

tab1, tab2, tab3, tab4 = st.tabs(["Contact list", "Why is this customer at risk?", "Model performance", "Fairness"])

# ---------------- tab 1: contact list ----------------
with tab1:
    worth = view[view["expected_net_value"] > 0]
    capacity = st.number_input("Call-centre capacity (max customers to contact)", 0, max(len(worth), 1), min(500, len(worth)), 50)
    contact = worth.head(int(capacity))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers matching filters", f"{len(view):,}")
    c2.metric("Worth contacting (expected gain > cost)", f"{len(worth):,}")
    c3.metric("Contact list size", f"{len(contact):,}", help="Highest expected net value first, limited by capacity")
    c4.metric("Expected net benefit of list", f"${contact['expected_net_value'].sum():,.0f}")
    cols = ["customerID", "risk_tier", "churn_probability", "expected_net_value", "driver_1", "driver_2", "driver_3",
            "Contract", "tenure", "MonthlyCharges", "InternetService"]
    st.dataframe(contact[cols], hide_index=True)
    st.download_button("Download contact list (CSV)", contact[cols].to_csv(index=False).encode(),
                       "retention_contact_list.csv", "text/csv")
    st.caption("Expected net value = probability x success rate x customer value - offer cost. "
               "Change the assumptions in the sidebar to see how the list changes.")
    mix = view["risk_tier"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0)
    st.bar_chart(mix, y_label="customers", x_label="risk tier")

# ---------------- tab 2: single customer ----------------
with tab2:
    ids = view["customerID"].tolist() or df["customerID"].tolist()
    cid = st.selectbox("Customer", ids)
    r = df[df["customerID"] == cid].iloc[0]
    a, b, c = st.columns(3)
    a.metric("Churn probability", f"{r['churn_probability']:.1%}")
    b.metric("Risk tier", r["risk_tier"])
    c.metric("Expected net value of an offer", f"${r['expected_net_value']:,.0f}")
    st.write(f"**Profile:** {r['Contract']} contract, {int(r['tenure'])} months with us, ${r['MonthlyCharges']:.2f}/month, "
             f"{r['InternetService']} internet.")
    drv = pd.DataFrame({"driver": [r["driver_1"], r["driver_2"], r["driver_3"]],
                        "impact on churn risk (SHAP)": [r["driver_1_effect"], r["driver_2_effect"], r["driver_3_effect"]]})
    st.subheader("Top 3 reasons this customer may leave")
    st.bar_chart(drv.set_index("driver"), horizontal=True)
    if r["split"] == "holdout":
        st.caption(f"Hold-out customer (never used in training). Actual outcome: {'churned' if r['actual_churn'] == 1 else 'stayed'}.")

# ---------------- tab 3: performance ----------------
with tab3:
    sc = pd.DataFrame([{"Metric": k, "Hold-out value": round(v["value"], 3), "BRD target": v["target"],
                        "Result": "PASS" if v["pass"] else "FAIL"} for k, v in metrics["scorecard"].items()])
    st.dataframe(sc, hide_index=True)
    ci = metrics["holdout_bootstrap_95ci"]
    st.caption("95% bootstrap intervals: " + "; ".join(f"{k} [{v[0]:.3f}, {v[1]:.3f}]" for k, v in ci.items()))
    st.image(str(REP / "figures" / "model_performance.png"))
    st.subheader("Do the risk tiers behave? (hold-out)")
    st.dataframe(tiers_val.round(3), hide_index=True)

# ---------------- tab 4: fairness ----------------
with tab4:
    st.write("Model quality by group on the hold-out set, at the operating threshold. `gender` and `SeniorCitizen` "
             "are **not** model inputs; they are used here only for auditing.")
    st.dataframe(fair.round(3), hide_index=True)
    st.image(str(REP / "figures" / "fairness_by_group.png"))
    st.info("Senior citizens are flagged more often and have a higher false-positive rate. See MODEL_CARD.md section 6 - "
            "Compliance should review before launch.")
