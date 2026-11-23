import streamlit as st
import pandas as pd
import joblib, json
from xgboost import XGBRegressor

st.set_page_config(page_title="Sri Lankan Car Price - Ensemble Dashboard", layout="wide")

model = XGBRegressor()
model.load_model("model.json")   # version-safe XGBoost format
cols = joblib.load("columns.pkl")
opt = json.load(open("options.json"))

st.title("🚗 Sri Lankan Used Car Price: Ensemble Learning Dashboard")
st.caption("Data: ikman.lk and Riyasewana.com listings (Dec 2024 - Feb 2025). "
           "Price is the asking price in lakhs (1 lakh = Rs. 100,000). "
           "Built with the help of AI (Claude).")

tab1, tab2, tab3, tab4 = st.tabs(
    ["💰 Predict Price", "📊 Model Comparison", "⭐ Feature Importance", "🔀 Hybrid Models"])

with tab1:
    st.subheader("Enter car details")
    c1, c2, c3 = st.columns(3)
    brand = c1.selectbox("Brand", opt["Brand"])
    model_name = c1.selectbox("Model", opt["Model"])
    town = c1.selectbox("Town", opt["Town"])
    age = c2.slider("Age (years)", opt["Age"][0], opt["Age"][1], min(8, opt["Age"][1]))
    engine = c2.number_input("Engine (cc)", opt["Engine"][0], opt["Engine"][1], 1000)
    gear = c2.selectbox("Gear", opt["Gear"])
    fuel = c3.selectbox("Fuel Type", opt["Fuel Type"])
    cond = c3.selectbox("Condition", opt["Condition"])
    leasing = c3.selectbox("Leasing", opt["Leasing"])
    e1, e2, e3, e4 = st.columns(4)
    ac = e1.selectbox("Air Condition", opt["AIR CONDITION"])
    ps = e2.selectbox("Power Steering", opt["POWER STEERING"])
    pm = e3.selectbox("Power Mirror", opt["POWER MIRROR"])
    pw = e4.selectbox("Power Window", opt["POWER WINDOW"])

    if st.button("Predict price", type="primary"):
        row = {"Brand": brand, "Model": model_name, "Engine (cc)": float(engine),
               "Gear": gear, "Fuel Type": fuel, "Town": town, "Leasing": leasing,
               "Condition": cond, "AIR CONDITION": ac, "POWER STEERING": ps,
               "POWER MIRROR": pm, "POWER WINDOW": pw, "Age": age}
        x = pd.get_dummies(pd.DataFrame([row]))
        x.columns = x.columns.str.replace(r"[^A-Za-z0-9_]", "_", regex=True)
        x = x.reindex(columns=cols, fill_value=0).astype(float)
        pred = max(float(model.predict(x)[0]), 0)
        m1, m2 = st.columns(2)
        m1.metric("Predicted price (lakhs)", f"{pred:,.1f}")
        m2.metric("Approx. price (Rs.)", f"Rs. {pred*100000:,.0f}")
        st.info("Model: XGBoost. This is an estimate of the asking price, not a final sale price. "
                "Very expensive cars tend to be under-predicted.")

with tab2:
    res = pd.read_csv("results.csv")
    st.subheader("Bagging vs Boosting vs Stacking (test set)")
    st.dataframe(res, use_container_width=True)
    st.bar_chart(res.set_index("Model")["R2"])
    st.write("Stacking had the best R2 (0.818), but XGBoost, CatBoost and LightGBM were almost "
             "as accurate and far faster to train.")

with tab3:
    imp = pd.read_csv("importance.csv", index_col=0)
    st.subheader("Top 10 features (Random Forest)")
    st.bar_chart(imp.iloc[:, 0])
    st.write("Age, manual gearbox, engine size and luxury brands drive prices the most.")

with tab4:
    hyb = pd.read_csv("hybrid_results.csv")
    st.subheader("RF + ANN hybrid (tabular regression)")
    st.dataframe(hyb, use_container_width=True)
    st.write("The hybrid did not beat Random Forest alone because the ANN was weaker. "
             "RF + ARIMA (on synthetic time-series data) reduced RMSE from 56.57 to 42.04.")
