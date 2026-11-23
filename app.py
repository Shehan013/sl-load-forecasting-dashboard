import streamlit as st
import pandas as pd

st.set_page_config(page_title="SL Load Forecasting Ensemble Dashboard", layout="wide")
st.title("Sri Lanka Daily Load Forecasting — Ensemble & Hybrid Dashboard")
st.caption("Chronological 80/20 split (1,571 train days, 393 test days) on daily-aggregated load, weather, calendar & economic features.")

df = pd.read_csv("ensemble_hybrid_results.csv")

tab1, tab2, tab3, tab4 = st.tabs([
    "Model Comparison",
    "Parameter Reasoning",
    "RF + ARIMA vs. RF + ANN",
    "Tabular vs. Image / Text"
])

with tab1:
    families = st.multiselect("Filter by Model Family", df["Family"].unique().tolist(), default=df["Family"].unique().tolist())
    filtered = df[df["Family"].isin(families)]
    c1, c2 = st.columns([1.2, 1.8])
    with c1:
        st.dataframe(filtered, use_container_width=True, hide_index=True)
    with c2:
        st.bar_chart(filtered.set_index("Model")[["Test_MAE", "Test_RMSE"]])

with tab2:
    st.markdown("- **Bagging & Random Forest (`n_estimators=200`, `max_depth=8`, `max_features=\'sqrt\'`):** Parallel bootstrap trees reduce variance; capping depth and sampling features prevents over-reliance on strong lag features.")
    st.markdown("- **Gradient Boosting / XGBoost / LightGBM / CatBoost (`learning_rate=0.05`, shallow depth/leaves):** Small step sizes with sequential residual trees allow smooth error reduction without overfitting.")
    st.markdown("- **Stacking (`RidgeCV` meta-learner + ordered 5-fold CV):** Blends diverse tree ensembles using L2 regularization so correlated learners don't overfit meta-weights.")

with tab3:
    st.markdown("- **Time Series (`RF + ARIMA` vs. `ARIMA`):** `RF + ARIMA` (`MAE 5.182`, `RMSE 7.508`) outperformed `ARIMA (1,0,1)` (`MAE 16.405`, `RMSE 20.478`) because Random Forest models residual non-linearities from lag, calendar, and weather features.")
    st.markdown("- **Tabular Regression (`RF + ANN` vs. `ANN`):** `ANN (MLP) Only` (`MAE 26.632`, `RMSE 33.667`) edged out `RF + ANN Hybrid` (`MAE 29.944`, `RMSE 38.143`), showing that adding tree predictions alongside correlated tabular features does not always help neural nets.")

with tab4:
    st.markdown("| Dimension | Tabular & Time-Series Data | Image & Text Data |\n| :--- | :--- | :--- |\n| **Structure** | Columns with distinct units/types (kW, °C, LKR, day-of-week) | 2D/3D spatial grids (pixels) or sequential token streams |\n| **Ensemble Strengths** | Trees handle mixed scales, monotonic transforms, and threshold rules natively with fast CPU training | Strong when paired with pre-extracted dense embeddings/features |\n| **Ensemble Limitations** | Struggles with smooth linear extrapolation far outside training bounds | Raw pixels/tokens lack spatial or attention inductive bias in standard trees; CNNs/Transformers fit better |")
