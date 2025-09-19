import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

st.set_page_config(page_title="Day 37 - ARIMA Forecasting", layout="wide")
st.title("📈 Day 37 — ARIMA: Classical Time-Series Forecasting")

st.markdown("""
**ARIMA (AutoRegressive Integrated Moving Average)** is a classical statistical model 
widely used for time-series forecasting.  

Upload your own CSV with **two columns**: `date`, `value` or use the sample dataset.
""")

# --- Sample dataset ---
@st.cache_data
def load_sample():
    dates = pd.date_range(start="2015-01-01", periods=100, freq="M")
    values = np.linspace(100, 200, 100) + np.random.normal(0, 5, 100)
    df = pd.DataFrame({"date": dates, "value": values})
    return df

# Upload or sample
col1, col2 = st.columns([3,1])
with col1:
    uploaded = st.file_uploader("Upload CSV (date, value)", type=["csv"])
with col2:
    use_sample = st.button("Load sample dataset")

df = None
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        if "date" not in df.columns or "value" not in df.columns:
            st.error("CSV must contain `date` and `value` columns.")
            st.stop()
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
elif use_sample:
    df = load_sample()

if df is None:
    st.info("Upload a CSV or click *Load sample dataset* to continue.")
    st.stop()

# Ensure datetime
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

st.subheader("📂 Dataset Preview")
st.dataframe(df.head())

# Train-test split
test_size = st.slider("Test set size (months)", 6, 24, 12)
train = df[:-test_size]
test = df[-test_size:]

# ARIMA parameters
st.sidebar.subheader("⚙️ ARIMA Parameters")
p = st.sidebar.number_input("AR order (p)", 0, 5, 1)
d = st.sidebar.number_input("Differencing (d)", 0, 2, 1)
q = st.sidebar.number_input("MA order (q)", 0, 5, 1)

if st.button("Train ARIMA Model"):
    try:
        with st.spinner("Training ARIMA model..."):
            model = ARIMA(train["value"], order=(p, d, q))
            model_fit = model.fit()

        # Forecast
        forecast = model_fit.forecast(steps=test_size)
        forecast_index = test["date"]

        # Plot forecast
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(train["date"], train["value"], label="Train")
        ax.plot(test["date"], test["value"], label="Test", color="orange")
        ax.plot(forecast_index, forecast, label="Forecast", color="green")
        ax.set_title("ARIMA Forecast")
        ax.legend()
        st.pyplot(fig)

        # Evaluation
        mape = mean_absolute_percentage_error(test["value"], forecast)
        rmse = np.sqrt(mean_squared_error(test["value"], forecast))
        st.subheader("📊 Evaluation Metrics")
        st.write(f"MAPE: {mape:.3f}")
        st.write(f"RMSE: {rmse:.3f}")

        # Download forecast
        forecast_df = pd.DataFrame({"date": forecast_index, "forecast": forecast})
        csv = forecast_df.to_csv(index=False)
        st.download_button("Download Forecast CSV", csv, "arima_forecast.csv", "text/csv")

    except Exception as e:
        st.error(f"Model training failed: {e}")
