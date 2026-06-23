import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime

# Reset to standard configuration to prevent CSS clashes
st.set_page_config(page_title="Forecasting Dashboard", layout="wide")

st.title("📈 Market Forecasting Dashboard")
st.subheader("High-Readability Configuration (ARIMA 4,1,1)")

# 1. Sidebar Ticker Entry
ticker = st.sidebar.text_input("Enter NSE Ticker Symbol:", value="RELIANCE.NS")

if ticker:
    try:
        # Load Data
        df = yf.Ticker(ticker).history(period="5y")['Close']
        series = df.resample('W-FRI').mean().dropna()
        
        # ARIMA(4,1,1) Model
        model = ARIMA(series, order=(4, 1, 1), trend='t', enforce_stationarity=False).fit()
        forecast = model.get_forecast(steps=52)
        
        # 2. High Contrast Chart
        st.write("### Primary Predictive Model")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series.index, y=series, name="Historical Data"))
        fig.add_trace(go.Scatter(x=forecast.predicted_mean.index, y=forecast.predicted_mean, name="Forecast", line=dict(width=3)))
        
        # Use a template that is always readable
        fig.update_layout(template="plotly_white", xaxis_title="Date", yaxis_title="Price (₹)")
        st.plotly_chart(fig, use_container_width=True)
        
        # 3. Readable Data Ledger
        st.write("### Forecast Data Ledger")
        results = pd.DataFrame({
            "Date": forecast.predicted_mean.index.strftime('%Y-%m-%d'),
            "Predicted Price (₹)": forecast.predicted_mean.values
        })
        # Standard table (no custom CSS)
        st.table(results.head(10))
        
    except Exception as e:
        st.error(f"Could not load data: {e}")
