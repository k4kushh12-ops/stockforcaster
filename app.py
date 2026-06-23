import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. PAGE CONFIGURATION & PREMIUM CSS
# ==========================================
st.set_page_config(page_title="Institutional Quant Pro", page_icon="📊", layout="wide")

st.markdown("""
    <style>
        .stApp { background: #f8fafc; font-family: 'Inter', sans-serif; }
        .metric-card { background: white; padding: 1.5rem; border-radius: 16px; border: 1px solid #e2e8f0; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
        .highlight { color: #3b82f6; font-weight: 800; }
        .summary-box { background: #eff6ff; padding: 20px; border-radius: 12px; border-left: 6px solid #3b82f6; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CORE ENGINE & INDICATORS
# ==========================================
def get_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# ==========================================
# 3. SIDEBAR
# ==========================================
with st.sidebar:
    ticker = st.text_input("Enter NSE Ticker:", value="RELIANCE").upper()
    if ticker and not ticker.endswith(".NS"): ticker += ".NS"
    st.markdown("---")
    mode = st.radio("Engine Configuration:", ["⚡ Locked ARIMA(4,1,1)", "🎛️ Manual Override"])

# ==========================================
# 4. EXECUTION
# ==========================================
if ticker:
    data = yf.Ticker(ticker).history(period="5y")['Close']
    data.index = data.index.tz_localize(None)
    series = data.resample('W-FRI').mean().dropna()

    # Model Params
    if mode == "⚡ Locked ARIMA(4,1,1)":
        p, d, q = 4, 1, 1
    else:
        p = st.slider("p", 0, 5, 1); d = st.slider("d", 0, 2, 1); q = st.slider("q", 0, 5, 1)

    model = ARIMA(series, order=(p, d, q), trend='t', enforce_stationarity=False).fit()
    forecast = model.get_forecast(steps=60)
    
    # UI Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price", f"₹{series.iloc[-1]:.2f}")
    col2.metric("2027 Projection", f"₹{forecast.predicted_mean.iloc[-1]:.2f}")
    rsi = get_rsi(series).iloc[-1]
    col3.metric("RSI (Momentum)", f"{rsi:.1f}", "Overbought > 70 | Oversold < 30")

    # Executive Summary
    st.markdown(f'<div class="summary-box">**Market Insight:** The engine is running with a high-order <b>ARIMA({p},{d},{q})</b> configuration. RSI is currently at <b>{rsi:.1f}</b>. Asset trend is currently {"Bullish" if rsi > 50 else "Bearish"}.</div>', unsafe_allow_html=True)

    # Tabs
    tab1, tab2 = st.tabs(["🔮 Predictive Model", "⚙️ Momentum & RSI"])

    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series.index, y=series, name="Historical", line=dict(color="#0f172a")))
        fig.add_trace(go.Scatter(x=forecast.predicted_mean.index, y=forecast.predicted_mean, name="Forecast", line=dict(color="#3b82f6", width=3, dash='dash')))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("SMA 10/40 Crossover", "Relative Strength Index (RSI)"))
        fig2.add_trace(go.Scatter(x=series.index, y=series.rolling(10).mean(), name="10W SMA", line=dict(color="#10b981")), row=1, col=1)
        fig2.add_trace(go.Scatter(x=series.index, y=series.rolling(40).mean(), name="40W SMA", line=dict(color="#ef4444")), row=1, col=1)
        fig2.add_trace(go.Scatter(x=series.index, y=get_rsi(series), name="RSI", line=dict(color="#f59e0b")), row=2, col=1)
        fig2.add_hline(y=70, line_dash="dash", row=2, col=1); fig2.add_hline(y=30, line_dash="dash", row=2, col=1)
        st.plotly_chart(fig2, use_container_width=True)
