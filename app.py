import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime

# 1. Page Configuration
st.set_page_config(
    page_title="NSE ARIMA Forecaster",
    page_icon="📈",
    layout="wide"
)

# 2. Custom Premium Styling
st.markdown("""
    <style>
        .main { background-color: #f9fbfd; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .header-container {
            background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
            padding: 2.5rem; border-radius: 12px; color: white; margin-bottom: 2rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }
        .header-title { font-size: 2.4rem; font-weight: 700; margin: 0; }
        .header-subtitle { font-size: 1.1rem; opacity: 0.85; margin-top: 0.5rem; }
        .card {
            background: white; padding: 1rem; border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.02); border: 1px solid #eef2f6; margin-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# 3. Application Banner
st.markdown("""
    <div class="header-container">
        <div class="header-title">📈 Indian Stock Market ARIMA Forecasting Tool</div>
        <div class="header-subtitle">Fits an ARIMA model on 5 years of historical Yahoo Finance data to project trends through June 2027.</div>
    </div>
""", unsafe_allow_html=True)

# 4. Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    ticker_input = st.text_input("Enter NSE Ticker Symbol (e.g., RELIANCE, TCS, INFY):", value="RELIANCE")
    
    ticker = ticker_input.strip().upper()
    if not ticker.endswith(".NS"):
        ticker = f"{ticker}.NS"
        
    st.markdown("---")
    st.markdown("### ARIMA Parameters")
    p = st.slider("Autoregressive Order (p)", 0, 5, 2)
    d = st.slider("Differencing Degree (d)", 0, 2, 1)
    q = st.slider("Moving Average Order (q)", 0, 5, 2)

# 5. Data Loading Function
@st.cache_data(ttl=86400)
def fetch_historical_data(symbol):
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - pd.DateOffset(years=5)).strftime('%Y-%m-%d')
    df = yf.download(symbol, start=start_date, end=end_date)
    return df

# 6. Execution Block
if ticker:
    with st.spinner(f"Pulling 5-year historical records for {ticker}..."):
        raw_data = fetch_historical_data(ticker)
        
    if raw_data.empty:
        st.error(f"No data discovered for ticker '{ticker}'. Please ensure the ticker code is listed on Yahoo Finance.")
    else:
        if isinstance(raw_data.columns, pd.MultiIndex):
            raw_data.columns = raw_data.columns.get_level_values(0)
            
        historical_series = raw_data['Close'].resample('W').mean().dropna()
        
        last_date = historical_series.index[-1]
        target_date = datetime(2027, 6, 30)
        
        delta_weeks = int(np.ceil((target_date - last_date).days / 7))
        
        if delta_weeks <= 0:
            st.warning("The target date (June 2027) has already passed or matches the historical timeline.")
        else:
            try:
                model = ARIMA(historical_series, order=(p, d, q))
                fitted_model = model.fit()
                
                predictions = fitted_model.get_forecast(steps=delta_weeks)
                forecast_index = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=delta_weeks, freq='W')
                forecast_values = predictions.predicted_mean
                forecast_values.index = forecast_index
                
                m1, m2, m3 = st.columns(3)
                
                with m1:
                    st.metric("Latest Close Price", f"₹{historical_series.iloc[-1]:.2f}", f"As of {last_date.strftime('%Y-%m-%d')}")
                with m2:
                    st.metric("June 2027 Projected Price", f"₹{forecast_values.iloc[-1]:.2f}")
                with m3:
                    st.metric("AIC Score (Lower is better)", f"{fitted_model.aic:.2f}")
                
                st.markdown("---")
                
                chart_col, table_col = st.columns([2, 1])
                
                with chart_col:
                    st.markdown('<div class="card"><h4 style="margin:0;">📊 Forecast Visualization</h4></div>', unsafe_allow_html=True)
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=historical_series.index, y=historical_series.values,
                        name="Historical Close (5 Years)", line=dict(color="#203a43", width=2)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_values.index, y=forecast_values.values,
                        name="ARIMA Forecast (thru June 2027)", line=dict(color="#e65c00", width=2.5, dash='dash')
                    ))
                    
                    fig.update_layout(
                        template="plotly_white",
                        xaxis_title="Timeline",
                        yaxis_title="Price (INR)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=10, r=10, t=10, b=10),
                        height=450
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                with table_col:
                    st.markdown('<div class="card"><h4 style="margin:0;">🔢 Numerical Estimates</h4></div>', unsafe_allow_html=True)
                    
                    output_df = pd.DataFrame({
                        "Date Target": forecast_values.index.strftime('%Y-%m-%d'),
                        "Forecasted Value (₹)": np.round(forecast_values.values, 2)
                    }).reset_index(drop=True)
                    
                    st.dataframe(output_df, use_container_width=True, height=450)
                    
            except Exception as e:
                st.error(f"Mathematical convergence failed for order configuration ({p},{d},{q}).")
                st.info("💡 **Optimization Tip:** Try lowering the differencing degree (d) or reduction of moving average inputs (q) to match stationarity constraints.")
