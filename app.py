import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
import warnings

# Suppress statistical warnings to keep the backend logs clean
warnings.filterwarnings("ignore")

# 1. Page Configuration
st.set_page_config(
    page_title="NSE Automated ARIMA Forecaster",
    page_icon="📈",
    layout="wide"
)

# 2. Custom Premium Styling
st.markdown("""
    <style>
        .main { background-color: #f9fbfd; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .header-container {
            background: linear-gradient(135deg, #1f4037 0%, #2a5298 100%);
            padding: 2.5rem; border-radius: 12px; color: white; margin-bottom: 2rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }
        .header-title { font-size: 2.4rem; font-weight: 700; margin: 0; color: #ffffff; }
        .header-subtitle { font-size: 1.1rem; opacity: 0.9; margin-top: 0.5rem; color: #ffffff; }
        .card {
            background: white; padding: 1.2rem; border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.02); border: 1px solid #eef2f6; margin-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# 3. Application Banner
st.markdown("""
    <div class="header-container">
        <div class="header-title">🤖 Automated Indian Stock Market Forecaster</div>
        <div class="header-subtitle">Optimized algorithmic tuning utilizing 5 years of historical data to project NSE stock trends through June 2027.</div>
    </div>
""", unsafe_allow_html=True)

# 4. Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    ticker_input = st.text_input("Enter NSE Ticker Symbol (e.g., RELIANCE, TCS, SBIN):", value="RELIANCE")
    
    ticker = ticker_input.strip().upper()
    if not ticker.endswith(".NS"):
        ticker = f"{ticker}.NS"
        
    st.markdown("---")
    st.markdown("### Model Control Mode")
    mode = st.radio("Choose Optimization Path:", ["⚡ Fast Auto-Tuning (Recommended)", "🎛️ Manual Specification"])
    
    if mode == "🎛️ Manual Specification":
        p = st.slider("Autoregressive Order (p)", 0, 5, 1)
        d = st.slider("Differencing Degree (d)", 0, 2, 1)
        q = st.slider("Moving Average Order (q)", 0, 5, 1)

# 5. Data Fetching Strategy
@st.cache_data(ttl=86400)
def fetch_historical_data(symbol):
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - pd.DateOffset(years=5)).strftime('%Y-%m-%d')
    df = yf.download(symbol, start=start_date, end=end_date)
    return df

# Fast, bounded grid-search that includes trend drift to prevent flatlining
def optimize_arima(series):
    best_aic = float("inf")
    best_order = (1, 1, 1) 
    
    p_values = [0, 1, 2]
    d_values = [1] 
    q_values = [0, 1, 2]
    
    total_iterations = len(p_values) * len(d_values) * len(q_values)
    
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    current_step = 0
    for p_val in p_values:
        for d_val in d_values:
            for q_val in q_values:
                current_step += 1
                progress_text.text(f"Scanning math parameters... {current_step}/{total_iterations}")
                try:
                    # Added trend='t' to force the model to capture long-term slope
                    tmp_model = ARIMA(series, order=(p_val, d_val, q_val), trend='t')
                    res = tmp_model.fit()
                    if res.aic < best_aic:
                        best_aic = res.aic
                        best_order = (p_val, d_val, q_val)
                except Exception:
                    continue
                
                progress_bar.progress(current_step / total_iterations)
                
    progress_text.empty()
    progress_bar.empty()
    
    return best_order

# 6. Primary Execution Block
if ticker:
    with st.spinner(f"Pulling 5-year historical records for {ticker}..."):
        raw_data = fetch_historical_data(ticker)
        
    if raw_data.empty:
        st.error(f"No data discovered for ticker '{ticker}'. Please ensure the ticker code is valid on Yahoo Finance.")
    else:
        if isinstance(raw_data.columns, pd.MultiIndex):
            raw_data.columns = raw_data.columns.get_level_values(0)
            
        historical_series = raw_data['Close'].resample('W').mean().dropna()
        
        last_date = historical_series.index[-1]
        target_date = datetime(2027, 6, 30)
        
        delta_weeks = int(np.ceil((target_date - last_date).days / 7))
        
        if delta_weeks <= 0:
            st.warning("The target forecasting date (June 2027) has already passed.")
        else:
            try:
                if mode == "⚡ Fast Auto-Tuning (Recommended)":
                    p, d, q = optimize_arima(historical_series)
                
                # Fit final model with trend='t' to prevent flatlines
                model = ARIMA(historical_series, order=(p, d, q), trend='t')
                fitted_model = model.fit()
                
                # Extract Predictions and 95% Confidence Intervals
                predictions = fitted_model.get_forecast(steps=delta_weeks)
                forecast_index = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=delta_weeks, freq='W')
                
                forecast_values = predictions.predicted_mean
                forecast_values.index = forecast_index
                
                conf_int = predictions.conf_int(alpha=0.05) # 95% Confidence Interval
                lower_bound = conf_int.iloc[:, 0]
                upper_bound = conf_int.iloc[:, 1]
                
                # Metric Cards Display
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Latest Close Price", f"₹{historical_series.iloc[-1]:.2f}", f"As of {last_date.strftime('%d %b %Y')}")
                with m2:
                    st.metric("June 2027 Projected Price", f"₹{forecast_values.iloc[-1]:.2f}")
                with m3:
                    st.metric("Model Configuration Used", f"ARIMA({p}, {d}, {q}) + Drift", f"AIC: {fitted_model.aic:.1f}", delta_color="off")
                
                st.markdown("---")
                
                # Visual Layout Split
                chart_col, table_col = st.columns([2, 1])
                
                with chart_col:
                    st.markdown('<div class="card"><h4 style="margin:0; color:#1f4037;">📊 Forecast & Confidence Interval</h4></div>', unsafe_allow_html=True)
                    fig = go.Figure()
                    
                    # 1. Shaded Confidence Interval
                    fig.add_trace(go.Scatter(
                        x=forecast_index.tolist() + forecast_index[::-1].tolist(),
                        y=upper_bound.tolist() + lower_bound[::-1].tolist(),
                        fill='toself',
                        fillcolor='rgba(230, 92, 0, 0.15)',
                        line=dict(color='rgba(255,255,255,0)'),
                        hoverinfo="skip",
                        showlegend=True,
                        name='95% Confidence Range'
                    ))
                    
                    # 2. Historical Data Line
                    fig.add_trace(go.Scatter(
                        x=historical_series.index, y=historical_series.values,
                        name="Historical Data", line=dict(color="#1f4037", width=2)
                    ))
                    
                    # 3. Forecast Line (Now with drift, no more flatline)
                    fig.add_trace(go.Scatter(
                        x=forecast_values.index, y=forecast_values.values,
                        name="Forward Projection", line=dict(color="#e65c00", width=2.5, dash='dash')
                    ))
                    
                    fig.update_layout(
                        template="plotly_white",
                        xaxis_title="Timeline Calendar",
                        yaxis_title="Price (INR)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=10, r=10, t=10, b=10),
                        height=420
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                with table_col:
                    st.markdown('<div class="card"><h4 style="margin:0; color:#1f4037;">🔢 Numeric Estimates</h4></div>', unsafe_allow_html=True)
                    
                    output_df = pd.DataFrame({
                        "Date Target": forecast_values.index.strftime('%Y-%m-%d'),
                        "Expected (₹)": np.round(forecast_values.values, 2),
                        "Low Range (₹)": np.round(lower_bound.values, 2),
                        "High Range (₹)": np.round(upper_bound.values, 2)
                    }).reset_index(drop=True)
                    
                    st.dataframe(output_df, use_container_width=True, height=420)
                    
            except Exception as e:
                st.error("The mathematical parameters failed to fit this specific stock's timeline.")
                st.info("💡 Try switching to 'Manual Specification' and testing an ARIMA(1, 1, 1) configuration.")
