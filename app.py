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

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="NSE Advanced Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .main { background-color: #f4f7f6; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .header-container {
            background: linear-gradient(135deg, #0b486b 0%, #f56217 100%);
            padding: 2.5rem; border-radius: 12px; color: white; margin-bottom: 2rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .header-title { font-size: 2.5rem; font-weight: 800; margin: 0; letter-spacing: -0.5px; }
        .header-subtitle { font-size: 1.1rem; opacity: 0.95; margin-top: 0.5rem; font-weight: 400; }
        .card {
            background: white; padding: 1.5rem; border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.03); border: 1px solid #e0e6ed; margin-bottom: 1rem;
        }
        .stButton>button { width: 100%; border-radius: 6px; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. HEADER BANNER
# ==========================================
st.markdown("""
    <div class="header-container">
        <div class="header-title">📈 Advanced NSE Market Forecaster</div>
        <div class="header-subtitle">Algorithmic ARIMA projections, confidence intervals, and moving average technicals through June 2027.</div>
    </div>
""", unsafe_allow_html=True)

# ==========================================
# 3. SIDEBAR CONFIGURATION
# ==========================================
with st.sidebar:
    st.header("⚙️ Target Asset")
    
    ticker_input = st.text_input("Enter NSE Ticker Symbol:", value="RELIANCE", help="Examples: TCS, INFY, HDFCBANK, TATAMOTORS")
    
    ticker = ticker_input.strip().upper()
    if not ticker.endswith(".NS"):
        ticker = f"{ticker}.NS"
        
    st.markdown("---")
    st.markdown("### 🧠 Forecasting Engine")
    mode = st.radio("Optimization Mode:", ["⚡ Auto-Tune (Recommended)", "🎛️ Manual Override"])
    
    if mode == "🎛️ Manual Override":
        st.markdown("Adjust mathematical parameters:")
        p = st.slider("Autoregressive (p)", 0, 5, 1, help="Lags of the stationary series. High values look deeper into past trends.")
        d = st.slider("Differencing (d)", 0, 2, 1, help="Degree of differencing. 1 is standard for stocks to remove random walk.")
        q = st.slider("Moving Average (q)", 0, 5, 1, help="Lags of the forecast errors. Smooths out sudden recent shocks.")

# ==========================================
# 4. CORE FUNCTIONS
# ==========================================
@st.cache_data(ttl=86400)
def fetch_historical_data(symbol):
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - pd.DateOffset(years=5)).strftime('%Y-%m-%d')
    df = yf.download(symbol, start=start_date, end=end_date)
    return df

def optimize_arima(series):
    best_aic = float("inf")
    best_order = (1, 1, 1) 
    
    p_values, d_values, q_values = [0, 1, 2], [1], [0, 1, 2]
    total_iters = len(p_values) * len(d_values) * len(q_values)
    
    prog_text = st.empty()
    prog_bar = st.progress(0)
    
    step = 0
    for p_val in p_values:
        for d_val in d_values:
            for q_val in q_values:
                step += 1
                prog_text.text(f"Optimizing Algorithm... Testing Matrix {step}/{total_iters}")
                try:
                    tmp_model = ARIMA(series, order=(p_val, d_val, q_val), trend='t')
                    res = tmp_model.fit()
                    if res.aic < best_aic:
                        best_aic = res.aic
                        best_order = (p_val, d_val, q_val)
                except Exception:
                    continue
                prog_bar.progress(step / total_iters)
                
    prog_text.empty()
    prog_bar.empty()
    return best_order

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# ==========================================
# 5. MAIN EXECUTION PIPELINE
# ==========================================
if ticker:
    with st.spinner(f"Establishing secure connection to Yahoo Finance for {ticker}..."):
        raw_data = fetch_historical_data(ticker)
        
    if raw_data.empty:
        st.error(f"No market data found for '{ticker}'. Ensure the ticker is actively listed on the NSE.")
    else:
        # Standardize DataFrame
        if isinstance(raw_data.columns, pd.MultiIndex):
            raw_data.columns = raw_data.columns.get_level_values(0)
            
        # Process primary historical series (Weekly average)
        historical_series = raw_data['Close'].resample('W').mean().dropna()
        
        # Calculate Technical Indicators (10-Week and 40-Week SMAs)
        sma_10 = historical_series.rolling(window=10).mean()
        sma_40 = historical_series.rolling(window=40).mean()
        
        last_date = historical_series.index[-1]
        target_date = datetime(2027, 6, 30)
        delta_weeks = int(np.ceil((target_date - last_date).days / 7))
        
        if delta_weeks <= 0:
            st.warning("The target forecasting date of June 2027 has already passed.")
        else:
            try:
                # 1. Hyperparameter Tuning
                if mode == "⚡ Auto-Tune (Recommended)":
                    p, d, q = optimize_arima(historical_series)
                
                # 2. Model Fitting
                model = ARIMA(historical_series, order=(p, d, q), trend='t')
                fitted_model = model.fit()
                
                # 3. Generating Predictions & Intervals
                predictions = fitted_model.get_forecast(steps=delta_weeks)
                forecast_index = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=delta_weeks, freq='W')
                
                forecast_values = predictions.predicted_mean
                forecast_values.index = forecast_index
                
                conf_int = predictions.conf_int(alpha=0.05)
                lower_bound, upper_bound = conf_int.iloc[:, 0], conf_int.iloc[:, 1]
                
                # 4. Calculate Growth Metrics
                current_price = historical_series.iloc[-1]
                future_price = forecast_values.iloc[-1]
                growth_pct = ((future_price - current_price) / current_price) * 100
                
                # ==========================================
                # 6. DASHBOARD RENDER
                # ==========================================
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Current Evaluation", f"₹{current_price:.2f}", f"As of {last_date.strftime('%d %b')}")
                with m2:
                    st.metric("Target 2027 Price", f"₹{future_price:.2f}", f"{growth_pct:+.2f}% Growth expected")
                with m3:
                    st.metric("Technical Trend (10W)", "Bullish 📈" if current_price > sma_10.iloc[-1] else "Bearish 📉")
                with m4:
                    st.metric("Model Architecture", f"ARIMA({p},{d},{q})", f"AIC Fit: {fitted_model.aic:.1f}", delta_color="off")
                
                st.markdown("---")
                chart_col, table_col = st.columns([2.5, 1])
                
                with chart_col:
                    st.markdown('<div class="card"><h4 style="margin:0;">📊 Precision Forecast Matrix</h4></div>', unsafe_allow_html=True)
                    fig = go.Figure()
                    
                    # Confidence Area
                    fig.add_trace(go.Scatter(
                        x=forecast_index.tolist() + forecast_index[::-1].tolist(),
                        y=upper_bound.tolist() + lower_bound[::-1].tolist(),
                        fill='toself', fillcolor='rgba(245, 98, 23, 0.15)', line=dict(color='rgba(255,255,255,0)'),
                        hoverinfo="skip", showlegend=True, name='95% Probability Zone'
                    ))
                    # Historical Data
                    fig.add_trace(go.Scatter(x=historical_series.index, y=historical_series.values, name="Actual Price", line=dict(color="#0b486b", width=2)))
                    # Technicals
                    fig.add_trace(go.Scatter(x=sma_40.index, y=sma_40.values, name="40-Week SMA (Macro Trend)", line=dict(color="#8e9eab", width=1.5, dash='dot')))
                    # Forecast
                    fig.add_trace(go.Scatter(x=forecast_values.index, y=forecast_values.values, name="Algorithmic Projection", line=dict(color="#f56217", width=3, dash='dash')))
                    
                    fig.update_layout(
                        template="plotly_white", xaxis_title="Market Timeline", yaxis_title="Price Output (INR)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=0, r=0, t=10, b=0), height=450, hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                with table_col:
                    st.markdown('<div class="card"><h4 style="margin:0;">🔢 Ledger Export</h4></div>', unsafe_allow_html=True)
                    
                    output_df = pd.DataFrame({
                        "Date": forecast_values.index.strftime('%Y-%m-%d'),
                        "Expected (₹)": np.round(forecast_values.values, 2),
                        "High (₹)": np.round(upper_bound.values, 2),
                        "Low (₹)": np.round(lower_bound.values, 2)
                    }).reset_index(drop=True)
                    
                    st.dataframe(output_df, use_container_width=True, height=380)
                    
                    # CSV Download Button
                    csv = convert_df_to_csv(output_df)
                    st.download_button(label="📥 Download Data (.csv)", data=csv, file_name=f"{ticker}_Forecast_2027.csv", mime="text/csv")
                
                # Educational Details Expander
                with st.expander("📖 View Historical Dataset & Analytical Details"):
                    st.markdown("""
                    **How to read this dashboard:**
                    * **Actual Price (Blue Line):** The 5-year historical weekly average of the selected stock.
                    * **40-Week SMA (Dotted Line):** The Simple Moving Average smoothing out 40 weeks of data to show macro-trends. 
                    * **Algorithmic Projection (Orange Dash):** The main trajectory calculated by the ARIMA engine utilizing historical drift (`trend='t'`).
                    * **95% Probability Zone (Shaded Area):** Statistically, there is a 95% chance that the actual future price will fall somewhere inside this growing shaded zone. The further into the future we predict, the wider the zone becomes due to market uncertainty.
                    """)
                    st.dataframe(historical_series.reset_index().rename(columns={'Close': 'Historical Close (₹)'}), use_container_width=True)
                    
            except Exception as e:
                st.error("Matrix compilation failed for this specific asset's volatility structure.")
                st.info("💡 Try reverting to 'Manual Override' mode and setting all sliders to 1.")
