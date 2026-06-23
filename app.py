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

# Suppress statistical warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="NSE Advanced Forecaster Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .main { background-color: #f4f7f6; font-family: 'Inter', 'Segoe UI', sans-serif; }
        .header-container {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            padding: 2.5rem; border-radius: 12px; color: white; margin-bottom: 2rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .header-title { font-size: 2.6rem; font-weight: 800; margin: 0; letter-spacing: -0.5px; }
        .header-subtitle { font-size: 1.1rem; opacity: 0.95; margin-top: 0.5rem; }
        .card {
            background: white; padding: 1.5rem; border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.03); border: 1px solid #e0e6ed; margin-bottom: 1rem;
        }
        /* Style Streamlit Tabs */
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] { 
            background-color: #ffffff; border-radius: 6px 6px 0 0; 
            padding: 10px 20px; box-shadow: 0 -2px 5px rgba(0,0,0,0.02);
        }
        .stTabs [aria-selected="true"] { background-color: #1e3c72; color: white !important; font-weight: 600;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. HEADER BANNER
# ==========================================
st.markdown("""
    <div class="header-container">
        <div class="header-title">📊 Advanced NSE Market Forecaster Pro</div>
        <div class="header-subtitle">Comprehensive algorithmic ARIMA projections, technical momentum, and volatility analysis.</div>
    </div>
""", unsafe_allow_html=True)

# ==========================================
# 3. SIDEBAR CONFIGURATION
# ==========================================
with st.sidebar:
    st.header("⚙️ Target Asset")
    ticker_input = st.text_input("Enter NSE Ticker Symbol:", value="RELIANCE", help="Examples: TCS, INFY, HDFCBANK")
    
    ticker = ticker_input.strip().upper()
    if not ticker.endswith(".NS"):
        ticker = f"{ticker}.NS"
        
    st.markdown("---")
    st.markdown("### 🧠 Forecasting Engine")
    mode = st.radio("Optimization Mode:", ["⚡ Auto-Tune (Recommended)", "🎛️ Manual Override"])
    
    if mode == "🎛️ Manual Override":
        p = st.slider("Autoregressive (p)", 0, 5, 1)
        d = st.slider("Differencing (d)", 0, 2, 1)
        q = st.slider("Moving Average (q)", 0, 5, 1)

# ==========================================
# 4. CORE DATA FUNCTIONS
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
        if isinstance(raw_data.columns, pd.MultiIndex):
            raw_data.columns = raw_data.columns.get_level_values(0)
            
        historical_series = raw_data['Close'].resample('W').mean().dropna()
        
        # Calculate Advanced Metrics
        sma_10 = historical_series.rolling(window=10).mean()
        sma_40 = historical_series.rolling(window=40).mean()
        weekly_returns = historical_series.pct_change().dropna() * 100
        
        last_date = historical_series.index[-1]
        target_date = datetime(2027, 6, 30)
        delta_weeks = int(np.ceil((target_date - last_date).days / 7))
        
        if delta_weeks <= 0:
            st.warning("The target forecasting date of June 2027 has already passed.")
        else:
            try:
                # ARIMA Modeling
                if mode == "⚡ Auto-Tune (Recommended)":
                    p, d, q = optimize_arima(historical_series)
                
                model = ARIMA(historical_series, order=(p, d, q), trend='t')
                fitted_model = model.fit()
                
                predictions = fitted_model.get_forecast(steps=delta_weeks)
                forecast_index = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=delta_weeks, freq='W')
                forecast_values = predictions.predicted_mean
                forecast_values.index = forecast_index
                
                conf_int = predictions.conf_int(alpha=0.05)
                lower_bound, upper_bound = conf_int.iloc[:, 0], conf_int.iloc[:, 1]
                
                current_price = historical_series.iloc[-1]
                future_price = forecast_values.iloc[-1]
                growth_pct = ((future_price - current_price) / current_price) * 100
                
                # Top Metrics Row
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Current Evaluation", f"₹{current_price:.2f}", f"As of {last_date.strftime('%d %b %Y')}")
                with m2:
                    st.metric("Target 2027 Price", f"₹{future_price:.2f}", f"{growth_pct:+.2f}% Expected Growth")
                with m3:
                    st.metric("Macro Trend (40W)", "Bullish 📈" if current_price > sma_40.iloc[-1] else "Bearish 📉")
                with m4:
                    st.metric("Model Architecture", f"ARIMA({p},{d},{q})", f"AIC Fit: {fitted_model.aic:.1f}", delta_color="off")
                
                st.markdown("---")
                
                # ==========================================
                # 6. TABBED DASHBOARD RENDERING
                # ==========================================
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "🔮 Primary Forecast", 
                    "📈 Technical Momentum", 
                    "🧩 Trend Decomposition", 
                    "⚖️ Risk Profile", 
                    "💾 Ledger Export"
                ])
                
                # TAB 1: Main Forecast
                with tab1:
                    st.markdown('<div class="card"><h4 style="margin:0;">Primary ARIMA Projection & Confidence Intervals</h4></div>', unsafe_allow_html=True)
                    fig1 = go.Figure()
                    fig1.add_trace(go.Scatter(
                        x=forecast_index.tolist() + forecast_index[::-1].tolist(),
                        y=upper_bound.tolist() + lower_bound[::-1].tolist(),
                        fill='toself', fillcolor='rgba(30, 60, 114, 0.15)', line=dict(color='rgba(255,255,255,0)'),
                        hoverinfo="skip", showlegend=True, name='95% Probability Zone'
                    ))
                    fig1.add_trace(go.Scatter(x=historical_series.index, y=historical_series.values, name="Historical Price", line=dict(color="#111111", width=2)))
                    fig1.add_trace(go.Scatter(x=forecast_values.index, y=forecast_values.values, name="Forecast Trajectory", line=dict(color="#1e3c72", width=3, dash='dash')))
                    fig1.update_layout(template="plotly_white", hovermode="x unified", height=500, margin=dict(t=20, b=20))
                    st.plotly_chart(fig1, use_container_width=True)
                
                # TAB 2: Moving Averages
                with tab2:
                    st.markdown('<div class="card"><h4 style="margin:0;">10-Week vs 40-Week Simple Moving Averages</h4></div>', unsafe_allow_html=True)
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=historical_series.index, y=historical_series.values, name="Price", line=dict(color="#e0e0e0", width=1.5)))
                    fig2.add_trace(go.Scatter(x=sma_10.index, y=sma_10.values, name="10-Week SMA (Fast)", line=dict(color="#00C853", width=2)))
                    fig2.add_trace(go.Scatter(x=sma_40.index, y=sma_40.values, name="40-Week SMA (Slow)", line=dict(color="#D50000", width=2)))
                    fig2.update_layout(template="plotly_white", hovermode="x unified", height=500, margin=dict(t=20, b=20))
                    st.plotly_chart(fig2, use_container_width=True)
                
                # TAB 3: Seasonal Decomposition
                with tab3:
                    st.markdown('<div class="card"><h4 style="margin:0;">Time-Series Decomposition (52-Week Period)</h4></div>', unsafe_allow_html=True)
                    try:
                        # Break data into Trend, Seasonality, and Residual noise
                        decomposition = seasonal_decompose(historical_series, model='additive', period=52)
                        
                        fig3 = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                                             subplot_titles=("Underlying Macro Trend", "Annual Seasonality Cycle", "Random Market Noise (Residuals)"))
                        
                        fig3.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, line=dict(color="#2980b9")), row=1, col=1)
                        fig3.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, line=dict(color="#27ae60")), row=2, col=1)
                        fig3.add_trace(go.Bar(x=decomposition.resid.index, y=decomposition.resid, marker_color="#c0392b"), row=3, col=1)
                        
                        fig3.update_layout(template="plotly_white", height=650, showlegend=False, margin=dict(t=40, b=20))
                        st.plotly_chart(fig3, use_container_width=True)
                    except Exception:
                        st.info("Insufficient data length to perform a 52-week seasonal decomposition.")

                # TAB 4: Risk Profile (Histogram)
                with tab4:
                    st.markdown('<div class="card"><h4 style="margin:0;">Volatility Profile (Weekly Returns Distribution)</h4></div>', unsafe_allow_html=True)
                    fig4 = go.Figure(data=[go.Histogram(x=weekly_returns, nbinsx=50, marker_color="#1e3c72", opacity=0.75)])
                    fig4.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Break Even (0%)")
                    fig4.update_layout(template="plotly_white", xaxis_title="Weekly Return (%)", yaxis_title="Frequency (Weeks)", height=500, margin=dict(t=20, b=20))
                    st.plotly_chart(fig4, use_container_width=True)
                    
                # TAB 5: Data Ledger & Export
                with tab5:
                    st.markdown('<div class="card"><h4 style="margin:0;">Raw Forecast Ledger</h4></div>', unsafe_allow_html=True)
                    output_df = pd.DataFrame({
                        "Date": forecast_values.index.strftime('%Y-%m-%d'),
                        "Expected Price (₹)": np.round(forecast_values.values, 2),
                        "Optimistic High (₹)": np.round(upper_bound.values, 2),
                        "Pessimistic Low (₹)": np.round(lower_bound.values, 2)
                    }).reset_index(drop=True)
                    
                    st.dataframe(output_df, use_container_width=True, height=400)
                    csv = convert_df_to_csv(output_df)
                    st.download_button(label="📥 Download Full Forecast (.csv)", data=csv, file_name=f"{ticker}_Forecast_2027.csv", mime="text/csv")
                    
            except Exception as e:
                st.error("Matrix compilation failed for this specific asset's volatility structure.")
                st.info("💡 Try reverting to 'Manual Override' mode and setting parameters to (1, 1, 1).")
