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

# Suppress all backend warnings for a clean execution environment
warnings.filterwarnings("ignore")

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Advanced Quant Forecaster",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .main { background-color: #f0f2f5; font-family: 'Inter', 'Segoe UI', sans-serif; }
        .header-container {
            background: linear-gradient(135deg, #111827 0%, #374151 100%);
            padding: 2.5rem; border-radius: 12px; color: white; margin-bottom: 2rem;
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        }
        .header-title { font-size: 2.6rem; font-weight: 800; margin: 0; letter-spacing: -0.5px; color: #f9fafb; }
        .header-subtitle { font-size: 1.1rem; opacity: 0.85; margin-top: 0.5rem; color: #9ca3af; }
        .card { background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.02); border: 1px solid #e5e7eb; margin-bottom: 1rem; }
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] { background-color: #ffffff; border-radius: 6px 6px 0 0; padding: 10px 20px; box-shadow: 0 -2px 5px rgba(0,0,0,0.02); }
        .stTabs [aria-selected="true"] { background-color: #111827; color: white !important; font-weight: 600;}
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="header-container">
        <div class="header-title">🧬 Institutional Quant Forecaster</div>
        <div class="header-subtitle">Stochastic Monte Carlo Risk Analysis, Probability Matrices, and Volatility Adjusted Projections.</div>
    </div>
""", unsafe_allow_html=True)

# ==========================================
# 2. SIDEBAR CONFIGURATION
# ==========================================
with st.sidebar:
    st.header("⚙️ Target Asset")
    ticker_input = st.text_input("Enter NSE Ticker Symbol:", value="RELIANCE")
    
    # Clean and format ticker safely
    ticker = ticker_input.strip().upper()
    if ticker and not ticker.endswith(".NS"):
        ticker = f"{ticker}.NS"
        
    st.markdown("---")
    st.markdown("### 🧠 Forecasting Engine")
    mode = st.radio("Optimization Mode:", ["⚡ Auto-Tune (Recommended)", "🎛️ Manual Override"])
    
    if mode == "🎛️ Manual Override":
        p = st.slider("Autoregressive Order (p)", 0, 5, 1)
        d = st.slider("Differencing Degree (d)", 0, 2, 1)
        q = st.slider("Moving Average Order (q)", 0, 5, 1)

# ==========================================
# 3. BULLETPROOF DATA FUNCTIONS
# ==========================================
@st.cache_data(ttl=3600)
def fetch_historical_data(symbol):
    """Safely extracts exactly 5 years of closing prices without MultiIndex errors."""
    try:
        tkr = yf.Ticker(symbol)
        df = tkr.history(period="5y")
        if df.empty or 'Close' not in df.columns:
            return pd.Series()
        
        # Scrub timezone to prevent plotting and ARIMA frequency errors
        df.index = df.index.tz_localize(None)
        return df['Close']
    except Exception:
        return pd.Series()

def optimize_arima(series):
    """Grid search that ignores invertibility/stationarity to prevent crashing."""
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
                    tmp_model = ARIMA(
                        series, order=(p_val, d_val, q_val), trend='t', 
                        enforce_stationarity=False, enforce_invertibility=False
                    )
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
    return df.to_csv(index=True).encode('utf-8')

# ==========================================
# 4. MAIN EXECUTION PIPELINE
# ==========================================
if ticker:
    with st.spinner(f"Establishing secure connection to market data for {ticker}..."):
        close_series = fetch_historical_data(ticker)
        
    if close_series.empty:
        st.error(f"Failed to retrieve data for '{ticker}'. Please verify the symbol is active on Yahoo Finance.")
    else:
        # Standardize strictly to Fridays to ensure math intervals are perfect
        historical_series = close_series.resample('W-FRI').mean().dropna()
        
        sma_10 = historical_series.rolling(window=10).mean()
        sma_40 = historical_series.rolling(window=40).mean()
        weekly_returns = historical_series.pct_change().dropna() * 100
        
        last_date = historical_series.index[-1]
        target_date = datetime(2027, 6, 30)
        delta_weeks = int(np.ceil((target_date - last_date).days / 7))
        
        if delta_weeks <= 0:
            st.warning("The target projection horizon (June 2027) is already in the past.")
        else:
            try:
                # 1. Configuration
                if mode == "⚡ Auto-Tune (Recommended)":
                    p, d, q = optimize_arima(historical_series)
                
                # 2. Model Fitting (Forced stability)
                model = ARIMA(
                    historical_series, order=(p, d, q), trend='t', 
                    enforce_stationarity=False, enforce_invertibility=False
                )
                fitted_model = model.fit()
                
                # 3. Expected Mean Baseline
                predictions = fitted_model.get_forecast(steps=delta_weeks)
                forecast_index = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=delta_weeks, freq='W-FRI')
                
                forecast_values = predictions.predicted_mean
                forecast_values.index = forecast_index

                # 4. Background Monte Carlo Engine (25 Paths)
                num_simulations = 25
                np.random.seed(42) 
                
                # Simulate paths and lock to the exact forecast index
                sims = fitted_model.simulate(nsimulations=delta_weeks, anchor='end', repetitions=num_simulations)
                sims.index = forecast_index
                sims.columns = [f"Path_{i+1}" for i in range(num_simulations)]
                
                sim_upper = sims.quantile(0.95, axis=1)
                sim_lower = sims.quantile(0.05, axis=1)
                
                current_price = historical_series.iloc[-1]
                future_price = forecast_values.iloc[-1]
                probability_of_profit = (sims.iloc[-1] > current_price).mean() * 100
                
                # ==========================================
                # 5. DASHBOARD UI RENDER
                # ==========================================
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Current Asset Value", f"₹{current_price:.2f}")
                with m2:
                    st.metric("Expected Mean (June 2027)", f"₹{future_price:.2f}")
                with m3:
                    st.metric("Probability of Profit", f"{probability_of_profit:.1f}%", "Chance to beat current price")
                with m4:
                    st.metric("Expected Shortfall (Worst 5%)", f"₹{sim_lower.iloc[-1]:.2f}", "95% Confidence floor", delta_color="inverse")
                
                st.markdown("---")
                
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "🔮 Volatility Projection", 
                    "📈 Moving Averages", 
                    "🧩 Trend Structure", 
                    "⚖️ Risk Distribution", 
                    "💾 Forecast Ledger"
                ])
                
                with tab1:
                    st.markdown('<div class="card"><h4 style="margin:0; color:#111827;">Stochastic Forward Projections</h4></div>', unsafe_allow_html=True)
                    fig1 = go.Figure()
                    
                    # Simulated Bounds
                    fig1.add_trace(go.Scatter(
                        x=forecast_index.tolist() + forecast_index[::-1].tolist(),
                        y=sim_upper.tolist() + sim_lower[::-1].tolist(),
                        fill='toself', fillcolor='rgba(17, 24, 39, 0.08)', line=dict(color='rgba(255,255,255,0)'),
                        hoverinfo="skip", showlegend=True, name='Simulated 95% Bounds'
                    ))
                    
                    # Historical Data
                    fig1.add_trace(go.Scatter(x=historical_series.index, y=historical_series.values, name="Historical Price", line=dict(color="#111111", width=2.5)))
                    
                    # Background Paths
                    for col in sims.columns:
                        fig1.add_trace(go.Scatter(
                            x=sims.index, y=sims[col], name=f"Simulated {col}", 
                            line=dict(width=1, color=f"rgba(220, 38, 38, 0.08)"), 
                            showlegend=False, hoverinfo="skip"
                        ))
                    
                    # Main Forecast Line
                    fig1.add_trace(go.Scatter(x=forecast_values.index, y=forecast_values.values, name="Expected Mean (Smoothed)", line=dict(color="#111827", width=3, dash='dash')))
                    
                    fig1.update_layout(template="plotly_white", xaxis_title="Timeline Calendar", yaxis_title="Price (INR)", hovermode="x unified", height=550, margin=dict(t=15, b=15))
                    st.plotly_chart(fig1, use_container_width=True)

                with tab2:
                    st.markdown('<div class="card"><h4 style="margin:0; color:#111827;">Asset Momentum via Simple Moving Averages</h4></div>', unsafe_allow_html=True)
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=historical_series.index, y=historical_series.values, name="Base Asset Price", line=dict(color="#e0e0e0", width=1.5)))
                    fig2.add_trace(go.Scatter(x=sma_10.index, y=sma_10.values, name="10-Week Fast SMA", line=dict(color="#10b981", width=2)))
                    fig2.add_trace(go.Scatter(x=sma_40.index, y=sma_40.values, name="40-Week Slow Macro SMA", line=dict(color="#ef4444", width=2)))
                    fig2.update_layout(template="plotly_white", xaxis_title="Timeline Calendar", yaxis_title="Price (INR)", hovermode="x unified", height=500, margin=dict(t=15, b=15))
                    st.plotly_chart(fig2, use_container_width=True)
                
                with tab3:
                    st.markdown('<div class="card"><h4 style="margin:0; color:#111827;">Additive Time-Series Decomposition (52-Week Filter)</h4></div>', unsafe_allow_html=True)
                    try:
                        decomposition = seasonal_decompose(historical_series, model='additive', period=52)
                        fig3 = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, subplot_titles=("Macro Trend Line", "Annual Cyclical Seasonality", "Unsystematic Random Market Residuals"))
                        fig3.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, line=dict(color="#3b82f6", width=2)), row=1, col=1)
                        fig3.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, line=dict(color="#10b981", width=1.5)), row=2, col=1)
                        fig3.add_trace(go.Bar(x=decomposition.resid.index, y=decomposition.resid, marker_color="#ef4444"), row=3, col=1)
                        fig3.update_layout(template="plotly_white", height=650, showlegend=False, margin=dict(t=40, b=15))
                        st.plotly_chart(fig3, use_container_width=True)
                    except Exception:
                        st.info("Insufficient historical tracking length to execute a full 52-week seasonal decomposition.")

                with tab4:
                    st.markdown('<div class="card"><h4 style="margin:0; color:#111827;">Historical Volatility Architecture</h4></div>', unsafe_allow_html=True)
                    fig4 = go.Figure(data=[go.Histogram(x=weekly_returns, nbinsx=60, marker_color="#374151", opacity=0.80)])
                    fig4.add_vline(x=0, line_dash="dash", line_color="#ef4444", annotation_text="Break-Even Threshold (0%)", annotation_position="top left")
                    fig4.update_layout(template="plotly_white", xaxis_title="Percentage Weekly Shift (%)", yaxis_title="Instance Frequency (Weeks)", height=500, margin=dict(t=15, b=15))
                    st.plotly_chart(fig4, use_container_width=True)
                    
                with tab5:
                    st.markdown('<div class="card"><h4 style="margin:0; color:#111827;">Target Analytical Output Dataframe</h4></div>', unsafe_allow_html=True)
                    output_df = pd.DataFrame({
                        "Target Date": forecast_values.index.strftime('%Y-%m-%d'),
                        "Expected Mean (₹)": np.round(forecast_values.values, 2),
                        "Simulated Upper Bound (₹)": np.round(sim_upper.values, 2),
                        "Simulated Lower Bound (₹)": np.round(sim_lower.values, 2)
                    }).set_index("Target Date")
                    
                    st.dataframe(output_df, use_container_width=True, height=400)
                    csv_bytes = convert_df_to_csv(output_df)
                    st.download_button(label="📥 Download Structured Forecast Ledger (.csv)", data=csv_bytes, file_name=f"{ticker}_quant_projections_2027.csv", mime="text/csv")
                    
            except Exception as core_err:
                st.error(f"Mathematical execution failed. System details: {str(core_err)}")
                st.info("💡 Try adjusting the manual parameters if the automatic matrix failed to converge.")
