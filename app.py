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

# Suppress backend warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. PAGE CONFIGURATION & PREMIUM CSS
# ==========================================
st.set_page_config(
    page_title="Global Pro Forecaster",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
        
        .stApp { font-family: 'Inter', sans-serif; background-color: #f8fafc; }
        
        .header-container {
            background: linear-gradient(135deg, #0f172a 0%, #3b82f6 100%);
            padding: 3rem; border-radius: 16px; color: white; margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(59, 130, 246, 0.25);
        }
        .header-title { font-size: 2.8rem; font-weight: 800; margin: 0; letter-spacing: -1px; }
        .header-subtitle { font-size: 1.1rem; font-weight: 400; opacity: 0.9; margin-top: 0.5rem; }
        
        [data-testid="stMetric"] {
            background-color: #ffffff; padding: 1.5rem; border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.04); border-left: 5px solid #3b82f6;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        [data-testid="stMetric"] label, [data-testid="stMetric"] div { color: #0f172a !important; }
        [data-testid="stMetric"]:hover { transform: translateY(-3px); box-shadow: 0 8px 15px rgba(0,0,0,0.08); }
        
        .card { background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.02); border: 1px solid #e2e8f0; margin-bottom: 1rem; color: #0f172a; }
        
        .stTabs [data-baseweb="tab-list"] { gap: 10px; }
        .stTabs [data-baseweb="tab"] { background-color: #ffffff; border-radius: 8px 8px 0 0; padding: 12px 24px; box-shadow: 0 -2px 5px rgba(0,0,0,0.02); border: 1px solid #e2e8f0; border-bottom: none; }
        .stTabs [data-baseweb="tab"] div { color: #0f172a; }
        .stTabs [aria-selected="true"] { background-color: #0f172a; border-color: #0f172a; }
        .stTabs [aria-selected="true"] div { color: white !important; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="header-container">
        <div class="header-title">🌍 Global Market Analytics</div>
        <div class="header-subtitle">Unrestricted Algorithmic ARIMA Modeling with Automated Native Currency Formatting.</div>
    </div>
""", unsafe_allow_html=True)

# ==========================================
# 2. MASTER GLOBAL CURRENCY REGISTRY
# ==========================================
GLOBAL_CURRENCY_REGISTRY = {
    "USD": {"symbol": "$", "name": "US Dollar"},
    "INR": {"symbol": "₹", "name": "Indian Rupee"},
    "EUR": {"symbol": "€", "name": "Euro"},
    "JPY": {"symbol": "¥", "name": "Japanese Yen"},
    "GBP": {"symbol": "£", "name": "British Pound"},
    "GBP": {"symbol": "p", "name": "British Pence"}, 
    "CAD": {"symbol": "C$", "name": "Canadian Dollar"},
    "AUD": {"symbol": "A$", "name": "Australian Dollar"},
    "CHF": {"symbol": "CHF ", "name": "Swiss Franc"},
    "CNY": {"symbol": "¥", "name": "Chinese Yuan"},
    "HKD": {"symbol": "HK$", "name": "Hong Kong Dollar"},
    "SGD": {"symbol": "S$", "name": "Singapore Dollar"},
    "KRW": {"symbol": "₩", "name": "South Korean Won"},
    "BRL": {"symbol": "R$", "name": "Brazilian Real"},
    "ZAR": {"symbol": "R ", "name": "South African Rand"},
    "MXN": {"symbol": "$", "name": "Mexican Peso"},
    "NZD": {"symbol": "NZ$", "name": "New Zealand Dollar"},
    "SEK": {"symbol": "kr ", "name": "Swedish Krona"},
    "NOK": {"symbol": "kr ", "name": "Norwegian Krone"},
    "DKK": {"symbol": "kr ", "name": "Danish Krone"},
    "AED": {"symbol": "DH ", "name": "UAE Dirham"},
    "SAR": {"symbol": "SR ", "name": "Saudi Riyal"},
    "TRY": {"symbol": "₺", "name": "Turkish Lira"},
    "TWD": {"symbol": "NT$", "name": "New Taiwan Dollar"}
}

# ==========================================
# 3. SIDEBAR CONFIGURATION
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2942/2942269.png", width=60)
    st.header("⚙️ Target Asset")
    
    st.markdown("**Enter Global Ticker Symbol:**")
    st.caption("Examples: AAPL (US), RELIANCE.NS (India), INFY.NS (India), BARC.L (UK), DAI.DE (Germany)")
    ticker_input = st.text_input("", value="RELIANCE.NS", label_visibility="collapsed")
    ticker = ticker_input.strip().upper()
        
    st.markdown("---")
    st.markdown("### 🧠 Forecasting Engine")
    mode = st.radio("Optimization Mode:", ["⚡ Auto-Tune (Recommended)", "🎛️ Manual Override"])
    
    if mode == "🎛️ Manual Override":
        p = st.slider("Autoregressive Order (p)", 0, 5, 1)
        d = st.slider("Differencing Degree (d)", 0, 2, 1)
        q = st.slider("Moving Average Order (q)", 0, 5, 1)

# ==========================================
# 4. DATA ENGINE FUNCTIONS
# ==========================================
@st.cache_data(ttl=3600)
def fetch_historical_data(symbol):
    try:
        tkr = yf.Ticker(symbol)
        df = tkr.history(period="5y")
        if df.empty or 'Close' not in df.columns:
            return pd.Series()
        df.index = df.index.tz_localize(None)
        return df['Close']
    except Exception:
        return pd.Series()

@st.cache_data(ttl=3600)
def fetch_currency_metadata(symbol):
    try:
        tkr = yf.Ticker(symbol)
        info = tkr.info
        code = info.get('currency', info.get('financialCurrency', 'USD')).upper()
        
        if code in GLOBAL_CURRENCY_REGISTRY:
            return code, GLOBAL_CURRENCY_REGISTRY[code]["symbol"], GLOBAL_CURRENCY_REGISTRY[code]["name"]
        else:
            return code, f"{code} ", f"Local Currency ({code})"
    except Exception:
        return 'USD', '$', 'US Dollar'

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
                prog_text.text(f"Scanning algorithms... {step}/{total_iters}")
                try:
                    # FIX: Changed trend='t' to trend='c' to prevent horizontal flat-lining
                    tmp_model = ARIMA(series, order=(p_val, d_val, q_val), trend='c', enforce_stationarity=False, enforce_invertibility=False)
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
    with st.spinner(f"Establishing connection & extracting global currency metrics for {ticker}..."):
        close_series = fetch_historical_data(ticker)
        currency_code, curr_sym, currency_name = fetch_currency_metadata(ticker)
        
    if close_series.empty:
        st.error(f"❌ Failed to retrieve data for '{ticker}'. Check that you are using the correct global suffix (e.g. '.NS' for India, '.L' for London).")
    else:
        historical_series = close_series.resample('W-FRI').mean().dropna()
        sma_10 = historical_series.rolling(window=10).mean()
        sma_40 = historical_series.rolling(window=40).mean()
        
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
                
                # 2. Backtesting for Accuracy Score
                train_size = int(len(historical_series) * 0.90)
                train, test = historical_series.iloc[:train_size], historical_series.iloc[train_size:]
                try:
                    # FIX: Changed trend='t' to trend='c'
                    val_model = ARIMA(train, order=(p,d,q), trend='c', enforce_stationarity=False, enforce_invertibility=False)
                    val_fit = val_model.fit()
                    val_preds = val_fit.forecast(steps=len(test))
                    mape = np.mean(np.abs((test - val_preds) / test)) * 100
                    accuracy_score = max(0, 100 - mape)
                except:
                    accuracy_score = 0
                
                # 3. Main Model Fitting
                # FIX: Changed trend='t' to trend='c' for the main model as well
                model = ARIMA(historical_series, order=(p, d, q), trend='c', enforce_stationarity=False, enforce_invertibility=False)
                fitted_model = model.fit()
                
                # 4. Base Projections & Confidence Bounds
                predictions = fitted_model.get_forecast(steps=delta_weeks)
                forecast_index = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=delta_weeks, freq='W-FRI')
                
                forecast_values = predictions.predicted_mean
                forecast_values.index = forecast_index
                
                conf_int = predictions.conf_int(alpha=0.05) 
                upper_bound = conf_int.iloc[:, 1]
                lower_bound = conf_int.iloc[:, 0].clip(lower=0) 
                
                # NEW FEATURE: Generate a realistic Stochastic Market Path based on the ARIMA variance
                np.random.seed(len(ticker) + int(historical_series.iloc[-1])) # Unique but stable seed per stock
                random_shocks = np.random.normal(0, fitted_model.resid.std(), delta_weeks)
                simulated_wobble = forecast_values + np.cumsum(random_shocks)

                # Calculations
                current_price = historical_series.iloc[-1]
                future_price = forecast_values.iloc[-1]
                growth_pct = ((future_price - current_price) / current_price) * 100
                
                # ==========================================
                # 6. DASHBOARD UI RENDER 
                # ==========================================
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric(f"Current Evaluation ({currency_name})", f"{curr_sym}{current_price:,.2f}", f"As of {last_date.strftime('%d %b %Y')}")
                with m2:
                    st.metric(f"Target Projection ({currency_code})", f"{curr_sym}{future_price:,.2f}", f"{growth_pct:+.2f}% Expected Return")
                with m3:
                    st.metric("Macro Trend (40W SMA)", "Bullish 📈" if current_price > sma_40.iloc[-1] else "Bearish 📉", "Technical Indicator")
                with m4:
                    st.metric("Historical Accuracy Score", f"{accuracy_score:.1f}%", "Based on hold-out backtest", delta_color="normal")
                
                trend_status = "an upward bullish" if current_price > sma_40.iloc[-1] else "a downward bearish"
                growth_dir = "grow" if growth_pct > 0 else "decline"
                
                st.info(f"**💡 AI Executive Summary:** Based on backtested data, **{ticker}** is currently in {trend_status} macro trend. Using an ARIMA({p},{d},{q}) mathematical structure (which has historically proven **{accuracy_score:.1f}% accurate** on this asset), the algorithm projects the asset will {growth_dir} by **{abs(growth_pct):.1f}%** to reach a final expected baseline of **{curr_sym}{future_price:,.2f} {currency_name}** by June 2027.")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                tab1, tab2, tab3, tab4 = st.tabs([
                    "🔮 Primary Forecast", 
                    "📈 Technical Momentum", 
                    "🧩 Trend Structure", 
                    "💾 Data Ledger"
                ])
                
                with tab1:
                    st.markdown('<div class="card"><h4 style="margin:0;">Primary Trajectory & 95% Confidence Bounds</h4></div>', unsafe_allow_html=True)
                    fig1 = go.Figure()
                    
                    # 95% Confidence Interval Area
                    fig1.add_trace(go.Scatter(
                        x=forecast_index.tolist() + forecast_index[::-1].tolist(),
                        y=upper_bound.tolist() + lower_bound[::-1].tolist(),
                        fill='toself', fillcolor='rgba(59, 130, 246, 0.15)', line=dict(color='rgba(255,255,255,0)'),
                        hoverinfo="skip", showlegend=True, name='95% Confidence Interval'
                    ))
                    
                    # Historical Data
                    fig1.add_trace(go.Scatter(x=historical_series.index, y=historical_series.values, name="Actual Price", line=dict(color="#0f172a", width=2.5)))
                    
                    # NEW: Simulated Market Volatility Path
                    fig1.add_trace(go.Scatter(x=forecast_values.index, y=simulated_wobble.values, name="Simulated Volatility", line=dict(color="#10b981", width=1.5)))
                    
                    # Expected Baseline (Mean)
                    fig1.add_trace(go.Scatter(x=forecast_values.index, y=forecast_values.values, name="Expected Baseline", line=dict(color="#3b82f6", width=3, dash='dash')))
                    
                    fig1.update_layout(template="plotly_white", xaxis_title="Market Timeline", yaxis_title=f"Asset Price ({currency_name})", hovermode="x unified", height=500, margin=dict(t=15, b=15))
                    st.plotly_chart(fig1, use_container_width=True)

                with tab2:
                    st.markdown('<div class="card"><h4 style="margin:0;">Momentum via Simple Moving Averages</h4></div>', unsafe_allow_html=True)
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=historical_series.index, y=historical_series.values, name="Asset Price", line=dict(color="#cbd5e1", width=1.5)))
                    fig2.add_trace(go.Scatter(x=sma_10.index, y=sma_10.values, name="10-Week SMA (Fast)", line=dict(color="#10b981", width=2)))
                    fig2.add_trace(go.Scatter(x=sma_40.index, y=sma_40.values, name="40-Week SMA (Macro)", line=dict(color="#ef4444", width=2)))
                    
                    fig2.update_layout(template="plotly_white", xaxis_title="Market Timeline", yaxis_title=f"Asset Price ({currency_name})", hovermode="x unified", height=500, margin=dict(t=15, b=15))
                    st.plotly_chart(fig2, use_container_width=True)
                
                with tab3:
                    st.markdown('<div class="card"><h4 style="margin:0;">Time-Series Decomposition (52-Week Filter)</h4></div>', unsafe_allow_html=True)
                    try:
                        decomposition = seasonal_decompose(historical_series, model='additive', period=52)
                        fig3 = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, subplot_titles=("Underlying Macro Trend", "Annual Cyclical Seasonality", "Unsystematic Random Noise"))
                        fig3.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, line=dict(color="#3b82f6", width=2)), row=1, col=1)
                        fig3.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, line=dict(color="#10b981", width=1.5)), row=2, col=1)
                        fig3.add_trace(go.Bar(x=decomposition.resid.index, y=decomposition.resid, marker_color="#f59e0b"), row=3, col=1)
                        fig3.update_layout(template="plotly_white", height=650, showlegend=False, margin=dict(t=40, b=15))
                        st.plotly_chart(fig3, use_container_width=True)
                    except Exception:
                        st.info("Insufficient historical tracking length to execute a full 52-week seasonal decomposition.")

                with tab4:
                    st.markdown('<div class="card"><h4 style="margin:0;">Target Analytical Dataframe</h4></div>', unsafe_allow_html=True)
                    
                    output_df = pd.DataFrame({
                        "Target Date": forecast_values.index,
                        "Expected Target": forecast_values.values,
                        "Optimistic Bound": upper_bound.values,
                        "Pessimistic Bound": lower_bound.values
                    })
                    
                    fig_table = go.Figure(data=[go.Table(
                        header=dict(
                            values=["<b>Target Date</b>", f"<b>Expected Price ({currency_name})</b>", "<b>High Risk Cap</b>", "<b>Low Risk Floor</b>"],
                            fill_color='#0f172a',
                            align='center',
                            font=dict(color='white', size=14)
                        ),
                        cells=dict(
                            values=[
                                output_df["Target Date"].dt.strftime('%b %d, %Y'),
                                output_df["Expected Target"].apply(lambda x: f"{curr_sym}{x:,.2f}"),
                                output_df["Optimistic Bound"].apply(lambda x: f"{curr_sym}{x:,.2f}"),
                                output_df["Pessimistic Bound"].apply(lambda x: f"{curr_sym}{x:,.2f}")
                            ],
                            fill_color='#f8fafc',
                            align='center',
                            font=dict(color='#0f172a', size=13),
                            height=30
                        )
                    )])
                    fig_table.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=400)
                    st.plotly_chart(fig_table, use_container_width=True)
                    
                    csv_bytes = convert_df_to_csv(output_df)
                    st.download_button(label=f"📥 Download Formatted Ledger ({currency_code})", data=csv_bytes, file_name=f"{ticker}_forecast_2027.csv", mime="text/csv")
                    
            except Exception as core_err:
                st.error(f"Execution failed. System details: {str(core_err)}")
