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
        .stTabs [data-baseweb="tab"] { background-color: #ffffff; border-radius: 6px 6px 0 0; padding: 10px 20px; box-
