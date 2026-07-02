import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Naira Exchange Rate Predictor",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0d0d1a 0%, #0f0f2e 50%, #0d0d1a 100%);
    color: #e0e0ff;
}

.hero {
    background: linear-gradient(135deg, #1a1a3e 0%, #2d1b69 50%, #1a1a3e 100%);
    border: 1px solid #4a3f8f;
    border-radius: 16px;
    padding: 2.5rem;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(100, 80, 200, 0.3);
}

.hero h1 {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}

.hero p {
    color: #9ca3af;
    font-size: 1rem;
    margin: 0;
}

.metric-card {
    background: linear-gradient(135deg, #1e1b4b, #1e1b3a);
    border: 1px solid #4338ca;
    border-radius: 12px;
    padding: 1.4rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(67, 56, 202, 0.2);
    margin-bottom: 1rem;
}

.metric-card .label {
    font-size: 0.75rem;
    color: #818cf8;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 0.5rem;
}

.metric-card .value {
    font-size: 1.9rem;
    font-weight: 700;
    color: #e0e7ff;
}

.metric-card .sub {
    font-size: 0.8rem;
    color: #6366f1;
    margin-top: 0.3rem;
}

.prediction-box {
    background: linear-gradient(135deg, #1e1b4b, #2e1b69);
    border: 1px solid #7c3aed;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(124, 58, 237, 0.3);
    margin: 1.5rem 0;
}

.prediction-box .pred-label {
    font-size: 0.85rem;
    color: #a78bfa;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.8rem;
}

.prediction-box .pred-value {
    font-size: 3rem;
    font-weight: 700;
    color: #c4b5fd;
}

.prediction-box .pred-sub {
    font-size: 0.9rem;
    color: #7c3aed;
    margin-top: 0.5rem;
}

.section-header {
    font-size: 1rem;
    font-weight: 600;
    color: #818cf8;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin: 1.5rem 0 1rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #2d2d6e;
}

.settings-card {
    background: linear-gradient(135deg, #1e1b4b, #1e1b3a);
    border: 1px solid #4338ca;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(67, 56, 202, 0.2);
}

.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.75rem 2rem;
    font-size: 1rem;
    font-weight: 600;
    width: 100%;
    transition: all 0.3s ease;
    letter-spacing: 0.5px;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.5);
    transform: translateY(-1px);
}

.about-card {
    background: linear-gradient(135deg, #1e1b4b, #1e1b3a);
    border: 1px solid #4338ca;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load("xgboost_usdngn_model.pkl")

@st.cache_data(ttl=3600)
def fetch_live_data():
    df = yf.download("USDNGN=X", start="2015-01-01", progress=False)
    df.columns = df.columns.get_level_values(0)
    df = df[["Close"]].copy()
    df = df[(df["Close"] >= 100) & (df["Close"] <= 2000)]
    return df

def build_features(df):
    df = df.copy()
    df["lag_1"]           = df["Close"].shift(1)
    df["lag_7"]           = df["Close"].shift(7)
    df["lag_30"]          = df["Close"].shift(30)
    df["lag_90"]          = df["Close"].shift(90)
    df["rolling_mean_30"] = df["Close"].shift(1).rolling(30).mean()
    df["day_of_week"]     = df.index.dayofweek
    df["month"]           = df.index.month
    df["year"]            = df.index.year
    df["Close_log"]       = np.log(df["Close"])
    return df.dropna()


# ── HERO ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1> Naira Exchange Rate Predictor</h1>
    <p> · Live market data · </p>
</div>
""", unsafe_allow_html=True)

# ── LOAD DATA ─────────────────────────────────────────────────
model  = load_model()
raw_df = fetch_live_data()
df     = build_features(raw_df)

current_rate = float(raw_df["Close"].iloc[-1])
rate_7d_ago  = float(raw_df["Close"].iloc[-7]) if len(raw_df) >= 7 else current_rate
change_7d    = ((current_rate - rate_7d_ago) / rate_7d_ago) * 100

# ── METRICS ───────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Live Rate</div>
        <div class="value">₦{current_rate:,.2f}</div>
        <div class="sub">per $1 USD</div>
    </div>""", unsafe_allow_html=True)

with col2:
    arrow = "▲" if change_7d > 0 else "▼"
    color = "#f87171" if change_7d > 0 else "#34d399"
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">7-Day Change</div>
        <div class="value" style="color:{color}">{arrow} {abs(change_7d):.2f}%</div>
        <div class="sub">vs last week</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Data Points</div>
        <div class="value">{len(df):,}</div>
        <div class="sub">trading days (2015–2026)</div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Model Accuracy</div>
        <div class="value">96.13%</div>
        <div class="sub">3.87% MAPE on test data</div>
    </div>""", unsafe_allow_html=True)


# ── SETTINGS ──────────────────────────────────────────────────
st.markdown('<div class="section-header">Forecast Settings</div>', unsafe_allow_html=True)

st.markdown('<div class="settings-card">', unsafe_allow_html=True)
set_col1, set_col2 = st.columns(2)
with set_col1:
    days_ahead = st.slider(
        " Forecast horizon (days)",
        min_value=1, max_value=30, value=7
    )
with set_col2:
    lookback = st.selectbox(
        " Historical period to display",
        ["3 Months", "6 Months", "1 Year", "Full History"],
        index=1
    )
st.markdown('</div>', unsafe_allow_html=True)


# ── FORECAST ──────────────────────────────────────────────────
st.markdown('<div class="section-header"> Forecast</div>', unsafe_allow_html=True)

if st.button(" Generate Forecast"):
    forecast_df = df.copy()
    predictions = []
    dates       = []

    for i in range(days_ahead):
        next_date = forecast_df.index[-1] + timedelta(days=1)
        features  = np.array([[
            float(forecast_df["Close"].iloc[-1]),
            float(forecast_df["Close"].iloc[-7]),
            float(forecast_df["Close"].iloc[-30]),
            float(forecast_df["Close"].iloc[-90]),
            float(forecast_df["Close"].shift(1).rolling(30).mean().iloc[-1]),
            next_date.dayofweek,
            next_date.month,
            next_date.year
        ]])

        pred_log  = model.predict(features)[0]
        pred_rate = float(np.exp(pred_log))

        predictions.append(pred_rate)
        dates.append(next_date)

        new_row = pd.DataFrame({
            "Close":           [pred_rate],
            "lag_1":           [float(forecast_df["Close"].iloc[-1])],
            "lag_7":           [float(forecast_df["Close"].iloc[-7])],
            "lag_30":          [float(forecast_df["Close"].iloc[-30])],
            "lag_90":          [float(forecast_df["Close"].iloc[-90])],
            "rolling_mean_30": [float(forecast_df["Close"].iloc[-30:].mean())],
            "day_of_week":     [next_date.dayofweek],
            "month":           [next_date.month],
            "year":            [next_date.year],
            "Close_log":       [pred_log]
        }, index=[next_date])

        forecast_df = pd.concat([forecast_df, new_row])

    final_pred = predictions[-1]
    direction  = "▲" if final_pred > current_rate else "▼"
    diff       = final_pred - current_rate
    color      = "#f87171" if final_pred > current_rate else "#34d399"

    st.markdown(f"""
    <div class="prediction-box">
        <div class="pred-label">Predicted rate in {days_ahead} day{"s" if days_ahead > 1 else ""}</div>
        <div class="pred-value" style="color:{color}">₦{final_pred:,.2f}</div>
        <div class="pred-sub">{direction} ₦{abs(diff):,.2f} from today's rate of ₦{current_rate:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

    forecast_table = pd.DataFrame({
        "Date":               [d.strftime("%b %d, %Y") for d in dates],
        "Predicted Rate (₦)": [f"₦{p:,.2f}" for p in predictions]
    })
    st.dataframe(forecast_table, use_container_width=True, hide_index=True)


# ── HISTORICAL CHART ──────────────────────────────────────────
st.markdown('<div class="section-header"> Historical Rate</div>', unsafe_allow_html=True)

lookback_map = {
    "3 Months":     90,
    "6 Months":     180,
    "1 Year":       365,
    "Full History": len(raw_df)
}
chart_df = raw_df.tail(lookback_map[lookback])[["Close"]].rename(columns={"Close": "USD/NGN Rate (₦)"})
st.line_chart(chart_df, color="#818cf8")


# ── ABOUT ─────────────────────────────────────────────────────
st.markdown('<div class="section-header">ℹ️ About This Model</div>', unsafe_allow_html=True)

st.markdown("""
<div class="about-card">
    <p style="color:#a78bfa; font-size:0.9rem; line-height:1.8; margin:0">
    Trained on daily USD/NGN data from <b>2015–2026</b> using <b>XGBoost</b> with lag features and a 30-day rolling mean.<br><br>
    The model predicts the next day's exchange rate based on historical patterns — 
    lag values from 1, 7, 30, and 90 days ago, plus time-based features like month and day of week.<br><br>
    <b style="color:#e0e7ff">Test MAE:</b> <span style="color:#34d399">₦55.18</span> &nbsp;·&nbsp;
    <b style="color:#e0e7ff">Test MAPE:</b> <span style="color:#34d399">3.87%</span> &nbsp;·&nbsp;
    <b style="color:#e0e7ff">Data source:</b> Yahoo Finance
    </p>
</div>
""", unsafe_allow_html=True)


# ── FOOTER ────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; color:#4338ca; font-size:0.78rem; margin-top:3rem; padding-top:1rem; border-top:1px solid #1e1b4b">
    Built by Mine Dema · XGBoost · Yahoo Finance API · Streamlit · 2026
</div>
""", unsafe_allow_html=True)
