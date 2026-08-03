"""
Live Mumbai AQI Prediction Dashboard
------------------------------------------------------------------------
Reads the data files produced by predict_once.py (run daily via GitHub
Actions) and displays them. This file itself does NOT call the WAQI API
or run the model - it just visualizes whatever the latest committed
data files contain, so it stays fast and lightweight.

To run locally:  streamlit run dashboard.py
To deploy:        push to GitHub, then connect the repo on
                   https://share.streamlit.io (point it at dashboard.py)
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Mumbai AQI Live Prediction", layout="wide")

st.title("🌫️ Mumbai Air Quality — Live LSTM Prediction")
st.caption("Auto-updates daily via a scheduled prediction job. "
           f"Dashboard last loaded: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# ---- Load data files (all produced by predict_once.py) ----
try:
    predictions = pd.read_csv("predictions_log.csv")
except FileNotFoundError:
    predictions = pd.DataFrame()

try:
    buffer = pd.read_csv("buffer.csv", parse_dates=["Date"])
except FileNotFoundError:
    buffer = pd.DataFrame()

try:
    stations = pd.read_csv("latest_stations.csv")
except FileNotFoundError:
    stations = pd.DataFrame()

# ---- Top row: latest prediction as a headline number ----
col1, col2, col3 = st.columns(3)

if not predictions.empty:
    latest = predictions.iloc[-1]
    col1.metric("Predicted AQI (tomorrow)", f"{latest['predicted_aqi']:.0f}",
                help=f"For {latest['predicted_aqi_for']}, based on data through {latest['based_on_date']}")
else:
    col1.warning("No predictions logged yet.")

if not buffer.empty:
    latest_actual = buffer.iloc[-1]
    col2.metric("Most recent actual AQI (city avg)", f"{latest_actual['AQI']:.0f}",
                help=f"Date: {latest_actual['Date'].date()}")

if not predictions.empty:
    col3.metric("Total predictions logged", len(predictions))

st.divider()

# ---- Historical AQI trend + prediction ----
st.subheader("Recent AQI Trend")

if not buffer.empty:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=buffer["Date"], y=buffer["AQI"],
        mode="lines+markers", name="Actual AQI (city average)",
        line=dict(color="#1f77b4")
    ))

    if not predictions.empty:
        last_pred = predictions.iloc[-1]
        fig.add_trace(go.Scatter(
            x=[pd.to_datetime(last_pred["predicted_aqi_for"])],
            y=[last_pred["predicted_aqi"]],
            mode="markers", name="Predicted (tomorrow)",
            marker=dict(color="red", size=14, symbol="star")
        ))

    fig.update_layout(xaxis_title="Date", yaxis_title="AQI",
                       height=400, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No buffer data yet - the first scheduled run will populate this.")

st.divider()

# ---- Today's per-station breakdown ----
st.subheader("Latest Station Readings")
if not stations.empty:
    st.dataframe(stations, use_container_width=True, hide_index=True)
else:
    st.info("No station-level data logged yet.")

st.divider()

# ---- Prediction history table ----
st.subheader("Prediction History")
if not predictions.empty:
    st.dataframe(predictions.sort_values("run_timestamp", ascending=False),
                 use_container_width=True, hide_index=True)
else:
    st.info("No predictions logged yet.")
