"""
Single-run version of the AQI prediction pipeline, designed to be
triggered by GitHub Actions on a daily schedule (see
.github/workflows/daily_predict.yml). Unlike realtime_aqi_system.py,
this does NOT loop forever - it runs once, updates the data files,
and exits. GitHub Actions handles "when to run", not this script.

The WAQI token is read from an environment variable (WAQI_TOKEN) so
it never needs to be committed to the repo in plain text - set it as
a GitHub Actions "secret" (see setup instructions).
"""

import os
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from tensorflow.keras.models import load_model

TOKEN = os.environ.get("WAQI_TOKEN", "b96baa0045a132d1ea15a9c8b45f0f390bb1d5b6")

STATION_UIDS = {
    "Kurla": "12454",
    "Bandra Kurla Complex": "13715",
    "Chhatrapati Shivaji Intl. Airport": "12456",
    "Sion": "12464",
    "Bandra": "8678",
}

MODEL_PATH = "mumbai_aqi_lstm_model_finetuned_recent.keras"
SCALER_PATH = "mumbai_aqi_scaler.save"
BUFFER_PATH = "buffer.csv"
PREDICTIONS_LOG_PATH = "predictions_log.csv"
STATIONS_LOG_PATH = "latest_stations.csv"

SEQ_LEN = 14
FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "O3", "AQI"]
TARGET_COL = "AQI"

import requests


def _to_valid_number(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v == 999:
        return None
    return v


def fetch_station_reading(uid):
    url = f"https://api.waqi.info/feed/@{uid}/?token={TOKEN}"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if data["status"] != "ok":
            print(f"  Station {uid}: API error - {data}")
            return None
        iaqi = data["data"].get("iaqi", {})
        return {
            "PM2.5": _to_valid_number(iaqi.get("pm25", {}).get("v")),
            "PM10": _to_valid_number(iaqi.get("pm10", {}).get("v")),
            "NO2": _to_valid_number(iaqi.get("no2", {}).get("v")),
            "SO2": _to_valid_number(iaqi.get("so2", {}).get("v")),
            "O3": _to_valid_number(iaqi.get("o3", {}).get("v")),
            "AQI": _to_valid_number(data["data"].get("aqi")),
        }
    except Exception as e:
        print(f"  Station {uid}: fetch failed - {e}")
        return None


def fetch_today_city_average():
    readings, station_rows = [], []
    for name, uid in STATION_UIDS.items():
        r = fetch_station_reading(uid)
        if r is not None:
            readings.append(r)
            station_rows.append({"station": name, **r})
            print(f"  {name}: {r}")

    if station_rows:
        pd.DataFrame(station_rows).to_csv(STATIONS_LOG_PATH, index=False)

    if not readings:
        print("  No stations returned data today.")
        return None

    df = pd.DataFrame(readings)
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    avg = df[FEATURES].mean(skipna=True)

    missing = avg[avg.isna()].index.tolist()
    if missing:
        print(f"  Warning: no valid reading from ANY station for: {missing}")
    return avg


def main():
    print(f"Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print("Loading model and scaler...")
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    target_idx = FEATURES.index(TARGET_COL)

    today = pd.Timestamp.now().normalize()

    print("Fetching live station data...")
    today_values = fetch_today_city_average()
    if today_values is None:
        print("No data available today - exiting without prediction.")
        return

    try:
        buffer = pd.read_csv(BUFFER_PATH, parse_dates=["Date"])
    except FileNotFoundError:
        print(f"No {BUFFER_PATH} found - creating a new one.")
        buffer = pd.DataFrame(columns=["Date"] + FEATURES)

    if today in buffer["Date"].values:
        print("Today's entry already logged - skipping duplicate.")
    else:
        new_row = {"Date": today, **today_values.to_dict()}
        buffer = pd.concat([buffer, pd.DataFrame([new_row])], ignore_index=True)
        buffer = buffer.sort_values("Date").tail(SEQ_LEN * 3)
        buffer.to_csv(BUFFER_PATH, index=False)
        print(f"Buffer updated - now {len(buffer)} days stored.")

    if len(buffer) < SEQ_LEN:
        print(f"Not enough history yet ({len(buffer)}/{SEQ_LEN} days).")
        return

    recent_window = buffer.tail(SEQ_LEN)[FEATURES].values
    if np.isnan(recent_window).any():
        recent_window = (
            pd.DataFrame(recent_window, columns=FEATURES)
            .interpolate(limit_direction="both")
            .values
        )

    scaled_window = scaler.transform(recent_window)
    X_input = scaled_window.reshape(1, SEQ_LEN, len(FEATURES))
    pred_scaled = model.predict(X_input, verbose=0).flatten()

    dummy = np.zeros((1, len(FEATURES)))
    dummy[:, target_idx] = pred_scaled
    pred_aqi = scaler.inverse_transform(dummy)[:, target_idx][0]

    print(f"PREDICTED AQI FOR TOMORROW: {pred_aqi:.1f}")

    log_row = {
        "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "based_on_date": today.strftime("%Y-%m-%d"),
        "predicted_aqi_for": (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        "predicted_aqi": round(float(pred_aqi), 1),
    }
    try:
        log_df = pd.read_csv(PREDICTIONS_LOG_PATH)
        log_df = pd.concat([log_df, pd.DataFrame([log_row])], ignore_index=True)
    except FileNotFoundError:
        log_df = pd.DataFrame([log_row])
    log_df.to_csv(PREDICTIONS_LOG_PATH, index=False)
    print(f"Logged to {PREDICTIONS_LOG_PATH}")


if __name__ == "__main__":
    main()
