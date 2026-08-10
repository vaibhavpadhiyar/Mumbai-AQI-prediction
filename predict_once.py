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
HISTORY_FULL_PATH = "history_full.csv"

SEQ_LEN = 14
FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "O3", "AQI"]
TARGET_COL = "AQI"

# WAQI stations don't always report daily - a station whose sensor is down
# just keeps serving its last known reading. If we don't check the age of
# that reading, one dead sensor can freeze the whole citywide average.
MAX_STATION_AGE_HOURS = 30

import requests


def _to_valid_number(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v == 999:
        return None
    return v


def _reading_age_hours(iso_timestamp):
    """How many hours old is this station's last reading, per WAQI's own
    reported observation time (data.time.iso in the API response)."""
    if not iso_timestamp:
        return None
    try:
        observed = pd.to_datetime(iso_timestamp, utc=True)
        now = pd.Timestamp.now(tz="UTC")
        return (now - observed).total_seconds() / 3600
    except Exception:
        return None


def fetch_station_reading(uid):
    url = f"https://api.waqi.info/feed/@{uid}/?token={TOKEN}"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if data["status"] != "ok":
            print(f"  Station {uid}: API error - {data}")
            return None

        observed_iso = data["data"].get("time", {}).get("iso")
        age_hours = _reading_age_hours(observed_iso)
        if age_hours is not None and age_hours > MAX_STATION_AGE_HOURS:
            print(
                f"  Station {uid}: last reading is {age_hours:.0f}h old "
                f"(observed {observed_iso}) - sensor looks offline, skipping "
                f"so it doesn't drag down the average with stale data."
            )
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


POLLUTANT_COLS = ["PM2.5", "PM10", "NO2", "SO2", "O3"]


def fetch_today_city_average():
    readings, station_rows = [], []
    for name, uid in STATION_UIDS.items():
        r = fetch_station_reading(uid)
        if r is not None:
            readings.append(r)
            station_rows.append({"station": name, **r})
            print(f"  {name}: {r}")

            # WAQI's per-station "aqi" field is sometimes wrong/stale relative
            # to that same station's own pollutant readings (e.g. reporting a
            # low AQI alongside a very high PM2.5). Flag it loudly so it shows
            # up in the Action logs instead of silently poisoning the average.
            station_aqi = r.get("AQI")
            pollutant_vals = [r[p] for p in POLLUTANT_COLS if r.get(p) is not None]
            if station_aqi is not None and pollutant_vals:
                implied_aqi = max(pollutant_vals)
                if implied_aqi - station_aqi > 100:
                    print(
                        f"    Warning: {name} reports AQI={station_aqi} but its own "
                        f"pollutant readings imply ~{implied_aqi:.0f} - likely a bad "
                        f"reading from this station, treating with suspicion."
                    )

    if station_rows:
        pd.DataFrame(station_rows).to_csv(STATIONS_LOG_PATH, index=False)

    if not readings:
        print("  No stations returned data today.")
        return None

    df = pd.DataFrame(readings)
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    avg = df[FEATURES].mean(skipna=True)

    # Recompute the citywide AQI from the averaged pollutant sub-indices
    # (standard AQI methodology: overall AQI = max of the individual
    # pollutant sub-indices) rather than averaging each station's own
    # reported AQI field. Averaging reported AQIs lets one bad station
    # (e.g. a low AQI reported alongside a very high PM2.5) drag the whole
    # city's number down to something inconsistent with the pollutant data.
    pollutant_avgs = avg[POLLUTANT_COLS].dropna()
    if not pollutant_avgs.empty:
        derived_aqi = pollutant_avgs.max()
        if pd.notna(avg.get("AQI")) and abs(derived_aqi - avg["AQI"]) > 50:
            print(
                f"  Note: station-reported AQI average ({avg['AQI']:.1f}) disagreed "
                f"with pollutant-derived AQI ({derived_aqi:.1f}) - using the "
                f"pollutant-derived value."
            )
        avg["AQI"] = derived_aqi

    missing = avg[avg.isna()].index.tolist()
    if missing:
        print(f"  Warning: no valid reading from ANY station for: {missing}")
    return avg


def check_for_stale_data(today_values: "pd.Series", buffer: pd.DataFrame) -> None:
    """Warn (but don't block) if today's fetch is suspiciously identical to
    recent buffer entries - a strong sign the WAQI API served cached/stale
    data instead of a fresh reading (e.g. an expired WAQI_TOKEN)."""
    if buffer.empty:
        return
    recent = buffer.tail(3)
    matches = 0
    for _, row in recent.iterrows():
        if all(
            pd.notna(row.get(col)) and pd.notna(today_values.get(col))
            and abs(row[col] - today_values[col]) < 1e-6
            for col in FEATURES
        ):
            matches += 1
    if matches >= 2:
        print(
            "  Warning: today's reading is identical to at least 2 of the last "
            "3 logged days. This usually means the WAQI API is returning stale "
            "or cached data - double check that the WAQI_TOKEN secret is set "
            "and valid in the repo's GitHub Actions settings."
        )


def update_full_history(today: pd.Timestamp, today_values: "pd.Series") -> None:
    """Append today's reading to a second, never-trimmed log (history_full.csv).

    buffer.csv is intentionally kept short (SEQ_LEN * 3 days) since that's all
    the model needs as input. But a website showing 6-month/1-year AQI trends
    needs the full history, so this keeps a separate, ever-growing file with
    the same columns instead of trimming it."""
    try:
        history_full = pd.read_csv(HISTORY_FULL_PATH, parse_dates=["Date"])
    except FileNotFoundError:
        print(f"No {HISTORY_FULL_PATH} found - creating a new one.")
        history_full = pd.DataFrame(columns=["Date"] + FEATURES)

    if today in history_full["Date"].values:
        return

    new_row = {"Date": today, **today_values.to_dict()}
    history_full = pd.concat([history_full, pd.DataFrame([new_row])], ignore_index=True)
    history_full = history_full.sort_values("Date")
    history_full.to_csv(HISTORY_FULL_PATH, index=False)
    print(f"Full history updated - now {len(history_full)} days stored in {HISTORY_FULL_PATH}.")


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

    check_for_stale_data(today_values, buffer)

    if today in buffer["Date"].values:
        print("Today's entry already logged - skipping duplicate.")
    else:
        new_row = {"Date": today, **today_values.to_dict()}
        buffer = pd.concat([buffer, pd.DataFrame([new_row])], ignore_index=True)
        buffer = buffer.sort_values("Date").tail(SEQ_LEN * 3)
        buffer.to_csv(BUFFER_PATH, index=False)
        print(f"Buffer updated - now {len(buffer)} days stored.")

    update_full_history(today, today_values)

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
