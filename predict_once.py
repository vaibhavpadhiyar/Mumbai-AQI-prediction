"""
Single-run version of the AQI prediction pipeline, designed to be
triggered by GitHub Actions on a daily schedule (see
.github/workflows/daily_predict.yml). Unlike realtime_aqi_system.py,
this does NOT loop forever - it runs once, updates the data files,
and exits. GitHub Actions handles "when to run", not this script.

The WAQI token is read from an environment variable (WAQI_TOKEN) so
it never needs to be committed to the repo in plain text - set it as
a GitHub Actions "secret" (see setup instructions).

STATION DISCOVERY:
Earlier versions of this script hardcoded 5 WAQI station UIDs
(Kurla, BKC, CSMIA, Sion, Bandra). That's fragile - if any station
goes permanently offline (as Bandra/8678 did in Dec 2021, and as
4 more did simultaneously in June 2026), there is no fixed-ID list
that stays valid forever.

Instead, this version queries WAQI's map/bounds endpoint for
whatever stations are CURRENTLY reporting inside Mumbai's bounding
box, and filters to only the ones with a recent timestamp. This
self-heals if stations die or new ones come online - no code change
needed on our end.
"""

import os
import numpy as np
import pandas as pd
import joblib
import requests
from datetime import datetime
from tensorflow.keras.models import load_model

TOKEN = os.environ.get("WAQI_TOKEN", "b96baa0045a132d1ea15a9c8b45f0f390bb1d5b6")

# Southwest lat,lng , Northeast lat,lng - covers Greater Mumbai
MUMBAI_BOUNDS = "18.85,72.75,19.35,73.05"

MODEL_PATH = "mumbai_aqi_lstm_model_finetuned_recent.keras"
SCALER_PATH = "mumbai_aqi_scaler.save"
BUFFER_PATH = "buffer.csv"
PREDICTIONS_LOG_PATH = "predictions_log.csv"
STATIONS_LOG_PATH = "latest_stations.csv"
HISTORY_FULL_PATH = "history_full.csv"

SEQ_LEN = 14
FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "O3", "AQI"]
TARGET_COL = "AQI"
POLLUTANT_COLS = ["PM2.5", "PM10", "NO2", "SO2", "O3"]

MAX_STALE_HOURS = 6          # a station is "offline" if its last reading is older than this
MIN_LIVE_STATIONS = 2        # don't trust the average if fewer than this many stations are live
MAX_AQI_MISMATCH = 100        # if a station's own AQI vs its pollutant-implied AQI differ by
                               # more than this, the station's own AQI value is corrupted -
                               # drop it and derive that station's AQI from pollutants instead
MAX_PLAUSIBLE_SUBINDEX = 500  # WAQI sub-indices top out at 500 (hazardous ceiling) - anything
                               # above that is a bad reading, not real air quality


def _to_valid_number(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v == 999:
        return None
    return v


def discover_live_stations():
    """Query WAQI's map/bounds endpoint for every station currently
    reporting inside Mumbai's bounding box, and keep only the ones whose
    last observation is fresh. Returns a list of {"uid", "name", "age_hours"}.
    """
    url = f"https://api.waqi.info/map/bounds/?latlng={MUMBAI_BOUNDS}&token={TOKEN}"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if data.get("status") != "ok":
            print(f"  Station discovery failed: {data}")
            return []
    except Exception as e:
        print(f"  Station discovery failed: {e}")
        return []

    live_stations = []
    for entry in data.get("data", []):
        aqi_val = entry.get("aqi")
        if aqi_val in (None, "-", "999"):
            continue  # WAQI's own "no data" placeholder for this station

        station_info = entry.get("station", {}) or {}
        time_str = station_info.get("time")
        if not time_str:
            continue
        try:
            obs_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue

        age_hours = (datetime.now() - obs_time).total_seconds() / 3600
        if age_hours > MAX_STALE_HOURS:
            print(
                f"  Discovered station {entry.get('uid')} ({station_info.get('name', '?')}): "
                f"{age_hours:.0f}h old - too stale, skipping."
            )
            continue

        live_stations.append({
            "uid": entry["uid"],
            "name": station_info.get("name", f"uid-{entry['uid']}"),
            "age_hours": age_hours,
        })

    return live_stations


def fetch_station_reading(uid):
    """Fetch pollutant breakdown for one station. A second, independent
    freshness check happens here too (defense in depth) in case the
    bounds endpoint's timestamp and the feed endpoint's timestamp disagree."""
    url = f"https://api.waqi.info/feed/@{uid}/?token={TOKEN}"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if data["status"] != "ok":
            print(f"  Station {uid}: API error - {data}")
            return None

        obs_time_str = data["data"].get("time", {}).get("iso")
        if obs_time_str:
            obs_time = datetime.fromisoformat(obs_time_str)
            now_utc = datetime.now(obs_time.tzinfo) if obs_time.tzinfo else datetime.now()
            age_hours = (now_utc - obs_time).total_seconds() / 3600
            if age_hours > MAX_STALE_HOURS:
                print(
                    f"  Station {uid}: last reading is {age_hours:.0f}h old "
                    f"(observed {obs_time_str}) - sensor looks offline, skipping "
                    f"so it doesn't drag down the average with stale data."
                )
                return None
        else:
            print(f"  Station {uid}: no timestamp in response - skipping to be safe.")
            return None

        iaqi = data["data"].get("iaqi", {})
        reading = {
            "PM2.5": _to_valid_number(iaqi.get("pm25", {}).get("v")),
            "PM10": _to_valid_number(iaqi.get("pm10", {}).get("v")),
            "NO2": _to_valid_number(iaqi.get("no2", {}).get("v")),
            "SO2": _to_valid_number(iaqi.get("so2", {}).get("v")),
            "O3": _to_valid_number(iaqi.get("o3", {}).get("v")),
            "AQI": _to_valid_number(data["data"].get("aqi")),
        }

        # Discard individual pollutant sub-indices outside the plausible WAQI
        # range (0-500). Real example that motivated this: a "live" station
        # reporting PM2.5=837 with no PM10 and no AQI at all - not staleness,
        # just a bad reading, and it would otherwise still poison the average.
        for p in POLLUTANT_COLS:
            v = reading.get(p)
            if v is not None and (v < 0 or v > MAX_PLAUSIBLE_SUBINDEX):
                print(f"    Station {uid}: {p}={v} is outside the plausible 0-{MAX_PLAUSIBLE_SUBINDEX} "
                      f"range - discarding that value (bad reading, not a real level).")
                reading[p] = None

        return reading
    except Exception as e:
        print(f"  Station {uid}: fetch failed - {e}")
        return None


def fetch_today_city_average():
    print("Discovering live stations near Mumbai...")
    candidates = discover_live_stations()
    if len(candidates) < MIN_LIVE_STATIONS:
        print(
            f"  Only {len(candidates)} live station(s) found near Mumbai "
            f"(need at least {MIN_LIVE_STATIONS}) - not enough to trust an average today."
        )
        return None

    print(f"  Found {len(candidates)} live stations: "
          + ", ".join(f"{c['name']} ({c['age_hours']:.1f}h old)" for c in candidates))

    readings, station_rows = [], []
    for c in candidates:
        r = fetch_station_reading(c["uid"])
        if r is not None:
            readings.append(r)
            station_rows.append({"station": c["name"], "uid": c["uid"], **r})
            print(f"  {c['name']}: {r}")

            station_aqi = r.get("AQI")
            pollutant_vals = [r[p] for p in POLLUTANT_COLS if r.get(p) is not None]
            if station_aqi is not None and pollutant_vals:
                implied_aqi = max(pollutant_vals)
                if abs(implied_aqi - station_aqi) > MAX_AQI_MISMATCH:
                    print(
                        f"    Warning: {c['name']} reports AQI={station_aqi} but its own "
                        f"pollutant readings imply ~{implied_aqi:.0f} - this station's AQI "
                        f"field is corrupted, discarding it (its pollutant readings are "
                        f"still used)."
                    )
                    # Drop only the bad AQI field, not the whole station - the
                    # pollutant sub-indices (PM2.5, PM10, etc.) are what actually
                    # drive the derived citywide AQI below, so they're still useful.
                    r["AQI"] = None

    if station_rows:
        pd.DataFrame(station_rows).to_csv(STATIONS_LOG_PATH, index=False)

    if len(readings) < MIN_LIVE_STATIONS:
        print(f"  Only {len(readings)} station(s) returned usable data - skipping today.")
        return None

    df = pd.DataFrame(readings)
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    avg = df[FEATURES].mean(skipna=True)

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
