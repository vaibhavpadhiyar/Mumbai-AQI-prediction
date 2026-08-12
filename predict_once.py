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
Earlier versions hardcoded 5 WAQI station UIDs (Kurla, BKC, CSMIA, Sion,
Bandra). That's fragile - if a station goes permanently offline (as
Bandra/8678 did in Dec 2021, and as 4 more did simultaneously in June
2026), a fixed-ID list never recovers.

This version samples several points spread across Mumbai using WAQI's
/feed/geo:LAT;LON/ endpoint (the same endpoint style as the original
@uid version, just with coordinates instead of a hardcoded ID). WAQI
resolves each point to whichever real station is currently nearest and
reporting, so this self-heals if stations die or new ones appear - no
code change needed on our end. (An earlier attempt used /map/bounds/
instead, but that endpoint returned 0 stations for this token/bbox in
testing, so /feed/ - already proven reliable - is used instead.)
"""

import os
import numpy as np
import pandas as pd
import joblib
import requests
from datetime import datetime
from tensorflow.keras.models import load_model

TOKEN = os.environ.get("WAQI_TOKEN", "b96baa0045a132d1ea15a9c8b45f0f390bb1d5b6")

# Points spread across Greater Mumbai. Each is used with WAQI's
# /feed/geo:LAT;LON/ endpoint, which resolves to whichever real station is
# nearest and currently reporting - so this list doesn't need to match real
# station locations exactly, it just needs decent geographic spread so we
# sample different parts of the city instead of repeatedly hitting one area.
CITY_SAMPLE_POINTS = [
    ("South Mumbai", 18.9220, 72.8347),
    ("Bandra", 19.0596, 72.8295),
    ("Kurla/BKC", 19.0728, 72.8826),
    ("Andheri", 19.1197, 72.8468),
    ("Powai", 19.1176, 72.9060),
    ("Borivali", 19.2307, 72.8567),
    ("Chembur", 19.0522, 72.9005),
    ("Navi Mumbai", 19.0330, 73.0297),
]

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
    """Find live Mumbai stations by sampling several points spread across the
    city with WAQI's /feed/geo:LAT;LON/ endpoint - the SAME endpoint style as
    the original fixed-UID version (/feed/@UID/), just with coordinates
    instead of a hardcoded ID. WAQI resolves each point to whichever real
    station is currently nearest and reporting; duplicates are merged.

    This replaces an earlier attempt using /map/bounds/, which returned 0
    stations for this token/bbox in testing (likely a plan restriction) -
    /feed/ is the endpoint already proven to work reliably with this token.
    Returns a list of {"uid", "name", "reading"} dicts.
    """
    found = {}
    for label, lat, lon in CITY_SAMPLE_POINTS:
        url = f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={TOKEN}"
        try:
            resp = requests.get(url, timeout=10)
            data = resp.json()
        except Exception as e:
            print(f"  {label}: fetch failed - {e}")
            continue

        parsed = _parse_feed_response(data, label)
        if parsed is None:
            continue
        uid, name, reading = parsed
        if uid in found:
            continue  # two sample points resolved to the same nearest station
        found[uid] = {"uid": uid, "name": name, "reading": reading}
        print(f"  {label} -> nearest live station: {name} (uid {uid})")

    print(f"  Discovery: sampled {len(CITY_SAMPLE_POINTS)} points across Mumbai, "
          f"found {len(found)} distinct live station(s).")
    return list(found.values())


def _parse_feed_response(data, source_label):
    """Shared parsing + validation for a single WAQI /feed/ response
    (works whether the URL used @uid or geo:lat;lon). Returns
    (uid, name, reading_dict) if the reading is fresh and usable, else None."""
    if data.get("status") != "ok":
        print(f"  {source_label}: API error - {data}")
        return None

    obs_time_str = data["data"].get("time", {}).get("iso")
    if not obs_time_str:
        print(f"  {source_label}: no timestamp in response - skipping to be safe.")
        return None

    obs_time = pd.to_datetime(obs_time_str, errors="coerce", utc=True)
    if pd.isna(obs_time):
        print(f"  {source_label}: unparseable timestamp '{obs_time_str}' - skipping.")
        return None

    age_hours = (pd.Timestamp.now(tz="UTC") - obs_time).total_seconds() / 3600
    if age_hours > MAX_STALE_HOURS:
        print(
            f"  {source_label}: last reading is {age_hours:.0f}h old "
            f"(observed {obs_time_str}) - sensor looks offline, skipping "
            f"so it doesn't drag down the average with stale data."
        )
        return None

    uid = data["data"].get("idx")
    name = data["data"].get("city", {}).get("name", f"station-{uid}")

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
            print(f"    {name}: {p}={v} is outside the plausible 0-{MAX_PLAUSIBLE_SUBINDEX} "
                  f"range - discarding that value (bad reading, not a real level).")
            reading[p] = None

    return uid, name, reading


def fetch_today_city_average():
    print("Discovering live stations near Mumbai (sampling multiple points across the city)...")
    candidates = discover_live_stations()
    if len(candidates) < MIN_LIVE_STATIONS:
        print(
            f"  Only {len(candidates)} live station(s) found near Mumbai "
            f"(need at least {MIN_LIVE_STATIONS}) - not enough to trust an average today."
        )
        return None

    readings, station_rows = [], []
    for c in candidates:
        r = c["reading"]
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
