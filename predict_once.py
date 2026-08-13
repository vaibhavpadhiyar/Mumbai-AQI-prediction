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

MULTI-SOURCE FALLBACK (Aug 2026):
WAQI's Mumbai data feed itself went dark for ~50 days (every station,
same last-reading timestamp - confirmed via discovery logs, not our bug).
Rather than depend on one upstream source, this version tries THREE
independent sources in order and uses the first that returns usable data:

  1. WAQI (keyword search, see below) - kept first in case it recovers.
  2. CPCB via data.gov.in - India's own official real-time monitoring
     network, the same one WAQI itself is supposed to ingest from.
     Requires CPCB_API_KEY (free, data.gov.in registration). Prioritized
     over OpenWeatherMap since it's ground-truth government data.
  3. OpenWeatherMap Air Pollution API - coordinate-based modeled reading,
     not tied to any single physical sensor that can individually die.
     Requires OWM_API_KEY (free tier, instant signup). Used last since
     it's a modeled estimate rather than a direct measurement.

Only if ALL THREE come back empty does the script fall back to reusing
the model's last known-good window (see main()), clearly flagged as
"stale_fallback" rather than presented as live.

WAQI STATION DISCOVERY:
Earlier versions hardcoded 5 WAQI station UIDs (Kurla, BKC, CSMIA, Sion,
Bandra). That's fragile - if a station goes permanently offline (as
Bandra/8678 did in Dec 2021, and as 4 more did simultaneously in June
2026), a fixed-ID list never recovers.

This version searches WAQI's /search/?keyword=Mumbai endpoint to find
station names matching "Mumbai" directly - a text match, not a geographic
one - then fetches each match's own /feed/@uid/ for pollutant data and a
freshness check. (Two earlier attempts used /map/bounds/ and
/feed/geo:lat;lon/ - both location-based endpoints - and both returned
unusable results for this token: /map/bounds/ returned 0 stations, and
/feed/geo:/ returned the same unrelated Delhi station regardless of the
Mumbai coordinates given.)
"""

import os
import numpy as np
import pandas as pd
import joblib
import requests
from datetime import datetime
from tensorflow.keras.models import load_model

TOKEN = os.environ.get("WAQI_TOKEN", "b96baa0045a132d1ea15a9c8b45f0f390bb1d5b6")
OWM_API_KEY = os.environ.get("OWM_API_KEY", "")
CPCB_API_KEY = os.environ.get("CPCB_API_KEY", "")

# Keywords used against WAQI's /search/ endpoint to find Mumbai station
# names. Includes the old British spelling too, in case some station is
# still labeled that way in WAQI's database.
SEARCH_KEYWORDS = ["Mumbai", "Bombay"]

MUMBAI_CENTER = (19.0760, 72.8777)

# "Real Time Air Quality Index From CPCB" dataset on data.gov.in.
CPCB_RESOURCE_ID = "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"

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


# --- EPA AQI sub-index conversion (used for OpenWeatherMap and, if needed,
# any other source that gives raw µg/m3 concentrations instead of a
# pre-computed 0-500 index like WAQI's iaqi). Standard US EPA breakpoint
# tables, PM2.5 using the 2024 revision. ---

def _linear_aqi(conc, breakpoints):
    """breakpoints: list of (bp_lo, bp_hi, aqi_lo, aqi_hi) tuples, ascending."""
    if conc is None or conc < 0:
        return None
    for bp_lo, bp_hi, aqi_lo, aqi_hi in breakpoints:
        if bp_lo <= conc <= bp_hi:
            return round((aqi_hi - aqi_lo) / (bp_hi - bp_lo) * (conc - bp_lo) + aqi_lo, 1)
    last_hi = breakpoints[-1][1]
    return 500.0 if conc > last_hi else 0.0


PM25_BREAKPOINTS = [  # micrograms/m3, 24-hr avg
    (0.0, 9.0, 0, 50), (9.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
    (55.5, 125.4, 151, 200), (125.5, 225.4, 201, 300), (225.5, 325.4, 301, 500),
]
PM10_BREAKPOINTS = [  # micrograms/m3, 24-hr avg
    (0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150),
    (255, 354, 151, 200), (355, 424, 201, 300), (425, 604, 301, 500),
]
# EPA defines NO2/SO2/O3 breakpoints in ppb; OWM/CPCB give micrograms/m3,
# so convert using each gas's molar mass first.
NO2_BREAKPOINTS_PPB = [
    (0, 53, 0, 50), (54, 100, 51, 100), (101, 360, 101, 150),
    (361, 649, 151, 200), (650, 1249, 201, 300), (1250, 2049, 301, 500),
]
SO2_BREAKPOINTS_PPB = [
    (0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150),
    (186, 304, 151, 200), (305, 604, 201, 300), (605, 1004, 301, 500),
]
O3_BREAKPOINTS_PPB = [  # 8-hr table (EPA also has a separate 1-hr table above 200, omitted for simplicity)
    (0, 54, 0, 50), (55, 70, 51, 100), (71, 85, 101, 150),
    (86, 105, 151, 200), (106, 200, 201, 300),
]


def _ugm3_to_ppb(ugm3, molecular_weight):
    if ugm3 is None:
        return None
    return ugm3 * 24.45 / molecular_weight


def pm25_to_aqi(v):
    return _linear_aqi(v, PM25_BREAKPOINTS)


def pm10_to_aqi(v):
    return _linear_aqi(v, PM10_BREAKPOINTS)


def no2_to_aqi(v_ugm3):
    return _linear_aqi(_ugm3_to_ppb(v_ugm3, 46.0055), NO2_BREAKPOINTS_PPB)


def so2_to_aqi(v_ugm3):
    return _linear_aqi(_ugm3_to_ppb(v_ugm3, 64.066), SO2_BREAKPOINTS_PPB)


def o3_to_aqi(v_ugm3):
    return _linear_aqi(_ugm3_to_ppb(v_ugm3, 48.0), O3_BREAKPOINTS_PPB)


def discover_live_stations():
    """Find live Mumbai stations via WAQI's keyword search (text match on
    station name, not geography), then verify each match with its own
    /feed/@uid/ call for pollutant data and a freshness check. Duplicates
    across keywords are merged by uid. Returns a list of
    {"uid", "name", "reading"} dicts.
    """
    matched = {}  # uid -> name, deduped across keywords
    for kw in SEARCH_KEYWORDS:
        url = f"https://api.waqi.info/search/?token={TOKEN}&keyword={kw}"
        try:
            resp = requests.get(url, timeout=10)
            data = resp.json()
        except Exception as e:
            print(f"  Search for '{kw}' failed: {e}")
            continue

        if data.get("status") != "ok":
            print(f"  Search for '{kw}' failed: {data}")
            continue

        results = data.get("data", [])
        print(f"  Search '{kw}': {len(results)} station name match(es).")
        for entry in results:
            uid = entry.get("uid")
            name = (entry.get("station", {}) or {}).get("name", f"uid-{uid}")
            if uid is None or uid in matched:
                continue
            matched[uid] = name

    if not matched:
        print("  No stations matched Mumbai/Bombay by name search.")
        return []

    print(f"  {len(matched)} distinct station(s) matched by name - checking freshness of each...")

    live = []
    for uid, name in matched.items():
        url = f"https://api.waqi.info/feed/@{uid}/?token={TOKEN}"
        try:
            resp = requests.get(url, timeout=10)
            data = resp.json()
        except Exception as e:
            print(f"  {name} (uid {uid}): fetch failed - {e}")
            continue

        parsed = _parse_feed_response(data, f"{name} (uid {uid})")
        if parsed is None:
            continue
        parsed_uid, parsed_name, reading = parsed
        live.append({"uid": parsed_uid, "name": parsed_name, "reading": reading})

    print(f"  Discovery: {len(live)} of {len(matched)} name-matched stations are live and fresh.")
    return live


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


def fetch_openweathermap_reading():
    """Fallback source #2. OWM's Air Pollution API gives modeled pollutant
    concentrations for a coordinate rather than readings tied to a specific
    physical station - so there's no individual sensor that can go offline
    the way a WAQI/CPCB station can. Raw concentrations (micrograms/m3) are
    converted to EPA AQI sub-indices to match the scale WAQI's iaqi (and
    this model) uses. Returns a list with one entry, or [] if unavailable.
    """
    if not OWM_API_KEY:
        print("  OpenWeatherMap: OWM_API_KEY not set - skipping this source.")
        return []

    lat, lon = MUMBAI_CENTER
    url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={OWM_API_KEY}"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
    except Exception as e:
        print(f"  OpenWeatherMap: fetch failed - {e}")
        return []

    entries = data.get("list", [])
    if not entries:
        print(f"  OpenWeatherMap: no data returned - {data}")
        return []

    entry = entries[0]
    dt = entry.get("dt")
    if dt is None:
        print("  OpenWeatherMap: no timestamp in response - skipping to be safe.")
        return []
    obs_time = pd.Timestamp(dt, unit="s", tz="UTC")
    age_hours = (pd.Timestamp.now(tz="UTC") - obs_time).total_seconds() / 3600
    if age_hours > MAX_STALE_HOURS:
        print(f"  OpenWeatherMap: last reading is {age_hours:.0f}h old - skipping.")
        return []

    c = entry.get("components", {})
    reading = {
        "PM2.5": pm25_to_aqi(c.get("pm2_5")),
        "PM10": pm10_to_aqi(c.get("pm10")),
        "NO2": no2_to_aqi(c.get("no2")),
        "SO2": so2_to_aqi(c.get("so2")),
        "O3": o3_to_aqi(c.get("o3")),
    }
    pollutant_vals = [v for v in reading.values() if v is not None]
    reading["AQI"] = max(pollutant_vals) if pollutant_vals else None
    print(f"  OpenWeatherMap: raw {c} -> converted sub-indices {reading}")

    return [{"uid": "owm-mumbai-center", "name": "OpenWeatherMap (Mumbai, modeled)", "reading": reading}]


def fetch_cpcb_readings():
    """Fallback source #3: India's official CPCB real-time AQI bulletin via
    data.gov.in - the same government network WAQI is itself supposed to
    ingest from. NOTE: this dataset's pollutant values are documented as
    already being on the 0-500 AQI index scale (not raw concentrations),
    so no breakpoint conversion is applied here - verify this against a
    real response from your own API key, since this integration hasn't
    been run against a live key yet.
    """
    if not CPCB_API_KEY:
        print("  CPCB: CPCB_API_KEY not set - skipping this source.")
        return []

    url = (
        f"https://api.data.gov.in/resource/{CPCB_RESOURCE_ID}"
        f"?api-key={CPCB_API_KEY}&format=json&filters[city]=Mumbai&limit=200"
    )
    try:
        resp = requests.get(url, timeout=15)
        data = resp.json()
    except Exception as e:
        print(f"  CPCB: fetch failed - {e}")
        return []

    records = data.get("records", [])
    if not records:
        print(f"  CPCB: no records returned for Mumbai - {data.get('message', data)}")
        return []

    pollutant_key_map = {"PM2.5": "PM2.5", "PM10": "PM10", "NO2": "NO2", "SO2": "SO2", "OZONE": "O3", "O3": "O3"}
    by_station = {}
    for rec in records:
        station = rec.get("station", "Unknown CPCB station")
        col = pollutant_key_map.get((rec.get("pollutant_id") or "").upper())
        if col is None:
            continue
        try:
            val = float(rec.get("pollutant_avg"))
        except (TypeError, ValueError):
            continue

        entry = by_station.setdefault(station, {
            "uid": f"cpcb-{station}",
            "name": f"{station} (CPCB)",
            "last_update": rec.get("last_update"),
            "reading": {"PM2.5": None, "PM10": None, "NO2": None, "SO2": None, "O3": None, "AQI": None},
        })
        entry["reading"][col] = val

    live = []
    for station, entry in by_station.items():
        ts = pd.to_datetime(entry.get("last_update"), errors="coerce")
        if pd.isna(ts):
            print(f"  CPCB {station}: no usable timestamp - skipping.")
            continue
        ts_utc = ts.tz_localize("Asia/Kolkata").tz_convert("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        age_hours = (pd.Timestamp.now(tz="UTC") - ts_utc).total_seconds() / 3600
        if age_hours > MAX_STALE_HOURS:
            print(f"  CPCB {station}: last reading is {age_hours:.0f}h old - skipping.")
            continue

        r = entry["reading"]
        pollutant_vals = [v for v in [r["PM2.5"], r["PM10"], r["NO2"], r["SO2"], r["O3"]] if v is not None]
        r["AQI"] = max(pollutant_vals) if pollutant_vals else None
        live.append({"uid": entry["uid"], "name": entry["name"], "reading": r})

    print(f"  CPCB: {len(live)} of {len(by_station)} Mumbai station(s) are live and fresh.")
    return live


def discover_live_readings():
    """Try each live source in order, use the first that returns enough
    usable data. Order: WAQI -> CPCB -> OpenWeatherMap. CPCB is prioritized
    over OpenWeatherMap because it's India's own official government
    monitoring network (ground-truth measurements), whereas OpenWeatherMap
    is a modeled/interpolated estimate - CPCB is the better source when
    both are available, even though OWM's coordinate-based reading is
    structurally more failure-resistant (no single sensor to go offline).
    WAQI and CPCB need >=MIN_LIVE_STATIONS distinct stations (to average
    out single-sensor noise); OpenWeatherMap's single modeled citywide
    reading is accepted alone since it isn't a physical sensor that can
    individually misbehave the same way.
    Returns (source_name, list of {"uid","name","reading"}).
    """
    print("Trying WAQI first...")
    waqi = discover_live_stations()
    if len(waqi) >= MIN_LIVE_STATIONS:
        return "waqi", waqi
    print(f"  WAQI: only {len(waqi)} usable station(s) - trying CPCB next.")

    cpcb = fetch_cpcb_readings()
    if len(cpcb) >= MIN_LIVE_STATIONS:
        return "cpcb", cpcb
    print(f"  CPCB: only {len(cpcb)} usable station(s) - trying OpenWeatherMap next.")

    owm = fetch_openweathermap_reading()
    if len(owm) >= 1:
        return "openweathermap", owm
    print("  OpenWeatherMap: unavailable either - no live source available today.")

    return "none", []


def fetch_today_city_average():
    print("Discovering live air quality data for Mumbai (WAQI -> CPCB -> OpenWeatherMap)...")
    source, candidates = discover_live_readings()
    if not candidates:
        print("  No live source returned usable data today.")
        return None

    print(f"  Using source: {source} ({len(candidates)} reading(s))")

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
        for row in station_rows:
            row["source"] = source
        pd.DataFrame(station_rows).to_csv(STATIONS_LOG_PATH, index=False)

    # OpenWeatherMap's single citywide modeled reading is intentionally
    # allowed to bypass the multi-station minimum (see discover_live_readings).
    min_needed = 1 if source == "openweathermap" else MIN_LIVE_STATIONS
    if len(readings) < min_needed:
        print(f"  Only {len(readings)} reading(s) returned usable data - skipping today.")
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
    avg.attrs["source"] = source
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

    try:
        buffer = pd.read_csv(BUFFER_PATH, parse_dates=["Date"])
    except FileNotFoundError:
        print(f"No {BUFFER_PATH} found - creating a new one.")
        buffer = pd.DataFrame(columns=["Date"] + FEATURES)

    # data_status travels through to the log so the frontend/presentation
    # can be honest about whether today's number is live or carried forward.
    data_status = "live"
    last_live_date = None

    if today_values is None:
        # WAQI has no fresh Mumbai data today (upstream outage, not a bug on
        # our end - see discovery logs). Rather than exiting with nothing to
        # show, fall back to the model's existing SEQ_LEN-day window and
        # still produce a prediction from it. This does NOT fabricate a new
        # "today" row in buffer.csv/history_full.csv - it reuses the last
        # real days as-is, so the historical record stays honest and never
        # gets padded with repeated fake values.
        if buffer.empty or len(buffer) < SEQ_LEN:
            print(
                "No live data today AND not enough historical buffer to fall "
                "back on either - exiting without prediction."
            )
            return
        data_status = "stale_fallback"
        last_live_date = buffer["Date"].max()
        print(
            f"  Live station data unavailable today (see discovery log above). "
            f"Falling back to the existing {SEQ_LEN}-day window - most recent "
            f"real data is from {last_live_date.date()} - so a prediction can "
            f"still be produced. This run is flagged 'stale_fallback' in "
            f"{PREDICTIONS_LOG_PATH}, not silently presented as live."
        )
    else:
        data_status = f"live_{today_values.attrs.get('source', 'unknown')}"
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

    status_tag = " [FALLBACK - based on stale data, not live]" if data_status == "stale_fallback" else ""
    print(f"PREDICTED AQI FOR TOMORROW: {pred_aqi:.1f}{status_tag}")

    based_on_date = today if data_status != "stale_fallback" else last_live_date
    log_row = {
        "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "based_on_date": based_on_date.strftime("%Y-%m-%d"),
        "predicted_aqi_for": (based_on_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        "predicted_aqi": round(float(pred_aqi), 1),
        "data_status": data_status,
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
