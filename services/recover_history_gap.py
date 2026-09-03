"""
Recover missing dates between the existing live history and today.

SOURCE PRIORITY
---------------
1. Existing live_history.csv
2. all_stations_combined.csv
3. OpenWeather historical pollution API

The script NEVER fabricates a long historical gap.

For each missing day:
    station aggregate is preferred
    OpenWeather historical is used only when station data is unavailable

AQI:
    CPCB AQI is calculated from pollutant concentrations.
    OpenWeather's own AQI field is NOT used.

Output:
    data/live_history.csv

Backup:
    data/live_history_backup_before_recovery.csv
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from dotenv import load_dotenv

from aqi.cpcb_aqi import calculate_cpcb_aqi

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

HISTORY_FILE = DATA_DIR / "live_history.csv"

STATION_FILE = DATA_DIR / "all_stations_combined.csv"

BACKUP_FILE = DATA_DIR / "live_history_backup_before_recovery.csv"


# ============================================================
# OPENWEATHER CONFIGURATION
# ============================================================

load_dotenv(PROJECT_ROOT / ".env")

OWM_API_KEY = os.getenv("OWM_API_KEY", "").strip()

OWM_URL = "https://api.openweathermap.org/data/2.5/air_pollution/history"

OWM_LAT = float(os.getenv("OWM_LAT", "19.0760"))
OWM_LON = float(os.getenv("OWM_LON", "72.8777"))

OWM_TIMEOUT = int(os.getenv("OWM_TIMEOUT", "60"))
OWM_RETRIES = int(os.getenv("OWM_RETRIES", "3"))
OWM_RETRY_DELAY = float(os.getenv("OWM_RETRY_DELAY", "3"))

# ============================================================
# RECOVERY CONFIGURATION
# ============================================================

# Recover historical data only up to the day before
# the current CPCB/live data begins.
RECOVERY_END_DATE = pd.Timestamp("2026-08-14")


# ============================================================
# PROJECT FEATURES
# ============================================================

FEATURES = [
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "O3",
    "AQI",
]


POLLUTANTS = [
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "O3",
]


HISTORY_COLUMNS = [
    "Date",
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "O3",
    "AQI",
    "source",
    "data_type",
    "timestamp",
    "station_count",
    "freshness_hours",
    "confidence",
]


# ============================================================
# HELPERS
# ============================================================


def calculate_aqi(
    pm25: float,
    pm10: float,
    no2: float,
    so2: float,
    o3: float,
) -> float:

    pollutants = {
        "PM2.5": pm25,
        "PM10": pm10,
        "NO2": no2,
        "SO2": so2,
        "O3": o3,
    }

    result = calculate_cpcb_aqi(pollutants)

    # calculate_cpcb_aqi() returns:
    # (overall_aqi, pollutant_sub_indices)

    if isinstance(result, tuple):
        return float(result[0])

    if isinstance(result, dict):
        for key in ("AQI", "aqi", "overall_aqi", "value"):
            if key in result:
                return float(result[key])

    return float(result)


def normalize_day(value) -> pd.Timestamp:

    value = pd.to_datetime(value, errors="coerce")

    if pd.isna(value):
        raise ValueError(f"Invalid date: {value}")

    return value.normalize()


def empty_history() -> pd.DataFrame:

    return pd.DataFrame(columns=HISTORY_COLUMNS)


# ============================================================
# LOAD EXISTING HISTORY
# ============================================================


def load_existing_history() -> pd.DataFrame:

    if not HISTORY_FILE.exists():

        print("No existing live_history.csv found.")

        return empty_history()

    df = pd.read_csv(
        HISTORY_FILE,
        parse_dates=["Date"],
    )

    for column in HISTORY_COLUMNS:

        if column not in df.columns:

            df[column] = None

    df = df[HISTORY_COLUMNS].copy()

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
    )

    df = df[df["Date"].notna()].copy()

    df["Date"] = df["Date"].dt.normalize()

    df = (
        df.sort_values("Date")
        .drop_duplicates(
            subset=["Date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return df


# ============================================================
# LOAD STATION DATA
# ============================================================


def load_station_data() -> pd.DataFrame:

    if not STATION_FILE.exists():

        raise FileNotFoundError(f"Station dataset not found:\n{STATION_FILE}")

    df = pd.read_csv(
        STATION_FILE,
        parse_dates=["date"],
    )

    required = [
        "date",
        "pm25",
        "pm10",
        "no2",
        "so2",
        "o3",
        "aqi",
    ]

    missing = [column for column in required if column not in df.columns]

    if missing:

        raise ValueError(f"Station dataset missing columns: {missing}")

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    ).dt.normalize()

    return df


# ============================================================
# PREPARE STATION DAILY DATA
# ============================================================


def prepare_station_daily(
    df: pd.DataFrame,
) -> pd.DataFrame:

    numeric_columns = [
        "pm25",
        "pm10",
        "no2",
        "so2",
        "o3",
        "aqi",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    daily = (
        df.groupby("date")
        .agg(
            pm25=("pm25", "mean"),
            pm10=("pm10", "mean"),
            no2=("no2", "mean"),
            so2=("so2", "mean"),
            o3=("o3", "mean"),
            aqi=("aqi", "mean"),
            station_count=("station_id", "nunique"),
        )
        .reset_index()
    )

    daily["usable"] = (
        daily[
            [
                "pm25",
                "pm10",
                "no2",
                "so2",
                "o3",
                "aqi",
            ]
        ]
        .notna()
        .all(axis=1)
    )

    return daily


# ============================================================
# OPENWEATHER FETCH
# ============================================================


def fetch_openweather_day(
    day: pd.Timestamp,
) -> pd.DataFrame:

    if not OWM_API_KEY:

        raise RuntimeError("OWM_API_KEY is missing from .env")

    # Query one UTC day.
    start_dt = datetime(
        day.year,
        day.month,
        day.day,
        tzinfo=timezone.utc,
    )

    end_dt = start_dt + timedelta(days=1)

    start = int(start_dt.timestamp())
    end = int(end_dt.timestamp())

    params = {
        "lat": OWM_LAT,
        "lon": OWM_LON,
        "start": start,
        "end": end,
        "appid": OWM_API_KEY,
    }

    for attempt in range(
        1,
        OWM_RETRIES + 1,
    ):

        try:

            print(f"    OpenWeather request " f"{attempt}/{OWM_RETRIES}")

            response = requests.get(
                OWM_URL,
                params=params,
                timeout=OWM_TIMEOUT,
            )

            response.raise_for_status()

            payload = response.json()

            observations = payload.get(
                "list",
                [],
            )

            if not observations:

                raise RuntimeError(
                    f"OpenWeather returned no observations " f"for {day.date()}"
                )

            rows = []

            for item in observations:

                components = item.get(
                    "components",
                    {},
                )

                rows.append(
                    {
                        "pm25": components.get("pm2_5"),
                        "pm10": components.get("pm10"),
                        "no2": components.get("no2"),
                        "so2": components.get("so2"),
                        "o3": components.get("o3"),
                        "dt": item.get("dt"),
                    }
                )

            result = pd.DataFrame(rows)

            for column in ["pm25", "pm10", "no2", "so2", "o3"]:

                result[column] = pd.to_numeric(
                    result[column],
                    errors="coerce",
                )

            result = result.dropna(
                subset=[
                    "pm25",
                    "pm10",
                    "no2",
                    "so2",
                    "o3",
                ]
            )

            if result.empty:

                raise RuntimeError(
                    f"OpenWeather observations for "
                    f"{day.date()} contain no complete "
                    f"pollutant rows."
                )

            return result

        except Exception as exc:

            print(f"    OpenWeather attempt failed: {exc}")

            if attempt < OWM_RETRIES:

                time.sleep(OWM_RETRY_DELAY)

    raise RuntimeError(f"Could not recover OpenWeather data " f"for {day.date()}")


# ============================================================
# BUILD OPENWEATHER DAILY ROW
# ============================================================


def build_openweather_row(
    day: pd.Timestamp,
    hourly: pd.DataFrame,
) -> dict:

    pm25 = float(hourly["pm25"].mean())
    pm10 = float(hourly["pm10"].mean())
    no2 = float(hourly["no2"].mean())
    so2 = float(hourly["so2"].mean())
    o3 = float(hourly["o3"].mean())

    aqi = calculate_aqi(
        pm25,
        pm10,
        no2,
        so2,
        o3,
    )

    return {
        "Date": day,
        "PM2.5": pm25,
        "PM10": pm10,
        "NO2": no2,
        "SO2": so2,
        "O3": o3,
        "AQI": aqi,
        "source": "openweathermap",
        "data_type": "historical_modeled",
        "timestamp": (f"{day.strftime('%Y-%m-%d')}" "T00:00:00+00:00"),
        "station_count": 0,
        "freshness_hours": None,
        "confidence": 0.60,
    }


# ============================================================
# BUILD STATION ROW
# ============================================================


def build_station_row(
    row: pd.Series,
) -> dict:

    day = normalize_day(row["date"])

    return {
        "Date": day,
        "PM2.5": float(row["pm25"]),
        "PM10": float(row["pm10"]),
        "NO2": float(row["no2"]),
        "SO2": float(row["so2"]),
        "O3": float(row["o3"]),
        "AQI": float(row["aqi"]),
        "source": "historical_station_aggregate",
        "data_type": "historical",
        "timestamp": (f"{day.strftime('%Y-%m-%d')}" "T00:00:00+05:30"),
        "station_count": int(row["station_count"]),
        "freshness_hours": None,
        "confidence": 0.90,
    }


# ============================================================
# VALIDATE ROW
# ============================================================


def validate_history_row(
    row: dict,
) -> bool:

    for column in FEATURES:

        value = pd.to_numeric(
            row.get(column),
            errors="coerce",
        )

        if pd.isna(value):

            return False

        if float(value) < 0:

            return False

    return True


# ============================================================
# BACKUP
# ============================================================


def create_backup() -> None:

    if not HISTORY_FILE.exists():

        print("No existing history to backup.")

        return

    shutil.copy2(
        HISTORY_FILE,
        BACKUP_FILE,
    )

    print()
    print("Backup created:")
    print(f"  {BACKUP_FILE}")


# ============================================================
# CONTINUITY VALIDATION
# ============================================================


def validate_continuity(
    history: pd.DataFrame,
) -> None:

    dates = pd.to_datetime(history["Date"]).sort_values().drop_duplicates()

    gaps = dates.diff().dropna()

    bad = gaps[gaps != pd.Timedelta(days=1)]

    print()
    print("=" * 70)
    print("FINAL CONTINUITY CHECK")
    print("=" * 70)

    print(f"Rows: {len(dates)}")

    print(f"Start: {dates.min().date()}")

    print(f"End: {dates.max().date()}")

    if bad.empty:

        print("✓ COMPLETE DAILY CONTINUITY")

    else:

        print(f"✗ {len(bad)} date gap(s) detected.")

        for index in bad.index[:20]:

            previous = dates.loc[index - 1]
            current = dates.loc[index]

            print(
                f"  {previous.date()} -> "
                f"{current.date()} "
                f"({(current - previous).days} days)"
            )

        raise RuntimeError(
            "Recovery produced a non-contiguous " "history. File was NOT saved."
        )


# ============================================================
# MAIN RECOVERY
# ============================================================


def main():

    print()
    print("=" * 70)
    print("MUMBAI AQI HISTORICAL GAP RECOVERY")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    history = load_existing_history()

    station_raw = load_station_data()

    station_daily = prepare_station_daily(station_raw)

    print()
    print(f"Existing history: " f"{len(history)} rows")

    print(
        f"Existing history range: "
        f"{history['Date'].min().date()} "
        f"-> "
        f"{history['Date'].max().date()}"
    )

    # --------------------------------------------------------
    # IMPORTANT
    # --------------------------------------------------------

    latest_existing = history["Date"].max()

    target_end = RECOVERY_END_DATE

    print()
    print(
        f"Recovery target: " f"{latest_existing.date()} " f"-> " f"{target_end.date()}"
    )

    # --------------------------------------------------------
    # BACKUP
    # --------------------------------------------------------

    create_backup()

    # --------------------------------------------------------
    # RECOVER EACH MISSING DATE
    # --------------------------------------------------------

    recovered_rows = []

    missing_dates = pd.date_range(
        latest_existing + pd.Timedelta(days=1),
        target_end - pd.Timedelta(days=1),
        freq="D",
    )

    print()
    print(f"Missing dates to recover: " f"{len(missing_dates)}")

    for day in missing_dates:

        day = normalize_day(day)

        print()
        print(f"[{day.date()}]")

        # ----------------------------------------------------
        # SOURCE 1 — STATION DATA
        # ----------------------------------------------------

        station_match = station_daily[
            (station_daily["date"] == day) & (station_daily["usable"] == True)
        ]

        if not station_match.empty:

            row = build_station_row(station_match.iloc[0])

            if validate_history_row(row):

                recovered_rows.append(row)

                print("  ✓ Source: " "historical station aggregate")

                continue

        # ----------------------------------------------------
        # SOURCE 2 — OPENWEATHER
        # ----------------------------------------------------

        print("  Station data unavailable.")

        print("  Trying OpenWeather historical...")

        hourly = fetch_openweather_day(day)

        row = build_openweather_row(
            day,
            hourly,
        )

        if not validate_history_row(row):

            raise RuntimeError(f"Invalid recovered row " f"for {day.date()}")

        recovered_rows.append(row)

        print(f"  ✓ OpenWeather recovered " f"{len(hourly)} hourly observations")

        print(
            f"    PM2.5={row['PM2.5']:.2f}, "
            f"PM10={row['PM10']:.2f}, "
            f"NO2={row['NO2']:.2f}, "
            f"SO2={row['SO2']:.2f}, "
            f"O3={row['O3']:.2f}, "
            f"AQI={row['AQI']:.2f}"
        )

    # --------------------------------------------------------
    # MERGE
    # --------------------------------------------------------

    recovered = pd.DataFrame(
        recovered_rows,
        columns=HISTORY_COLUMNS,
    )

    combined = pd.concat(
        [
            history,
            recovered,
        ],
        ignore_index=True,
    )

    combined["Date"] = pd.to_datetime(
        combined["Date"],
        errors="coerce",
    ).dt.normalize()

    combined = (
        combined.sort_values("Date")
        .drop_duplicates(
            subset=["Date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    validate_continuity(combined)

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    combined.to_csv(
        HISTORY_FILE,
        index=False,
    )

    print()
    print("=" * 70)
    print("RECOVERY COMPLETE")
    print("=" * 70)

    print(f"Saved: {HISTORY_FILE}")

    print(f"Total rows: {len(combined)}")

    print(
        f"Date range: "
        f"{combined['Date'].min().date()} "
        f"-> "
        f"{combined['Date'].max().date()}"
    )

    print()
    print("Source distribution:")

    print(combined["source"].value_counts().to_string())

    print()
    print("Last 20 rows:")

    print(combined.tail(20).to_string(index=False))


if __name__ == "__main__":

    main()
