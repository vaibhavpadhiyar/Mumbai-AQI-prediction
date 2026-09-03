"""
CPCB live data connector for Mumbai AQI V3.

Responsibilities
----------------
1. Fetch live CPCB data from data.gov.in
2. Parse station-level pollutant records
3. Keep only Mumbai stations
4. Reject stale observations
5. Convert station records into raw pollutant concentrations
6. Aggregate Mumbai station values
7. Calculate CPCB AQI using aqi/cpcb_aqi.py
8. Return a canonical reading

IMPORTANT
---------
This module NEVER converts pollutant concentrations into AQI
and then stores those AQI values as pollutant concentrations.

The canonical structure remains:

PM2.5 -> raw concentration
PM10  -> raw concentration
NO2   -> raw concentration
SO2   -> raw concentration
O3    -> raw concentration
AQI   -> calculated AQI
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

# Load project-root .env
load_dotenv()

from aqi.cpcb_aqi import calculate_cpcb_aqi
from services.common import (
    MAX_STALE_HOURS,
    make_canonical_reading,
    parse_timestamp,
    to_float,
)

# ============================================================
# CONFIGURATION
# ============================================================

# ============================================================
# CPCB / DATA.GOV.IN CONFIGURATION
# ============================================================

# Official data.gov.in resource:
# "Real time Air Quality Index from various locations"
#
# Resource ID:
# 3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69
#
# The URL is intentionally kept inside the connector.
# The user only needs to provide CPCB_API_KEY in .env.

CPCB_RESOURCE_ID = "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"

CPCB_API_URL = f"https://api.data.gov.in/resource/{CPCB_RESOURCE_ID}"

CPCB_API_KEY = os.getenv("CPCB_API_KEY", "").strip()
# Request settings.
CPCB_TIMEOUT = int(os.getenv("CPCB_TIMEOUT", "15"))

CPCB_RETRIES = int(os.getenv("CPCB_RETRIES", "2"))

CPCB_RETRY_DELAY = float(os.getenv("CPCB_RETRY_DELAY", "2"))


# ============================================================
# MUMBAI IDENTIFICATION
# ============================================================

MUMBAI_NAMES = {
    "mumbai",
    "bombay",
}

MUMBAI_STATION_KEYWORDS = [
    "mumbai",
    "bombay",
    "bandra",
    "kurla",
    "sion",
    "worli",
    "colaba",
    "powai",
    "deonar",
    "borivali",
    "andheri",
    "chakala",
    "vile parle",
]


# ============================================================
# POLLUTANT NAME NORMALIZATION
# ============================================================

POLLUTANT_ALIASES = {
    "pm2.5": "PM2.5",
    "pm25": "PM2.5",
    "pm_2_5": "PM2.5",
    "pm2_5": "PM2.5",
    "pm10": "PM10",
    "pm_10": "PM10",
    "no2": "NO2",
    "no_2": "NO2",
    "so2": "SO2",
    "so_2": "SO2",
    "o3": "O3",
    "ozone": "O3",
}


# ============================================================
# BASIC HELPERS
# ============================================================


def normalize_text(value: Any) -> str:
    """
    Normalize arbitrary text safely.
    """

    if value is None:
        return ""

    return str(value).strip().lower()


def normalize_pollutant_name(value: Any) -> Optional[str]:
    """
    Convert CPCB pollutant identifiers into our canonical names.
    """

    if value is None:
        return None

    text = normalize_text(value)

    return POLLUTANT_ALIASES.get(text)


def is_mumbai_station(record: Dict[str, Any]) -> bool:
    """
    Determine whether a CPCB record belongs to Mumbai.

    We check city/state/station fields because CPCB responses can
    vary slightly in naming.
    """

    city = normalize_text(record.get("city"))

    state = normalize_text(record.get("state"))

    station = normalize_text(record.get("station"))

    location = normalize_text(record.get("location"))

    combined = " ".join(
        [
            city,
            station,
            location,
        ]
    )

    # Explicit Mumbai/Bombay city match.
    if city in MUMBAI_NAMES:
        return True

    # Station-level Mumbai keyword matching.
    for keyword in MUMBAI_STATION_KEYWORDS:
        if keyword in combined:
            return True

    return False


# ============================================================
# TIMESTAMP EXTRACTION
# ============================================================


def extract_timestamp(record: Dict[str, Any]) -> Optional[str]:
    """
    Extract the CPCB observation timestamp.

    CPCB commonly provides `last_update`.
    We also check a few alternate field names.
    """

    possible_fields = [
        "last_update",
        "last_updated",
        "timestamp",
        "date",
        "updated_at",
    ]

    for field in possible_fields:

        value = record.get(field)

        if value:
            parsed = parse_timestamp(value)

            if parsed is not None:
                return parsed.isoformat()

    return None


# ============================================================
# FRESHNESS
# ============================================================


def is_fresh_record(record: Dict[str, Any]) -> bool:
    """
    Check whether a CPCB record is recent enough.
    """

    timestamp = extract_timestamp(record)

    if timestamp is None:
        return False

    parsed = parse_timestamp(timestamp)

    if parsed is None:
        return False

    now = parse_timestamp(
        __import__("datetime").datetime.now().astimezone().isoformat()
    )

    if now is None:
        return False

    age_hours = (now - parsed).total_seconds() / 3600

    return 0 <= age_hours <= MAX_STALE_HOURS


# ============================================================
# HTTP FETCH
# ============================================================


def fetch_cpcb_records() -> List[Dict[str, Any]]:
    """
    Fetch records from CPCB/data.gov.in.

    Returns:
        list of raw API records

    Raises:
        RuntimeError when all attempts fail.
    """

    if not CPCB_API_KEY:
        raise RuntimeError("CPCB_API_KEY is not configured.")

    params = {
        "api-key": CPCB_API_KEY,
        "format": "json",
        "offset": 0,
        "limit": 1000,
        "filters[state]": "Maharashtra",
        "filters[city]": "Mumbai",
    }

    last_error = None

    for attempt in range(1, CPCB_RETRIES + 1):

        try:

            print(f"  CPCB: request attempt " f"{attempt}/{CPCB_RETRIES}...")

            response = requests.get(
                CPCB_API_URL,
                params=params,
                timeout=CPCB_TIMEOUT,
                headers={"User-Agent": "Mumbai-AQI-Prediction/3.0"},
            )
            print("\n========== CPCB RAW RESPONSE DEBUG ==========")

            try:
                debug_data = response.json()

                print("HTTP status:", response.status_code)
                print("Top-level keys:", list(debug_data.keys()))

                records = debug_data.get("records", [])

                print("Record count:", len(records))
                if records:
                    print("\nFIRST RAW RECORD:")
                    print(records[0])

            except Exception as debug_exc:
                print("Could not inspect CPCB response:", debug_exc)

            print("=============================================\n")
            response.raise_for_status()

            payload = response.json()

            records = payload.get("records", [])

            if not isinstance(records, list):
                raise ValueError("CPCB response contains invalid " "'records' field.")

            print(f"  CPCB: received " f"{len(records)} raw record(s).")

            return records

        except Exception as exc:

            last_error = exc

            print(f"  CPCB: attempt {attempt} failed: " f"{exc}")

            if attempt < CPCB_RETRIES:
                time.sleep(CPCB_RETRY_DELAY)

    raise RuntimeError(
        f"CPCB failed after " f"{CPCB_RETRIES} attempts: " f"{last_error}"
    )


# ============================================================
# GROUP CPCB RECORDS
# ============================================================


def extract_mumbai_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Keep only fresh Mumbai station records.
    """

    mumbai_records = []

    city_count = 0
    stale_count = 0

    for record in records:

        if not isinstance(record, dict):
            continue

        if not is_mumbai_station(record):
            continue

        city_count += 1

        if not is_fresh_record(record):
            stale_count += 1
            continue

        mumbai_records.append(record)

    print(f"  CPCB: {city_count} Mumbai-related " f"record(s) found.")

    print(f"  CPCB: {stale_count} stale/invalid " f"record(s) discarded.")

    print(f"  CPCB: {len(mumbai_records)} " f"fresh record(s) remaining.")

    return mumbai_records


# ============================================================
# CONVERT CPCB RECORDS INTO STATION DATA
# ============================================================


def build_station_readings(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Convert CPCB pollutant records into station-level
    pollutant dictionaries.

    Example output:

    {
        "Bandra": {
            "PM2.5": 25.4,
            "PM10": 62.1,
            "NO2": 17.3
        }
    }
    """

    stations = defaultdict(dict)

    for record in records:

        station = (
            record.get("station")
            or record.get("location")
            or record.get("station_name")
        )

        if not station:
            continue

        station = str(station).strip()

        pollutant = normalize_pollutant_name(
            record.get("pollutant_id")
            or record.get("pollutant")
            or record.get("parameter")
        )

        if pollutant is None:
            continue

        # IMPORTANT:
        #
        # pollutant_avg is treated as a raw
        # concentration, NOT an AQI.
        #
        value = to_float(
            record.get("avg_value")
            if record.get("avg_value") is not None
            else record.get("pollutant_avg")
        )

        if value is None:
            continue

        stations[station][pollutant] = value

    return dict(stations)


# ============================================================
# CITY AGGREGATION
# ============================================================


def aggregate_station_readings(
    stations: Dict[str, Dict[str, Any]],
) -> Dict[str, Optional[float]]:
    """
    Aggregate Mumbai station pollutant concentrations.

    We use the median rather than a simple mean.

    Why?

    A single abnormal station can distort a city-wide average.
    Median is more robust against one bad sensor.
    """

    values = defaultdict(list)

    for station_name, reading in stations.items():

        for pollutant, value in reading.items():

            value = to_float(value)

            if value is None:
                continue

            values[pollutant].append(value)

    result = {}

    for pollutant, pollutant_values in values.items():

        if not pollutant_values:
            result[pollutant] = None
            continue

        sorted_values = sorted(pollutant_values)

        n = len(sorted_values)

        middle = n // 2

        if n % 2 == 1:

            median = sorted_values[middle]

        else:

            median = (sorted_values[middle - 1] + sorted_values[middle]) / 2

        result[pollutant] = round(median, 3)

    return result


# ============================================================
# PUBLIC FUNCTION
# ============================================================


def get_mumbai_cpcb_reading() -> Dict[str, Any]:
    """
    Main CPCB connector.

    Returns a canonical reading.

    Raises RuntimeError if CPCB cannot provide a usable
    fresh Mumbai observation.
    """

    print("  CPCB: fetching Mumbai station data...")

    records = fetch_cpcb_records()

    records = extract_mumbai_records(records)

    if not records:

        raise RuntimeError("CPCB returned no fresh Mumbai records.")

    stations = build_station_readings(records)

    if not stations:

        raise RuntimeError(
            "CPCB returned fresh records but " "no usable station pollutant data."
        )

    print(f"  CPCB: {len(stations)} usable " f"station(s).")

    aggregated = aggregate_station_readings(stations)

    print("  CPCB: aggregated raw pollutants:")

    for pollutant in [
        "PM2.5",
        "PM10",
        "NO2",
        "SO2",
        "O3",
    ]:

        print(f"    {pollutant}: " f"{aggregated.get(pollutant)}")

    # --------------------------------------------------------
    # Calculate CPCB AQI
    # --------------------------------------------------------

    aqi, sub_indices = calculate_cpcb_aqi(aggregated)

    if aqi is None:

        raise RuntimeError("CPCB pollutant data is insufficient " "to calculate AQI.")

    print(f"  CPCB: calculated AQI = {aqi}")

    print("  CPCB: sub-indices = " f"{sub_indices}")
    # --------------------------------------------------------
    # Determine latest CPCB observation timestamp
    # --------------------------------------------------------

    timestamps = []

    for record in records:

        timestamp = extract_timestamp(record)

        if timestamp is not None:
            parsed = parse_timestamp(timestamp)

            if parsed is not None:
                timestamps.append(parsed)

    latest_timestamp = None

    if timestamps:
        latest_timestamp = max(timestamps).isoformat()

    print(f"  CPCB: latest observation timestamp = {latest_timestamp}")
    # --------------------------------------------------------
    # Build canonical reading
    # --------------------------------------------------------

    return make_canonical_reading(
        aggregated,
        aqi=aqi,
        source="cpcb",
        data_type="ground_station",
        timestamp=latest_timestamp,
        station_count=len(stations),
        confidence=0.95,
    )


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("CPCB CONNECTOR TEST")
    print("=" * 60)

    try:

        result = get_mumbai_cpcb_reading()

        print()
        print("SUCCESS")
        print("-" * 60)

        for key, value in result.items():
            print(f"{key}: {value}")

    except Exception as exc:

        print()
        print("CPCB TEST FAILED")
        print("-" * 60)
        print(exc)
