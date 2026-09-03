"""
WAQI live data connector for Mumbai AQI V3.

Architecture:

WAQI
  ↓
discover Mumbai stations
  ↓
fetch station feeds
  ↓
freshness validation
  ↓
raw pollutant extraction
  ↓
quality validation
  ↓
Mumbai aggregation
  ↓
CPCB AQI calculation
  ↓
canonical reading

IMPORTANT:
WAQI values used for LSTM features remain pollutant
concentrations where available.

WAQI's own AQI is NOT blindly copied into our
canonical AQI field because WAQI and CPCB AQI
methodologies can differ.
"""

from __future__ import annotations

import os
import statistics
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
import requests

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

WAQI_TOKEN = os.getenv("WAQI_TOKEN", "").strip()


BASE_URL = "https://api.waqi.info"


REQUEST_TIMEOUT = int(os.getenv("WAQI_TIMEOUT", "10"))


MAX_RETRIES = int(os.getenv("WAQI_RETRIES", "2"))


RETRY_DELAY = float(os.getenv("WAQI_RETRY_DELAY", "1.5"))


# Mumbai bounding box.
#
# We deliberately use a geographic area rather than
# relying only on station names.
#
# north, west, south, east
#
MUMBAI_BOUNDS = "19.35,72.75,18.85,73.20"


# We don't want a single old/broken station
# to dominate the city reading.
MIN_STATIONS = 1


# ============================================================
# HTTP
# ============================================================


def request_json(
    url: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Execute a WAQI request with limited retry.
    """

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = requests.get(
                url,
                params=params,
                timeout=REQUEST_TIMEOUT,
                headers={"User-Agent": "Mumbai-AQI-Prediction/3.0"},
            )

            response.raise_for_status()

            payload = response.json()

            if not isinstance(payload, dict):
                raise ValueError("WAQI returned invalid JSON structure.")

            if payload.get("status") != "ok":

                reason = payload.get("data", "Unknown WAQI error")

                raise RuntimeError(f"WAQI API error: {reason}")

            return payload

        except Exception as exc:

            last_error = exc

            print(f"  WAQI: attempt " f"{attempt}/{MAX_RETRIES} failed: " f"{exc}")

            if attempt < MAX_RETRIES:

                time.sleep(RETRY_DELAY)

    raise RuntimeError(
        f"WAQI request failed after " f"{MAX_RETRIES} attempts: " f"{last_error}"
    )


# ============================================================
# DISCOVER STATIONS BY GEOGRAPHIC AREA
# ============================================================


def discover_stations_by_bounds() -> List[Dict[str, Any]]:
    """
    Discover WAQI stations inside the Mumbai bounding box.

    This avoids depending entirely on the station name.
    """

    url = f"{BASE_URL}/v2/map/bounds/"

    payload = request_json(
        url,
        {
            "latlng": MUMBAI_BOUNDS,
            "token": WAQI_TOKEN,
        },
    )

    stations = payload.get("data", [])

    if not isinstance(stations, list):
        return []

    return stations


# ============================================================
# DISCOVER BY SEARCH
# ============================================================


def search_stations(keyword: str) -> List[Dict[str, Any]]:
    """
    Search WAQI stations by keyword.

    Used as a secondary discovery mechanism.
    """

    url = f"{BASE_URL}/v2/search/"

    payload = request_json(
        url,
        {
            "keyword": keyword,
            "token": WAQI_TOKEN,
        },
    )

    stations = payload.get("data", [])

    if not isinstance(stations, list):
        return []

    return stations


# ============================================================
# COMBINE DISCOVERY
# ============================================================


def discover_mumbai_stations() -> List[Dict[str, Any]]:
    """
    Discover Mumbai stations using multiple methods.

    Priority:
        1. Geographic discovery
        2. Mumbai search
        3. Bombay search

    Duplicate UIDs are removed.
    """

    found = {}

    # --------------------------------------------------------
    # Geographic discovery
    # --------------------------------------------------------

    try:

        stations = discover_stations_by_bounds()

        print(f"  WAQI: geographic discovery " f"found {len(stations)} station(s).")

        for station in stations:

            uid = station.get("uid")

            if uid is not None:
                found[str(uid)] = station

    except Exception as exc:

        print(f"  WAQI: geographic discovery failed: " f"{exc}")

    # --------------------------------------------------------
    # Name search
    # --------------------------------------------------------

    for keyword in [
        "Mumbai",
        "Bombay",
    ]:

        try:

            stations = search_stations(keyword)

            print(f"  WAQI: search '{keyword}' " f"found {len(stations)} station(s).")

            for station in stations:

                uid = station.get("uid")

                if uid is not None:
                    found[str(uid)] = station

        except Exception as exc:

            print(f"  WAQI: search '{keyword}' " f"failed: {exc}")

    result = list(found.values())

    print(f"  WAQI: {len(result)} " f"distinct station(s) discovered.")

    return result


# ============================================================
# FETCH INDIVIDUAL STATION
# ============================================================


def fetch_station(uid: Any) -> Dict[str, Any]:
    """
    Fetch the detailed station feed.

    WAQI documents station feeds using:
        /feed/@UID/
    """

    url = f"{BASE_URL}/feed/@{uid}/"

    payload = request_json(
        url,
        {
            "token": WAQI_TOKEN,
        },
    )

    data = payload.get("data")

    if not isinstance(data, dict):
        raise RuntimeError(f"WAQI station {uid} " f"returned invalid data.")

    return data


# ============================================================
# TIMESTAMP
# ============================================================


def get_station_timestamp(data: Dict[str, Any]) -> Optional[str]:
    """
    Extract WAQI station observation time.
    """

    time_data = data.get("time", {})

    if not isinstance(time_data, dict):
        return None

    timestamp = time_data.get("iso") or time_data.get("s")

    if timestamp is None:
        return None

    parsed = parse_timestamp(timestamp)

    if parsed is None:
        return None

    return parsed.isoformat()


# ============================================================
# FRESHNESS
# ============================================================


def is_fresh(timestamp: Optional[str]) -> bool:
    """
    Reject old WAQI station observations.
    """

    if timestamp is None:
        return False

    parsed = parse_timestamp(timestamp)

    if parsed is None:
        return False

    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)

    age_hours = (now - parsed.to_pydatetime()).total_seconds() / 3600

    return 0 <= age_hours <= MAX_STALE_HOURS


# ============================================================
# POLLUTANT EXTRACTION
# ============================================================


def extract_pollutants(data: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """
    Extract raw pollutant concentrations from WAQI.

    WAQI's `iaqi` object contains pollutant measurements.

    NOTE:
    The API may not provide all pollutants at every station.
    """

    iaqi = data.get("iaqi", {})

    if not isinstance(iaqi, dict):
        iaqi = {}

    def value(key: str) -> Optional[float]:

        item = iaqi.get(key)

        if isinstance(item, dict):
            return to_float(item.get("v"))

        return to_float(item)

    return {
        "PM2.5": value("pm25"),
        "PM10": value("pm10"),
        "NO2": value("no2"),
        "SO2": value("so2"),
        "O3": value("o3"),
    }


# ============================================================
# VALIDATE STATION
# ============================================================


def build_valid_station(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Convert WAQI station data into a validated station record.

    This function intentionally prints the exact reason a
    station is rejected so that the live-data pipeline is
    debuggable.
    """

    uid = data.get("idx")

    timestamp = get_station_timestamp(data)

    print()
    print(f"  WAQI station {uid}:")

    # --------------------------------------------------------
    # Timestamp validation
    # --------------------------------------------------------

    if timestamp is None:

        print("    timestamp: MISSING")

        print("    status: REJECTED " "(no observation timestamp)")

        return None

    print(f"    timestamp: {timestamp}")

    parsed = parse_timestamp(timestamp)

    if parsed is None:

        print("    status: REJECTED " "(invalid timestamp)")

        return None

    from datetime import datetime, timezone

    parsed_dt = parsed.to_pydatetime()

    if parsed_dt.tzinfo is None:

        parsed_dt = parsed_dt.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)

    age_hours = (now - parsed_dt).total_seconds() / 3600

    print(f"    age: {age_hours:.1f} hours")

    if age_hours < 0:

        print("    status: REJECTED " "(future timestamp)")

        return None

    if age_hours > MAX_STALE_HOURS:

        print(f"    status: REJECTED " f"(STALE; limit={MAX_STALE_HOURS}h)")

        return None

    # --------------------------------------------------------
    # Pollutant extraction
    # --------------------------------------------------------

    pollutants = extract_pollutants(data)

    print("    pollutants:")

    for pollutant, value in pollutants.items():

        print(f"      {pollutant}: {value}")

    valid_pollutants = {
        key: value for key, value in pollutants.items() if value is not None
    }

    print(f"    usable pollutant count: " f"{len(valid_pollutants)}/5")

    # --------------------------------------------------------
    # Minimum pollutant requirement
    # --------------------------------------------------------

    if len(valid_pollutants) < 3:

        print("    status: REJECTED " "(INSUFFICIENT POLLUTANTS)")

        return None

    # --------------------------------------------------------
    # Station name
    # --------------------------------------------------------

    city = data.get("city", {})

    if isinstance(city, dict):

        station_name = city.get("name")

    else:

        station_name = str(city)

    # --------------------------------------------------------
    # Accept station
    # --------------------------------------------------------

    print("    status: ACCEPTED")

    return {
        "uid": uid,
        "station": station_name,
        "timestamp": timestamp,
        "pollutants": pollutants,
        "waqi_aqi": to_float(data.get("aqi")),
    }
    """
    Convert WAQI station data into a validated station record.
    """

    timestamp = get_station_timestamp(data)

    if not is_fresh(timestamp):

        return None

    pollutants = extract_pollutants(data)

    valid_pollutants = [value for value in pollutants.values() if value is not None]

    # Need at least three pollutants.
    if len(valid_pollutants) < 3:

        return None

    city = data.get("city", {})

    if isinstance(city, dict):
        station_name = city.get("name")
    else:
        station_name = str(city)

    return {
        "uid": data.get("idx"),
        "station": station_name,
        "timestamp": timestamp,
        "pollutants": pollutants,
        # WAQI AQI retained only as metadata.
        #
        # We DON'T use it as our canonical AQI.
        "waqi_aqi": to_float(data.get("aqi")),
    }


# ============================================================
# AGGREGATE STATIONS
# ============================================================


def aggregate_stations(stations: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """
    Aggregate Mumbai station pollutant concentrations.

    Median is used to reduce the impact of one abnormal
    station.
    """

    pollutant_values = {
        "PM2.5": [],
        "PM10": [],
        "NO2": [],
        "SO2": [],
        "O3": [],
    }

    for station in stations:

        pollutants = station["pollutants"]

        for pollutant in pollutant_values:

            value = pollutants.get(pollutant)

            if value is not None:

                pollutant_values[pollutant].append(value)

    result = {}

    for pollutant, values in pollutant_values.items():

        if not values:

            result[pollutant] = None

        else:

            result[pollutant] = round(statistics.median(values), 3)

    return result


# ============================================================
# MAIN PUBLIC FUNCTION
# ============================================================


def get_mumbai_waqi_reading() -> Dict[str, Any]:
    """
    Fetch and aggregate fresh Mumbai WAQI observations.

    Raises RuntimeError when no usable current station
    data is available.
    """

    if not WAQI_TOKEN:

        raise RuntimeError("WAQI_TOKEN is not configured.")

    print("  WAQI: discovering Mumbai stations...")

    discovered = discover_mumbai_stations()

    if not discovered:

        raise RuntimeError("WAQI discovered no Mumbai stations.")

    valid_stations = []

    stale_count = 0
    invalid_count = 0

    for station in discovered:

        uid = station.get("uid")

        if uid is None:
            continue

        try:

            data = fetch_station(uid)

            validated = build_valid_station(data)

            if validated is None:

                stale_count += 1

                print(
                    f"  WAQI: station {uid} "
                    f"not usable (stale or insufficient data)."
                )

                continue

            valid_stations.append(validated)

            print(
                f"  WAQI: station " f"{validated['station']} " f"(uid {uid}) is fresh."
            )

        except Exception as exc:

            invalid_count += 1

            print(f"  WAQI: station {uid} " f"fetch failed: {exc}")

    print(f"  WAQI: {len(valid_stations)} " f"fresh usable station(s).")

    print(f"  WAQI: {stale_count} stale/invalid " f"station(s).")

    print(f"  WAQI: {invalid_count} station " f"request failure(s).")

    if len(valid_stations) < MIN_STATIONS:

        raise RuntimeError("WAQI has no fresh usable Mumbai stations.")

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    aggregated = aggregate_stations(valid_stations)

    print("  WAQI: aggregated raw pollutants:")

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

        raise RuntimeError(
            "WAQI data does not contain enough " "pollutants to calculate CPCB AQI."
        )

    print(f"  WAQI: CPCB-compatible AQI = {aqi}")

    print(f"  WAQI: CPCB sub-indices = " f"{sub_indices}")

    # --------------------------------------------------------
    # Return canonical data
    # --------------------------------------------------------

    return make_canonical_reading(
        aggregated,
        aqi=aqi,
        source="waqi",
        data_type="ground_station",
        timestamp=None,
        station_count=len(valid_stations),
        confidence=0.90,
    )


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("WAQI CONNECTOR TEST")
    print("=" * 60)

    try:

        result = get_mumbai_waqi_reading()

        print()
        print("SUCCESS")
        print("-" * 60)

        for key, value in result.items():

            print(f"{key}: {value}")

    except Exception as exc:

        print()
        print("WAQI TEST FAILED")
        print("-" * 60)

        print(exc)
