"""
OpenWeatherMap air-pollution connector.

Purpose
-------
Use OpenWeatherMap as a fallback source when fresh ground
station observations from CPCB / WAQI are unavailable.

IMPORTANT
---------
OpenWeatherMap provides modeled air-pollution concentrations.

We preserve those raw concentrations as the model features.

We DO NOT replace PM2.5/PM10/etc. with their AQI sub-indices.

Flow:

OWM API
   ↓
raw pollutant concentrations
   ↓
validation
   ↓
CPCB AQI calculation
   ↓
canonical reading
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from dotenv import load_dotenv
import requests

load_dotenv()

from aqi.cpcb_aqi import calculate_cpcb_aqi
from services.common import (
    make_canonical_reading,
    to_float,
)

# ============================================================
# CONFIGURATION
# ============================================================

OWM_API_KEY = os.getenv("OWM_API_KEY", "").strip()

OWM_LAT = float(os.getenv("OWM_LAT", "19.0760"))

OWM_LON = float(os.getenv("OWM_LON", "72.8777"))

OWM_TIMEOUT = int(os.getenv("OWM_TIMEOUT", "10"))

OWM_RETRIES = int(os.getenv("OWM_RETRIES", "2"))

OWM_RETRY_DELAY = float(os.getenv("OWM_RETRY_DELAY", "1.5"))


AIR_POLLUTION_URL = "https://api.openweathermap.org/data/2.5/air_pollution"


# ============================================================
# HTTP REQUEST
# ============================================================


def fetch_air_pollution() -> Dict[str, Any]:
    """
    Fetch current OpenWeatherMap air pollution data.

    OpenWeatherMap's current air pollution endpoint returns
    pollutant concentrations under:

        list[0]["components"]

    """

    if not OWM_API_KEY:

        raise RuntimeError("OWM_API_KEY is not configured.")

    params = {
        "lat": OWM_LAT,
        "lon": OWM_LON,
        "appid": OWM_API_KEY,
    }

    last_error = None

    for attempt in range(1, OWM_RETRIES + 1):

        try:

            print(f"  OpenWeatherMap: request " f"attempt {attempt}/{OWM_RETRIES}...")

            response = requests.get(
                AIR_POLLUTION_URL,
                params=params,
                timeout=OWM_TIMEOUT,
                headers={"User-Agent": "Mumbai-AQI-Prediction/3.0"},
            )

            response.raise_for_status()

            payload = response.json()

            if not isinstance(payload, dict):

                raise ValueError("OpenWeatherMap returned " "invalid JSON.")

            return payload

        except Exception as exc:

            last_error = exc

            print(f"  OpenWeatherMap: attempt " f"{attempt} failed: {exc}")

            if attempt < OWM_RETRIES:

                time.sleep(OWM_RETRY_DELAY)

    raise RuntimeError(
        "OpenWeatherMap request failed after " f"{OWM_RETRIES} attempts: {last_error}"
    )


# ============================================================
# EXTRACT RAW POLLUTANTS
# ============================================================


def extract_raw_pollutants(payload: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """
    Extract raw pollutant concentrations.

    OpenWeatherMap component units are μg/m³.

    IMPORTANT:
    These values are kept exactly as pollutant
    concentrations.

    They are NOT AQI values.
    """

    entries = payload.get("list")

    if not isinstance(entries, list) or not entries:

        raise RuntimeError(
            "OpenWeatherMap response contains " "no air-pollution records."
        )

    first = entries[0]

    if not isinstance(first, dict):

        raise RuntimeError("OpenWeatherMap pollution record " "has invalid structure.")

    components = first.get("components", {})

    if not isinstance(components, dict):

        raise RuntimeError("OpenWeatherMap components field " "is invalid.")

    # --------------------------------------------------------
    # IMPORTANT
    #
    # These are RAW concentrations.
    #
    # Do not convert them here into AQI
    # sub-indices.
    # --------------------------------------------------------

    pollutants = {
        "PM2.5": to_float(components.get("pm2_5")),
        "PM10": to_float(components.get("pm10")),
        "NO2": to_float(components.get("no2")),
        "SO2": to_float(components.get("so2")),
        "O3": to_float(components.get("o3")),
    }

    return pollutants


# ============================================================
# VALIDATE POLLUTANTS
# ============================================================


def validate_pollutants(pollutants: Dict[str, Optional[float]]) -> None:
    """
    Ensure the returned pollutant concentrations are usable.
    """

    for pollutant, value in pollutants.items():

        if value is None:
            continue

        if value < 0:

            raise RuntimeError(
                f"OpenWeatherMap returned negative " f"value for {pollutant}: {value}"
            )

    available = [value for value in pollutants.values() if value is not None]

    if len(available) < 3:

        raise RuntimeError(
            "OpenWeatherMap returned fewer than " "3 usable pollutant concentrations."
        )


# ============================================================
# API TIMESTAMP
# ============================================================


def extract_timestamp(payload: Dict[str, Any]) -> Optional[str]:
    """
    Extract the timestamp supplied by OpenWeatherMap.
    """

    entries = payload.get("list")

    if not isinstance(entries, list) or not entries:

        return None

    first = entries[0]

    if not isinstance(first, dict):

        return None

    dt = first.get("dt")

    if dt is None:
        return None

    try:

        dt = int(dt)

        timestamp = datetime.fromtimestamp(dt, tz=timezone.utc)

        return timestamp.isoformat()

    except Exception:

        return None


# ============================================================
# MAIN PUBLIC FUNCTION
# ============================================================


def get_mumbai_openweather_reading() -> Dict[str, Any]:
    """
    Get the current OpenWeatherMap modeled pollution
    observation for Mumbai.

    Returns the same canonical structure used by CPCB
    and WAQI connectors.
    """

    print("  OpenWeatherMap: fetching modeled " "Mumbai pollution...")

    payload = fetch_air_pollution()

    pollutants = extract_raw_pollutants(payload)

    validate_pollutants(pollutants)

    timestamp = extract_timestamp(payload)

    # --------------------------------------------------------
    # PRINT RAW VALUES
    # --------------------------------------------------------

    print("  OpenWeatherMap: RAW pollutant " "concentrations:")

    for pollutant in [
        "PM2.5",
        "PM10",
        "NO2",
        "SO2",
        "O3",
    ]:

        print(f"    {pollutant}: " f"{pollutants.get(pollutant)}")

    # --------------------------------------------------------
    # CALCULATE OUR AQI
    # --------------------------------------------------------

    aqi, sub_indices = calculate_cpcb_aqi(pollutants)

    if aqi is None:

        raise RuntimeError(
            "OpenWeatherMap pollutants are insufficient " "for CPCB AQI calculation."
        )

    print(f"  OpenWeatherMap: calculated AQI = {aqi}")

    print(f"  OpenWeatherMap: CPCB sub-indices = " f"{sub_indices}")

    # --------------------------------------------------------
    # CANONICAL OUTPUT
    # --------------------------------------------------------

    result = make_canonical_reading(
        pollutants,
        aqi=aqi,
        source="openweathermap",
        data_type="modeled",
        timestamp=timestamp,
        station_count=0,
        confidence=0.60,
    )

    return result


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("OPENWEATHERMAP CONNECTOR TEST")
    print("=" * 60)

    try:

        result = get_mumbai_openweather_reading()

        print()
        print("SUCCESS")
        print("-" * 60)

        for key, value in result.items():

            print(f"{key}: {value}")

    except Exception as exc:

        print()
        print("OPENWEATHERMAP TEST FAILED")
        print("-" * 60)

        print(exc)
