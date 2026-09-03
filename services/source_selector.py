"""
Central live-data source selector.

Priority:
    1. CPCB
    2. WAQI
    3. OpenWeatherMap

The selector does NOT modify the LSTM architecture.

Its only job is to provide one validated canonical
observation to the prediction pipeline.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

# ============================================================
# SOURCE IMPORTS
# ============================================================

from services.cpcb import get_mumbai_cpcb_reading
from services.waqi import get_mumbai_waqi_reading
from services.openweather import get_mumbai_openweather_reading

# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_ORDER = [
    "cpcb",
    "waqi",
    "openweathermap",
]


# Minimum number of pollutants required.
MIN_POLLUTANTS = 3


# ============================================================
# VALIDATION
# ============================================================

REQUIRED_POLLUTANTS = [
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "O3",
]


def validate_reading(reading: Dict[str, Any]) -> tuple[bool, str]:
    """
    Validate a canonical source reading.

    Returns:
        (True, "OK")
    or:
        (False, "reason")
    """

    if not isinstance(reading, dict):

        return (False, "reading is not a dictionary")

    # --------------------------------------------------------
    # Source
    # --------------------------------------------------------

    source = reading.get("source")

    if not source:

        return (False, "missing source")

    # --------------------------------------------------------
    # Pollutants
    # --------------------------------------------------------

    available = {}

    for pollutant in REQUIRED_POLLUTANTS:

        value = reading.get(pollutant)

        if value is None:
            continue

        try:

            value = float(value)

        except (TypeError, ValueError):

            return (False, f"{pollutant} is not numeric")

        if value < 0:

            return (False, f"{pollutant} is negative")

        available[pollutant] = value

    if len(available) < MIN_POLLUTANTS:

        return (False, "insufficient pollutant data")

    # --------------------------------------------------------
    # AQI
    # --------------------------------------------------------

    aqi = reading.get("AQI")

    if aqi is None:

        return (False, "AQI is missing")

    try:

        aqi = float(aqi)

    except (TypeError, ValueError):

        return (False, "AQI is not numeric")

    if aqi < 0:

        return (False, "AQI is negative")

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    if not reading.get("Date"):

        return (False, "Date is missing")

    return (True, "OK")


# ============================================================
# SOURCE FETCH
# ============================================================


def fetch_source(source: str) -> Dict[str, Any]:

    if source == "cpcb":

        return get_mumbai_cpcb_reading()

    if source == "waqi":

        return get_mumbai_waqi_reading()

    if source == "openweathermap":

        return get_mumbai_openweather_reading()

    raise ValueError(f"Unknown source: {source}")


# ============================================================
# MAIN SELECTOR
# ============================================================


def get_best_mumbai_reading() -> Dict[str, Any]:
    """
    Select the best available live Mumbai AQI reading.

    Priority:
        CPCB → WAQI → OpenWeatherMap

    Each source is independently fetched and validated.

    If a source fails, the next source is tried.
    """

    print()
    print("=" * 70)
    print("LIVE MUMBAI AIR-QUALITY SOURCE SELECTION")
    print("=" * 70)

    failures = []

    for source in SOURCE_ORDER:

        print()
        print(f"[SOURCE] Trying {source.upper()}...")

        try:

            reading = fetch_source(source)

        except Exception as exc:

            reason = f"request/connector failure: " f"{exc}"

            print(f"  ✗ {source.upper()} failed")

            print(f"    Reason: {reason}")

            failures.append(
                {
                    "source": source,
                    "reason": reason,
                }
            )

            continue

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        valid, reason = validate_reading(reading)

        if not valid:

            print(f"  ✗ {source.upper()} rejected")

            print(f"    Reason: {reason}")

            failures.append(
                {
                    "source": source,
                    "reason": reason,
                }
            )

            continue

        # ----------------------------------------------------
        # Success
        # ----------------------------------------------------

        print(f"  ✓ {source.upper()} accepted")

        print(f"    AQI: {reading.get('AQI')}")

        print(f"    Date: {reading.get('Date')}")

        print(f"    Data type: " f"{reading.get('data_type')}")

        print(f"    Confidence: " f"{reading.get('confidence')}")

        print()
        print(f"SELECTED SOURCE: " f"{source.upper()}")

        return reading

    # --------------------------------------------------------
    # Everything failed
    # --------------------------------------------------------

    print()
    print("✗ ALL LIVE DATA SOURCES FAILED")

    print()

    for failure in failures:

        print(f"  {failure['source']}: " f"{failure['reason']}")

    raise RuntimeError("No usable Mumbai air-quality source " "was available.")


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    try:

        reading = get_best_mumbai_reading()

        print()
        print("=" * 70)
        print("FINAL CANONICAL READING")
        print("=" * 70)

        for key, value in reading.items():

            print(f"{key}: {value}")

    except Exception as exc:

        print()
        print("=" * 70)
        print("SOURCE SELECTION FAILED")
        print("=" * 70)

        print(exc)
