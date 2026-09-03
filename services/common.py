"""
Common utilities for Mumbai AQI V3 pipeline.

Responsibilities:
- canonical feature definitions
- timestamp parsing
- freshness validation
- pollutant validation
- numeric normalization
- common source metadata
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

import math
import pandas as pd

# ============================================================
# CANONICAL MODEL FEATURES
# ============================================================

POLLUTANT_COLS = [
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "O3",
]

FEATURES = [
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "O3",
    "AQI",
]

SEQ_LEN = 14


# ============================================================
# DATA QUALITY SETTINGS
# ============================================================

# We will reject observations older than this.
MAX_STALE_HOURS = 48

# Minimum number of pollutant values required for a useful
# observation.
MIN_POLLUTANTS_REQUIRED = 3


# ============================================================
# SAFE NUMBER CONVERSION
# ============================================================


def to_float(value: Any) -> Optional[float]:
    """
    Convert a value to float safely.

    Returns None if conversion fails or value is invalid.
    """

    if value is None:
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(value):
        return None

    return value


# ============================================================
# TIMESTAMP PARSING
# ============================================================


def parse_timestamp(value):
    """
    Parse timestamps from all supported sources.

    Supported examples:
        14-08-2026 10:00:00
        2026-08-14 10:00:00
        2026-08-14T10:00:00+05:30
        2026-08-14T10:00:00.123+05:30

    Returns:
        timezone-aware pandas Timestamp, or None.
    """

    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    # --------------------------------------------------------
    # 1. ISO-8601 timestamps
    # --------------------------------------------------------
    # Examples:
    # 2026-08-14T10:00:00+05:30
    # 2026-08-14T10:00:00.123+05:30
    #
    # These must NOT use dayfirst=True.
    # --------------------------------------------------------

    if "T" in text:
        try:
            ts = pd.to_datetime(text, errors="coerce", utc=True)

            if not pd.isna(ts):
                return ts

        except Exception:
            pass

    # --------------------------------------------------------
    # 2. CPCB / Indian date format
    # --------------------------------------------------------
    # Example:
    # 14-08-2026 10:00:00
    # --------------------------------------------------------

    try:
        ts = pd.to_datetime(text, format="%d-%m-%Y %H:%M:%S", errors="coerce")

        if not pd.isna(ts):
            return ts.tz_localize("Asia/Kolkata")

    except Exception:
        pass

    # --------------------------------------------------------
    # 3. Standard date/time fallback
    # --------------------------------------------------------

    try:
        ts = pd.to_datetime(text, errors="coerce")

        if pd.isna(ts):
            return None

        if ts.tzinfo is None:
            ts = ts.tz_localize("Asia/Kolkata")

        return ts.tz_convert("UTC")

    except Exception:
        return None


# ============================================================
# FRESHNESS CHECK
# ============================================================


def freshness_hours(timestamp: Any) -> Optional[float]:
    """
    Return age of observation in hours.
    """

    ts = parse_timestamp(timestamp)

    if ts is None:
        return None

    now = pd.Timestamp.now(tz="UTC")

    age = (now - ts).total_seconds() / 3600

    return age


def is_fresh(
    timestamp: Any,
    max_stale_hours: float = MAX_STALE_HOURS,
) -> bool:
    """
    Check whether an observation is recent enough.
    """

    age = freshness_hours(timestamp)

    if age is None:
        return False

    return 0 <= age <= max_stale_hours


# ============================================================
# POLLUTANT VALIDATION
# ============================================================


def validate_pollutants(
    reading: Dict[str, Any],
) -> Dict[str, Optional[float]]:
    """
    Normalize pollutant values.

    IMPORTANT:
    These values must represent RAW pollutant concentrations,
    NOT AQI sub-indices.
    """

    result = {}

    for pollutant in POLLUTANT_COLS:
        value = to_float(reading.get(pollutant))

        # Negative pollutant concentration is invalid.
        if value is not None and value < 0:
            value = None

        result[pollutant] = value

    return result


# ============================================================
# COUNT AVAILABLE POLLUTANTS
# ============================================================


def count_valid_pollutants(
    reading: Dict[str, Any],
) -> int:
    """
    Count how many pollutant concentrations are usable.
    """

    normalized = validate_pollutants(reading)

    return sum(value is not None for value in normalized.values())


# ============================================================
# GENERIC READING VALIDATION
# ============================================================


def validate_reading(
    reading: Dict[str, Any],
    timestamp: Any = None,
) -> tuple[bool, str]:
    """
    Generic validation for a source reading.

    Returns:
        (True, "ok")
    or:
        (False, reason)
    """

    pollutants = validate_pollutants(reading)

    valid_count = sum(value is not None for value in pollutants.values())

    if valid_count < MIN_POLLUTANTS_REQUIRED:
        return (False, f"only {valid_count} usable pollutants")

    if timestamp is not None:
        age = freshness_hours(timestamp)

        if age is None:
            return False, "invalid timestamp"

        if age < 0:
            return False, "future timestamp"

        if age > MAX_STALE_HOURS:
            return (False, f"stale observation ({age:.1f}h old)")

    return True, "ok"


# ============================================================
# CANONICAL READING
# ============================================================


def make_canonical_reading(
    pollutants: Dict[str, Any],
    *,
    aqi: Optional[float],
    source: str,
    data_type: str,
    timestamp: Any = None,
    station_count: int = 1,
    confidence: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Build the standard record used by the rest of the pipeline.

    Canonical meaning:

    PM2.5 = raw concentration
    PM10  = raw concentration
    NO2   = raw concentration
    SO2   = raw concentration
    O3    = raw concentration
    AQI   = calculated AQI
    """

    normalized = validate_pollutants(pollutants)

    age = freshness_hours(timestamp) if timestamp is not None else None

    return {
        "Date": (pd.Timestamp.now(tz="Asia/Kolkata").strftime("%Y-%m-%d")),
        "PM2.5": normalized["PM2.5"],
        "PM10": normalized["PM10"],
        "NO2": normalized["NO2"],
        "SO2": normalized["SO2"],
        "O3": normalized["O3"],
        "AQI": to_float(aqi),
        "source": source,
        "data_type": data_type,
        "timestamp": (
            parse_timestamp(timestamp).isoformat()
            if timestamp is not None and parse_timestamp(timestamp) is not None
            else None
        ),
        "station_count": station_count,
        "freshness_hours": (round(age, 2) if age is not None else None),
        "confidence": (to_float(confidence) if confidence is not None else None),
    }
