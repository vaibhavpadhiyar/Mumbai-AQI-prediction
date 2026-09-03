"""
Rolling historical buffer for the Mumbai AQI LSTM pipeline.

Purpose
-------
Maintain enough chronological observations for the LSTM
without fabricating repeated values.

The buffer:
    1. Loads existing history
    2. Adds the newest live observation
    3. Removes duplicate timestamps
    4. Sorts chronologically
    5. Saves the updated history
    6. Returns the latest sequence required by the model
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

DATA_DIR.mkdir(parents=True, exist_ok=True)


HISTORY_FILE = DATA_DIR / "live_history.csv"


# IMPORTANT:
# Change this ONLY if your existing LSTM uses another
# sequence length.
SEQUENCE_LENGTH = 14
# ============================================================
# SOURCE QUALITY PRIORITY
# ============================================================
#
# Higher number = better source.
#
# Observed station data is preferred over modeled data.
#
SOURCE_PRIORITY = {
    "cpcb": 3,
    "waqi": 2,
    "openweathermap": 1,
    "historical_station_aggregate": 3,
}


def _source_priority(source: str) -> int:
    """Return priority for a data source."""
    return SOURCE_PRIORITY.get(str(source).strip().lower(), 0)


# Same feature order used by the model.
FEATURE_COLUMNS = [
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "O3",
    "AQI",
]


# ============================================================
# CANONICAL COLUMNS
# ============================================================

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
# LOAD HISTORY
# ============================================================


def load_history(path: Path = HISTORY_FILE) -> pd.DataFrame:
    """
    Load existing historical/live buffer.

    Returns an empty DataFrame when the file doesn't exist.
    """

    if not path.exists():

        return pd.DataFrame(columns=HISTORY_COLUMNS)

    try:

        df = pd.read_csv(path)

    except Exception as exc:

        raise RuntimeError(f"Could not read history file " f"{path}: {exc}")

    # Make sure required columns exist.
    for column in HISTORY_COLUMNS:

        if column not in df.columns:

            df[column] = None

    df = df[HISTORY_COLUMNS]

    # Parse date.
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Remove completely invalid dates.
    df = df[df["Date"].notna()].copy()

    # Numeric feature columns.
    for column in FEATURE_COLUMNS:

        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Sort.
    df = df.sort_values("Date")

    return df.reset_index(drop=True)


# ============================================================
# CONVERT CANONICAL READING TO ROW
# ============================================================


def reading_to_row(reading: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert canonical source reading into one history row.
    """

    return {
        "Date": reading.get("Date"),
        "PM2.5": reading.get("PM2.5"),
        "PM10": reading.get("PM10"),
        "NO2": reading.get("NO2"),
        "SO2": reading.get("SO2"),
        "O3": reading.get("O3"),
        "AQI": reading.get("AQI"),
        "source": reading.get("source"),
        "data_type": reading.get("data_type"),
        "timestamp": reading.get("timestamp"),
        "station_count": reading.get("station_count"),
        "freshness_hours": reading.get("freshness_hours"),
        "confidence": reading.get("confidence"),
    }


# ============================================================
# VALIDATE LIVE READING
# ============================================================


def validate_live_reading(reading: Dict[str, Any]) -> None:
    """
    Validate the canonical reading before inserting it.
    """

    if not isinstance(reading, dict):

        raise ValueError("Reading must be a dictionary.")

    if not reading.get("Date"):

        raise ValueError("Reading has no Date.")

    available = 0

    for column in FEATURE_COLUMNS:

        value = reading.get(column)

        if value is None:
            continue

        try:

            value = float(value)

        except (TypeError, ValueError):

            raise ValueError(f"{column} is not numeric.")

        if value < 0:

            raise ValueError(f"{column} cannot be negative.")

        available += 1

    if available < 3:

        raise ValueError(
            "Live reading contains fewer than " "3 usable numeric features."
        )


# ============================================================
# ADD LIVE READING
# ============================================================


def append_reading(
    reading: Dict[str, Any],
    path: Path = HISTORY_FILE,
) -> pd.DataFrame:
    """
    Add one canonical reading to the history.

    Duplicate dates are handled using source quality.

    Higher-quality sources are preserved:

        CPCB / historical observed
            >
        WAQI
            >
        OpenWeatherMap modeled

    If the existing row has higher priority than the
    incoming row, the existing row is retained.

    If the incoming row has equal or higher priority,
    the incoming row replaces the existing row.
    """

    validate_live_reading(reading)

    df = load_history(path)

    new_row = pd.DataFrame([reading_to_row(reading)])

    new_row["Date"] = pd.to_datetime(
        new_row["Date"],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Reject invalid incoming date
    # --------------------------------------------------------

    if new_row["Date"].isna().any():

        raise ValueError("Incoming reading has an invalid Date.")

    incoming_date = new_row.iloc[0]["Date"]

    incoming_source = new_row.iloc[0]["source"]

    incoming_priority = _source_priority(incoming_source)

    # --------------------------------------------------------
    # Check whether this date already exists
    # --------------------------------------------------------

    existing = df[df["Date"] == incoming_date]

    if not existing.empty:

        existing_row = existing.iloc[-1]

        existing_source = existing_row["source"]

        existing_priority = _source_priority(existing_source)

        print()
        print(f"History already contains " f"{incoming_date.date()}")

        print(f"  Existing source: " f"{existing_source}")

        print(f"  Incoming source: " f"{incoming_source}")

        print(f"  Existing priority: " f"{existing_priority}")

        print(f"  Incoming priority: " f"{incoming_priority}")

        # ----------------------------------------------------
        # Keep existing better-quality observation
        # ----------------------------------------------------

        if existing_priority > incoming_priority:

            print("  → Existing higher-quality " "reading retained.")

            return df

        # ----------------------------------------------------
        # Otherwise replace it
        # ----------------------------------------------------

        print("  → Incoming reading accepted.")

        df = df[df["Date"] != incoming_date].copy()

    # --------------------------------------------------------
    # Add incoming row
    # --------------------------------------------------------

    combined = pd.concat(
        [
            df,
            new_row,
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Remove invalid dates
    # --------------------------------------------------------

    combined = combined[combined["Date"].notna()].copy()

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    combined = combined.sort_values("Date").reset_index(drop=True)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        path,
        index=False,
    )

    print()
    print(f"History saved: {path}")

    print(f"Total rows: {len(combined)}")

    return combined
    """
    Add one canonical live reading to the history.

    Duplicate dates are replaced by the newest reading.

    This prevents repeated GitHub Actions runs from creating
    multiple identical rows for the same date.
    """

    validate_live_reading(reading)

    df = load_history(path)

    new_row = pd.DataFrame([reading_to_row(reading)])

    new_row["Date"] = pd.to_datetime(new_row["Date"], errors="coerce")

    combined = pd.concat(
        [
            df,
            new_row,
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Remove invalid dates
    # --------------------------------------------------------

    combined = combined[combined["Date"].notna()].copy()

    # --------------------------------------------------------
    # Remove duplicate dates
    #
    # Keep the newest inserted observation.
    # --------------------------------------------------------

    combined = combined.drop_duplicates(subset=["Date"], keep="last")

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    combined = combined.sort_values("Date")

    combined = combined.reset_index(drop=True)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    path.parent.mkdir(parents=True, exist_ok=True)

    combined.to_csv(path, index=False)

    return combined


# ============================================================
# GET LSTM SEQUENCE
# ============================================================


def get_latest_sequence(
    df: pd.DataFrame, sequence_length: int = SEQUENCE_LENGTH
) -> pd.DataFrame:
    """
    Return the latest contiguous sequence required by the LSTM.

    The LSTM was trained with:
        14 timesteps
        6 features

    No artificial rows are created here.

    If historical data contains a large gap, the function starts
    from the latest available observation and looks backward only
    within the latest contiguous daily block.
    """

    if df.empty:
        raise RuntimeError(
            "History is empty. No observations are available for " "the LSTM sequence."
        )

    if "Date" not in df.columns:
        raise RuntimeError("History does not contain a Date column.")

    working = df.copy()

    # --------------------------------------------------------
    # Normalize and sort dates
    # --------------------------------------------------------

    working["Date"] = pd.to_datetime(
        working["Date"],
        errors="coerce",
    )

    working = working.dropna(subset=["Date"])

    working = working.sort_values("Date").reset_index(drop=True)

    # --------------------------------------------------------
    # Find the latest contiguous daily block
    # --------------------------------------------------------

    latest_block = []

    for index in range(len(working) - 1, -1, -1):

        current_row = working.iloc[index]

        if not latest_block:
            latest_block.append(current_row)
            continue

        previous_row = working.iloc[index + 1]

        gap = previous_row["Date"] - current_row["Date"]

        if gap == pd.Timedelta(days=1):
            latest_block.append(current_row)

        else:
            break

    latest_block = list(reversed(latest_block))

    latest_block = pd.DataFrame(latest_block)

    # --------------------------------------------------------
    # Check whether the latest contiguous block is long enough
    # --------------------------------------------------------

    if len(latest_block) < sequence_length:

        latest_start = latest_block["Date"].min().date()
        latest_end = latest_block["Date"].max().date()

        raise RuntimeError(
            "Not enough consecutive observations in the latest "
            "continuous block for the LSTM sequence. "
            f"Required={sequence_length}, "
            f"Available={len(latest_block)}, "
            f"Block={latest_start} to {latest_end}."
        )

    # --------------------------------------------------------
    # Take the latest 14 observations from the latest block
    # --------------------------------------------------------

    sequence = latest_block.tail(sequence_length).copy()

    sequence = sequence.sort_values("Date").reset_index(drop=True)

    # --------------------------------------------------------
    # Final date continuity check
    # --------------------------------------------------------

    dates = pd.to_datetime(sequence["Date"])

    date_gaps = dates.diff().dropna()

    if not (date_gaps == pd.Timedelta(days=1)).all():

        raise RuntimeError(
            "The latest LSTM window contains a date gap. "
            "Do not pass a non-contiguous sequence to the model."
        )

    # --------------------------------------------------------
    # Numeric model features
    # --------------------------------------------------------

    for column in FEATURE_COLUMNS:

        sequence[column] = pd.to_numeric(
            sequence[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Missing-value validation
    # --------------------------------------------------------

    if sequence[FEATURE_COLUMNS].isnull().any().any():

        missing = sequence[FEATURE_COLUMNS].isnull().sum()

        missing = missing[missing > 0]

        raise RuntimeError(
            "LSTM sequence contains missing values: " f"{missing.to_dict()}"
        )

    return sequence


# ============================================================
# STATUS
# ============================================================


def get_history_status(
    df: pd.DataFrame, sequence_length: int = SEQUENCE_LENGTH
) -> Dict[str, Any]:
    """
    Return useful diagnostics about the history buffer.
    """

    if df.empty:

        return {
            "rows": 0,
            "ready": False,
            "required": sequence_length,
            "latest_date": None,
            "oldest_date": None,
        }

    return {
        "rows": len(df),
        "ready": len(df) >= sequence_length,
        "required": sequence_length,
        "latest_date": (df["Date"].max().isoformat()),
        "oldest_date": (df["Date"].min().isoformat()),
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)

    print("HISTORY BUFFER TEST")

    print("=" * 60)

    df = load_history()

    print(f"History file: {HISTORY_FILE}")

    print(f"Rows available: {len(df)}")

    print(get_history_status(df))

    if len(df) >= SEQUENCE_LENGTH:

        sequence = get_latest_sequence(df)

        print()
        print(f"Latest {SEQUENCE_LENGTH}-row sequence:")

        print(sequence)

    else:

        print()
        print("LSTM sequence is not ready yet.")

        print(f"Need {SEQUENCE_LENGTH} real rows.")
