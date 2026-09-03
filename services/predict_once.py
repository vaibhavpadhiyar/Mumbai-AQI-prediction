"""
End-to-end single prediction pipeline for Mumbai AQI.

Pipeline:

    CPCB
      ↓
    WAQI
      ↓
    OpenWeatherMap
      ↓
    Canonical live reading
      ↓
    History continuity check
      ↓
    Append only if safe
      ↓
    Latest 14-day sequence
      ↓
    Existing LSTM
      ↓
    Tomorrow's AQI prediction

IMPORTANT
---------
This file does NOT change the LSTM architecture.

It also does NOT fabricate missing historical observations.

If the live observation cannot safely be added to the
chronological history, the pipeline stops instead of
feeding invalid data to the LSTM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

HISTORY_PATH = PROJECT_ROOT / "data" / "live_history.csv"


# ============================================================
# IMPORT PIPELINE COMPONENTS
# ============================================================

from services.source_selector import get_best_mumbai_reading
from services.history_buffer import (
    load_history,
    append_reading,
    get_latest_sequence,
    SEQUENCE_LENGTH,
)
from services.lstm_predictor import (
    load_artifacts,
    prepare_input,
    predict_tomorrow,
)

# ============================================================
# CONFIGURATION
# ============================================================

# Maximum gap allowed between the existing history and the
# new live observation.
#
# We only want to append a live observation if it is the next
# expected daily observation.
MAX_APPEND_GAP_DAYS = 1


# ============================================================
# PRINT READING
# ============================================================


def print_live_reading(reading: Dict[str, Any]) -> None:

    print()
    print("=" * 70)
    print("LIVE CANONICAL READING")
    print("=" * 70)

    fields = [
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

    for field in fields:

        if field in reading:

            print(f"{field}: {reading.get(field)}")


# ============================================================
# DATE NORMALIZATION
# ============================================================


def normalize_date(value: Any) -> pd.Timestamp:
    """
    Convert source date into a normalized pandas Timestamp.

    We compare dates rather than timestamps because the LSTM
    history is one observation per day.
    """

    date = pd.to_datetime(value, errors="coerce")

    if pd.isna(date):

        raise ValueError(f"Invalid observation date received from source: {value}")

    return date.normalize()


# ============================================================
# CHECK HISTORY CONTINUITY
# ============================================================


def check_live_reading_continuity(
    history: pd.DataFrame,
    reading: Dict[str, Any],
) -> tuple[bool, str]:

    if history.empty:

        return (
            False,
            "History is empty. A bootstrap dataset is required before "
            "live observations can be appended.",
        )

    if "Date" not in history.columns:

        return (
            False,
            "History does not contain a Date column.",
        )

    history_dates = pd.to_datetime(
        history["Date"],
        errors="coerce",
    ).dropna()

    if history_dates.empty:

        return (
            False,
            "History contains no valid dates.",
        )

    latest_history_date = history_dates.max().normalize()

    live_date = normalize_date(reading.get("Date"))

    # --------------------------------------------------------
    # Duplicate date
    # --------------------------------------------------------

    if live_date == latest_history_date:

        return (
            True,
            "Live observation belongs to the latest existing date. "
            "It can replace/update that day's observation.",
        )

    # --------------------------------------------------------
    # Older observation
    # --------------------------------------------------------

    if live_date < latest_history_date:

        return (
            False,
            f"Live observation date {live_date.date()} is older than "
            f"latest history date {latest_history_date.date()}.",
        )

        # --------------------------------------------------------
    # Future gap
    # --------------------------------------------------------

    gap_days = (live_date - latest_history_date).days

    if gap_days > MAX_APPEND_GAP_DAYS:

        return (
            True,
            f"History gap detected: latest history is "
            f"{latest_history_date.date()}, but live data is "
            f"{live_date.date()} ({gap_days} days later). "
            "Starting a new live observation block. "
            "No historical rows will be fabricated.",
        )

    # --------------------------------------------------------
    # Exactly next day
    # --------------------------------------------------------

    return (
        True,
        f"Live observation is the next daily observation after "
        f"{latest_history_date.date()}.",
    )


# ============================================================
# APPEND LIVE READING SAFELY
# ============================================================


def update_history_with_live_reading(
    reading: Dict[str, Any],
) -> tuple[pd.DataFrame, bool]:

    print()
    print("=" * 70)
    print("HISTORY UPDATE")
    print("=" * 70)

    history = load_history(HISTORY_PATH)

    print(f"Existing history rows: {len(history)}")

    if not history.empty:

        print("Latest history date: " f"{history['Date'].max().date()}")

    allowed, reason = check_live_reading_continuity(
        history,
        reading,
    )

    print(f"Continuity check: {reason}")

    if not allowed:

        print()
        print("⚠ LIVE READING WILL NOT BE APPENDED.")
        print("This prevents an artificial gap from entering " "the LSTM history.")

        return history, False

    # --------------------------------------------------------
    # Append/update
    # --------------------------------------------------------

    updated_history = append_reading(
        reading,
        HISTORY_PATH,
    )

    print()
    print("✓ Live reading added/updated successfully.")

    print(f"Updated history rows: {len(updated_history)}")

    print("Updated latest date: " f"{updated_history['Date'].max().date()}")

    return updated_history, True


# ============================================================
# VALIDATE LSTM SEQUENCE
# ============================================================


def get_valid_lstm_sequence(
    history: pd.DataFrame,
) -> pd.DataFrame:

    print()
    print("=" * 70)
    print("LSTM HISTORY VALIDATION")
    print("=" * 70)

    print(f"Available history rows: {len(history)}")

    print(f"Required sequence length: {SEQUENCE_LENGTH}")

    sequence = get_latest_sequence(
        history,
        SEQUENCE_LENGTH,
    )

    print()
    print(f"✓ Valid {SEQUENCE_LENGTH}-day contiguous sequence found.")

    print(f"Sequence start: {sequence['Date'].min().date()}")

    print(f"Sequence end: {sequence['Date'].max().date()}")

    return sequence


# ============================================================
# RUN LSTM
# ============================================================


def run_prediction(
    sequence: pd.DataFrame,
) -> float:

    print()
    print("=" * 70)
    print("RUNNING LSTM")
    print("=" * 70)

    model, scaler = load_artifacts()

    X = prepare_input(
        sequence,
        scaler,
    )

    predicted_aqi = predict_tomorrow(
        model,
        scaler,
        X,
    )

    return predicted_aqi


# ============================================================
# MAIN PIPELINE
# ============================================================


def main():

    print()
    print("=" * 70)
    print("MUMBAI AQI END-TO-END PREDICTION PIPELINE")
    print("=" * 70)

    print()
    print("Step 1/5: Discovering best live AQI source...")

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    try:

        reading = get_best_mumbai_reading()

    except Exception as exc:

        print()
        print("✗ LIVE SOURCE SELECTION FAILED")
        print()
        print(exc)

        return 1

    print_live_reading(reading)

    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    print()
    print("Step 2/5: Updating historical buffer...")

    history, live_added = update_history_with_live_reading(reading)

    # --------------------------------------------------------
    # IMPORTANT SAFETY STOP
    # --------------------------------------------------------

    if not live_added:

        print()
        print("=" * 70)
        print("PIPELINE STOPPED SAFELY")
        print("=" * 70)

        print()
        print(
            "The live reading was NOT inserted because the "
            "historical timeline is not continuous."
        )

        print()
        print("Current history ends at:")

        print(f"  {history['Date'].max().date()}")

        print()
        print("Live observation is:")

        print(f"  {normalize_date(reading['Date']).date()}")

        print()
        print(
            "Next step: recover the missing historical period "
            "before allowing live data into the LSTM sequence."
        )

        return 2

    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    print()
    print("Step 3/5: Building LSTM sequence...")

    try:

        sequence = get_valid_lstm_sequence(history)

    except Exception as exc:

        print()
        print("✗ LSTM HISTORY VALIDATION FAILED")
        print()
        print(exc)

        return 3

    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    print()
    print("Step 4/5: Predicting tomorrow's AQI...")

    try:

        predicted_aqi = run_prediction(sequence)

    except Exception as exc:

        print()
        print("✗ LSTM PREDICTION FAILED")
        print()
        print(exc)

        return 4

    # --------------------------------------------------------
    # STEP 5
    # --------------------------------------------------------

    print()
    print("Step 5/5: Prediction complete.")

    print()
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print(f"Live source: {reading.get('source')}")

    print(f"Live AQI: {float(reading.get('AQI')):.2f}")

    print(f"Live date: {normalize_date(reading.get('Date')).date()}")

    print(f"LSTM sequence end: " f"{sequence['Date'].max().date()}")

    print(f"PREDICTED AQI FOR TOMORROW: " f"{predicted_aqi:.2f}")

    print("=" * 70)

    return 0


# ============================================================
# ENTRY POINT
# ============================================================


if __name__ == "__main__":

    raise SystemExit(main())
