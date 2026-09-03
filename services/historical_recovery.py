"""
Historical recovery service for Mumbai AQI.

Purpose
-------
Recover missing historical observations required to make the
LSTM inference timeline continuous.

IMPORTANT
---------
This module NEVER fabricates long historical gaps.

It can use:
    1. Existing all_stations_combined.csv
    2. A separately supplied historical CSV

The historical source must contain genuine observations.

Expected raw columns:

    date
    station_name
    pm25
    pm10
    no2
    so2
    o3
    aqi

Output format:

    Date
    PM2.5
    PM10
    NO2
    SO2
    O3
    AQI
    source
    data_type
    timestamp
    station_count
    freshness_hours
    confidence
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

CURRENT_HISTORY = DATA_DIR / "live_history.csv"

EXISTING_RAW = DATA_DIR / "all_stations_combined.csv"

# Optional external historical file.
#
# We will put a genuinely recovered CPCB/CCR historical CSV here.
RECOVERY_FILE = DATA_DIR / "historical_recovery.csv"


# ============================================================
# CONFIGURATION
# ============================================================

FEATURES = [
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "O3",
    "AQI",
]

SEQ_LEN = 14

MAX_INTERPOLATION_GAP = 3


# ============================================================
# STANDARDIZATION
# ============================================================


def standardize_raw_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rename_map = {
        "date": "Date",
        "pm25": "PM2.5",
        "pm10": "PM10",
        "no2": "NO2",
        "so2": "SO2",
        "o3": "O3",
        "aqi": "AQI",
    }

    df = df.rename(columns=rename_map)

    required = [
        "Date",
        "PM2.5",
        "PM10",
        "NO2",
        "SO2",
        "O3",
        "AQI",
    ]

    missing = [column for column in required if column not in df.columns]

    if missing:

        raise ValueError(
            "Historical recovery file is missing " f"required columns: {missing}"
        )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
    )

    for column in FEATURES:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df[df["Date"].notna()].copy()

    return df


# ============================================================
# LOAD HISTORICAL SOURCE
# ============================================================


def load_recovery_source(
    path: Path = RECOVERY_FILE,
) -> pd.DataFrame:

    if not path.exists():

        raise FileNotFoundError(
            "\nNo historical recovery file found.\n\n"
            f"Expected:\n  {path}\n\n"
            "Place a genuine historical CPCB/CCR/MPCB "
            "dataset here before running recovery."
        )

    print()
    print("Loading historical recovery source:")
    print(f"  {path}")

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    print(f"Recovery raw rows: {len(df)}")

    df = standardize_raw_columns(df)

    print(
        "Recovery date range: "
        f"{df['Date'].min().date()} → "
        f"{df['Date'].max().date()}"
    )

    return df


# ============================================================
# AGGREGATE STATIONS
# ============================================================


def aggregate_city_daily(
    df: pd.DataFrame,
) -> pd.DataFrame:

    print()
    print("Aggregating recovery source into " "one Mumbai city-level value per day...")

    daily = df.groupby("Date")[FEATURES].mean().reset_index()

    daily = daily.sort_values("Date").reset_index(drop=True)

    return daily


# ============================================================
# VALIDATE DAILY DATA
# ============================================================


def validate_daily_data(
    daily: pd.DataFrame,
) -> pd.DataFrame:

    print()
    print("Validating recovered daily data...")

    daily = daily.copy()

    # A day is valid only if ALL six LSTM features exist.
    daily["is_valid"] = daily[FEATURES].notna().all(axis=1)

    valid_count = int(daily["is_valid"].sum())

    invalid_count = int((~daily["is_valid"]).sum())

    print(f"Valid days: {valid_count}")

    print(f"Invalid days: {invalid_count}")

    return daily


# ============================================================
# FIND CONTINUOUS BLOCKS
# ============================================================


def find_continuous_blocks(
    daily: pd.DataFrame,
) -> list[pd.DataFrame]:

    valid = daily[daily["is_valid"]].copy()

    if valid.empty:

        return []

    valid = valid.sort_values("Date")

    gap = valid["Date"].diff().dt.days

    valid["block"] = gap.fillna(1).ne(1).cumsum()

    blocks = []

    for _, group in valid.groupby("block"):

        group = group.drop(columns=["block"]).reset_index(drop=True)

        if len(group) >= SEQ_LEN:

            blocks.append(group)

    return blocks


# ============================================================
# REPORT CONTINUITY
# ============================================================


def print_blocks(
    blocks: list[pd.DataFrame],
) -> None:

    print()
    print(f"Continuous blocks >= {SEQ_LEN} days:")

    if not blocks:

        print("  NONE")

        return

    for index, block in enumerate(blocks):

        print(
            f"  Block {index}: "
            f"{block['Date'].min().date()} → "
            f"{block['Date'].max().date()} "
            f"({len(block)} days)"
        )


# ============================================================
# MERGE WITH CURRENT HISTORY
# ============================================================


def merge_recovered_history(
    recovered: pd.DataFrame,
) -> pd.DataFrame:

    if CURRENT_HISTORY.exists():

        current = pd.read_csv(
            CURRENT_HISTORY,
            parse_dates=["Date"],
        )

        print()
        print(f"Existing live history rows: " f"{len(current)}")

    else:

        current = pd.DataFrame()

    recovered = recovered.copy()

    recovered["Date"] = pd.to_datetime(
        recovered["Date"],
        errors="coerce",
    )

    recovered = recovered[recovered["Date"].notna()].copy()

    if not current.empty:

        current["Date"] = pd.to_datetime(
            current["Date"],
            errors="coerce",
        )

        combined = pd.concat(
            [
                current,
                recovered,
            ],
            ignore_index=True,
        )

    else:

        combined = recovered

    combined = combined[combined["Date"].notna()].copy()

    # Keep one row per date.
    combined = (
        combined.sort_values("Date")
        .drop_duplicates(
            subset=["Date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return combined


# ============================================================
# BUILD CANONICAL RECOVERY ROWS
# ============================================================


def build_canonical_recovery(
    daily: pd.DataFrame,
) -> pd.DataFrame:

    output = daily[["Date"] + FEATURES].copy()

    output["source"] = "historical_cpcb_recovery"

    output["data_type"] = "historical"

    output["timestamp"] = output["Date"].dt.strftime("%Y-%m-%dT00:00:00+05:30")

    output["station_count"] = None

    output["freshness_hours"] = None

    output["confidence"] = None

    return output


# ============================================================
# MAIN RECOVERY
# ============================================================


def recover_history(
    path: Path = RECOVERY_FILE,
) -> pd.DataFrame:

    print()
    print("=" * 70)
    print("MUMBAI AQI HISTORICAL RECOVERY")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    raw = load_recovery_source(path)

    # --------------------------------------------------------
    # AGGREGATE
    # --------------------------------------------------------

    daily = aggregate_city_daily(raw)

    print()
    print(f"Recovered unique days: " f"{len(daily)}")

    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    daily = validate_daily_data(daily)

    # --------------------------------------------------------
    # DO NOT INTERPOLATE LONG GAPS
    # --------------------------------------------------------

    print()
    print("No long-gap interpolation will be performed.")

    # --------------------------------------------------------
    # CONTINUOUS BLOCKS
    # --------------------------------------------------------

    blocks = find_continuous_blocks(daily)

    print_blocks(blocks)

    if not blocks:

        raise RuntimeError(
            "Historical recovery source does not "
            "contain a continuous 14-day usable block."
        )

    # --------------------------------------------------------
    # SELECT LATEST BLOCK
    # --------------------------------------------------------

    latest = max(
        blocks,
        key=lambda block: block["Date"].max(),
    )

    print()
    print("Latest recovered continuous block:")

    print(f"  Start: " f"{latest['Date'].min().date()}")

    print(f"  End: " f"{latest['Date'].max().date()}")

    print(f"  Days: " f"{len(latest)}")

    # --------------------------------------------------------
    # CANONICAL FORMAT
    # --------------------------------------------------------

    recovered = build_canonical_recovery(latest)

    # --------------------------------------------------------
    # MERGE
    # --------------------------------------------------------

    merged = merge_recovered_history(recovered)

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    merged.to_csv(
        CURRENT_HISTORY,
        index=False,
    )

    print()
    print("✓ Historical recovery merged.")

    print(f"History saved: " f"{CURRENT_HISTORY}")

    print(f"Total rows: " f"{len(merged)}")

    print(
        "Final history range: "
        f"{merged['Date'].min().date()} → "
        f"{merged['Date'].max().date()}"
    )

    return merged


# ============================================================
# DIRECT TEST
# ============================================================


if __name__ == "__main__":

    try:

        recover_history()

        print()
        print("=" * 70)
        print("HISTORICAL RECOVERY SUCCESS")
        print("=" * 70)

    except Exception as exc:

        print()
        print("=" * 70)
        print("HISTORICAL RECOVERY FAILED")
        print("=" * 70)

        print()
        print(exc)

        raise SystemExit(1)
