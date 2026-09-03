"""
Bootstrap the live LSTM history from the same historical
data methodology used during fine-tuning.

Source:
    data/all_stations_combined.csv

Output:
    data/live_history.csv

The processing intentionally follows the fine-tuning pipeline:

1. Load all Mumbai station observations
2. Keep 2021 onward
3. Aggregate stations by date
4. Reindex to a complete daily calendar
5. Interpolate ONLY short gaps <= 3 days
6. Reject long gaps
7. Find the latest continuous valid block
8. Export that history for live inference

LSTM configuration:
    Sequence length = 14
    Features =
        PM2.5
        PM10
        NO2
        SO2
        O3
        AQI
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

INPUT_FILE = DATA_DIR / "all_stations_combined.csv"

OUTPUT_FILE = DATA_DIR / "live_history.csv"


# ============================================================
# MODEL CONFIGURATION
# ============================================================

SEQ_LEN = 14

FORECAST_HORIZON = 1

FEATURES = [
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "O3",
    "AQI",
]

RECENT_CUTOFF = pd.Timestamp("2021-01-01")

MAX_GAP_DAYS = 3


# ============================================================
# LOAD DATA
# ============================================================


def load_raw_data() -> pd.DataFrame:

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"\nHistorical dataset not found:\n"
            f"{INPUT_FILE}\n\n"
            f"Place all_stations_combined.csv inside:\n"
            f"{DATA_DIR}"
        )

    print(f"Loading historical dataset:\n" f"  {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, parse_dates=["date"])

    print(f"Raw rows: {len(df)}")

    return df


# ============================================================
# STANDARDIZE COLUMNS
# ============================================================


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:

    rename_map = {
        "pm25": "PM2.5",
        "pm10": "PM10",
        "no2": "NO2",
        "so2": "SO2",
        "o3": "O3",
        "aqi": "AQI",
        "date": "Date",
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
            "Historical dataset is missing " f"required columns: {missing}"
        )

    return df


# ============================================================
# FILTER RECENT TRAINING WINDOW
# ============================================================


def filter_recent_data(df: pd.DataFrame) -> pd.DataFrame:

    df = df[df["Date"] >= RECENT_CUTOFF].copy()

    df = df.sort_values("Date")

    print()
    print(f"Rows from {RECENT_CUTOFF.date()}: " f"{len(df)}")

    if df.empty:

        raise RuntimeError("No data exists from 2021 onward.")

    return df


# ============================================================
# DAILY CITY AGGREGATION
# ============================================================


def aggregate_city_daily(df: pd.DataFrame) -> pd.DataFrame:

    print()
    print("Aggregating Mumbai stations " "into one city-level value per day...")

    # Same methodology used during fine-tuning.
    daily = df.groupby("Date")[FEATURES].mean().reset_index()

    daily = daily.sort_values("Date").reset_index(drop=True)

    print(f"Unique days after aggregation: " f"{len(daily)}")

    print(
        f"Date range: "
        f"{daily['Date'].min().date()} "
        f"→ "
        f"{daily['Date'].max().date()}"
    )

    return daily


# ============================================================
# REINDEX TO DAILY CALENDAR
# ============================================================


def create_daily_calendar(daily: pd.DataFrame) -> pd.DataFrame:

    full_range = pd.date_range(daily["Date"].min(), daily["Date"].max(), freq="D")

    daily = daily.set_index("Date").reindex(full_range)

    daily.index.name = "Date"

    missing_days = daily[FEATURES[0]].isna().sum()

    print()
    print(f"Calendar days with no station data: " f"{missing_days} / {len(daily)}")

    return daily


# ============================================================
# INTERPOLATE ONLY SHORT GAPS
# ============================================================


def interpolate_short_gaps(daily: pd.DataFrame) -> pd.DataFrame:

    print()
    print(f"Interpolating short gaps " f"(maximum {MAX_GAP_DAYS} consecutive days)...")

    daily[FEATURES] = daily[FEATURES].interpolate(
        method="linear", limit=MAX_GAP_DAYS, limit_direction="both"
    )

    return daily


# ============================================================
# FIND CONTIGUOUS VALID BLOCKS
# ============================================================


def find_valid_blocks(daily: pd.DataFrame) -> list[pd.DataFrame]:

    daily = daily.copy()

    daily["is_valid"] = daily[FEATURES].notna().all(axis=1)

    # Every invalid day starts a new block.
    daily["block"] = (~daily["is_valid"]).cumsum()

    valid_blocks = daily[daily["is_valid"]].groupby("block")

    chunks = []

    for _, group in valid_blocks:

        group = group.reset_index()

        if len(group) >= SEQ_LEN:

            chunks.append(group)

    print()
    print(f"Continuous usable blocks " f"(>= {SEQ_LEN} days): " f"{len(chunks)}")

    for index, chunk in enumerate(chunks):

        print(
            f"  Block {index}: "
            f"{chunk['Date'].min().date()} "
            f"→ "
            f"{chunk['Date'].max().date()} "
            f"({len(chunk)} days)"
        )

    return chunks


# ============================================================
# SELECT LATEST BLOCK
# ============================================================


def select_latest_block(chunks: list[pd.DataFrame]) -> pd.DataFrame:

    if not chunks:

        raise RuntimeError(f"No continuous block contains " f"{SEQ_LEN} usable days.")

    # We want the most recent valid block,
    # not necessarily the longest block.
    latest = max(chunks, key=lambda x: x["Date"].max())

    print()
    print("Selected latest continuous block:")

    print(f"  Start: " f"{latest['Date'].min().date()}")

    print(f"  End: " f"{latest['Date'].max().date()}")

    print(f"  Days: " f"{len(latest)}")

    return latest


# ============================================================
# BUILD OUTPUT
# ============================================================


def build_history_output(chunk: pd.DataFrame) -> pd.DataFrame:

    output = chunk[["Date"] + FEATURES].copy()

    output["source"] = "historical_station_aggregate"

    output["data_type"] = "historical"

    output["timestamp"] = output["Date"].dt.strftime("%Y-%m-%dT00:00:00+05:30")

    output["station_count"] = None

    output["freshness_hours"] = None

    output["confidence"] = None

    return output


# ============================================================
# SAVE
# ============================================================


def save_history(history: pd.DataFrame) -> None:

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    history.to_csv(OUTPUT_FILE, index=False)

    print()
    print("History saved successfully:")

    print(f"  {OUTPUT_FILE}")

    print()
    print(f"Rows saved: {len(history)}")

    print(
        f"Date range: "
        f"{history['Date'].min().date()} "
        f"→ "
        f"{history['Date'].max().date()}"
    )


# ============================================================
# MAIN
# ============================================================


def main():

    print()
    print("=" * 70)

    print("MUMBAI AQI HISTORY BOOTSTRAP")

    print("=" * 70)

    # 1. Load
    df = load_raw_data()

    # 2. Standardize
    df = standardize_columns(df)

    # 3. Recent data
    df = filter_recent_data(df)

    # 4. Aggregate stations
    daily = aggregate_city_daily(df)

    # 5. Complete daily calendar
    daily = create_daily_calendar(daily)

    # 6. Short-gap interpolation
    daily = interpolate_short_gaps(daily)

    # 7. Find valid continuous blocks
    chunks = find_valid_blocks(daily)

    # 8. Select most recent block
    latest = select_latest_block(chunks)

    # 9. Build canonical history
    history = build_history_output(latest)

    # 10. Save
    save_history(history)

    # 11. Print latest rows
    print()
    print("=" * 70)

    print("LATEST HISTORY")

    print("=" * 70)

    print(history.tail(min(14, len(history))).to_string(index=False))

    print()
    print("Bootstrap complete.")


if __name__ == "__main__":

    main()
