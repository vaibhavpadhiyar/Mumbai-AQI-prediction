"""
One-time migration: converts the existing repo's history_full.csv (flat,
7 columns) into the corrected pipeline's data/live_history.csv format
(adds the source/confidence metadata columns services/history_buffer.py
expects). Run this ONCE, from the repo root, after copying in the
corrected services/aqi/model folders and BEFORE deleting history_full.csv.

Usage (from repo root):
    python migrate_history.py
"""

from pathlib import Path
import pandas as pd

SOURCE = Path("history_full.csv")
DEST = Path("data/live_history.csv")

HISTORY_COLUMNS = [
    "Date", "PM2.5", "PM10", "NO2", "SO2", "O3", "AQI",
    "source", "data_type", "timestamp", "station_count",
    "freshness_hours", "confidence",
]


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit(f"{SOURCE} not found — run this from the repo root.")

    df = pd.read_csv(SOURCE, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Sanity check: the LSTM needs a contiguous daily block. Fail loudly
    # rather than silently migrating a gap.
    gaps = df["Date"].diff().dropna()
    bad = gaps[gaps != pd.Timedelta(days=1)]
    if not bad.empty:
        print("WARNING: history_full.csv has non-consecutive dates at:")
        print(df.loc[bad.index, "Date"].to_string(index=False))
        print("Migrating anyway — services/history_buffer.py will only use")
        print("the latest contiguous block when building the LSTM sequence.")

    df["source"] = "historical_migrated"
    df["data_type"] = "observed"
    df["timestamp"] = df["Date"].astype(str)
    df["station_count"] = None
    df["freshness_hours"] = None
    df["confidence"] = None

    df = df[HISTORY_COLUMNS]

    DEST.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DEST, index=False)

    print(f"Migrated {len(df)} rows: {df['Date'].min().date()} -> {df['Date'].max().date()}")
    print(f"Written to {DEST}")


if __name__ == "__main__":
    main()
