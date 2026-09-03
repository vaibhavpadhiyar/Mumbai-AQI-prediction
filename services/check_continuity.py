from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "all_stations_combined.csv"

FEATURES = [
    "pm25",
    "pm10",
    "no2",
    "so2",
    "o3",
    "aqi",
]


def main():

    print("=" * 70)
    print("CHECKING HISTORICAL CONTINUITY")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    # Aggregate all Mumbai stations by date
    daily = df.groupby("date")[FEATURES].mean().sort_index()

    # Only dates where ALL model features exist
    complete = daily.dropna(subset=FEATURES)

    print()
    print(f"Complete daily observations: {len(complete)}")
    print(f"First complete date: {complete.index.min().date()}")
    print(f"Last complete date: {complete.index.max().date()}")

    # --------------------------------------------------------
    # Find continuous daily runs
    # --------------------------------------------------------

    dates = list(complete.index)

    runs = []

    start = dates[0]
    previous = dates[0]

    for current in dates[1:]:

        difference = current - previous

        if difference != pd.Timedelta(days=1):

            runs.append(
                (
                    start,
                    previous,
                    (previous - start).days + 1,
                )
            )

            start = current

        previous = current

    runs.append(
        (
            start,
            previous,
            (previous - start).days + 1,
        )
    )

    runs.sort(key=lambda x: x[2], reverse=True)

    print()
    print("=" * 70)
    print("LONGEST CONTINUOUS DAILY RUNS")
    print("=" * 70)

    for start, end, length in runs[:10]:

        print(f"{start.date()} -> {end.date()} " f"({length} consecutive days)")

    # --------------------------------------------------------
    # Latest continuous run
    # --------------------------------------------------------

    latest_start, latest_end, latest_length = max(runs, key=lambda x: x[1])

    print()
    print("=" * 70)
    print("LATEST CONTINUOUS RUN")
    print("=" * 70)

    print(f"Start: {latest_start.date()}")
    print(f"End:   {latest_end.date()}")
    print(f"Days:  {latest_length}")

    print()
    print("Latest 20 rows:")
    print(complete.loc[latest_start:latest_end].tail(20).to_string())


if __name__ == "__main__":
    main()
