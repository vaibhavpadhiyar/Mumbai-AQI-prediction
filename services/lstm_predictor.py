from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "model" / "mumbai_aqi_lstm_model_finetuned_recent.keras"

SCALER_PATH = PROJECT_ROOT / "model" / "mumbai_aqi_scaler.save"

HISTORY_PATH = PROJECT_ROOT / "data" / "live_history.csv"


# ============================================================
# MODEL CONTRACT
# ============================================================

SEQ_LEN = 14

FEATURES = [
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "O3",
    "AQI",
]


# ============================================================
# LOAD MODEL + SCALER
# ============================================================


def load_artifacts():

    print("Loading LSTM model...")

    model = tf.keras.models.load_model(MODEL_PATH)

    print("✓ LSTM model loaded")

    print("Loading scaler...")

    scaler = joblib.load(SCALER_PATH)

    print("✓ Scaler loaded")

    return model, scaler


# ============================================================
# LOAD HISTORY
# ============================================================


def load_history():

    if not HISTORY_PATH.exists():

        raise FileNotFoundError(f"History file not found:\n" f"{HISTORY_PATH}")

    df = pd.read_csv(HISTORY_PATH, parse_dates=["Date"])

    df = df.sort_values("Date").reset_index(drop=True)

    return df


# ============================================================
# VALIDATE HISTORY
# ============================================================


def validate_history(df: pd.DataFrame):

    if len(df) < SEQ_LEN:

        raise RuntimeError(
            f"Not enough rows for LSTM.\n"
            f"Required: {SEQ_LEN}\n"
            f"Available: {len(df)}"
        )

    missing = [column for column in FEATURES if column not in df.columns]

    if missing:

        raise RuntimeError(f"Missing model features: {missing}")

    latest = df.tail(SEQ_LEN).copy()

    # --------------------------------------------------------
    # Check dates
    # --------------------------------------------------------

    dates = pd.to_datetime(latest["Date"])

    gaps = dates.diff().dropna()

    if not (gaps == pd.Timedelta(days=1)).all():

        raise RuntimeError("Latest 14 rows are not " "consecutive daily observations.")

    # --------------------------------------------------------
    # Check numeric values
    # --------------------------------------------------------

    values = latest[FEATURES].apply(pd.to_numeric, errors="coerce")

    if values.isnull().any().any():

        bad = values.isnull().sum()

        bad = bad[bad > 0]

        raise RuntimeError(
            "Missing/non-numeric values " f"in LSTM input: {bad.to_dict()}"
        )

    return latest


# ============================================================
# PREPARE INPUT
# ============================================================


def prepare_input(latest: pd.DataFrame, scaler):

    raw_values = latest[FEATURES].astype(float).values

    print()
    print("Raw LSTM input:")

    print(latest[["Date"] + FEATURES].to_string(index=False))

    # --------------------------------------------------------
    # IMPORTANT:
    # Use the EXISTING scaler.
    #
    # DO NOT call scaler.fit().
    # --------------------------------------------------------

    scaled = scaler.transform(raw_values)

    X = scaled.reshape(1, SEQ_LEN, len(FEATURES))

    print()
    print(f"LSTM input shape: {X.shape}")

    return X


# ============================================================
# PREDICT
# ============================================================


def predict_tomorrow(model, scaler, X):

    prediction_scaled = model.predict(X, verbose=0)

    print()
    print("Scaled model output:")

    print(prediction_scaled)

    # --------------------------------------------------------
    # Model predicts scaled AQI.
    #
    # To inverse transform only AQI,
    # create a six-feature placeholder row.
    # --------------------------------------------------------

    placeholder = np.zeros((1, len(FEATURES)), dtype=float)

    aq_index = FEATURES.index("AQI")

    placeholder[0, aq_index] = prediction_scaled[0, 0]

    inverse = scaler.inverse_transform(placeholder)

    predicted_aqi = float(inverse[0, aq_index])

    # AQI cannot be negative.
    predicted_aqi = max(0.0, predicted_aqi)

    return predicted_aqi


# ============================================================
# MAIN
# ============================================================


def main():

    print()
    print("=" * 70)

    print("LSTM PREDICTOR TEST")

    print("=" * 70)

    # 1. Load model and scaler
    model, scaler = load_artifacts()

    # 2. Load history
    history = load_history()

    print()
    print(f"History rows: {len(history)}")

    print(f"Latest historical date: " f"{history['Date'].max().date()}")

    # 3. Validate latest 14 rows
    latest = validate_history(history)

    # 4. Prepare LSTM input
    X = prepare_input(latest, scaler)

    # 5. Predict
    predicted_aqi = predict_tomorrow(model, scaler, X)

    print()
    print("=" * 70)

    print(f"PREDICTED NEXT-DAY AQI: " f"{predicted_aqi:.2f}")

    print("=" * 70)


if __name__ == "__main__":
    main()
