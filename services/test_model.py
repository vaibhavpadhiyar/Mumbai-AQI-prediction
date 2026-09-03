from pathlib import Path

import joblib
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "model" / "mumbai_aqi_lstm_model_finetuned_recent.keras"

SCALER_PATH = PROJECT_ROOT / "model" / "mumbai_aqi_scaler.save"


print("=" * 70)
print("MODEL + SCALER TEST")
print("=" * 70)

print()
print("Model:")
print(MODEL_PATH)

print()
print("Scaler:")
print(SCALER_PATH)


if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found:\n{MODEL_PATH}")

if not SCALER_PATH.exists():
    raise FileNotFoundError(f"Scaler not found:\n{SCALER_PATH}")


print()
print("Loading LSTM model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("✓ Model loaded successfully")


print()
print("Model input shape:")
print(model.input_shape)

print()
print("Model output shape:")
print(model.output_shape)


print()
print("Loading scaler...")

scaler = joblib.load(SCALER_PATH)

print("✓ Scaler loaded successfully")


print()
print("Scaler type:")
print(type(scaler))

print()
print("Scaler feature count:")

if hasattr(scaler, "n_features_in_"):
    print(scaler.n_features_in_)
else:
    print("n_features_in_ not available")


print()
print("=" * 70)
print("MODEL + SCALER TEST COMPLETE")
print("=" * 70)
