"""
REST API wrapper around the existing Mumbai AQI pipeline.

Drop this file in the ROOT of your Mumbai_AQI_Prediction project
(next to the `services/`, `aqi/`, `model/` and `data/` folders) and run:

    pip install fastapi "uvicorn[standard]"
    uvicorn api:app --reload --port 8000

It does NOT change any pipeline logic — it only exposes what already
exists (services/predict_once.py, services/lstm_predictor.py,
services/history_buffer.py) over HTTP so the AirSense frontend can call it.

Endpoints
---------
GET  /api/health            liveness check
GET  /api/latest            most recent canonical reading from live_history.csv
GET  /api/history?days=365  chronological readings, most recent `days` days
GET  /api/predict           tomorrow's AQI from the trained LSTM (cached daily)
POST /api/refresh           pull a fresh live reading, append to history,
                             and refresh the cached prediction.
                             Protect this in production (see REFRESH_TOKEN below)
                             and call it from a daily cron / GitHub Action —
                             the LSTM + live-source calls are too slow for
                             every dashboard page load.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from services.history_buffer import load_history, HISTORY_FILE
from services.lstm_predictor import (
    load_artifacts,
    prepare_input,
    predict_tomorrow,
)
from services.predict_once import (
    update_history_with_live_reading,
    get_valid_lstm_sequence,
)
from services.source_selector import get_best_mumbai_reading

# ============================================================
# CONFIG
# ============================================================

# Comma-separated list of allowed frontend origins, e.g.
#   ALLOWED_ORIGINS="http://localhost:5173,https://your-frontend.example.com"
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")

# Set this in your environment and pass the same value as the
# `x-refresh-token` header when calling POST /api/refresh from your
# cron job / GitHub Action. Leave unset only for local development.
REFRESH_TOKEN = os.getenv("REFRESH_TOKEN")

app = FastAPI(title="Mumbai AQI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Model + scaler are loaded once at startup (loading the .keras model on
# every request would be far too slow for a dashboard).
_model = None
_scaler = None

# Cheap in-memory cache for the last prediction, so /api/predict doesn't
# re-run the LSTM on every page load.
_prediction_cache: Dict[str, Any] = {"date": None, "value": None}


@app.on_event("startup")
def _load_model_once() -> None:
    global _model, _scaler
    _model, _scaler = load_artifacts()


# ============================================================
# SCHEMAS
# ============================================================


class Reading(BaseModel):
    date: str
    pm25: Optional[float] = None
    pm10: Optional[float] = None
    no2: Optional[float] = None
    so2: Optional[float] = None
    o3: Optional[float] = None
    aqi: Optional[float] = None
    source: Optional[str] = None


class PredictionResponse(BaseModel):
    date: str
    predictedTomorrow: float
    basedOn: str


# ============================================================
# HELPERS
# ============================================================


def _row_to_reading(row: pd.Series) -> Reading:
    return Reading(
        date=row["Date"].date().isoformat() if hasattr(row["Date"], "date") else str(row["Date"]),
        pm25=_safe_float(row.get("PM2.5")),
        pm10=_safe_float(row.get("PM10")),
        no2=_safe_float(row.get("NO2")),
        so2=_safe_float(row.get("SO2")),
        o3=_safe_float(row.get("O3")),
        aqi=_safe_float(row.get("AQI")),
        source=row.get("source"),
    )


def _safe_float(value: Any) -> Optional[float]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return round(float(value), 2)


def _run_prediction() -> float:
    history = load_history(HISTORY_FILE)
    latest_sequence = get_valid_lstm_sequence(history)
    X = prepare_input(latest_sequence, _scaler)
    return predict_tomorrow(_model, _scaler, X)


# ============================================================
# ENDPOINTS
# ============================================================


@app.get("/api/health")
def health() -> Dict[str, Any]:
    history = load_history(HISTORY_FILE)
    return {
        "status": "ok",
        "historyRows": len(history),
        "latestDate": history["Date"].max().date().isoformat() if not history.empty else None,
    }


@app.get("/api/latest", response_model=Reading)
def latest() -> Reading:
    history = load_history(HISTORY_FILE)
    if history.empty:
        raise HTTPException(status_code=404, detail="No history available yet.")
    return _row_to_reading(history.iloc[-1])


@app.get("/api/history", response_model=List[Reading])
def history_range(days: int = 365) -> List[Reading]:
    history = load_history(HISTORY_FILE)
    if history.empty:
        return []
    sliced = history.tail(max(1, days))
    return [_row_to_reading(row) for _, row in sliced.iterrows()]


@app.get("/api/predict", response_model=PredictionResponse)
def predict() -> PredictionResponse:
    history = load_history(HISTORY_FILE)
    if history.empty:
        raise HTTPException(status_code=404, detail="No history available yet.")
    latest_date = history["Date"].max().date().isoformat()

    if _prediction_cache["date"] != latest_date:
        try:
            value = _run_prediction()
        except Exception as exc:  # noqa: BLE001 - surface pipeline error to caller
            raise HTTPException(status_code=503, detail=f"Prediction unavailable: {exc}") from exc
        _prediction_cache["date"] = latest_date
        _prediction_cache["value"] = round(value, 2)

    return PredictionResponse(
        date=latest_date,
        predictedTomorrow=_prediction_cache["value"],
        basedOn="mumbai_aqi_lstm_model_finetuned_recent",
    )


@app.post("/api/refresh")
def refresh(x_refresh_token: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    if REFRESH_TOKEN and x_refresh_token != REFRESH_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid refresh token.")

    reading = get_best_mumbai_reading()
    history, added = update_history_with_live_reading(reading)

    if not added:
        return {"added": False, "reason": "History gap or duplicate — see server logs.", "latestDate": history["Date"].max().date().isoformat()}

    # Invalidate the cached prediction so the next /api/predict call recomputes it.
    _prediction_cache["date"] = None

    return {
        "added": True,
        "latestDate": history["Date"].max().date().isoformat(),
        "source": reading.get("source"),
        "aqi": reading.get("AQI"),
    }
