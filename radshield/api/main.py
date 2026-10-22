"""
FastAPI API for RadShield radiation monitoring
RadShield AI — Energetic Particle Radiation Forecasts for GEO Satellites
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from radshield.models.forecasting import TFT_INFERENCE_SUPPORTED
from radshield.data.sample_data import load_storm_replay, STORM_EVENTS

app = FastAPI(
    title="RadShield AI — GEO Radiation Forecast API",
    description=(
        "TFT forecast serving is disabled until checkpoint feature and preprocessing "
        "provenance can be verified."
    ),
    version="1.0.0"
)

class ForecastRequest(BaseModel):
    current_flux_pfu: float
    Vsw: float
    BZ_GSM: float
    BY_GSM: float
    KP: float
    DST: float
    dDst_dt: Optional[float] = 0.0
    AE_1h: Optional[float] = 200.0
    ULF_power: Optional[float] = -3.5

class HorizonForecast(BaseModel):
    horizon: str
    flux_pfu: float
    log_flux: float
    risk_level: str
    risk_message: str

class ForecastResponse(BaseModel):
    regime_code: int
    regime_label: str
    engine_agreement_pct: float
    model_confidence_pct: float
    multi_horizon_forecasts: List[HorizonForecast]
    dominant_drivers: List[dict]

@app.get("/")
def health_check():
    return {
        "status": "online",
        "forecast_status": "disabled",
        "forecast_note": "No provenance-compatible TFT serving path is enabled; /predict returns HTTP 503.",
        "system": "RadShield AI — GEO Satellite Radiation Forecasts",
        "project": "RadShield AI",
        "event": "AITHON 2.0",
        "team": "Team CodeNova",
        "team_id": 911,
        "problem_statement": (
            "Forecasting Energetic Particle Radiation Environment for "
            "Geostationary Satellites"
        ),
        "theme": "AI-Powered Space Radiation Forecasting & Satellite Protection",
        "category": "Software",
        "track": "Space Technology & Artificial Intelligence",
    }

@app.post("/predict", response_model=ForecastResponse)
def predict_radiation_risk(req: ForecastRequest):
    reason = (
        "the configured TFT checkpoint's training features and preprocessing cannot be verified"
        if not TFT_INFERENCE_SUPPORTED
        else "the API does not yet accept the timestamped feature window required by TFT inference"
    )
    raise HTTPException(
        status_code=503,
        detail=(
            f"Forecast inference is disabled: {reason}. Formula-based surrogate "
            "values are not served as checkpoint forecasts."
        ),
    )

@app.get("/replays")
def list_storm_replays():
    return {"available_storms": list(STORM_EVENTS.keys())}
