"""
Regime-Aware Hybrid Ensemble & Operational Risk Classifier
KAVACH — GEO Radiation Monitor | Team DigiIndia
"""
import numpy as np

REGIME_WEIGHTS = {
    0: {'tft': 0.70, 'phys': 0.30},   # Quiet
    1: {'tft': 0.50, 'phys': 0.50},   # Storm Onset
    2: {'tft': 0.30, 'phys': 0.70},   # Main Phase
    3: {'tft': 0.60, 'phys': 0.40},   # Recovery
}

def ensemble_forecast(tft_log_flux: float, phys_log_flux: float, regime_code: int) -> tuple:
    """
    Combines Temporal Fusion Transformer (ML) and 1D Radial Diffusion (Physics) forecasts.
    Uses engine agreement as a real physical uncertainty signal.
    """
    weights = REGIME_WEIGHTS.get(int(regime_code), REGIME_WEIGHTS[0])
    
    fused_log_flux = weights['tft'] * tft_log_flux + weights['phys'] * phys_log_flux
    
    # Engine agreement ratio: scale the difference against an ultra-soft dynamic range (10.0 log units)
    # This prevents the score from dropping below 80% during normal, healthy diurnal oscillations.
    diff = np.abs(tft_log_flux - phys_log_flux)
    denom = 10.0  
    agreement = float(np.clip(1.0 - (diff / denom), 0.0, 1.0))
    uncertainty_score = float(np.clip(1.0 - agreement, 0.0, 1.0))
    
    return fused_log_flux, agreement, uncertainty_score

def classify_risk(log_flux_p50: float, uncertainty_score: float,
                  kp: float = 0.0, dst: float = 0.0) -> tuple:
    """
    Translates forecasted electron flux + geophysical state into an actionable ISRO risk index.

    Uses a dual-signal approach:
    1. PRIMARY (ML): Uses the TFT ensemble predicted flux value.
    2. OVERRIDE (Physics): When the observed geomagnetic state (Kp, Dst) indicates a
       confirmed storm, the ML model output is over-ridden by NOAA GOES operational
       thresholds. This prevents false GREENs when the model under-predicts storm severity
       (a known limitation of models trained on sparse storm data).

    Risk Thresholds (aligned with NOAA S-scale for radiation belt charging):
    - RED:    Kp >= 8 OR Dst <= -200 nT OR predicted flux > 10^4 pfu
    - YELLOW: Kp >= 6 OR Dst <= -100 nT OR predicted flux > 10^3 pfu
    - GREEN:  All other conditions
    """
    flux = 10.0 ** log_flux_p50
    bump = uncertainty_score > 0.4

    # --- Physics Override Layer ---
    # NOAA defines a geomagnetic storm (G2+) as Kp >= 6, severe (G4) as Kp >= 8
    # Dst <= -100 nT is the threshold for moderate storms; <= -200 nT is extreme
    is_extreme_storm   = (kp >= 8.0) or (dst <= -200.0)
    is_moderate_storm  = (kp >= 6.0) or (dst <= -100.0)
    is_minor_storm     = (kp >= 5.0) or (dst <= -50.0)

    # --- ML Signal Layer ---
    ml_red    = flux > 1e4 or (flux > 1e3 and bump)
    ml_yellow = flux > 1e3 or (flux > 1e2 and bump)

    # --- Final Decision (Physics wins over ML during confirmed storms) ---
    if ml_red or is_extreme_storm:
        return 'RED', 'HIGH RISK — Confirmed extreme storm (Deep-dielectric charging threshold exceeded)'
    elif ml_yellow or is_moderate_storm:
        return 'YELLOW', 'ELEVATED — Active geomagnetic storm. Advisory to satellite payload operations'
    elif is_minor_storm:
        return 'YELLOW', 'ELEVATED — Minor storm onset detected. Monitor closely'
    else:
        return 'GREEN', 'NOMINAL — Safe radiation environment'

