"""
KAVACH — Alert Dispatcher Module (SMS / Email / Ops Webhook)
Bharatiya Antariksh Hackathon 2026 | Team DigiIndia | PS-14 ISRO

Handles automated satellite operator alert notifications when hazard risk levels 
breach operational thresholds (RED / YELLOW advisories).
"""
import json
import logging
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("KAVACH_ALERT_DISPATCH")

def dispatch_satellite_alert(
    risk_level: str,
    horizon: str,
    flux_value: float,
    uncertainty_band: tuple,
    satellite_name: str = "GSAT-19 (48°E GEO)",
    webhook_url: str = None
):
    """
    Constructs and dispatches satellite operational alert payloads for RED/YELLOW risks.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    
    payload = {
        "system": "KAVACH_GEO_RADIATION_MONITOR",
        "timestamp_utc": timestamp,
        "satellite": satellite_name,
        "risk_level": risk_level,
        "forecast_horizon": horizon,
        "predicted_flux_pfu": round(flux_value, 2),
        "uncertainty_band_pfu": [round(uncertainty_band[0], 2), round(uncertainty_band[1], 2)],
        "action_required": risk_level == "RED",
        "advisory_message": (
            "CRITICAL WARNING: Deep-dielectric charging hazard detected. Prepare sensor payload safing."
            if risk_level == "RED" else
            "MODERATE ADVISORY: Elevated electron flux detected. Monitor payload telemetry."
        )
    }
    
    # Log alert payload
    logger.info(f"ALERT DISPATCHED [{risk_level}]: {json.dumps(payload, indent=2)}")
    
    # Simulated SMS/Email/Ops Webhook integration
    if webhook_url:
        try:
            import requests
            requests.post(webhook_url, json=payload, timeout=3)
            logger.info("Webhook alert delivered successfully.")
        except Exception as e:
            logger.warning(f"Webhook delivery failed: {e}")
            
    return payload
