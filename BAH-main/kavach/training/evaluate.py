"""
Storm Replay Validation & Metric Evaluation Engine
KAVACH — GEO Radiation Monitor | Team DigiIndia
"""
import numpy as np


class EvaluationUnavailableError(RuntimeError):
    """Raised when no provenance-valid model evaluation can be performed."""


def evaluate_metrics(
    obs_log_flux: np.ndarray,
    pred_log_flux: np.ndarray,
    p10: np.ndarray | None = None,
    p90: np.ndarray | None = None,
) -> dict:
    """
    Computes all operational metrics specified in Section 9.1 of KAVACH developer context:
    - Log-RMSE (< 0.5 target)
    - Threat Score (TS >= 0.40 target at >10^4 pfu)
    - False Alarm Rate (FAR < 0.30 target)
    - Prediction Interval Coverage Probability (PICP 90% band >= 85% target)
    - Persistence Skill score (> 0.20 target)
    """
    # 1. Log-space RMSE
    rmse = np.sqrt(np.mean((pred_log_flux - obs_log_flux) ** 2))

    # 2. Threat Score (TS) & False Alarm Rate (FAR) for storm flux threshold (>10,000 pfu => log_flux > 4.0)
    thresh = 4.0
    obs_storm = obs_log_flux >= thresh
    pred_storm = pred_log_flux >= thresh

    tp = np.sum(obs_storm & pred_storm)
    fp = np.sum((~obs_storm) & pred_storm)
    fn = np.sum(obs_storm & (~pred_storm))

    threat_score = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 1.0
    false_alarm_rate = float(fp / (tp + fp)) if (tp + fp) > 0 else 0.0

    # 3. PICP (Prediction Interval Coverage Probability)
    if (p10 is None) != (p90 is None):
        raise ValueError("Both p10 and p90 arrays are required to calculate PICP.")
    if p10 is not None and p90 is not None:
        if p10.shape != obs_log_flux.shape or p90.shape != obs_log_flux.shape:
            raise ValueError("Observation and quantile arrays must have matching shapes.")
        inside = (obs_log_flux >= p10) & (obs_log_flux <= p90)
        picp = float(np.mean(inside) * 100.0)
    else:
        picp = None

    # 4. Persistence Baseline Skill Score
    pers_log_flux = np.roll(obs_log_flux, 6) # 30-min persistence
    pers_log_flux[:6] = obs_log_flux[0]
    rmse_pers = np.sqrt(np.mean((pers_log_flux - obs_log_flux) ** 2))
    
    skill_score = (
        float((rmse_pers - rmse) / rmse_pers)
        if rmse_pers > 0
        else None
    )

    metrics = {
        'RMSE (log-space)': round(float(rmse), 4),
        'Threat Score (TS)': round(float(threat_score), 4) if tp + fp + fn > 0 else None,
        'False Alarm Rate (FAR)': round(float(false_alarm_rate), 4) if tp + fp > 0 else None,
        'Persistence Skill Score': round(float(skill_score), 4) if skill_score is not None else None,
    }
    if picp is not None:
        metrics['PICP (provided interval %)'] = round(picp, 2)
    return metrics

def run_storm_replay_benchmark(storm_name: str = 'Gannon Storm (May 2024)') -> dict:
    """Refuse the legacy replay until real checkpoint predictions can be evaluated."""
    raise EvaluationUnavailableError(
        f"Cannot evaluate {storm_name!r}: the configured TFT checkpoint is disabled, "
        "and the former replay used heuristic TFT values and hand-built quantile bands "
        "instead of model outputs."
    )

if __name__ == '__main__':
    raise SystemExit(
        "Evaluation unavailable: checkpoint provenance and timestamp-aligned "
        "held-out observations are required."
    )
