"""
KAVACH — Reliability & Quantile Calibration Curve Generator
Bharatiya Antariksh Hackathon 2026 | Team DigiIndia | PS-14 ISRO

Evaluates predicted quantile coverage (P10, P25, P50, P75, P90) against 
empirical observed rates to compute the Prediction Interval Coverage Probability (PICP).
"""
import numpy as np
import pandas as pd

def evaluate_quantile_calibration(predictions: np.ndarray, targets: np.ndarray, quantiles=[0.10, 0.25, 0.50, 0.75, 0.90]):
    """
    Computes calibration metrics comparing predicted quantiles vs empirical coverage.
    
    Args:
        predictions: Array of shape [N, num_quantiles]
        targets: Array of shape [N]
        quantiles: List of target quantile levels
        
    Returns:
        dict containing calibration table and Coverage Error metric.
    """
    calibration_results = []
    
    for i, q in enumerate(quantiles):
        pred_q = predictions[:, i]
        empirical_coverage = np.mean(targets <= pred_q)
        coverage_error = empirical_coverage - q
        
        calibration_results.append({
            "Target_Quantile": q,
            "Empirical_Coverage": round(float(empirical_coverage), 4),
            "Coverage_Error": round(float(coverage_error), 4)
        })
        
    df_calib = pd.DataFrame(calibration_results)
    mean_abs_calibration_error = float(df_calib["Coverage_Error"].abs().mean())
    
    return {
        "calibration_table": df_calib,
        "mean_calibration_error": round(mean_abs_calibration_error, 4)
    }

if __name__ == "__main__":
    # Unit test calibration check
    np.random.seed(42)
    y_true = np.random.normal(2.5, 0.5, 1000)
    preds = np.column_stack([np.percentile(y_true, q * 100) + np.random.normal(0, 0.02, 1000) for q in [0.10, 0.25, 0.50, 0.75, 0.90]])
    res = evaluate_quantile_calibration(preds, y_true)
    print("KAVACH Calibration Suite Check:\n", res["calibration_table"])
