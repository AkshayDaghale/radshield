"""
Comprehensive End-to-End Unit Test Suite for KAVACH
"""
import unittest
import numpy as np
import pandas as pd
import torch

from kavach.data.features import engineer_features, classify_regime
from kavach.data.sample_data import generate_synthetic_dataset, load_storm_replay
from kavach.models.radial_diff import run_physics_forecast, compute_DLL
from kavach.models.tft_model import KAVACH_TFT, PinballLoss
from kavach.models.ensemble import ensemble_forecast, classify_risk
from kavach.training.evaluate import (
    EvaluationUnavailableError,
    evaluate_metrics,
    run_storm_replay_benchmark,
)

class TestKavachPipeline(unittest.TestCase):

    def test_feature_engineering(self):
        df = generate_synthetic_dataset(days=2)
        self.assertIn('log_flux', df.columns)
        self.assertIn('Ec', df.columns)
        self.assertIn('Pdyn', df.columns)
        self.assertIn('Bz_neg_dur', df.columns)
        self.assertIn('regime', df.columns)
        self.assertGreaterEqual(len(df.columns), 19)

    def test_radial_diffusion_physics(self):
        L_grid = np.array([3.0, 4.0, 5.0, 6.6])
        dll = compute_DLL(L_grid, Kp=4.0)
        self.assertTrue((dll > 0).all())

        phys_res = run_physics_forecast(current_log_flux=3.5, current_Kp=5.0)
        self.assertIn('T+30m', phys_res)
        self.assertIn('T+6h', phys_res)
        self.assertIn('T+12h', phys_res)
        self.assertGreater(phys_res['T+30m'], 0)

    def test_tft_forward_pass(self):
        model = KAVACH_TFT(num_features=19, hidden_size=64)
        dummy_x = torch.randn(2, 288, 19)
        q_out, attn = model(dummy_x)
        self.assertEqual(q_out.shape, (2, 144, 5))

    def test_ensemble_and_risk(self):
        fused, agreement, uncert = ensemble_forecast(tft_log_flux=4.5, phys_log_flux=4.3, regime_code=1)
        self.assertGreaterEqual(agreement, 0.0)
        self.assertLessEqual(agreement, 1.0)
        
        risk_level, msg = classify_risk(fused, uncert)
        self.assertIn(risk_level, ['RED', 'YELLOW', 'GREEN'])

    def test_evaluation_metrics(self):
        obs = np.array([2.0, 3.0, 4.2, 4.5, 3.8])
        pred = np.array([2.1, 2.9, 4.1, 4.4, 3.9])
        metrics = evaluate_metrics(obs, pred)
        self.assertLess(metrics['RMSE (log-space)'], 0.5)
        self.assertNotIn('PICP (90% band %)', metrics)

        with self.assertRaisesRegex(ValueError, "Both p10 and p90"):
            evaluate_metrics(obs, pred, p10=np.zeros_like(obs))

    def test_storm_replay_benchmark(self):
        with self.assertRaises(EvaluationUnavailableError):
            run_storm_replay_benchmark("Gannon Storm (May 2024)")

if __name__ == '__main__':
    unittest.main()
