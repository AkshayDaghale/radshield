import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import torch

import kavach.models.forecasting as forecasting
from kavach.models.forecasting import (
    EXPERIMENTAL_FLUX_THRESHOLDS_PFU,
    ForecastInputError,
    TFT_DASHBOARD_TRAINING_FEATURE_COLUMNS,
    TFT_FEATURE_COLUMNS,
    TFT_FEATURE_SCHEMA_MATCHES_PACKAGE_FINETUNE,
    TFT_PACKAGE_FINETUNE_FEATURE_COLUMNS,
    TFT_PREPROCESSING_MATCHES_CHECKPOINT,
    classify_experimental_flux_screening,
    load_tft_checkpoint,
    run_tft_forecast,
)


class StubTFT:
    def eval(self):
        return self

    def __call__(self, tensor):
        self.input_shape = tuple(tensor.shape)
        output = torch.zeros((1, 144, 5), dtype=torch.float32)
        output[0, :, :] = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        return output, None


class WrongShapeTFT(StubTFT):
    def __call__(self, tensor):
        return torch.zeros((1, 143, 5), dtype=torch.float32), None


def valid_frame():
    index = pd.date_range(
        end="2025-01-02 00:00:00+00:00",
        periods=288,
        freq="5min",
    )
    frame = pd.DataFrame(index=index)
    for column, value in {
        "log_flux": 1.0,
        "BY_GSM": 2.0,
        "BZ_GSM": -3.0,
        "Pdyn": 2.5,
        "Vsw": 400.0,
        "AE": 100.0,
        "DST": -10.0,
        "F10.7_index": 70.0,
        "observed_flux": 10.0,
        "observed_BY_GSM": 2.0,
        "observed_BZ_GSM": -3.0,
        "observed_Vsw": 400.0,
    }.items():
        frame[column] = value
    return frame


def run_with_test_model(model, frame, **kwargs):
    with patch.object(forecasting, "TFT_INFERENCE_SUPPORTED", True):
        return run_tft_forecast(model, frame, **kwargs)


class ForecastingTests(unittest.TestCase):
    def test_inference_output_shape_and_utc_forecast_timestamps(self):
        model = StubTFT()
        generated_at = datetime(2025, 1, 2, 0, 1, tzinfo=timezone.utc)
        result = run_with_test_model(
            model,
            valid_frame(),
            satellite_lon=-75.0,
            generated_at=generated_at,
        )

        self.assertEqual(model.input_shape, (1, 288, 10))
        self.assertEqual(result.quantiles_log10_pfu.shape, (144, 5))
        self.assertEqual(
            result.forecast_timestamps_utc[0],
            pd.Timestamp("2025-01-02 00:05:00+00:00"),
        )
        self.assertEqual(
            result.forecast_timestamps_utc[-1],
            pd.Timestamp("2025-01-02 12:00:00+00:00"),
        )
        self.assertEqual(result.generated_at_utc, generated_at)
        self.assertEqual(result.direct_flux_observation_count, 288)

    def test_rejects_missing_required_features_and_irregular_time_steps(self):
        frame = valid_frame().drop(columns="Vsw")
        with self.assertRaisesRegex(ForecastInputError, "Vsw"):
            run_with_test_model(StubTFT(), frame, satellite_lon=0.0)

        frame = valid_frame()
        frame.index = frame.index.where(
            np.arange(len(frame)) != 100,
            frame.index[100] + pd.Timedelta(minutes=1),
        )
        with self.assertRaisesRegex(ForecastInputError, "five-minute sequence"):
            run_with_test_model(StubTFT(), frame, satellite_lon=0.0)

    def test_rejects_missing_direct_flux_and_unavailable_model(self):
        frame = valid_frame()
        frame["observed_flux"] = np.nan
        with self.assertRaisesRegex(ForecastInputError, "No direct GOES flux"):
            run_with_test_model(StubTFT(), frame, satellite_lon=0.0)
        with self.assertRaisesRegex(ForecastInputError, "checkpoint is unavailable"):
            run_with_test_model(None, valid_frame(), satellite_lon=0.0)

    def test_rejects_inference_output_with_wrong_shape(self):
        with self.assertRaisesRegex(ForecastInputError, "output shape must be"):
            run_with_test_model(WrongShapeTFT(), valid_frame(), satellite_lon=0.0)

    def test_missing_checkpoint_is_reported_not_silently_replaced(self):
        missing_path = Path(tempfile.gettempdir()) / "radshield_checkpoint_missing_test.pth"
        model, status = load_tft_checkpoint(missing_path)

        self.assertIsNone(model)
        self.assertFalse(status["available"])
        self.assertIsNotNone(status["error"])

    def test_feature_order_matches_dashboard_but_not_package_finetuner(self):
        self.assertEqual(TFT_FEATURE_COLUMNS, (
            "log_electron_flux",
            "BY_GSM",
            "BZ_GSM",
            "Pdyn",
            "Vsw",
            "AE",
            "DST",
            "F10.7_index",
            "MLT_sin",
            "MLT_cos",
        ))
        self.assertEqual(TFT_DASHBOARD_TRAINING_FEATURE_COLUMNS, TFT_FEATURE_COLUMNS)
        self.assertEqual(TFT_PACKAGE_FINETUNE_FEATURE_COLUMNS, (
            "log_electron_flux",
            "Flow_Speed",
            "Bz_GSM",
            "Proton_Density",
            "Temperature",
            "Flow_Pressure",
            "log_flux_t-1h",
            "log_flux_t-3h",
            "log_flux_t-24h",
            "ULF_Power",
        ))
        self.assertFalse(TFT_FEATURE_SCHEMA_MATCHES_PACKAGE_FINETUNE)

    def test_target_transform_and_window_normalization_match_dashboard_code(self):
        frame = valid_frame()
        frame["log_flux"] = np.arange(288, dtype=np.float64)
        frame["BY_GSM"] = 2.0
        normalized, *_ = forecasting._prepare_input_matrix(frame, satellite_lon=-75.0)

        expected_target = frame["log_flux"].rolling(
            window=24, min_periods=1, center=False
        ).median().to_numpy()
        np.testing.assert_allclose(normalized[:, 0], expected_target.astype(np.float32))
        np.testing.assert_allclose(normalized[:, 1], 0.0)

        scaler_path = Path(__file__).resolve().parents[1] / "weights" / "scaler.pkl"
        scaler = joblib.load(scaler_path)
        sidecar_normalized_by = (
            frame["BY_GSM"].iloc[-288:].to_numpy() - scaler["mean"][0, 1]
        ) / scaler["std"][0, 1]
        self.assertGreater(abs(float(sidecar_normalized_by[0])), 0.1)
        self.assertFalse(TFT_PREPROCESSING_MATCHES_CHECKPOINT)

    def test_nonfinite_source_features_are_rejected(self):
        frame = valid_frame()
        frame.loc[frame.index[-1], "BZ_GSM"] = np.nan
        with self.assertRaisesRegex(ForecastInputError, "non-finite values"):
            run_with_test_model(StubTFT(), frame, satellite_lon=0.0)

    def test_actual_checkpoint_loads_architecture_but_remains_inference_blocked(self):
        checkpoint = (
            Path(__file__).resolve().parents[1]
            / "weights"
            / "finetuned_gsat19_grasp_ulf.pth"
        )
        model, status = load_tft_checkpoint(checkpoint)

        self.assertIsNotNone(model)
        self.assertTrue(status["available"])
        self.assertTrue(status["architecture_compatible"])
        self.assertFalse(status["training_metadata_present"])
        self.assertFalse(status["training_provenance_verified"])
        self.assertFalse(status["scaler_provenance_verified"])
        self.assertFalse(status["inference_supported"])
        self.assertIn("Multiple repository pipelines", status["inference_error"])

        scaler_path = checkpoint.parent / "scaler.pkl"
        scaler = joblib.load(scaler_path)
        self.assertEqual(set(scaler), {"mean", "std"})
        self.assertEqual(scaler["mean"].shape, (1, 10))
        self.assertEqual(scaler["std"].shape, (1, 10))

    def test_inference_gate_blocks_even_a_shape_compatible_model(self):
        model = StubTFT()
        with self.assertRaisesRegex(ForecastInputError, "provenance"):
            run_tft_forecast(model, valid_frame(), satellite_lon=0.0)
        self.assertFalse(hasattr(model, "input_shape"))

    def test_threshold_boundaries_are_experimental_and_compatibility_is_unverified(self):
        lower, upper = EXPERIMENTAL_FLUX_THRESHOLDS_PFU
        self.assertIn("BELOW", classify_experimental_flux_screening(lower))
        self.assertIn("ELEVATED", classify_experimental_flux_screening(lower + 0.01))
        self.assertIn("ELEVATED", classify_experimental_flux_screening(upper))
        self.assertIn("HIGH", classify_experimental_flux_screening(upper + 0.01))
        self.assertFalse(TFT_PREPROCESSING_MATCHES_CHECKPOINT)


if __name__ == "__main__":
    unittest.main()
