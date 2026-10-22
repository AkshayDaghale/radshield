import unittest

from fastapi import HTTPException

from radshield.api.main import ForecastRequest, health_check, predict_radiation_risk


class ForecastServingTests(unittest.TestCase):
    def test_predict_endpoint_refuses_unverified_checkpoint(self):
        request = ForecastRequest(
            current_flux_pfu=100.0,
            Vsw=400.0,
            BZ_GSM=-2.0,
            BY_GSM=1.0,
            KP=3.0,
            DST=-20.0,
        )
        with self.assertRaises(HTTPException) as context:
            predict_radiation_risk(request)

        self.assertEqual(context.exception.status_code, 503)
        self.assertIn("disabled", context.exception.detail)

    def test_api_health_reports_forecast_inference_disabled(self):
        health = health_check()
        self.assertEqual(health["forecast_status"], "disabled")
        self.assertEqual(health["project"], "RadShield AI")
        self.assertEqual(health["event"], "AITHON 2.0")
        self.assertEqual(health["team"], "Team CodeNova")
        self.assertEqual(health["team_id"], 911)
        self.assertEqual(
            health["problem_statement"],
            "Forecasting Energetic Particle Radiation Environment for "
            "Geostationary Satellites",
        )


if __name__ == "__main__":
    unittest.main()
