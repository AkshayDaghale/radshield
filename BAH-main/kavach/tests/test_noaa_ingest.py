import unittest
from unittest.mock import Mock, patch

import numpy as np

from kavach.data.noaa_ingest import (
    NOAA_GOES_MAG_URL,
    NOAA_GOES_URL,
    NOAA_KP_URL,
    NOAA_MAG_URL,
    NOAA_PLASMA_URL,
    NOAA_ELECTRON_CONTAMINATION_WARNING,
    NOAA_ELECTRON_UNITS,
    fetch_live_noaa_telemetry,
)


class TestNoaaIngest(unittest.TestCase):
    @patch("kavach.data.noaa_ingest.requests.get")
    def test_keeps_direct_flux_gaps_and_reports_each_feed(self, mock_get):
        feeds = {
            NOAA_PLASMA_URL: [
                {"time_tag": "2024-05-01T00:00:00Z", "proton_speed": 400, "proton_density": 5},
                {"time_tag": "2024-05-01T00:05:00Z", "proton_speed": 410, "proton_density": 6},
                {"time_tag": "2024-05-01T00:10:00Z", "proton_speed": 420, "proton_density": 7},
            ],
            NOAA_MAG_URL: [
                {"time_tag": "2024-05-01T00:00:00Z", "bx_gsm": 1, "by_gsm": 2, "bz_gsm": -3},
                {"time_tag": "2024-05-01T00:05:00Z", "bx_gsm": 2, "by_gsm": 3, "bz_gsm": -4},
                {"time_tag": "2024-05-01T00:10:00Z", "bx_gsm": 3, "by_gsm": 4, "bz_gsm": -5},
            ],
            NOAA_GOES_URL: [
                {"time_tag": "2024-05-01T00:00:00Z", "energy": ">=2 MeV", "flux": 100.0, "satellite": 18},
                {"time_tag": "2024-05-01T00:10:00Z", "energy": ">=2 MeV", "flux": 200.0, "satellite": 18},
            ],
            NOAA_GOES_MAG_URL: [],
            NOAA_KP_URL: [
                ["time_tag", "Kp", "a_running"],
                ["2024-05-01T00:00:00Z", "3.0", "3.0"],
                ["2024-05-01T00:03:00Z", "4.0", "3.5"],
            ],
        }

        def response_for_url(url, **_kwargs):
            response = Mock(status_code=200)
            response.json.return_value = feeds[url]
            return response

        mock_get.side_effect = response_for_url
        df, status = fetch_live_noaa_telemetry()

        self.assertIsNotNone(df)
        self.assertEqual(set(status), {
            "solar_wind", "solar_wind_magnetic_field", "goes_electron_flux",
            "goes_magnetometer", "planetary_kp",
        })
        self.assertTrue(all(feed["connected"] for feed in status.values()))
        self.assertEqual(status["goes_electron_flux"]["valid_observations"], 2)
        self.assertEqual(
            status["goes_electron_flux"]["last_observation_timestamp_utc"],
            "2024-05-01T00:10:00+00:00",
        )
        self.assertIsNotNone(status["goes_electron_flux"]["fetch_timestamp_utc"])
        self.assertEqual(status["goes_electron_flux"]["satellite_ids"], ["18"])
        self.assertIsNone(status["goes_electron_flux"]["instrument"])
        self.assertEqual(status["goes_electron_flux"]["units"], NOAA_ELECTRON_UNITS)
        self.assertEqual(
            status["goes_electron_flux"]["quality_warning"],
            NOAA_ELECTRON_CONTAMINATION_WARNING,
        )
        self.assertEqual(df["observed_flux"].notna().sum(), 2)
        self.assertTrue(np.isnan(df.loc["2024-05-01T00:05:00Z", "observed_flux"]))
        self.assertTrue(np.isfinite(df.loc["2024-05-01T00:05:00Z", "flux"]))
        self.assertEqual(df.attrs["feed_status"], status)

    @patch("kavach.data.noaa_ingest.requests.get")
    def test_required_feed_failure_is_reported_without_synthetic_data(self, mock_get):
        failed_response = Mock(status_code=503)
        mock_get.return_value = failed_response

        df, status = fetch_live_noaa_telemetry()

        self.assertIsNone(df)
        self.assertFalse(status["solar_wind"]["connected"])
        self.assertIn("HTTP 503", status["solar_wind"]["error"])
        self.assertFalse(status["solar_wind_magnetic_field"]["connected"])

    @patch("kavach.data.noaa_ingest.requests.get")
    def test_optional_goes_electron_failure_is_reported_without_substitute(self, mock_get):
        timestamps = [
            "2024-05-01T00:00:00Z",
            "2024-05-01T00:05:00Z",
            "2024-05-01T00:10:00Z",
        ]
        payloads = {
            NOAA_PLASMA_URL: [
                {"time_tag": time, "proton_speed": 400, "proton_density": 5}
                for time in timestamps
            ],
            NOAA_MAG_URL: [
                {"time_tag": time, "bx_gsm": 1, "by_gsm": 2, "bz_gsm": -3}
                for time in timestamps
            ],
            NOAA_GOES_MAG_URL: [],
            NOAA_KP_URL: [["time_tag", "Kp", "a_running"]],
        }

        def response_for_url(url, **_kwargs):
            response = Mock(status_code=503 if url == NOAA_GOES_URL else 200)
            if url in payloads:
                response.json.return_value = payloads[url]
            return response

        mock_get.side_effect = response_for_url
        frame, status = fetch_live_noaa_telemetry()

        self.assertIsNotNone(frame)
        self.assertFalse(status["goes_electron_flux"]["connected"])
        self.assertIn("HTTP 503", status["goes_electron_flux"]["error"])
        self.assertTrue(frame["observed_flux"].isna().all())
        self.assertTrue(np.isfinite(frame["flux"]).all())

    @patch("kavach.data.noaa_ingest.requests.get")
    def test_empty_and_invalid_timestamp_goes_rows_remain_unobserved(self, mock_get):
        times = [
            "2024-05-01T00:00:00Z",
            "2024-05-01T00:05:00Z",
            "2024-05-01T00:10:00Z",
        ]
        payloads = {
            NOAA_PLASMA_URL: [
                {"time_tag": time, "proton_speed": 400, "proton_density": 5}
                for time in times
            ],
            NOAA_MAG_URL: [
                {"time_tag": time, "bx_gsm": 1, "by_gsm": 2, "bz_gsm": -3}
                for time in times
            ],
            NOAA_GOES_URL: [
                {"time_tag": "not-a-time", "energy": ">=2 MeV", "flux": 8.0, "satellite": 19},
                {"time_tag": times[0], "energy": ">=2 MeV", "flux": "invalid", "satellite": 19},
            ],
            NOAA_GOES_MAG_URL: [],
            NOAA_KP_URL: [["time_tag", "Kp", "a_running"]],
        }

        def response_for_url(url, **_kwargs):
            response = Mock(status_code=200)
            response.json.return_value = payloads[url]
            return response

        mock_get.side_effect = response_for_url
        frame, status = fetch_live_noaa_telemetry()

        self.assertIsNotNone(frame)
        self.assertEqual(status["goes_electron_flux"]["valid_observations"], 0)
        self.assertEqual(status["goes_electron_flux"]["satellite_ids"], [])
        self.assertIn("invalid timestamps or flux values", status["goes_electron_flux"]["error"])
        self.assertTrue(frame["observed_flux"].isna().all())
        self.assertTrue(np.isfinite(frame["flux"]).all())

    @patch("kavach.data.noaa_ingest.requests.get")
    def test_empty_goes_response_is_reported_as_no_measurements(self, mock_get):
        times = [
            "2024-05-01T00:00:00Z",
            "2024-05-01T00:05:00Z",
            "2024-05-01T00:10:00Z",
        ]
        payloads = {
            NOAA_PLASMA_URL: [
                {"time_tag": time, "proton_speed": 400, "proton_density": 5}
                for time in times
            ],
            NOAA_MAG_URL: [
                {"time_tag": time, "bx_gsm": 1, "by_gsm": 2, "bz_gsm": -3}
                for time in times
            ],
            NOAA_GOES_URL: [],
            NOAA_GOES_MAG_URL: [],
            NOAA_KP_URL: [["time_tag", "Kp", "a_running"]],
        }

        def response_for_url(url, **_kwargs):
            response = Mock(status_code=200)
            response.json.return_value = payloads[url]
            return response

        mock_get.side_effect = response_for_url
        frame, status = fetch_live_noaa_telemetry()

        self.assertIsNotNone(frame)
        self.assertTrue(status["goes_electron_flux"]["connected"])
        self.assertEqual(status["goes_electron_flux"]["valid_observations"], 0)
        self.assertIn("No >=2 MeV", status["goes_electron_flux"]["error"])
        self.assertTrue(frame["observed_flux"].isna().all())


if __name__ == "__main__":
    unittest.main()
