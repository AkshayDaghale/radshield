import unittest
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from kavach.data.noaa_ingest import NOAA_ELECTRON_CONTAMINATION_WARNING
from kavach.data.radiation_observation_status import (
    STALE_AFTER_MINUTES,
    build_radiation_observation_status,
)


class TestRadiationObservationStatus(unittest.TestCase):
    def setUp(self):
        self.feed = {
            "connected": True,
            "url": "https://services.swpc.noaa.gov/json/goes/primary/integral-electrons-7-day.json",
            "fetch_timestamp_utc": "2024-05-01T00:14:00+00:00",
            "satellite_ids": ["18"],
            "instrument": None,
            "quality_flag_available": False,
            "quality_warning": NOAA_ELECTRON_CONTAMINATION_WARNING,
        }

    def test_fresh_status_uses_last_direct_value_and_preserves_gaps(self):
        index = pd.to_datetime(
            ["2024-05-01T00:00:00Z", "2024-05-01T00:05:00Z", "2024-05-01T00:10:00Z"]
        )
        frame = pd.DataFrame(
            {
                "observed_flux": [12.5, np.nan, 25.0],
                "flux": [12.5, 20.0, 25.0],
            },
            index=index,
        )
        result = build_radiation_observation_status(
            frame,
            {"goes_electron_flux": self.feed},
            now=datetime(2024, 5, 1, 0, 14, tzinfo=timezone.utc),
        )

        self.assertEqual(result["latest_value"], 25.0)
        self.assertEqual(result["valid_observation_count"], 2)
        self.assertEqual(result["data_quality_status"], "fresh; event quality unknown")
        self.assertEqual(result["satellite_ids"], ["18"])
        self.assertIsNone(result["instrument"])
        self.assertFalse(result["quality_flag_available"])
        self.assertIn("protons", result["quality_warning"])
        self.assertIn("seven-day", result["history_limit"])

    def test_stale_observation_and_timezone_normalization(self):
        frame = pd.DataFrame(
            {"observed_flux": [4.0]},
            index=pd.DatetimeIndex(["2024-05-01T00:00:00-04:00"]),
        )
        result = build_radiation_observation_status(
            frame,
            {"goes_electron_flux": self.feed},
            now=datetime(2024, 5, 1, 4, 20, tzinfo=timezone.utc),
        )

        self.assertEqual(result["data_quality_status"], "stale")
        self.assertEqual(result["latest_observation_timestamp_utc"], "2024-05-01T04:00:00+00:00")
        self.assertEqual(result["age_minutes"], 20.0)
        self.assertEqual(result["stale_after_minutes"], STALE_AFTER_MINUTES)

    def test_feed_failure_and_model_fill_are_not_reported_as_observations(self):
        frame = pd.DataFrame(
            {"observed_flux": [np.nan], "flux": [123.0]},
            index=pd.to_datetime(["2024-05-01T00:00:00Z"]),
        )
        failed = build_radiation_observation_status(
            frame,
            {"goes_electron_flux": {**self.feed, "connected": False, "error": "HTTP 503"}},
            now=datetime(2024, 5, 1, 0, 1, tzinfo=timezone.utc),
        )
        connected_empty = build_radiation_observation_status(
            frame,
            {"goes_electron_flux": self.feed},
            now=datetime(2024, 5, 1, 0, 1, tzinfo=timezone.utc),
        )

        self.assertEqual(failed["data_quality_status"], "feed unavailable")
        self.assertEqual(failed["feed_error"], "HTTP 503")
        self.assertEqual(failed["valid_observation_count"], 0)
        self.assertIsNone(failed["latest_value"])
        self.assertEqual(connected_empty["data_quality_status"], "no valid measured observation")

    def test_rejects_naive_clock_and_future_observation_timestamps(self):
        frame = pd.DataFrame(
            {"observed_flux": [4.0]},
            index=pd.to_datetime(["2024-05-01T00:10:00Z"]),
        )
        with self.assertRaises(ValueError):
            build_radiation_observation_status(
                frame,
                {"goes_electron_flux": self.feed},
                now=datetime(2024, 5, 1, 0, 11),
            )
        result = build_radiation_observation_status(
            frame,
            {"goes_electron_flux": self.feed},
            now=datetime(2024, 5, 1, 0, 9, tzinfo=timezone.utc),
        )
        self.assertEqual(
            result["data_quality_status"],
            "invalid future observation timestamp",
        )

    def test_invalid_observation_timestamps_and_missing_observed_column(self):
        invalid_time = pd.DataFrame(
            {"observed_flux": [99.0], "flux": [101.0]},
            index=pd.Index(["not-a-timestamp"]),
        )
        no_direct_column = pd.DataFrame(
            {"flux": [101.0]},
            index=pd.to_datetime(["2024-05-01T00:00:00Z"]),
        )

        for frame in (invalid_time, no_direct_column):
            result = build_radiation_observation_status(
                frame,
                {"goes_electron_flux": self.feed},
                now=datetime(2024, 5, 1, 0, 1, tzinfo=timezone.utc),
            )
            self.assertEqual(result["valid_observation_count"], 0)
            self.assertIsNone(result["latest_value"])
            self.assertEqual(
                result["data_quality_status"],
                "no valid measured observation",
            )


if __name__ == "__main__":
    unittest.main()
