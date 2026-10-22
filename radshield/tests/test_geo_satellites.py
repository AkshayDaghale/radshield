import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

from radshield.data.geo_satellites import (
    CELESTRAK_GEO_TLE_URL,
    classify_catalogue_freshness,
    fetch_geo_satellite_positions,
    parse_geo_tle_catalog,
)


# Public CelesTrak catalogue TLEs captured for deterministic parser tests; not live fallbacks.
CATALOGUE_TLE_FIXTURE = """TDRS 3
1 19548U 88091B   26292.04929292 -.00000293  00000+0  00000+0 0  9994
2 19548  12.5107 340.1739 0039961 352.3659  23.7400  1.00280636126633
ABS-6
1 25924U 99053A   26292.17571233 -.00000116  00000+0  00000+0 0  9995
2 25924   0.0507  86.6470 0002051 158.1151   5.1736  1.00272074 99088
"""


class TestGeoSatellites(unittest.TestCase):
    def test_parses_and_propagates_only_geostationary_candidates(self):
        position_time = datetime(2026, 10, 19, 8, 0, tzinfo=timezone.utc)

        positions = parse_geo_tle_catalog(CATALOGUE_TLE_FIXTURE, position_time)

        self.assertEqual(len(positions), 1)
        satellite = positions[0]
        self.assertEqual(satellite["name"], "ABS-6")
        self.assertEqual(satellite["norad_id"], 25924)
        self.assertEqual(satellite["position_timestamp_utc"], position_time.isoformat())
        self.assertEqual(satellite["position_type"], "SGP4 propagated from TLE")
        self.assertEqual(satellite["source"], CELESTRAK_GEO_TLE_URL)
        self.assertGreaterEqual(satellite["orbital_longitude_deg"], -180)
        self.assertLessEqual(satellite["orbital_longitude_deg"], 180)
        self.assertLess(abs(satellite["latitude_deg"]), 1)
        self.assertGreater(satellite["altitude_km"], 30_000)

    def test_rejects_bad_checksums_and_returns_empty_for_no_valid_geo_data(self):
        invalid_tle = CATALOGUE_TLE_FIXTURE.replace("9995", "9990", 1)

        self.assertEqual(
            parse_geo_tle_catalog(
                invalid_tle,
                datetime(2026, 10, 19, 8, 0, tzinfo=timezone.utc),
            ),
            [],
        )
        self.assertEqual(
            parse_geo_tle_catalog(
                "",
                datetime(2026, 10, 19, 8, 0, tzinfo=timezone.utc),
            ),
            [],
        )

    @patch("radshield.data.geo_satellites.requests.get")
    def test_fetches_from_configured_source_and_propagates(self, mock_get):
        response = Mock(status_code=200, text=CATALOGUE_TLE_FIXTURE)
        response.raise_for_status.return_value = None
        mock_get.return_value = response

        positions = fetch_geo_satellite_positions(
            datetime(2026, 10, 19, 8, 0, tzinfo=timezone.utc)
        )

        self.assertEqual([satellite["norad_id"] for satellite in positions], [25924])
        mock_get.assert_called_once()
        self.assertEqual(mock_get.call_args.args[0], CELESTRAK_GEO_TLE_URL)

    @patch("radshield.data.geo_satellites.requests.get")
    def test_source_http_error_is_propagated(self, mock_get):
        import requests

        response = Mock()
        response.raise_for_status.side_effect = requests.HTTPError("HTTP 503")
        mock_get.return_value = response

        with self.assertRaises(requests.HTTPError):
            fetch_geo_satellite_positions(
                datetime(2026, 10, 19, 8, 0, tzinfo=timezone.utc)
            )

    def test_freshness_reports_fresh_stale_and_rejects_invalid_epochs(self):
        now = datetime(2026, 10, 19, 12, 0, tzinfo=timezone.utc)
        fresh_epoch = now - timedelta(hours=6)
        stale_epoch = now - timedelta(hours=25)

        self.assertEqual(classify_catalogue_freshness(fresh_epoch, now), (False, 6.0))
        stale, age_hours = classify_catalogue_freshness(stale_epoch, now)
        self.assertTrue(stale)
        self.assertEqual(age_hours, 25.0)
        with self.assertRaises(ValueError):
            classify_catalogue_freshness(datetime(2026, 10, 19, 0, 0), now)
        with self.assertRaises(ValueError):
            classify_catalogue_freshness(now + timedelta(seconds=1), now)

    def test_position_time_must_be_timezone_aware(self):
        with self.assertRaises(ValueError):
            parse_geo_tle_catalog(CATALOGUE_TLE_FIXTURE, datetime(2026, 10, 19, 8, 0))


if __name__ == "__main__":
    unittest.main()
