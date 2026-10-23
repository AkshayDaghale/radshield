"""Deterministic tests for CelesTrak TLE parsing and SGP4 orbital positions."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock, patch

import requests
from sgp4.io import compute_checksum

from kavach.data import orbital
from kavach.data.orbital import (
    CELESTRAK_GEO_URL,
    REQUEST_TIMEOUT_SECONDS,
    build_geo_globe_figure,
    fetch_celestrak_catalog,
    parse_tle_catalog,
    propagate_catalog,
    propagate_satellite,
    teme_to_ecef_km,
    tle_epoch_status,
)

# Vallado's SGP4 verification-set Vanguard 1 TLE; used only as a fixed
# propagation test vector, never as application catalogue data.
VANGUARD_TLE = (
    "VANGUARD 1",
    "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753",
    "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667",
)
FETCHED_AT = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _tle_text(record=VANGUARD_TLE):
    return "\n".join(record)


class TestCelesTrakTLEParsing(TestCase):
    def test_parses_name_id_and_epoch(self):
        snapshot = parse_tle_catalog(_tle_text(), fetched_at=FETCHED_AT)

        self.assertEqual(len(snapshot.satellites), 1)
        satellite = snapshot.satellites[0]
        self.assertEqual(satellite.name, "VANGUARD 1")
        self.assertEqual(satellite.norad_id, "00005")
        self.assertEqual(
            satellite.tle_epoch,
            datetime(2000, 6, 27, 18, 50, 19, 733_568, tzinfo=timezone.utc),
        )
        self.assertEqual(satellite.fetched_at, FETCHED_AT)

    def test_rejects_invalid_and_incomplete_records_without_losing_valid_ones(self):
        bad_checksum = VANGUARD_TLE[1][:-1] + "0"
        response = "\n".join((*VANGUARD_TLE, "BROKEN SATELLITE", "1 12345U"))
        response = response.replace(VANGUARD_TLE[1], bad_checksum)

        snapshot = parse_tle_catalog(response, fetched_at=FETCHED_AT)

        self.assertEqual(snapshot.satellites, ())
        self.assertIsNotNone(snapshot.error)
        self.assertTrue(snapshot.parse_errors)

    def test_keeps_valid_records_when_neighboring_records_are_malformed(self):
        response = "\n".join((*VANGUARD_TLE, "BROKEN SATELLITE", "1 12345U"))

        snapshot = parse_tle_catalog(response, fetched_at=FETCHED_AT)

        self.assertEqual([record.name for record in snapshot.satellites], ["VANGUARD 1"])
        self.assertIsNone(snapshot.error)
        self.assertTrue(snapshot.parse_errors)

    def test_rejects_mismatched_catalogue_identifiers(self):
        changed_line2 = VANGUARD_TLE[2].replace("00005", "00006", 1)
        changed_line2 = changed_line2[:68] + str(compute_checksum(changed_line2))
        mismatched = (
            VANGUARD_TLE[0],
            VANGUARD_TLE[1],
            changed_line2,
        )

        snapshot = parse_tle_catalog(_tle_text(mismatched), fetched_at=FETCHED_AT)

        self.assertFalse(snapshot.satellites)
        self.assertIn("mismatched NORAD", snapshot.parse_errors[0])

    def test_empty_response_is_unavailable(self):
        snapshot = parse_tle_catalog(" \r\n", fetched_at=FETCHED_AT)
        self.assertFalse(snapshot.satellites)
        self.assertIn("empty", snapshot.error)


class TestCelesTrakFetch(TestCase):
    def test_fetches_configured_geo_feed_and_parses_response(self):
        response = Mock()
        response.text = _tle_text()
        response.raise_for_status.return_value = None
        with TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "geo_tle_cache.json"
            with patch.object(orbital, "CELESTRAK_CACHE_PATH", cache_path):
                with patch(
                    "kavach.data.orbital.requests.get",
                    return_value=response,
                ) as get:
                    snapshot = fetch_celestrak_catalog(fetched_at=FETCHED_AT)

            get.assert_called_once_with(
                CELESTRAK_GEO_URL,
                headers={"User-Agent": "RadShield-AI/1.0 (GEO TLE catalogue)"},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            self.assertTrue(cache_path.is_file())
        self.assertEqual(snapshot.satellites[0].name, "VANGUARD 1")

    def test_timeout_is_reported(self):
        with patch(
            "kavach.data.orbital.requests.get",
            side_effect=requests.Timeout("slow"),
        ):
            snapshot = fetch_celestrak_catalog(fetched_at=FETCHED_AT)
        self.assertIn("timed out", snapshot.error)

    @patch("kavach.data.orbital.requests.get")
    def test_http_error_is_reported(self, get):
        response = Mock()
        response.raise_for_status.side_effect = requests.HTTPError("503 unavailable")
        get.return_value = response

        snapshot = fetch_celestrak_catalog(fetched_at=FETCHED_AT)

        self.assertIn("HTTP error", snapshot.error)

    @patch("kavach.data.orbital.requests.get")
    def test_empty_http_response_is_reported(self, get):
        response = Mock()
        response.text = ""
        response.raise_for_status.return_value = None
        get.return_value = response

        snapshot = fetch_celestrak_catalog(fetched_at=FETCHED_AT)

        self.assertFalse(snapshot.satellites)
        self.assertIn("empty", snapshot.error)

    def test_successful_fetch_is_persisted_and_reused_after_timeout(self):
        response = Mock()
        response.text = _tle_text()
        response.raise_for_status.return_value = None
        with TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "geo_tle_cache.json"
            with patch("kavach.data.orbital.CELESTRAK_CACHE_PATH", cache_path):
                with patch(
                    "kavach.data.orbital.requests.get",
                    return_value=response,
                ):
                    fresh_snapshot = fetch_celestrak_catalog(fetched_at=FETCHED_AT)

                self.assertTrue(cache_path.is_file())
                with patch(
                    "kavach.data.orbital.requests.get",
                    side_effect=requests.Timeout("offline"),
                ):
                    outage_snapshot = fetch_celestrak_catalog(
                        fetched_at=FETCHED_AT + timedelta(hours=1)
                    )

        self.assertIsNone(fresh_snapshot.error)
        self.assertEqual(len(outage_snapshot.satellites), 1)
        self.assertEqual(outage_snapshot.satellites[0].name, "VANGUARD 1")
        self.assertEqual(outage_snapshot.fetched_at, FETCHED_AT)
        self.assertIn("timed out", outage_snapshot.error)
        self.assertEqual(
            outage_snapshot.checked_at,
            FETCHED_AT + timedelta(hours=1),
        )

    def test_legacy_cache_without_source_metadata_is_not_used(self):
        with TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "geo_tle_cache.json"
            cache_path.write_text(json.dumps({
                "fetched_at": FETCHED_AT.isoformat(),
                "tle_text": _tle_text(),
            }), encoding="utf-8")
            with patch.object(orbital, "CELESTRAK_CACHE_PATH", cache_path):
                with patch(
                    "kavach.data.orbital.requests.get",
                    side_effect=requests.Timeout("offline"),
                ):
                    snapshot = fetch_celestrak_catalog(fetched_at=FETCHED_AT)

        self.assertEqual(snapshot.satellites, ())
        self.assertIn("timed out", snapshot.error)
        self.assertIn("unsupported schema", snapshot.error)


class TestSGP4PropagationAndGlobe(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = parse_tle_catalog(
            _tle_text(), fetched_at=FETCHED_AT
        ).satellites[0]

    def test_successful_propagation_keeps_source_and_utc_timestamp(self):
        propagation_time = datetime(2000, 6, 27, 19, 0, tzinfo=timezone.utc)
        position = propagate_satellite(self.record, propagation_time)

        self.assertIs(position.source_tle, self.record)
        self.assertEqual(position.propagated_at, propagation_time)
        self.assertTrue(-180.0 <= position.longitude_deg < 180.0)
        self.assertTrue(-90.0 <= position.latitude_deg <= 90.0)

    def test_propagation_error_is_isolated_in_catalogue_report(self):
        invalid_record = self.record.__class__(
            name="BAD RECORD",
            norad_id=self.record.norad_id,
            line1=self.record.line1[:-1] + "0",
            line2=self.record.line2,
            tle_epoch=self.record.tle_epoch,
            fetched_at=self.record.fetched_at,
        )
        report = propagate_catalog(
            [self.record, invalid_record],
            datetime(2000, 6, 27, 19, 0, tzinfo=timezone.utc),
        )

        self.assertEqual(len(report.positions), 1)
        self.assertEqual(len(report.failures), 1)
        self.assertEqual(report.failures[0].name, "BAD RECORD")

    @patch("kavach.data.orbital.Satrec.twoline2rv")
    def test_sgp4_error_code_is_reported(self, twoline2rv):
        satrec = Mock()
        satrec.sgp4.return_value = (6, (1.0, 2.0, 3.0), (0.0, 0.0, 0.0))
        twoline2rv.return_value = satrec

        with self.assertRaisesRegex(ValueError, "SGP4 error 6"):
            propagate_satellite(
                self.record,
                datetime(2000, 6, 27, 19, 0, tzinfo=timezone.utc),
            )

    @patch("kavach.data.orbital.Satrec.twoline2rv")
    def test_non_finite_sgp4_coordinates_are_rejected(self, twoline2rv):
        satrec = Mock()
        satrec.sgp4.return_value = (0, (float("nan"), 2.0, 3.0), (0.0, 0.0, 0.0))
        twoline2rv.return_value = satrec

        with self.assertRaisesRegex(ValueError, "non-finite"):
            propagate_satellite(
                self.record,
                datetime(2000, 6, 27, 19, 0, tzinfo=timezone.utc),
            )

    def test_propagation_requires_explicit_utc_timestamp(self):
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            propagate_satellite(self.record, datetime(2000, 6, 27, 19, 0))

    def test_teme_to_ecef_rotation_and_longitude_wrapping(self):
        x_ecef, y_ecef, z_ecef = teme_to_ecef_km((1.0, 0.0, 2.0), 3.0 * 3.141592653589793 / 2.0)
        self.assertAlmostEqual(x_ecef, 0.0, places=12)
        self.assertAlmostEqual(y_ecef, 1.0, places=12)
        self.assertEqual(z_ecef, 2.0)

        wrapped = parse_tle_catalog(_tle_text(), fetched_at=FETCHED_AT)
        position = propagate_satellite(
            wrapped.satellites[0],
            datetime(2000, 6, 27, 19, 0, tzinfo=timezone.utc),
        )
        self.assertGreaterEqual(position.longitude_deg, -180.0)
        self.assertLess(position.longitude_deg, 180.0)

    def test_old_tle_is_marked_stale(self):
        status = tle_epoch_status(
            self.record.tle_epoch,
            self.record.tle_epoch + timedelta(days=8),
        )
        self.assertEqual(status, "STALE TLE")

    def test_globe_markers_match_catalogue_propagation(self):
        propagation_time = datetime(2000, 6, 27, 19, 0, tzinfo=timezone.utc)
        report = propagate_catalog([self.record], propagation_time)
        figure = build_geo_globe_figure(report.positions)
        trace = figure.data[0]

        self.assertEqual(list(trace.text), [self.record.name])
        self.assertEqual(list(trace.lon), [report.positions[0].longitude_deg])
        self.assertEqual(list(trace.lat), [report.positions[0].latitude_deg])
        self.assertEqual(trace.customdata[0][0], self.record.norad_id)
        self.assertIn(
            "STALE TLE",
            build_geo_globe_figure(
                propagate_catalog(
                    [self.record],
                    self.record.tle_epoch + timedelta(days=8),
                ).positions
            ).data[0].customdata[0][4],
        )

    def test_globe_keeps_earth_visible_without_satellite_positions(self):
        figure = build_geo_globe_figure(())

        self.assertIn("Earth view", figure.layout.title.text)
        self.assertTrue(figure.layout.geo.showland)
        self.assertEqual(len(figure.data), 1)
        self.assertEqual(list(figure.data[0].lon), [])
        self.assertEqual(list(figure.data[0].lat), [])
        self.assertIn("no validated satellite positions", figure.layout.annotations[0].text)


if __name__ == "__main__":
    import unittest

    unittest.main()
