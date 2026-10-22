import unittest

import pandas as pd

from kavach.training.dataset_audit import (
    chronological_split_plan,
    inspect_dataframe,
    target_valid_window_starts,
)


class DatasetAuditTests(unittest.TestCase):
    def test_timestamp_integrity_records_parse_order_duplicate_and_timezone(self):
        frame = pd.DataFrame(
            {
                "datetime": [
                    "2024-01-01 00:00:00",
                    "2024-01-01 00:10:00",
                    "not-a-time",
                    "2024-01-01 00:10:00",
                ],
                "flux": [1.0, 2.0, 3.0, 4.0],
            }
        )

        result = inspect_dataframe(frame)

        self.assertEqual(result["timestamps"]["parse_failures"], 1)
        self.assertEqual(result["timestamps"]["duplicate_timestamp_rows"], 1)
        self.assertFalse(result["timestamps"]["chronological_in_input_order"])
        self.assertIsNone(result["columns"]["datetime"]["numeric_parse_failures"])
        self.assertEqual(
            result["timestamps"]["timezone_status"],
            "naive_timezone_unspecified",
        )

    def test_missing_target_is_reported_not_imputed(self):
        frame = pd.DataFrame(
            {
                "datetime": ["2024-01-01 00:00:00", "2024-01-01 00:05:00"],
                "Electron_Flux": [10.0, None],
            }
        )

        result = inspect_dataframe(frame)

        self.assertEqual(result["raw_target_columns"]["Electron_Flux"]["missing_values"], 1)
        self.assertEqual(result["columns"]["Electron_Flux"]["missing_values"], 1)
        self.assertEqual(frame["Electron_Flux"].isna().sum(), 1)

    def test_cadence_anomaly_reports_missing_slots_without_resampling(self):
        frame = pd.DataFrame(
            {
                "datetime": [
                    "2024-01-01 00:00:00",
                    "2024-01-01 00:05:00",
                    "2024-01-01 00:15:00",
                ],
                "flux": [1.0, 2.0, 3.0],
            }
        )

        result = inspect_dataframe(frame)

        self.assertEqual(
            result["timestamps"]["non_expected_interval_count"],
            1,
        )
        self.assertEqual(
            result["timestamps"]["missing_expected_slots_from_exact_multiple_gaps"],
            1,
        )
        self.assertEqual(result["row_count"], 3)

    def test_chronological_split_windows_never_cross_partition_boundaries(self):
        timestamps = pd.date_range("2024-01-01", periods=100, freq="5min")
        result = chronological_split_plan(
            timestamps,
            sequence_length=4,
            prediction_length=2,
            train_fraction=0.6,
            validation_fraction=0.2,
            stride=1,
            target_values=[1.0] * len(timestamps),
        )

        partitions = result["partitions"]
        for part in partitions.values():
            start = part["row_start_inclusive"]
            end = part["row_end_exclusive"]
            window_count = part[
                "complete_non_crossing_window_count_before_missing_target_filter"
            ]
            self.assertEqual(
                part["finite_target_feature_window_count"],
                window_count,
            )
            if window_count:
                last_start = start + window_count - 1
                self.assertGreaterEqual(start, part["row_start_inclusive"])
                self.assertLessEqual(last_start + 4 + 2, end)

        self.assertEqual(partitions["train"]["row_end_exclusive"], 60)
        self.assertEqual(partitions["validation"]["row_start_inclusive"], 60)
        self.assertEqual(partitions["validation"]["row_end_exclusive"], 80)
        self.assertEqual(partitions["test"]["row_start_inclusive"], 80)
        self.assertEqual(result["status"], "provisional_design_only_not_approved_for_training")

    def test_missing_target_history_or_horizon_excludes_window_without_filling(self):
        target = [1.0, 2.0, 3.0, 4.0, float("nan"), 6.0, 7.0, 8.0]

        starts = target_valid_window_starts(
            target,
            partition_start=0,
            partition_end=len(target),
            sequence_length=2,
            prediction_length=2,
            stride=1,
        )

        self.assertEqual(starts, [0])
        self.assertTrue(pd.isna(target[4]))


if __name__ == "__main__":
    unittest.main()
