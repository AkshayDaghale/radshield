import unittest

from radshield.training.dataset_provenance import (
    PROVENANCE_FIELDS,
    build_provenance_report,
)


class DatasetProvenanceTests(unittest.TestCase):
    def test_report_rows_have_required_table_columns(self):
        report = build_provenance_report()

        self.assertEqual(report["provenance_columns"], list(PROVENANCE_FIELDS))
        self.assertGreater(len(report["provenance_report"]), 0)
        for row in report["provenance_report"]:
            self.assertEqual(set(row), set(PROVENANCE_FIELDS))
            self.assertTrue(row["field"])
            self.assertTrue(row["source_file"])
            self.assertTrue(row["confidence_status"])
            self.assertTrue(row["unresolved_question"])
            self.assertTrue(row["action_required"])

    def test_target_and_feature_provenance_are_explicitly_unresolved(self):
        report = build_provenance_report()
        rows = {row["field"]: row for row in report["provenance_report"]}

        target = rows["Electron_Flux (fine-tuning target)"]
        self.assertIn("1,166", target["evidence_found"])
        self.assertIn("unverified", target["confidence_status"])
        self.assertTrue(target["action_required"])

        for field in (
            "BY / BY_GSM",
            "BZ / BZ_GSM",
            "V / Flow_Speed / Vsw",
            "Density / Proton_Density",
            "AE",
            "SYM_H",
            "ULF_Power",
        ):
            self.assertIn("unverified", rows[field]["confidence_status"])
            self.assertTrue(rows[field]["unresolved_question"])

    def test_missing_archives_and_scaler_provenance_are_explicit(self):
        report = build_provenance_report()

        self.assertIn(
            "Kaggle_PreTraining_Dataset.csv",
            report["local_artifact_search"]["missing_referenced_archives"],
        )
        self.assertFalse(
            report["local_artifact_search"][
                "matching_data_dictionary_or_run_manifest_found"
            ]
        )
        for scaler in report["local_artifact_search"]["scaler_sidecars"]:
            self.assertIsNone(scaler["feature_names"])
            self.assertIsNone(scaler["checkpoint_hash"])

    def test_report_keeps_model_approval_blocked(self):
        status = build_provenance_report()["training_approval"]

        self.assertEqual(status["status"], "blocked_pending_provenance")
        self.assertFalse(status["electron_flux_measurement_definition_verified"])
        self.assertFalse(status["feature_units_and_semantics_verified"])
        self.assertFalse(status["checkpoint_or_weights_modified"])
        self.assertFalse(status["tft_inference_enabled"])


if __name__ == "__main__":
    unittest.main()
