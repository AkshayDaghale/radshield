import unittest

from kavach.training.radiation_dataset_source_audit import (
    MINIMUM_TARGET,
    build_source_audit,
    evaluate_training_candidate,
)


class RadiationDatasetSourceAuditTests(unittest.TestCase):
    def test_report_has_required_sections_and_source_record_fields(self):
        report = build_source_audit()

        self.assertEqual(report["required_radiation_target"], MINIMUM_TARGET)
        self.assertGreaterEqual(len(report["source_candidates"]), 3)
        for candidate in report["source_candidates"]:
            for field in (
                "source_id",
                "exact_product_name",
                "instrument",
                "channel",
                "units",
                "cadence",
                "temporal_coverage",
                "timestamp",
                "missing_and_quality_semantics",
                "orbital_relevance",
                "access_method",
                "official_source_urls",
                "candidate_assessment",
                "training_eligibility",
                "scope_decision",
            ):
                self.assertIn(field, candidate)

    def test_unknown_units_or_target_semantics_reject_training_candidate(self):
        base = {
            "target_semantics_verified": True,
            "units_verified": True,
            "energy_channel_verified": True,
            "measurement_source_verified": True,
            "timestamp_and_cadence_verified": True,
            "missing_value_semantics_verified": True,
            "training_period_available": True,
            "raw_or_processing_lineage_verified": True,
        }
        unknown_units = {**base, "units_verified": False}
        unknown_target = {**base, "target_semantics_verified": False}

        self.assertFalse(evaluate_training_candidate(unknown_units)["eligible_for_training"])
        self.assertIn(
            "units_verified",
            evaluate_training_candidate(unknown_units)["rejected_requirements"],
        )
        self.assertFalse(evaluate_training_candidate(unknown_target)["eligible_for_training"])
        self.assertIn(
            "target_semantics_verified",
            evaluate_training_candidate(unknown_target)["rejected_requirements"],
        )

    def test_unverified_local_target_and_live_feed_are_not_training_ready(self):
        report = build_source_audit()
        candidates = {item["source_id"]: item for item in report["source_candidates"]}

        local = candidates["local_kaggle_finetuning_csv"]
        self.assertIsNone(local["instrument"])
        self.assertIsNone(local["channel"])
        self.assertIsNone(local["units"])
        self.assertFalse(
            local["training_eligibility"]["eligible_for_training"]
        )
        self.assertIn(
            "filled model-ready flux must not",
            candidates["noaa_swpc_integral_electrons_live"][
                "repository_integration_behavior"
            ],
        )
        self.assertFalse(
            candidates["noaa_swpc_integral_electrons_live"]["training_eligibility"]["eligible_for_training"]
        )
        self.assertFalse(report["decision"]["training_ready"])
        self.assertFalse(report["decision"]["inference_enabled"])

    def test_generic_solar_wind_is_not_accepted_as_electron_flux_target(self):
        report = build_source_audit()
        candidate = next(
            item
            for item in report["source_candidates"]
            if item["source_id"] == "solar_wind_and_geomagnetic_covariates"
        )

        self.assertIsNone(candidate["channel"])
        self.assertFalse(candidate["candidate_assessment"]["target_semantics_verified"])
        self.assertFalse(candidate["training_eligibility"]["eligible_for_training"])


if __name__ == "__main__":
    unittest.main()
