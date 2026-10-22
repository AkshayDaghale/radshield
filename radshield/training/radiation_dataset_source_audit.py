"""Source audit and training-eligibility checks for GEO energetic-electron data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SOURCE_AUDIT_VERSION = 1
MINIMUM_TARGET = {
    "particle": "electrons",
    "quantity": "integral particle flux",
    "energy_threshold_mev": 2.0,
    "energy_channel": "all electrons with energy greater than 2 MeV (or NOAA API label >=2 MeV)",
    "unit": "electrons/(cm^2 s sr)",
    "abbreviation": "pfu",
    "canonical_cadence": "5-minute averaged observations",
    "timestamp_requirement": (
        "One timestamp per observation with source timezone explicitly documented; "
        "normalize to UTC without changing source time semantics. Preserve gaps."
    ),
    "geographic_or_orbital_relevance": (
        "Measurements representative of the outer electron radiation belt at "
        "geostationary orbit; satellite/instrument identity must accompany each row."
    ),
    "target_transform": (
        "Any log transform must be declared and applied only to valid positive "
        "measured flux; missing/quality-invalid target samples remain missing."
    ),
}


def evaluate_training_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """Reject candidates without verified target semantics and unit evidence."""
    required = (
        "target_semantics_verified",
        "units_verified",
        "energy_channel_verified",
        "measurement_source_verified",
        "timestamp_and_cadence_verified",
        "missing_value_semantics_verified",
        "training_period_available",
        "raw_or_processing_lineage_verified",
    )
    missing = [name for name in required if candidate.get(name) is not True]
    return {
        "eligible_for_training": not missing,
        "rejected_requirements": missing,
        "status": "eligible_only_after_other_pipeline_checks" if not missing else "rejected_pending_verification",
    }


def build_source_audit() -> dict[str, Any]:
    sources = [
        {
            "source_id": "noaa_swpc_integral_electrons_live",
            "exact_product_name": "NOAA SWPC GOES integral electrons (7-day) JSON",
            "instrument": {
                "status": "not exposed by this endpoint",
                "evidence": (
                    "The live payload has satellite and energy fields. The separately "
                    "linked SWPC instrument-sources mapping identifies the primary "
                    "electron satellite, but does not name the detector/instrument."
                ),
            },
            "channel": "Integral electrons energy >=2 MeV",
            "units": {
                "value": "electrons/(cm^2 s sr)",
                "abbreviation": "pfu",
                "verification": "SWPC product page documents these units.",
            },
            "cadence": {
                "value": "5-minute averaged integral electron flux",
                "verification": "SWPC product page; inspected JSON samples also show 5-minute timestamps.",
            },
            "temporal_coverage": {
                "value": "Endpoint filename advertises a rolling 7-day product; this is not a historical training archive.",
                "historical_2017_2018_available_from_this_endpoint": False,
            },
            "timestamp": "JSON time_tag is ISO-8601 UTC with Z in inspected payload.",
            "missing_and_quality_semantics": (
                "The endpoint record schema provides time_tag, satellite, flux, "
                "energy; it does not expose a quality flag in the inspected records. "
                "SWPC warns the >2 MeV electron channel is invalid during significant "
                "proton events due to proton contamination. Missing-sample code semantics "
                "are not documented in this JSON feed."
            ),
            "repository_integration_behavior": (
                "radshield/data/noaa_ingest.py resamples feed values onto a 5-minute "
                "grid, retains observed_flux separately, and also forward/back-fills "
                "or synthesizes model-ready flux values. Only observed_flux is a "
                "candidate measurement series; filled model-ready flux must not be "
                "treated as measured training target."
            ),
            "orbital_relevance": (
                "SWPC states GOES electron measurements indicate the outer electron "
                "radiation belt at geostationary orbit."
            ),
            "access_method": "Public HTTPS JSON; no authentication observed.",
            "official_source_urls": [
                "https://services.swpc.noaa.gov/json/goes/primary/integral-electrons-7-day.json",
                "https://www.swpc.noaa.gov/products/goes-electron-flux",
                "https://services.swpc.noaa.gov/json/goes/instrument-sources.json",
            ],
            "candidate_assessment": {
                "target_semantics_verified": True,
                "units_verified": True,
                "energy_channel_verified": True,
                "measurement_source_verified": False,
                "timestamp_and_cadence_verified": True,
                "missing_value_semantics_verified": False,
                "training_period_available": False,
                "raw_or_processing_lineage_verified": False,
            },
            "scope_decision": (
                "Suitable source for current live observation display/short live history "
                "with explicit quality caveat. Not eligible as a historical training "
                "dataset, and not proven instrument-compatible with the 2017-18 CSV."
            ),
        },
        {
            "source_id": "noaa_ncei_goes_13_15_epead_science_quality",
            "exact_product_name": (
                "NCEI GOES 13-15 EPEAD science-quality 1-minute average E1/E2 "
                "electron flux product"
            ),
            "instrument": "Energetic Proton, Electron and Alpha Detector (EPEAD); E2 electron channel",
            "channel": "E2 integral electrons >2 MeV",
            "units": {
                "value": "electrons/(cm^2 s sr), generally called pfu",
                "verification": "SWPC product description documents integral-channel units; verify exact field units in each science-quality file metadata before use.",
            },
            "cadence": {
                "value": "1-minute averaged files",
                "verification": "NCEI GOES 13-15 product documentation.",
            },
            "temporal_coverage": {
                "value": (
                    "Monthly files are documented for GOES 13-15; exact available "
                    "dates for a selected satellite/file must be confirmed in the "
                    "archive before treating 2017-18 coverage as available."
                ),
                "historical_2017_2018_verified": False,
            },
            "timestamp": "Science-quality monthly NetCDF; confirm epoch convention/timezone from each file's metadata.",
            "missing_and_quality_semantics": (
                "NCEI describes science-quality reprocessing, dead-time correction, "
                "and near-background values replaced with fill values; product quality "
                "metadata and proton contamination handling must be read from the actual file/ATBD."
            ),
            "orbital_relevance": (
                "NCEI documents E2 as responding to trapped outer radiation-belt "
                "particles; GOES is geostationary."
            ),
            "access_method": "Public NCEI NetCDF archive; monthly file naming pattern is documented.",
            "official_source_urls": [
                "https://www.ncei.noaa.gov/products/goes-1-15/space-weather-instruments",
                "https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/avg/",
                "https://archive.data.noaa.gov/satellite-spaceweather/GOES1-15/docs/goesnop_algo_theoretical/EPEAD_Electron_Science_Reprocessing_ATBD_v1.0.pdf",
            ],
            "candidate_assessment": {
                "target_semantics_verified": True,
                "units_verified": False,
                "energy_channel_verified": True,
                "measurement_source_verified": True,
                "timestamp_and_cadence_verified": False,
                "missing_value_semantics_verified": True,
                "training_period_available": False,
                "raw_or_processing_lineage_verified": True,
            },
            "scope_decision": (
                "A strong authoritative candidate for science-quality historical "
                "data, but not yet accepted: actual dates, per-file units/time metadata, "
                "available 2017-18 coverage, and compatibility with the existing CSV "
                "must be checked from the retrieved file metadata. Do not combine "
                "with GOES-R operational data without cross-calibration."
            ),
        },
        {
            "source_id": "local_kaggle_finetuning_csv",
            "exact_product_name": "Kaggle_FineTuning_Dataset.csv (repository filename only)",
            "instrument": None,
            "channel": None,
            "units": None,
            "cadence": "Timestamps are regularly spaced at 5 minutes; measurement averaging process is unknown.",
            "temporal_coverage": "2017-07-01 through 2018-08-31, according to parsed local timestamps.",
            "timestamp": "Naive datetime values; timezone convention is undocumented.",
            "missing_and_quality_semantics": (
                "1,166 Electron_Flux blanks. The source meaning is undocumented. "
                "Training code in one path interpolates/fills the target."
            ),
            "repository_integration_behavior": (
                "The package fine-tuner log-transforms and drops rows with missing "
                "derived fields. The Kaggle pipeline interpolates numeric fields "
                "and forward/back-fills, including the target column."
            ),
            "orbital_relevance": (
                "Kaggle training-script comments claim GOES-16 plus ATHA ULF, but "
                "the CSV itself contains no source/instrument/channel metadata; "
                "geostationary target relevance is therefore unverified."
            ),
            "access_method": "Local CSV only; repository has no citation or source URL for this file.",
            "official_source_urls": [],
            "candidate_assessment": {
                "target_semantics_verified": False,
                "units_verified": False,
                "energy_channel_verified": False,
                "measurement_source_verified": False,
                "timestamp_and_cadence_verified": False,
                "missing_value_semantics_verified": False,
                "training_period_available": True,
                "raw_or_processing_lineage_verified": False,
            },
            "scope_decision": (
                "Rejected for training as a measured electron-flux target until "
                "the exact source product, channel, units, quality flags, timestamps, "
                "and processing lineage are supplied."
            ),
        },
        {
            "source_id": "solar_wind_and_geomagnetic_covariates",
            "exact_product_name": (
                "NOAA SWPC real-time solar wind, magnetic field, and planetary Kp feeds"
            ),
            "instrument": "Solar-wind and geomagnetic monitoring feeds; not an electron-flux detector.",
            "channel": None,
            "units": "Field-specific; not equivalent to electron flux.",
            "cadence": "1-minute solar-wind feeds; Kp has its own product cadence.",
            "temporal_coverage": "Configured real-time feeds only; not a historical target archive.",
            "timestamp": "SWPC time_tag values parsed as UTC by current ingestion code.",
            "missing_and_quality_semantics": (
                "Current ingestion retains direct observed columns separately from "
                "derived features; it also fills some model-ready covariates. These "
                "feeds do not provide the target particle-flux measurement."
            ),
            "orbital_relevance": "Potential predictor covariates only; not measured GEO electron flux.",
            "access_method": "Public SWPC JSON endpoints configured in radshield/data/noaa_ingest.py.",
            "official_source_urls": [
                "https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json",
                "https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json",
                "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json",
            ],
            "candidate_assessment": {
                "target_semantics_verified": False,
                "units_verified": False,
                "energy_channel_verified": False,
                "measurement_source_verified": False,
                "timestamp_and_cadence_verified": False,
                "missing_value_semantics_verified": False,
                "training_period_available": False,
                "raw_or_processing_lineage_verified": False,
            },
            "scope_decision": (
                "Not a candidate target dataset. May be considered as separate "
                "predictor features only after the feature schema and time alignment "
                "are scientifically documented."
            ),
        },
    ]
    for source in sources:
        source["training_eligibility"] = evaluate_training_candidate(
            source["candidate_assessment"]
        )

    requirements = [
        "Dataset citation, official landing page/download URL, version, license, and file checksum.",
        "Satellite/platform, detector/instrument, operating mode, energy channel response and electron energy threshold.",
        "Target definition and exact units, including whether values are differential or integral flux and angular convention.",
        "Calibration, dead-time, contamination, background, saturation and quality-flag definitions; identify solar-proton contamination intervals.",
        "Original timestamp meaning, timezone, cadence/averaging window, timestamp label convention, and leap-second/time conversion policy where relevant.",
        "Explicit missing/fill-value meanings and quality flags; preserve missing target intervals without interpolation or forward/back-fill.",
        "Raw-to-file processing provenance: resampling, averaging, calibration, clipping, smoothing, merging, and any transformations.",
        "Per-satellite and per-instrument provenance; cross-calibration evidence before joining or splicing missions/instruments.",
        "Chronological coverage, duplicate/out-of-order timestamps, cadence gaps, valid-target counts/ranges, fill sentinels, nonfinite/invalid values, and exact split boundary/window checks.",
        "Checkpoint training manifest and scaler metadata tied to hashes of the exact input dataset and preprocessing.",
    ]

    return {
        "report_version": SOURCE_AUDIT_VERSION,
        "research_checked_date": "2026-10-19",
        "scope": (
            "Source claims are limited to local repository evidence and the official "
            "NOAA URLs listed per candidate. No datasets were downloaded or merged."
        ),
        "required_radiation_target": MINIMUM_TARGET,
        "source_candidates": sources,
        "candidate_acceptance_rule": {
            "required_verified_fields": [
                "target_semantics_verified",
                "units_verified",
                "energy_channel_verified",
                "measurement_source_verified",
                "timestamp_and_cadence_verified",
                "missing_value_semantics_verified",
                "training_period_available",
                "raw_or_processing_lineage_verified",
            ],
            "rule": (
                "A candidate is not training-eligible unless every required field "
                "is explicitly True. Similar names, feature shapes, or nominal "
                "cadence are not evidence of target equivalence."
            ),
        },
        "comparison_with_existing_finetuning_csv": {
            "local_target_column": "Electron_Flux",
            "known_local_properties": {
                "timestamps": "2017-07-01 through 2018-08-31; naive; 5-minute timestamp spacing",
                "missing_target_rows": 1166,
                "channel": None,
                "units": None,
                "instrument": None,
                "upstream_processing": None,
                "missing_value_semantics": None,
            },
            "noaa_swpc_live_comparison": (
                "SWPC is documented as >2 MeV, pfu, five-minute averaged, and GEO "
                "outer-belt relevant. The local CSV cannot be claimed to match these "
                "semantics. The live endpoint only supplies a rolling 7-day window."
            ),
            "ncei_epead_comparison": (
                "NCEI documents a GOES 13-15 EPEAD E2 >2 MeV science-quality product "
                "at 1-minute cadence. It uses a distinct quality-processing pathway "
                "and near-background fill values; archive dates and file metadata "
                "must be verified before use. Do not assume it generated the local CSV."
            ),
            "solar_wind_comparison": (
                "Solar-wind/geomagnetic feeds are different physical quantities. "
                "They cannot replace measured electron flux as the forecast target."
            ),
        },
        "minimum_provenance_metadata": requirements,
        "manual_validation_checklist": [
            "Obtain source owner/publisher and confirm official product landing page and version.",
            "Verify detector/instrument and satellite from each data file's metadata, not filename guesses.",
            "Confirm the target is integral electron flux above 2 MeV in electrons/(cm^2 s sr) (pfu), not another electron channel, differential flux, proton flux, or a proxy.",
            "Confirm averaging interval, time-label convention and timezone; convert only after preserving source timestamps.",
            "Read all quality flags, fill codes and contamination notes; reject invalid target periods without filling.",
            "Confirm every requested historical date exists for the specific satellite and product.",
            "Validate monotonic unique timestamps, expected cadence, gaps, duplicate records, valid finite positive targets, and per-satellite coverage before any split.",
            "Compare source and local CSV values only after provenance is proven; quantify exact timestamp and measurement agreement without silently merging.",
            "Keep separate instrument/satellite series unless documented cross-calibration supports concatenation.",
            "Obtain the checkpoint-linked training manifest and scaler mapping; keep inference disabled until schema and preprocessing are compatible.",
        ],
        "decision": {
            "training_ready": False,
            "selected_historical_training_dataset": None,
            "live_observation_candidate": "noaa_swpc_integral_electrons_live",
            "historical_archive_candidate_for_manual_followup": "noaa_ncei_goes_13_15_epead_science_quality",
            "inference_enabled": False,
            "weights_modified": False,
            "remaining_blockers": [
                "Electron_Flux source identity, instrument, energy channel, unit, processing lineage and missing-value semantics remain unknown.",
                "The SWPC endpoint is a rolling seven-day feed, not the required 2017-18 training archive.",
                "NCEI EPEAD candidate file dates, exact units/timestamps, quality flags, and 2017-18 coverage must be verified from actual files.",
                "No cross-calibration or equivalence evidence ties either NOAA candidate to the local Electron_Flux values.",
                "Training feature provenance, missing 11-year pretraining archive, checkpoint producer and scaler pairing remain unresolved.",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("radshield/data/historical/radiation_dataset_source_audit.json"),
        help="Output JSON path, relative to the project root by default.",
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    output_path = args.output
    if not output_path.is_absolute():
        output_path = project_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = build_source_audit()
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Radiation dataset source audit written: {output_path}")
    print(f"Candidates audited: {len(report['source_candidates'])}")
    print(f"Training ready: {report['decision']['training_ready']}")


if __name__ == "__main__":
    main()
