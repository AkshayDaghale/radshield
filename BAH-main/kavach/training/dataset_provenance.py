"""Build a local-evidence-only provenance report for training datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROVENANCE_FIELDS = (
    "field",
    "evidence_found",
    "source_file",
    "confidence_status",
    "unresolved_question",
    "action_required",
)


def _row(
    field: str,
    evidence_found: str,
    source_file: str,
    confidence_status: str,
    unresolved_question: str,
    action_required: str,
) -> dict[str, str]:
    return {
        "field": field,
        "evidence_found": evidence_found,
        "source_file": source_file,
        "confidence_status": confidence_status,
        "unresolved_question": unresolved_question,
        "action_required": action_required,
    }


def build_provenance_report() -> dict[str, Any]:
    """Return claims bounded to repository files; no external metadata inferred."""
    rows = [
        _row(
            "Electron_Flux (fine-tuning target)",
            "The CSV header has Electron_Flux; values are numeric where present and 1,166 entries are blank. The package fine-tuner computes log10(max(value, 1e-5)). The Kaggle prepare_finetune path interpolates and forward/back-fills numeric columns before that transform.",
            "DataSets/Kaggle_FineTuning_Dataset.csv; kavach/training/train_tft.py; kavach_kaggle_train.py",
            "Partially documented; physical provenance unverified",
            "What instrument/product, satellite, energy channel or threshold, flux definition, and physical unit does each value represent? Are blank entries instrument gaps, rejected measurements, or another condition?",
            "Obtain the original publisher citation/URL, exact product and channel identifier, data dictionary, unit definition, and missing-value documentation. Do not approve training before confirmation.",
        ),
        _row(
            "Electron_Flux raw-versus-processed status",
            "The local CSV has a 5-minute timestamp cadence. The package fine-tuner reads rows, sorts timestamps, computes a log transform, and does not resample. The Kaggle pipeline fills numeric values before log transformation. The repository contains no upstream preparation manifest for this CSV.",
            "DataSets/Kaggle_FineTuning_Dataset.csv; kavach/training/train_tft.py; kavach_kaggle_train.py",
            "Processing in local scripts documented; upstream processing unknown",
            "Were the supplied values raw instrument observations, pre-averaged/resampled measurements, or previously cleaned values before this CSV was created?",
            "Obtain the data-export/preprocessing recipe and source-product metadata, including averaging interval and any calibration, clipping, smoothing, or resampling steps.",
        ),
        _row(
            "datetime",
            "The fine-tuning file parses with no failures, is chronologically ordered, has no duplicates, and has 122,975 consecutive 5-minute intervals. The timestamps have no timezone marker; audit code does not convert or assume UTC.",
            "DataSets/Kaggle_FineTuning_Dataset.csv; kavach/data/historical/dataset_audit.json",
            "Observed file facts; timezone meaning unverified",
            "Which timezone and timestamp convention does the source dataset use?",
            "Obtain source timestamp documentation and preserve explicit timezone handling in the future data contract.",
        ),
        _row(
            "BY / BY_GSM",
            "Fine-tuning CSV has BY. The package fine-tuner does not use it. Kaggle training maps BY to BY_GSM or substitutes zero when unavailable.",
            "DataSets/Kaggle_FineTuning_Dataset.csv; kavach/training/train_tft.py; kavach_kaggle_train.py",
            "Column and script mapping observed; physical semantics and units unverified",
            "Is BY in GSM coordinates, what unit is used, and are any numeric sentinels missing-value codes?",
            "Obtain the source column dictionary, coordinate frame, units, and sentinel/missing rules before treating it as BY_GSM.",
        ),
        _row(
            "BZ / BZ_GSM",
            "Fine-tuning CSV has BZ. The package fine-tuner renames it to Bz_GSM. Kaggle training maps it to BZ_GSM and uses a sentinel replacement list.",
            "DataSets/Kaggle_FineTuning_Dataset.csv; kavach/training/train_tft.py; kavach_kaggle_train.py",
            "Script rename observed; GSM frame, units and sentinels unverified",
            "Does the source field use GSM coordinates and what unit/sentinel convention applies?",
            "Obtain the original column definition and coordinate/unit/sentinel documentation.",
        ),
        _row(
            "V / Flow_Speed / Vsw",
            "Fine-tuning CSV has V. The package path renames it to Flow_Speed. Kaggle's pressure formula comments assume V is km/s, and the Kaggle path maps V to Vsw.",
            "DataSets/Kaggle_FineTuning_Dataset.csv; kavach/training/train_tft.py; kavach_kaggle_train.py",
            "Column and code assumptions found; units remain unverified",
            "Is V solar-wind bulk speed in km/s, and what sentinel codes indicate invalid samples?",
            "Obtain the source data dictionary and missing-value convention; verify before applying the pressure formula or model mapping.",
        ),
        _row(
            "Density / Proton_Density",
            "Fine-tuning CSV has Density. Package training renames it to Proton_Density. Kaggle's pressure formula comments assume cm^-3.",
            "DataSets/Kaggle_FineTuning_Dataset.csv; kavach/training/train_tft.py; kavach_kaggle_train.py",
            "Column and code assumptions found; physical definition and units remain unverified",
            "Does Density mean proton number density, in cm^-3, and which sentinel/missing codes apply?",
            "Obtain source field documentation and confirm the assumed physical quantity, units, and missing-value codes.",
        ),
        _row(
            "AE",
            "Fine-tuning CSV has AE; Kaggle training can read it directly or substitute 100. Numeric values matching 99 and 999 occur, and a script sentinel list contains those values, but the source convention is not documented.",
            "DataSets/Kaggle_FineTuning_Dataset.csv; kavach_kaggle_train.py; kavach/data/historical/dataset_audit.json",
            "Field presence observed; unit/source and sentinel meanings unverified",
            "Are 99 and 999 valid AE measurements or fill codes, and which AE product/units are represented?",
            "Obtain the source product, units, and authoritative fill-value rules before cleaning or training.",
        ),
        _row(
            "SYM_H",
            "Fine-tuning CSV has SYM_H. Kaggle's 10-feature schema requires DST; no repository mapping establishes SYM_H as DST.",
            "DataSets/Kaggle_FineTuning_Dataset.csv; kavach_kaggle_train.py",
            "Column presence observed; equivalence to DST unverified and unsupported",
            "What index/product is SYM_H, and is there a scientifically justified conversion to the model's DST feature?",
            "Obtain source metadata; do not substitute SYM_H for DST without an explicitly validated mapping.",
        ),
        _row(
            "ULF_Power",
            "Fine-tuning CSV has ULF_Power. Package training consumes it directly. A code comment says a former proxy was removed because data now contains 'true ULF power', but gives no source, frequency band, estimator, or unit.",
            "DataSets/Kaggle_FineTuning_Dataset.csv; kavach/training/train_tft.py",
            "Direct-use code observed; measured/proxy status unverified",
            "How was ULF_Power measured or calculated, from which instrument and frequency band, and in what units/log convention?",
            "Obtain the ULF data source and processing/data dictionary, including spectral definition, units, and missing-value rules.",
        ),
        _row(
            "Temperature (package feature)",
            "The package fine-tuner sets Temperature to constant 100000.0 and explicitly calls it a placeholder; no temperature column exists in the fine-tuning CSV.",
            "kavach/training/train_tft.py; DataSets/Kaggle_FineTuning_Dataset.csv",
            "Placeholder status documented in code; not an observation",
            "Was the actual checkpoint trained with this constant, another feature, or a different schema?",
            "Recover the exact checkpoint-producing run configuration and feature manifest; do not claim this placeholder is measured temperature.",
        ),
        _row(
            "Flow_Pressure / Pdyn",
            "Package training derives Flow_Pressure from Density and V with an SI conversion expression. Kaggle training comments assume cm^-3 and km/s for its corresponding formula; no source file metadata confirms these assumptions.",
            "kavach/training/train_tft.py; kavach_kaggle_train.py",
            "Formula/code intent observed; physical units and parity unverified",
            "What are the exact input units, formula convention, and pressure unit in the source and checkpoint training run?",
            "Obtain documented input units and the producer preprocessing configuration; verify formula outputs in nPa before feature use.",
        ),
        _row(
            "log_flux_t-1h / log_flux_t-3h / log_flux_t-24h",
            "Package script shifts the transformed target by 12, 36 and 288 rows, respectively, assuming a 5-minute cadence. It drops rows with nulls before constructing sequences.",
            "kavach/training/train_tft.py; DataSets/Kaggle_FineTuning_Dataset.csv",
            "Transformation observed; timestamp alignment is not guaranteed after row drops",
            "Were source gaps or missing target rows removed upstream, and were lags computed from timestamps or row positions in the checkpoint-producing run?",
            "Obtain the exact feature-generation manifest. Derive lags by exact timestamp joins and retain gaps/missing targets rather than row-shifting across them.",
        ),
        _row(
            "Kaggle schema: BY_GSM, BZ_GSM, Pdyn, Vsw, AE, DST, F10.7_index, MLT_sin, MLT_cos",
            "The root Kaggle script defines this ordered schema. It uses defaults/proxies for several fields, computes MLT using an assumed -75 degree longitude, and fits its scaler across pretraining and fine-tuning matrices.",
            "kavach_kaggle_train.py",
            "Script implementation observed; source equivalence and checkpoint attribution unverified",
            "Which of these channels were measured versus derived in the original checkpoint run, and what exact longitude/source data were used?",
            "Obtain the complete run manifest, feature definitions, source versions, transformation rules, and scaler fit dataset hashes.",
        ),
        _row(
            "Kaggle_Validation_March2015.csv target",
            "The CSV contains electron_flux and log_electron_flux columns. The available file is irregular at the expected 5-minute cadence and the repository does not document its source or role in the checkpoint's training history.",
            "DataSets/Kaggle_Validation_March2015.csv; kavach/README.md",
            "File structure observed; target provenance and independence unverified",
            "What source/product generated this file, and was March 2015 excluded from all training and preprocessing?",
            "Obtain source/version metadata and training-run split manifest; do not use as held-out evaluation until independence is shown.",
        ),
        _row(
            "May/October 2024 benchmark flux",
            "The extraction script builds a 5-minute index, uses forward filling and fallback flux, clips flux to at least 1 pfu, applies a centered rolling median, then computes log_flux. Benchmark CSVs store derived features.",
            "kavach/data/extract_2024_storms.py; kavach/data/historical/may_2024_benchmark.csv; kavach/data/historical/oct_2024_benchmark.csv",
            "Local processing code documents derived/filled status; underlying source lineage incomplete",
            "What exact source files, instrument channels and units provided these values, and are the timestamps/measurements independently verified?",
            "Obtain source archive manifests and preserve raw values/gaps; do not use these processed files as direct observations or raw held-out targets.",
        ),
        _row(
            "Kaggle_PreTraining_Dataset.csv / 11-year archive",
            "Training scripts reference these archive names, but no matching CSV exists under the project root. The local 11-year training path may invoke a bootstrap generator when its input is absent.",
            "kavach_kaggle_train.py; kavach/training/train_11yr_model.py; kavach/data/download_11yr_archive.py",
            "Referenced source absent locally",
            "What exact archive and version were used by the checkpoint producer? Is the bootstrap output measured-source data or synthetic?",
            "Obtain the exact training archive, source citation, data dictionary, processing history, and checksum; distinguish generated bootstrap data from sourced observations.",
        ),
        _row(
            "GRASP fine-tuning archive",
            "A transfer-learning script references a GRASP archive path and calls an archive bootstrap if missing; another example fine-tuner explicitly uses generated synthetic data. These do not establish the contents of the configured checkpoint's training data.",
            "kavach/training/transfer_learning_gsat19.py; kavach/training/finetune_grasp.py; kavach/data/download_11yr_archive.py",
            "Checkpoint relationship unverified; some code path is explicitly synthetic",
            "Was real GSAT-19 GRASP data used for the configured checkpoint, and if so where are its source files and preprocessing records?",
            "Obtain the actual source data and run manifest. Exclude synthetic examples and bootstrap data from claims of measured-data training.",
        ),
        _row(
            "Scaler sidecars and checkpoint pairing",
            "Local scaler.pkl, scaler (4).pkl and scaler_mentor.pkl each load as a dictionary with only mean/std arrays of shape 1x10; no feature labels, source dataset ID, or checkpoint hash is present. Training scripts save scaler sidecars to common filenames.",
            "kavach/weights/scaler.pkl; kavach/weights/scaler (4).pkl; kavach/weights/scaler_mentor.pkl; kavach/training/train_tft.py; kavach_kaggle_train.py",
            "Artifact structure verified; producer and pairing unverified",
            "Which scaler belongs to which checkpoint and ordered feature schema, and which exact rows were used to fit it?",
            "Obtain a checkpoint-linked manifest containing ordered feature names, scaler parameters, fit-data IDs/hashes, preprocessing version, and checkpoint hash.",
        ),
        _row(
            "Checkpoint-producing run",
            "The configured checkpoint is a tensor-only state dictionary; repository scripts have conflicting schemas and output paths. No local manifest attributes this checkpoint to one specific training run.",
            "kavach/weights/finetuned_gsat19_grasp_ulf.pth; kavach/training/train_tft.py; kavach_kaggle_train.py; kavach/models/forecasting.py",
            "Unverified; inference disabled",
            "Which source commit, script, exact input files, feature order, preprocessing, scaler, seed, and run produced these weights?",
            "Obtain the original run manifest and artifact hashes; maintain the inference-disabled gate until compatibility is demonstrated.",
        ),
    ]

    missing_local_artifacts = [
        "Kaggle_PreTraining_Dataset.csv",
        "DataSets/Kaggle_PreTraining_Dataset.csv",
        "kavach/data/archive_11yr.csv",
        "kavach/data/archive_11yr_goes_grasp.csv",
        "kavach/data/grasp_2yr.csv",
    ]
    external_requirements = [
        {
            "requirement": "Dataset citation and immutable identity",
            "needed": "Publisher/project, dataset landing-page or download URL, dataset slug/DOI/version, release date, license, and file checksum for each training/validation file.",
        },
        {
            "requirement": "Electron_Flux target definition",
            "needed": "Instrument/platform, product and channel identifier, energy threshold/range, flux quantity and unit, calibration/averaging method, and the meaning of blank or rejected samples.",
        },
        {
            "requirement": "Feature data dictionary",
            "needed": "Definitions, units, coordinate frames, source/product identifiers, timestamp convention/timezone, fill codes and validity ranges for every used input column, especially BY/BZ/V/Density/AE/SYM_H/ULF_Power.",
        },
        {
            "requirement": "Processing lineage",
            "needed": "Whether inputs are raw, calibrated, resampled, averaged, smoothed, clipped, merged, or interpolated; exact procedure and preservation of missing intervals.",
        },
        {
            "requirement": "Exact training run and artifacts",
            "needed": "Checkpoint-linked run manifest with script/revision, ordered features, target transform, windowing, missing-data policy, scaler values and fit rows/data hashes, model configuration, random seed, and checkpoint/scaler hashes.",
        },
        {
            "requirement": "Missing pretraining data",
            "needed": "The exact cited pretraining archive(s) used by the selected training recipe, including matching data dictionary, processing lineage, and checksums.",
        },
    ]
    return {
        "report_version": 1,
        "scope": "Local repository evidence only; no replacement data downloaded and no external source asserted.",
        "provenance_columns": list(PROVENANCE_FIELDS),
        "provenance_report": rows,
        "local_artifact_search": {
            "scope": "Project root only",
            "missing_referenced_archives": missing_local_artifacts,
            "scaler_sidecars": [
                {
                    "path": "kavach/weights/scaler.pkl",
                    "observed_contents": "Dictionary containing mean and std arrays, each shape 1x10.",
                    "feature_names": None,
                    "dataset_identifier": None,
                    "checkpoint_hash": None,
                },
                {
                    "path": "kavach/weights/scaler (4).pkl",
                    "observed_contents": "Dictionary containing mean and std arrays, each shape 1x10.",
                    "feature_names": None,
                    "dataset_identifier": None,
                    "checkpoint_hash": None,
                },
                {
                    "path": "kavach/weights/scaler_mentor.pkl",
                    "observed_contents": "Dictionary containing mean and std arrays, each shape 1x10.",
                    "feature_names": None,
                    "dataset_identifier": None,
                    "checkpoint_hash": None,
                },
            ],
            "matching_data_dictionary_or_run_manifest_found": False,
        },
        "external_documentation_required": external_requirements,
        "training_approval": {
            "status": "blocked_pending_provenance",
            "electron_flux_measurement_definition_verified": False,
            "feature_units_and_semantics_verified": False,
            "pretraining_archive_available": False,
            "checkpoint_training_run_identified": False,
            "checkpoint_or_weights_modified": False,
            "tft_inference_enabled": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("kavach/data/historical/dataset_provenance.json"),
        help="Output JSON path, relative to the project root by default.",
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    output_path = args.output
    if not output_path.is_absolute():
        output_path = project_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = build_provenance_report()
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Dataset provenance report written: {output_path}")
    print(f"Evidence rows: {len(report['provenance_report'])}")
    print(f"Training approval: {report['training_approval']['status']}")


if __name__ == "__main__":
    main()
