"""Read-only integrity audit for locally available training and validation CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


EXPECTED_CADENCE_MINUTES = 5.0
TARGET_COLUMNS = ("Electron_Flux", "electron_flux", "flux")
SCRIPT_SENTINEL_CANDIDATES = (
    9999999.0,
    99999.9,
    99999.0,
    9999.99,
    9999.0,
    999.99,
    999.9,
    999.0,
    99.99,
    99.0,
)
PACKAGE_FEATURES = (
    "log_electron_flux",
    "Flow_Speed",
    "Bz_GSM",
    "Proton_Density",
    "Temperature",
    "Flow_Pressure",
    "log_flux_t-1h",
    "log_flux_t-3h",
    "log_flux_t-24h",
    "ULF_Power",
)
KAGGLE_FEATURES = (
    "log_electron_flux",
    "BY_GSM",
    "BZ_GSM",
    "Pdyn",
    "Vsw",
    "AE",
    "DST",
    "F10.7_index",
    "MLT_sin",
    "MLT_cos",
)


def _timestamp_summary(
    values: pd.Series, expected_cadence_minutes: float
) -> dict[str, Any]:
    parsed = pd.to_datetime(values, errors="coerce")
    valid = parsed.dropna()
    timezone = getattr(parsed.dtype, "tz", None)
    if timezone is None and len(valid):
        timezone_status = "naive_timezone_unspecified"
    elif timezone is not None:
        timezone_status = f"timezone_aware:{timezone}"
    else:
        timezone_status = "unavailable"

    ordered = bool(parsed.notna().all() and parsed.is_monotonic_increasing)
    duplicate_count = int(valid.duplicated().sum())
    sorted_valid = valid.sort_values()
    intervals = (
        sorted_valid.diff().dropna().dt.total_seconds().div(60.0)
        if len(sorted_valid) > 1
        else pd.Series(dtype=float)
    )
    interval_counts = intervals.value_counts().sort_index()
    missing_slots = 0
    for minutes in intervals:
        if minutes > expected_cadence_minutes and np.isclose(
            minutes / expected_cadence_minutes,
            round(minutes / expected_cadence_minutes),
        ):
            missing_slots += int(round(minutes / expected_cadence_minutes)) - 1

    return {
        "timestamp_column": str(values.name),
        "parse_failures": int(parsed.isna().sum()),
        "timezone_status": timezone_status,
        "timezone_assumption": (
            "No timezone conversion or UTC assumption applied."
            if timezone is None
            else "Source timezone retained; no conversion applied."
        ),
        "chronological_in_input_order": ordered,
        "duplicate_timestamp_rows": duplicate_count,
        "first_timestamp": valid.min().isoformat() if len(valid) else None,
        "last_timestamp": valid.max().isoformat() if len(valid) else None,
        "valid_timestamp_rows": int(len(valid)),
        "expected_cadence_minutes_for_diagnostic_only": expected_cadence_minutes,
        "interval_count": int(len(intervals)),
        "interval_minutes_counts": {
            str(float(interval)): int(count)
            for interval, count in interval_counts.items()
        },
        "non_expected_interval_count": int(
            (intervals != expected_cadence_minutes).sum()
        ),
        "gap_over_expected_count": int(
            (intervals > expected_cadence_minutes).sum()
        ),
        "missing_expected_slots_from_exact_multiple_gaps": missing_slots,
    }


def _column_summary(frame: pd.DataFrame) -> dict[str, Any]:
    columns: dict[str, Any] = {}
    for name in frame.columns:
        raw = frame[name]
        is_timestamp = name in ("datetime", "time_tag")
        if is_timestamp:
            converted = pd.Series(dtype=np.float64)
            parse_failures = None
            nonfinite_count = None
            finite = pd.Series(dtype=np.float64)
        else:
            converted = pd.to_numeric(raw, errors="coerce")
            nonblank = raw.notna() & raw.astype(str).str.strip().ne("")
            parse_failures = int((nonblank & converted.isna()).sum())
            finite_values = converted.dropna().to_numpy(dtype=np.float64)
            nonfinite_count = int((~np.isfinite(finite_values)).sum())
            finite = converted[np.isfinite(converted)]
        is_target = name in TARGET_COLUMNS
        columns[str(name)] = {
            "dtype_in_csv_reader": str(raw.dtype),
            "numeric_checks_applicable": not is_timestamp,
            "missing_values": int(raw.isna().sum()),
            "numeric_parse_failures": parse_failures,
            "nonfinite_numeric_values": nonfinite_count,
            "negative_numeric_values_not_assumed_invalid": (
                (int((finite < 0).sum()) if len(finite) else 0)
                if not is_timestamp
                else None
            ),
            "zero_numeric_values": (
                (int((finite == 0).sum()) if len(finite) else 0)
                if not is_timestamp
                else None
            ),
            "minimum_finite_numeric_value": (
                float(finite.min()) if len(finite) else None
            ),
            "maximum_finite_numeric_value": (
                float(finite.max()) if len(finite) else None
            ),
            "nonpositive_values_in_raw_target": (
                int((finite <= 0).sum()) if is_target else None
            ),
            "script_sentinel_candidates_present": {
                str(sentinel): int((finite == sentinel).sum())
                for sentinel in SCRIPT_SENTINEL_CANDIDATES
                if not is_timestamp and bool((finite == sentinel).any())
            },
        }
    return columns


def inspect_dataframe(
    frame: pd.DataFrame,
    *,
    timestamp_column: str = "datetime",
    expected_cadence_minutes: float = EXPECTED_CADENCE_MINUTES,
) -> dict[str, Any]:
    """Summarize timestamps and values without altering or filling the frame."""
    if timestamp_column not in frame.columns:
        raise ValueError(f"Missing timestamp column: {timestamp_column}")
    timestamp = _timestamp_summary(frame[timestamp_column], expected_cadence_minutes)
    target_details = {
        column: {
            "present": column in frame.columns,
            "missing_values": (
                int(frame[column].isna().sum()) if column in frame.columns else None
            ),
            "nonpositive_values": (
                int(
                    (
                        pd.to_numeric(frame[column], errors="coerce").dropna()
                        <= 0
                    ).sum()
                )
                if column in frame.columns
                else None
            ),
        }
        for column in TARGET_COLUMNS
        if column in frame.columns
    }
    return {
        "row_count": int(len(frame)),
        "column_names": [str(column) for column in frame.columns],
        "timestamps": timestamp,
        "columns": _column_summary(frame),
        "raw_target_columns": target_details,
    }


def audit_csv(path: Path, project_root: Path) -> dict[str, Any]:
    frame = pd.read_csv(path)
    timestamp_column = next(
        (column for column in ("datetime", "time_tag") if column in frame.columns),
        None,
    )
    relative_path = path.resolve().relative_to(project_root.resolve()).as_posix()
    result: dict[str, Any] = {
        "path": relative_path,
        "file_size_bytes": int(path.stat().st_size),
        "format": "CSV",
    }
    if timestamp_column is None:
        result.update(
            {
                "audit_error": "No recognized datetime or time_tag column.",
                "row_count": int(len(frame)),
                "column_names": [str(column) for column in frame.columns],
            }
        )
        return result
    result.update(
        inspect_dataframe(frame, timestamp_column=timestamp_column)
    )
    result["timestamp_column"] = timestamp_column
    return result


def target_valid_window_starts(
    target: Sequence[float],
    *,
    partition_start: int,
    partition_end: int,
    sequence_length: int,
    prediction_length: int,
    stride: int = 1,
) -> list[int]:
    """Return starts with finite target history and forecast labels; never fill."""
    if sequence_length <= 0 or prediction_length <= 0 or stride <= 0:
        raise ValueError("Window lengths and stride must be positive.")
    if not 0 <= partition_start <= partition_end <= len(target):
        raise ValueError("Partition bounds must be valid half-open row indices.")

    starts: list[int] = []
    final_start = partition_end - sequence_length - prediction_length
    for start in range(partition_start, final_start + 1, stride):
        target_end = start + sequence_length + prediction_length
        values = np.asarray(target[start:target_end], dtype=np.float64)
        if np.isfinite(values).all():
            starts.append(start)
    return starts


def chronological_split_plan(
    timestamps: Sequence[Any],
    *,
    sequence_length: int,
    prediction_length: int,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    stride: int = 1,
    target_values: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Plan chronological, non-crossing windows; this does not make a split final."""
    parsed = pd.DatetimeIndex(pd.to_datetime(list(timestamps), errors="coerce"))
    if parsed.isna().any():
        raise ValueError("All timestamps must parse before planning a split.")
    if parsed.has_duplicates or not parsed.is_monotonic_increasing:
        raise ValueError("Split planning requires unique chronological timestamps.")
    if sequence_length <= 0 or prediction_length <= 0 or stride <= 0:
        raise ValueError("Window lengths and stride must be positive.")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Split fractions must be between zero and one.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must leave test rows.")

    row_count = len(parsed)
    train_end = int(row_count * train_fraction)
    validation_end = train_end + int(row_count * validation_fraction)
    bounds = {
        "train": (0, train_end),
        "validation": (train_end, validation_end),
        "test": (validation_end, row_count),
    }
    partitions: dict[str, Any] = {}
    for name, (start, end) in bounds.items():
        last_start = end - sequence_length - prediction_length
        windows = (
            max(0, (last_start - start) // stride + 1)
            if last_start >= start
            else 0
        )
        partitions[name] = {
            "row_start_inclusive": start,
            "row_end_exclusive": end,
            "first_timestamp": parsed[start].isoformat() if start < end else None,
            "last_timestamp": parsed[end - 1].isoformat() if start < end else None,
            "complete_non_crossing_window_count_before_missing_target_filter": windows,
        }
        if target_values is not None:
            if len(target_values) != row_count:
                raise ValueError("Target values must align one-to-one with timestamps.")
            partition_target = np.asarray(
                target_values[start:end], dtype=np.float64
            )
            partitions[name]["missing_or_nonfinite_target_rows"] = int(
                (~np.isfinite(partition_target)).sum()
            )
            partitions[name]["finite_target_feature_window_count"] = len(
                target_valid_window_starts(
                    target_values,
                    partition_start=start,
                    partition_end=end,
                    sequence_length=sequence_length,
                    prediction_length=prediction_length,
                    stride=stride,
                )
            )
    return {
        "status": "provisional_design_only_not_approved_for_training",
        "method": "chronological_row_blocks_with_only_fully_contained_windows",
        "row_count": row_count,
        "sequence_length_rows": sequence_length,
        "prediction_length_rows": prediction_length,
        "stride_rows": stride,
        "fractions": {
            "train": train_fraction,
            "validation": validation_fraction,
            "test": round(1.0 - train_fraction - validation_fraction, 12),
        },
        "leakage_policy": (
            "Each input-plus-target window must be wholly contained in one block; "
            "no window crosses a boundary. Exclude windows whose target-feature "
            "history or target horizon contains missing/nonfinite values; do not "
            "fill targets."
        ),
        "partitions": partitions,
    }


def build_report(project_root: Path) -> dict[str, Any]:
    datasets = sorted(project_root.rglob("*.csv"))
    dataset_reports = [audit_csv(path, project_root) for path in datasets]
    fine_tune_path = project_root / "DataSets" / "Kaggle_FineTuning_Dataset.csv"
    split_plan: dict[str, Any] | None = None
    if fine_tune_path.exists():
        fine_tune_frame = pd.read_csv(
            fine_tune_path, usecols=["datetime", "Electron_Flux"]
        )
        # The 288/144 choice is the package fine-tuner's candidate window, not
        # evidence that this recipe produced the configured checkpoint.
        split_plan = chronological_split_plan(
            fine_tune_frame["datetime"],
            sequence_length=288,
            prediction_length=144,
            stride=36,
            target_values=fine_tune_frame["Electron_Flux"],
        )

    return {
        "report_version": 1,
        "audit_policy": (
            "Read-only local CSV inspection. No resampling, interpolation, filling, "
            "timezone conversion, or target repair is performed. Negative values "
            "are counted but not treated as invalid without field-specific rules."
        ),
        "dataset_files_found": len(dataset_reports),
        "datasets": dataset_reports,
        "target_semantics": {
            "candidate_raw_target_columns": list(TARGET_COLUMNS),
            "electron_flux_target_definition": (
                "The 2017-18 file contains Electron_Flux and 1,166 blank entries. "
                "The training scripts log-transform this column, but the CSV has "
                "no data dictionary, instrument/channel definition, units, source "
                "identifier, or missing-value explanation. Project code uses pfu "
                "for electron flux generally, but that does not verify this file's "
                "target units or measurement semantics."
            ),
            "unit_status": (
                "Dataset-level units are undocumented. The Kaggle training script "
                "comments that Density is cm^-3 and V is km/s when calculating "
                "Pdyn; this script comment does not establish the source file's "
                "metadata. Other column units and target units are unverified."
            ),
            "missing_target_policy": "Keep missing raw targets missing; never interpolate or fill.",
            "units_and_provenance_by_dataset": {
                "DataSets/Kaggle_FineTuning_Dataset.csv": {
                    "Electron_Flux": "Not documented in this CSV; source, energy channel and units are unverified.",
                    "V": "The Kaggle training script assumes km/s when deriving Pdyn; not verified by file metadata.",
                    "Density": "The Kaggle training script assumes cm^-3 when deriving Pdyn; not verified by file metadata.",
                    "BY, BZ, AE, SYM_H, ULF_Power": "No dataset-level units/source metadata found.",
                },
                "DataSets/Kaggle_Validation_March2015.csv": {
                    "electron_flux, log_electron_flux": "No dataset-level units/source metadata found; physical flux source and log convention are not established for this file.",
                    "Flow_Speed, Proton_Density, Flow_Pressure, Bz_GSM, ULF_Power": "Training-script naming or formula assumptions exist, but this CSV has no data dictionary confirming units or source.",
                },
                "radshield/data/historical/may_2024_benchmark.csv": {
                    "flux": "The local benchmark extraction script labels its flux floor in pfu; this stored series is clipped and rolling-median smoothed, and is not raw direct observation.",
                    "log_flux": "Derived log10 of the clipped/smoothed flux series; not a raw observation.",
                    "other_columns": "Units/source fields are not documented in this CSV; extraction code includes fallback values, derived proxies and fills.",
                },
                "radshield/data/historical/oct_2024_benchmark.csv": {
                    "flux": "The local benchmark extraction script labels its flux floor in pfu; this stored series is clipped and rolling-median smoothed, and is not raw direct observation.",
                    "log_flux": "Derived log10 of the clipped/smoothed flux series; not a raw observation.",
                    "other_columns": "Units/source fields are not documented in this CSV; extraction code includes fallback values, derived proxies and fills.",
                },
            },
        },
        "training_schema_comparison": {
            "checkpoint_provenance": "unverified; neither schema is attributed to the checkpoint",
            "package_finetune_script": {
                "script": "radshield/training/train_tft.py",
                "features_in_order": list(PACKAGE_FEATURES),
                "input_length_rows": 288,
                "prediction_length_rows": 144,
                "preprocessing_observed": (
                    "log10(max(Electron_Flux, 1e-5)); direct rename of V/BZ/Density; "
                    "constant Temperature=100000; formula-derived Flow_Pressure; "
                    "row-shifted lag fields; dropna across derived frame; per-file "
                    "mean/std normalization; shuffled batches."
                ),
            },
            "root_kaggle_script": {
                "script": "radshield_kaggle_train.py",
                "features_in_order": list(KAGGLE_FEATURES),
                "input_length_rows": 288,
                "prediction_length_rows": 144,
                "preprocessing_observed": (
                    "Interpolates and forward/back-fills numeric fields; uses "
                    "zero/default/proxy fields including BY, AE, DST and F10.7; "
                    "derives MLT at an assumed -75 degree longitude; one global "
                    "scaler is fitted across pretraining and fine-tuning."
                ),
            },
            "other_training_window": {
                "script": "radshield/training/train_11yr_model.py",
                "input_length_rows": 2016,
                "prediction_length_rows": 144,
                "preprocessing_observed": (
                    "Interpolates and fills numeric columns before an 80/20 row "
                    "split, then independently computes normalization in each "
                    "dataset object; this is not shared with inference."
                ),
            },
        },
        "feature_availability": {
            "package_finetune_candidate": [
                {
                    "feature": "log_electron_flux",
                    "source_or_transform": "Electron_Flux -> log10(max(value, 1e-5))",
                    "availability": "Arithmetic transform possible for 121,810 nonmissing positive rows; source, units and blank semantics unverified.",
                },
                {
                    "feature": "Flow_Speed",
                    "source_or_transform": "V renamed directly",
                    "availability": "Column exists; unit and physical field mapping are not confirmed by dataset metadata.",
                },
                {
                    "feature": "Bz_GSM",
                    "source_or_transform": "BZ renamed directly",
                    "availability": "Column exists; GSM coordinate-frame equivalence and unit are not documented.",
                },
                {
                    "feature": "Proton_Density",
                    "source_or_transform": "Density renamed directly",
                    "availability": "Column exists; physical meaning and unit are not confirmed by dataset metadata.",
                },
                {
                    "feature": "Temperature",
                    "source_or_transform": "Constant 100000.0",
                    "availability": "Not measured in the CSV; unsupported placeholder.",
                },
                {
                    "feature": "Flow_Pressure",
                    "source_or_transform": "Formula from Density and V",
                    "availability": "Formula is reproducible only under assumed density/velocity units; the dataset does not establish them, so physical value is unverified.",
                },
                {
                    "feature": "log_flux_t-1h / log_flux_t-3h / log_flux_t-24h",
                    "source_or_transform": "Lagged transformed Electron_Flux",
                    "availability": "Requires exact timestamp-aligned observed targets 12/36/288 five-minute steps earlier; row shifting across missing target/cadence is unsafe.",
                },
                {
                    "feature": "ULF_Power",
                    "source_or_transform": "ULF_Power column consumed directly",
                    "availability": "Column exists; measurement definition, source and units are undocumented.",
                },
            ],
            "root_kaggle_candidate": [
                {
                    "feature": "log_electron_flux",
                    "source_or_transform": "log10 of Electron_Flux/electron_flux or benchmark flux",
                    "availability": "Transform can be computed where raw values exist, but source equivalence across datasets and units are unverified.",
                },
                {
                    "feature": "BY_GSM / BZ_GSM / Vsw",
                    "source_or_transform": "BY/BZ/V renamed where present",
                    "availability": "Potential raw columns exist in the fine-tuning CSV; coordinate frame, units and provenance are not documented.",
                },
                {
                    "feature": "Pdyn",
                    "source_or_transform": "Uses Pdyn/Flow_Pressure if present, otherwise derives from Density and V",
                    "availability": "Requires field unit and formula validation; no verified physical equivalence is available.",
                },
                {
                    "feature": "AE",
                    "source_or_transform": "AE column or default/proxy",
                    "availability": "AE column exists in the fine-tuning CSV, but unit/source are undocumented; proxy/default is not observation.",
                },
                {
                    "feature": "DST",
                    "source_or_transform": "DST/Dst column or default",
                    "availability": "No DST/Dst column in the fine-tuning CSV; SYM_H is not established as equivalent. Default -10 is not an observation.",
                },
                {
                    "feature": "F10.7_index",
                    "source_or_transform": "F10.7_index/F10.7 column or default",
                    "availability": "No corresponding field in the fine-tuning CSV; default 70 is not an observation.",
                },
                {
                    "feature": "MLT_sin / MLT_cos",
                    "source_or_transform": "Timestamp-derived MLT at assumed -75 degree longitude",
                    "availability": "Numerically derivable from timestamps, but satellite longitude and intended semantics are assumptions.",
                },
            ],
            "potentially_derivable_without_external_data": [
                "A log transform from positive observed flux values is arithmetically reproducible, but units, source and target semantics remain unknown; missing measurements remain missing.",
                "Timestamp-derived calendar features can be calculated from parseable timestamps, but the assumed satellite longitude and feature meaning must be stated.",
                "Lag values can be derived only where the exact prior timestamp exists and the source target is observed; row shifts across missing cadence or missing target are invalid.",
            ],
            "not_verified_or_requires_assumptions": [
                "Electron_Flux source/instrument/channel/units and blank-value semantics.",
                "Meaning, units, source and sentinel semantics for most training columns, including ULF_Power.",
                "Temperature=100000 and zero/default AE, DST, F10.7, BY values are substitutions/proxies, not observed measurements.",
                "Flow_Pressure/Pdyn equivalence requires verified input units and formula conventions.",
                "March 2015 validation source and its relationship to any pretraining data.",
                "May/October 2024 benchmark provenance; included files contain derived feature fields and are not established as independent raw telemetry.",
            ],
        },
        "referenced_but_missing_inputs": [
            "Kaggle_PreTraining_Dataset.csv",
            "DataSets/Kaggle_PreTraining_Dataset.csv",
            "radshield/data/archive_11yr.csv",
            "radshield/data/archive_11yr_goes_grasp.csv",
            "radshield/data/grasp_2yr.csv",
        ],
        "synthetic_or_nonraw_evaluation_artifacts": [
            {
                "path": "radshield/data/historical/goes_train_val_test_eval.json",
                "status": (
                    "Not treated as a dataset or evaluation result: "
                    "scripts/generate_eval_dataset.py constructs calibrated "
                    "prediction series and synthetic training observations."
                ),
            }
        ],
        "provisional_split_design": split_plan,
        "training_readiness": {
            "status": "not_ready",
            "reasons": [
                "Target source, physical definition, units and missing-value semantics are undocumented.",
                "The required pretraining archive is absent.",
                "Conflicting feature schemas and preprocessing recipes remain.",
                "Existing scripts interpolate or substitute values; a no-target-fill, provenance-valid pipeline is not established.",
                "The split plan is provisional and cannot be approved until target/feature schema and source dependencies are resolved.",
            ],
            "checkpoint_or_model_weights_modified": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("radshield/data/historical/dataset_audit.json"),
        help="Output JSON path, relative to the current working directory by default.",
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    output_path = args.output
    if not output_path.is_absolute():
        output_path = project_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = build_report(project_root)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Dataset audit written: {output_path}")
    print(f"CSV datasets audited: {report['dataset_files_found']}")
    print(f"Training readiness: {report['training_readiness']['status']}")


if __name__ == "__main__":
    main()
