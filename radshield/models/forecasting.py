"""Validated-input wrapper for the existing RadShield TFT checkpoint."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from radshield.models.tft_model import build_tft


TFT_INPUT_STEPS = 288
TFT_OUTPUT_STEPS = 144
TFT_SAMPLE_INTERVAL = pd.Timedelta(minutes=5)
TFT_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
TFT_FEATURE_COLUMNS = (
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
TFT_PACKAGE_FINETUNE_FEATURE_COLUMNS = (
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
TFT_DASHBOARD_TRAINING_FEATURE_COLUMNS = TFT_FEATURE_COLUMNS
TFT_FEATURE_SCHEMA_MATCHES_PACKAGE_FINETUNE = (
    TFT_FEATURE_COLUMNS == TFT_PACKAGE_FINETUNE_FEATURE_COLUMNS
)
TFT_CHECKPOINT_TRAINING_PROVENANCE_VERIFIED = False
TFT_CHECKPOINT_SCALER_PROVENANCE_VERIFIED = False
TFT_PREPROCESSING_MATCHES_CHECKPOINT = False
TFT_INFERENCE_SUPPORTED = (
    TFT_CHECKPOINT_TRAINING_PROVENANCE_VERIFIED
    and TFT_CHECKPOINT_SCALER_PROVENANCE_VERIFIED
    and TFT_PREPROCESSING_MATCHES_CHECKPOINT
)
TFT_PACKAGE_FINETUNE_FEATURE_AUDIT = (
    {
        "Feature": "log_electron_flux",
        "Physical meaning": "Log-transformed electron flux",
        "Unit": "log10(dataset value); physical flux unit not documented",
        "Package fine-tune source": "Electron_Flux column",
        "Training transformation": "log10(max(Electron_Flux, 1e-5)); target and input",
        "Dashboard availability": "GOES >=2 MeV flux is present, but raw-vs-filled history and dataset unit equivalence are not established",
    },
    {
        "Feature": "Flow_Speed",
        "Physical meaning": "Solar-wind flow speed",
        "Unit": "Not documented in training CSV; NOAA Vsw is km/s",
        "Package fine-tune source": "V column",
        "Training transformation": "Direct rename; no scaling until feature standardization",
        "Dashboard availability": "NOAA observed_Vsw exists; source feature units/provenance equivalence not verified",
    },
    {
        "Feature": "Bz_GSM",
        "Physical meaning": "GSM-frame interplanetary magnetic-field Bz",
        "Unit": "Not documented in training CSV",
        "Package fine-tune source": "BZ column",
        "Training transformation": "Direct rename; no scaling until feature standardization",
        "Dashboard availability": "NOAA observed_BZ_GSM exists; source equivalence not verified",
    },
    {
        "Feature": "Proton_Density",
        "Physical meaning": "Solar-wind proton number density",
        "Unit": "Not documented in training CSV",
        "Package fine-tune source": "Density column",
        "Training transformation": "Direct rename; no scaling until feature standardization",
        "Dashboard availability": "NOAA observed_Np exists; source equivalence not verified",
    },
    {
        "Feature": "Temperature",
        "Physical meaning": "Solar-wind temperature placeholder",
        "Unit": "Not documented",
        "Package fine-tune source": "No source column; hard-coded constant",
        "Training transformation": "Set to 100000.0 for every row",
        "Dashboard availability": "Unavailable; no measured temperature feature in the dashboard input frame",
    },
    {
        "Feature": "Flow_Pressure",
        "Physical meaning": "Dynamic solar-wind pressure",
        "Unit": "Formula intends nPa; source density/speed units are not documented",
        "Package fine-tune source": "Derived from Density and V",
        "Training transformation": "0.5*m_p*n*V^2 with unit conversion; no clipping",
        "Dashboard availability": "Pdyn is derived from NOAA Np/Vsw but clipped to 0.1–50 nPa; source semantics are not verified identical",
    },
    {
        "Feature": "log_flux_t-1h",
        "Physical meaning": "One-hour prior electron-flux value",
        "Unit": "log10(dataset flux value); physical flux unit not documented",
        "Package fine-tune source": "Electron_Flux history",
        "Training transformation": "Shift log_electron_flux by 12 rows (nominal five-minute cadence)",
        "Dashboard availability": "Derivable only from an adequate, gap-free direct flux history; not present as this feature",
    },
    {
        "Feature": "log_flux_t-3h",
        "Physical meaning": "Three-hour prior electron-flux value",
        "Unit": "log10(dataset flux value); physical flux unit not documented",
        "Package fine-tune source": "Electron_Flux history",
        "Training transformation": "Shift log_electron_flux by 36 rows (nominal five-minute cadence)",
        "Dashboard availability": "Derivable only from an adequate, gap-free direct flux history; not present as this feature",
    },
    {
        "Feature": "log_flux_t-24h",
        "Physical meaning": "Twenty-four-hour prior electron-flux value",
        "Unit": "log10(dataset flux value); physical flux unit not documented",
        "Package fine-tune source": "Electron_Flux history",
        "Training transformation": "Shift log_electron_flux by 288 rows (nominal five-minute cadence)",
        "Dashboard availability": "Derivable only from an adequate, gap-free direct flux history; not present as this feature",
    },
    {
        "Feature": "ULF_Power",
        "Physical meaning": "Ultra-low-frequency power",
        "Unit": "Not documented",
        "Package fine-tune source": "ULF_Power column",
        "Training transformation": "Consumed directly; no conversion in package fine-tuner",
        "Dashboard availability": "Dashboard ULF_power is a GOES-Hp/Bz variance proxy, not verified equivalent to training ULF_Power",
    },
)
TFT_REQUIRED_SOURCE_COLUMNS = (
    "log_flux",
    "BY_GSM",
    "BZ_GSM",
    "Pdyn",
    "Vsw",
    "AE",
    "DST",
    "F10.7_index",
)
EXPERIMENTAL_FLUX_THRESHOLDS_PFU = (1_000.0, 10_000.0)


class ForecastInputError(ValueError):
    """Raised when telemetry cannot safely be passed to the TFT."""


@dataclass(frozen=True)
class TFTForecast:
    quantiles_log10_pfu: np.ndarray
    forecast_timestamps_utc: pd.DatetimeIndex
    generated_at_utc: datetime
    input_end_timestamp_utc: datetime
    direct_flux_observation_count: int | None
    latest_direct_flux_timestamp_utc: datetime | None
    direct_flux_age_minutes: float | None
    missing_direct_input_counts: dict[str, int]
    warnings: tuple[str, ...]


def calculate_mlt_vectorized(
    dt_index: pd.DatetimeIndex,
    satellite_lon: float,
) -> np.ndarray:
    """Calculate the model's magnetic-local-time sine/cosine inputs."""
    dt_index = pd.DatetimeIndex(dt_index)
    if dt_index.tz is None:
        raise ForecastInputError("Input timestamps must be timezone-aware.")
    dt_index = dt_index.tz_convert("UTC")

    ut_hours = (
        dt_index.hour
        + dt_index.minute / 60.0
        + dt_index.second / 3600.0
    )
    day_of_year = dt_index.day_of_year
    year_fraction = (2 * np.pi / 365.0) * (
        day_of_year - 1 + (ut_hours - 12.0) / 24.0
    )
    equation_of_time = 229.18 * (
        0.000075
        + 0.001868 * np.cos(year_fraction)
        - 0.032077 * np.sin(year_fraction)
        - 0.014615 * np.cos(2 * year_fraction)
        - 0.040849 * np.sin(2 * year_fraction)
    )
    subsolar_longitude = -15.0 * (
        ut_hours - 12.0 + equation_of_time / 60.0
    )
    magnetic_local_time = ut_hours + (satellite_lon - subsolar_longitude) / 15.0
    return np.mod(magnetic_local_time, 24.0)


def load_tft_checkpoint(
    checkpoint_path: str | Path,
) -> tuple[Any | None, dict[str, Any]]:
    """Strictly load the configured 10-feature/5-quantile checkpoint."""
    path = Path(checkpoint_path)
    status = {
        "checkpoint": str(path),
        "available": False,
        "inference_supported": TFT_INFERENCE_SUPPORTED,
        "training_provenance_verified": TFT_CHECKPOINT_TRAINING_PROVENANCE_VERIFIED,
        "scaler_provenance_verified": TFT_CHECKPOINT_SCALER_PROVENANCE_VERIFIED,
        "training_metadata_present": False,
        "architecture_compatible": False,
        "error": None,
        "inference_error": None,
    }
    if not TFT_INFERENCE_SUPPORTED:
        status["inference_error"] = (
            "Inference disabled: checkpoint training and scaler provenance are unverified. "
            "Multiple repository pipelines write this checkpoint name with conflicting feature "
            "schemas and preprocessing; the tensor-only checkpoint and unlabeled scaler cannot "
            "identify which pipeline produced them."
        )
    try:
        model = build_tft(num_features=len(TFT_FEATURE_COLUMNS), num_quantiles=5)
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        status["training_metadata_present"] = bool(
            isinstance(state_dict, dict)
            and any(not isinstance(value, torch.Tensor) for value in state_dict.values())
        )
        model.load_state_dict(state_dict, strict=True)
        model.eval()
    except (OSError, RuntimeError, ValueError, TypeError, ImportError) as exc:
        status["error"] = str(exc)
        return None, status

    status["available"] = True
    status["architecture_compatible"] = True
    status["error"] = None
    return model, status


def _prepare_input_matrix(
    frame: pd.DataFrame,
    satellite_lon: float,
) -> tuple[np.ndarray, dict[str, int], list[str], int | None, datetime | None]:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ForecastInputError("TFT input must have a DatetimeIndex.")
    if frame.index.tz is None:
        raise ForecastInputError("TFT input timestamps must be timezone-aware.")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ForecastInputError("TFT input timestamps must be increasing and unique.")
    if len(frame) < TFT_INPUT_STEPS:
        raise ForecastInputError(
            f"TFT requires {TFT_INPUT_STEPS} five-minute samples "
            f"(24 hours); only {len(frame)} are available."
        )

    history = frame.iloc[-TFT_INPUT_STEPS:].copy()
    sample_deltas = history.index.to_series().diff().dropna()
    if not sample_deltas.eq(TFT_SAMPLE_INTERVAL).all():
        raise ForecastInputError(
            "The most recent 24-hour input window is not a complete five-minute sequence."
        )

    missing_source_columns = [
        name for name in TFT_REQUIRED_SOURCE_COLUMNS if name not in history.columns
    ]
    if missing_source_columns:
        raise ForecastInputError(
            "Missing TFT source feature columns: " + ", ".join(missing_source_columns)
        )

    feature_frame = pd.DataFrame(index=history.index)
    feature_frame["log_electron_flux"] = pd.to_numeric(
        history["log_flux"], errors="coerce"
    ).rolling(window=24, min_periods=1, center=False).median()
    for column in TFT_REQUIRED_SOURCE_COLUMNS[1:]:
        feature_frame[column] = pd.to_numeric(history[column], errors="coerce")

    mlt = calculate_mlt_vectorized(history.index, satellite_lon)
    feature_frame["MLT_sin"] = np.sin(mlt * 2.0 * np.pi / 24.0)
    feature_frame["MLT_cos"] = np.cos(mlt * 2.0 * np.pi / 24.0)
    data_matrix = feature_frame.loc[:, TFT_FEATURE_COLUMNS].to_numpy(dtype=np.float64)

    if not np.isfinite(data_matrix).all():
        bad_columns = feature_frame.columns[
            ~np.isfinite(data_matrix).all(axis=0)
        ].tolist()
        raise ForecastInputError(
            "TFT source features contain missing or non-finite values: "
            + ", ".join(bad_columns)
        )

    warnings: list[str] = []
    missing_direct_counts: dict[str, int] = {}
    raw_columns = {
        "log_electron_flux": "observed_flux",
        "BY_GSM": "observed_BY_GSM",
        "BZ_GSM": "observed_BZ_GSM",
        "Vsw": "observed_Vsw",
    }
    for feature_name, raw_column in raw_columns.items():
        if raw_column in history:
            raw_values = pd.to_numeric(history[raw_column], errors="coerce")
            missing_count = int(raw_values.isna().sum())
            missing_direct_counts[feature_name] = missing_count
            if missing_count:
                warnings.append(
                    f"{feature_name}: {missing_count} of {TFT_INPUT_STEPS} "
                    "direct input observations are missing; the existing model-ready "
                    "feature pipeline may have filled them."
                )
    if "observed_flux" in history:
        observed_flux = pd.to_numeric(history["observed_flux"], errors="coerce")
        valid_flux = observed_flux.dropna()
        if valid_flux.empty:
            raise ForecastInputError(
                "No direct GOES flux observations exist in the 24-hour TFT input window."
            )
        latest_flux_time = valid_flux.index[-1].to_pydatetime().astimezone(timezone.utc)
        flux_count: int | None = int(valid_flux.size)
    else:
        latest_flux_time = None
        flux_count = None
        warnings.append(
            "Raw flux provenance is unavailable for this processed historical dataset."
        )

    mean = np.mean(data_matrix, axis=0, keepdims=True)
    std = np.maximum(np.std(data_matrix, axis=0, keepdims=True), 1e-2)
    mean[:, 0] = 0.0
    std[:, 0] = 1.0
    normalized = ((data_matrix - mean) / std).astype(np.float32)
    return (
        normalized,
        missing_direct_counts,
        warnings,
        flux_count,
        latest_flux_time,
    )


def run_tft_forecast(
    model: Any,
    frame: pd.DataFrame,
    satellite_lon: float,
    generated_at: datetime | None = None,
) -> TFTForecast:
    """Run the loaded TFT and retain its raw output quantiles without blending."""
    if not TFT_INFERENCE_SUPPORTED:
        raise ForecastInputError(
            "TFT inference is disabled because checkpoint training and scaler "
            "provenance have not been verified."
        )
    if model is None:
        raise ForecastInputError("TFT checkpoint is unavailable.")
    if generated_at is None:
        generated_at = datetime.now(timezone.utc)
    elif generated_at.tzinfo is None:
        raise ForecastInputError("Forecast generation timestamp must include a timezone.")
    else:
        generated_at = generated_at.astimezone(timezone.utc)

    (
        normalized,
        missing_direct_counts,
        warnings,
        direct_flux_count,
        latest_direct_flux_time,
    ) = _prepare_input_matrix(frame, satellite_lon)
    tensor = torch.from_numpy(normalized).unsqueeze(0)
    try:
        model.eval()
        with torch.inference_mode():
            output, _attention = model(tensor)
    except (RuntimeError, ValueError, TypeError, IndexError) as exc:
        raise ForecastInputError(f"TFT inference failed: {exc}") from exc

    if not isinstance(output, torch.Tensor) or tuple(output.shape) != (
        1,
        TFT_OUTPUT_STEPS,
        len(TFT_QUANTILES),
    ):
        shape = tuple(output.shape) if hasattr(output, "shape") else type(output).__name__
        raise ForecastInputError(
            f"TFT output shape must be (1, {TFT_OUTPUT_STEPS}, "
            f"{len(TFT_QUANTILES)}); received {shape}."
        )
    quantiles = output.squeeze(0).detach().cpu().numpy()
    if not np.isfinite(quantiles).all():
        raise ForecastInputError("TFT output contains non-finite quantile values.")

    latest_input_time = frame.index[-1].to_pydatetime().astimezone(timezone.utc)
    forecast_timestamps = pd.date_range(
        start=frame.index[-1] + TFT_SAMPLE_INTERVAL,
        periods=TFT_OUTPUT_STEPS,
        freq=TFT_SAMPLE_INTERVAL,
    ).tz_convert("UTC")
    flux_age_minutes = (
        (generated_at - latest_direct_flux_time).total_seconds() / 60.0
        if latest_direct_flux_time is not None
        else None
    )
    if flux_age_minutes is not None and flux_age_minutes < 0:
        warnings.append("Latest direct GOES flux timestamp is later than generation time.")

    return TFTForecast(
        quantiles_log10_pfu=quantiles,
        forecast_timestamps_utc=forecast_timestamps,
        generated_at_utc=generated_at,
        input_end_timestamp_utc=latest_input_time,
        direct_flux_observation_count=direct_flux_count,
        latest_direct_flux_timestamp_utc=latest_direct_flux_time,
        direct_flux_age_minutes=flux_age_minutes,
        missing_direct_input_counts=missing_direct_counts,
        warnings=tuple(warnings),
    )


def classify_experimental_flux_screening(flux_pfu: float) -> str:
    """Return project heuristic screening only; this is not an operational safety alert."""
    if not math.isfinite(flux_pfu) or flux_pfu < 0:
        raise ValueError("flux_pfu must be a finite, non-negative value.")
    if flux_pfu > EXPERIMENTAL_FLUX_THRESHOLDS_PFU[1]:
        return "HIGH — experimental project heuristic"
    if flux_pfu > EXPERIMENTAL_FLUX_THRESHOLDS_PFU[0]:
        return "ELEVATED — experimental project heuristic"
    return "BELOW PROJECT SCREENING LEVELS — not a safety determination"
