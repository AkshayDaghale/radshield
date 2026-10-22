"""Prepare and evaluate satellite-specific TFTs from quality-flagged CDAWeb data."""

from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from radshield.data.cdaweb_archive import GOES_DATASETS
from radshield.models.tft_model import PhysicsInformedPinballLoss, build_tft


INPUT_STEPS = 288
OUTPUT_STEPS = 144
FEATURE_COLUMNS = (
    "log_electron_flux",
    "BY_GSM",
    "BZ_GSM",
    "flow_speed",
    "proton_density",
    "Pressure",
    "AE_INDEX",
    "SYM_H",
    "log_flux_lag_1h",
    "log_flux_lag_3h",
)
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


def _read_monthly_csv(directory: Path) -> pd.DataFrame:
    paths = sorted(directory.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No normalized CDAWeb CSV files found in {directory}.")
    frames = [pd.read_csv(path, parse_dates=["timestamp_utc"]) for path in paths]
    frame = pd.concat(frames, ignore_index=True)
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    return frame.sort_values("timestamp_utc").drop_duplicates(
        "timestamp_utc", keep="last"
    )


def prepare_satellite_dataset(
    archive_dir: str | Path,
    satellite: str,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Join measured GOES target and time-shifted OMNI predictors on a 5-minute grid."""
    if satellite not in GOES_DATASETS:
        raise ValueError(
            f"Unsupported satellite {satellite!r}; choose one of {tuple(GOES_DATASETS)}."
        )
    root = Path(archive_dir)
    goes_product = f"goes_{satellite.lower().replace('-', '')}"
    goes = _read_monthly_csv(root / "csv" / goes_product)
    omni = _read_monthly_csv(root / "csv" / "omni_hro_5min")

    goes["timestamp_utc"] = goes["timestamp_utc"].dt.floor("5min")
    goes["electron_flux_pfu"] = pd.to_numeric(
        goes["electron_flux_pfu"], errors="coerce"
    )
    goes["quality_flag"] = pd.to_numeric(goes["quality_flag"], errors="coerce")
    goes.loc[
        (goes["quality_flag"] != 0)
        | ~np.isfinite(goes["electron_flux_pfu"])
        | (goes["electron_flux_pfu"] <= 0),
        "electron_flux_pfu",
    ] = np.nan
    flux = (
        goes.groupby("timestamp_utc", as_index=False)
        .agg(
            electron_flux_pfu=("electron_flux_pfu", "mean"),
            valid_goes_minutes=("electron_flux_pfu", "count"),
        )
        .sort_values("timestamp_utc")
    )

    omni["timestamp_utc"] = omni["timestamp_utc"].dt.floor("5min")
    numeric_columns = (
        "BY_GSM",
        "BZ_GSM",
        "Pressure",
        "flow_speed",
        "proton_density",
        "AE_INDEX",
        "SYM_H",
        "IMF",
        "PLS",
        "percent_interp",
    )
    for column in numeric_columns:
        omni[column] = pd.to_numeric(omni[column], errors="coerce")
    omni = omni.replace([np.inf, -np.inf], np.nan)
    omni.loc[(omni["IMF"] == 99) | (omni["PLS"] == 99), list(FEATURE_COLUMNS[1:8])] = np.nan
    omni = omni.groupby("timestamp_utc", as_index=False).mean(numeric_only=True)

    frame = flux.merge(omni, on="timestamp_utc", how="left", validate="one_to_one")
    frame["log_electron_flux"] = np.where(
        frame["electron_flux_pfu"] > 0,
        np.log10(frame["electron_flux_pfu"]),
        np.nan,
    )
    frame["log_flux_lag_1h"] = frame["log_electron_flux"].shift(12)
    frame["log_flux_lag_3h"] = frame["log_electron_flux"].shift(36)
    frame["satellite"] = satellite
    frame["target_source"] = GOES_DATASETS[satellite]
    frame = frame.set_index("timestamp_utc").sort_index()

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination, index_label="timestamp_utc")
    return frame


def _make_windows(
    frame: pd.DataFrame,
    start: int,
    stop: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = frame.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    target = frame["log_electron_flux"].to_numpy(dtype=np.float32)
    timestamps_ns = frame.index.as_unit("ns").asi8
    sample_ns = 5 * 60 * 1_000_000_000
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    input_end_times: list[np.datetime64] = []
    last_start = min(stop - INPUT_STEPS - OUTPUT_STEPS, len(frame) - INPUT_STEPS - OUTPUT_STEPS)
    for window_start in range(start, last_start + 1, stride):
        input_end = window_start + INPUT_STEPS
        target_end = input_end + OUTPUT_STEPS
        x = features[window_start:input_end]
        y = target[input_end:target_end]
        if not np.all(np.diff(timestamps_ns[window_start:target_end]) == sample_ns):
            continue
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            continue
        xs.append(x)
        ys.append(y)
        input_end_times.append(frame.index[input_end].to_datetime64())
    if not xs:
        return (
            np.empty((0, INPUT_STEPS, len(FEATURE_COLUMNS)), dtype=np.float32),
            np.empty((0, OUTPUT_STEPS), dtype=np.float32),
            np.empty((0,), dtype="datetime64[ns]"),
        )
    return np.stack(xs), np.stack(ys), np.asarray(input_end_times)


def _fit_input_scaler(training_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flattened = training_x.reshape(-1, training_x.shape[-1]).astype(np.float64)
    mean = flattened.mean(axis=0)
    std = flattened.std(axis=0)
    mean[0] = 0.0
    std[0] = 1.0
    std = np.maximum(std, 1e-6)
    return mean.astype(np.float32), std.astype(np.float32)


def _normalize(
    values: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    normalized = (values - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
    return normalized.astype(np.float32)


def _evaluate(
    model: torch.nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    if not len(x):
        raise ValueError("The held-out test partition has no complete input/target windows.")
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(x), 32):
            tensor = torch.from_numpy(x[start : start + 32]).to(device)
            output, _ = model(tensor)
            predictions.append(output.cpu().numpy())
    pred = np.sort(np.concatenate(predictions, axis=0), axis=-1)
    horizon_indices = {"30m": 5, "6h": 71, "12h": 143}
    metrics: dict[str, Any] = {"window_count": int(len(x)), "horizons": {}}
    for label, index in horizon_indices.items():
        truth = y[:, index]
        median = pred[:, index, 2]
        persistence = x[:, -1, 0]
        metrics["horizons"][label] = {
            "mae_log10_pfu": float(np.mean(np.abs(median - truth))),
            "persistence_mae_log10_pfu": float(
                np.mean(np.abs(persistence - truth))
            ),
            "p10_to_p90_coverage": float(
                np.mean((truth >= pred[:, index, 0]) & (truth <= pred[:, index, 4]))
            ),
            "median_predicted_flux_pfu": float(np.median(np.power(10.0, median))),
            "median_observed_flux_pfu": float(np.median(np.power(10.0, truth))),
        }
    return metrics


def train_archive_tft(
    archive_dir: str | Path,
    satellite: str,
    output_dir: str | Path = "radshield/weights/archive_tft",
    epochs: int = 3,
    window_stride: int = 12,
    hidden_size: int = 32,
    batch_size: int = 16,
    seed: int = 42,
) -> dict[str, Any]:
    """Train one GOES-specific model and save its scaler and reproducible run manifest."""
    if epochs < 1 or window_stride < 1 or hidden_size < 8 or batch_size < 1 or seed < 0:
        raise ValueError(
            "epochs, stride, batch size, and hidden size must be positive; "
            "seed must be non-negative."
        )
    archive_manifest_path = Path(archive_dir) / "archive_manifest.json"
    if not archive_manifest_path.is_file():
        raise FileNotFoundError(
            f"Archive provenance manifest not found: {archive_manifest_path}."
        )
    torch.manual_seed(seed)
    np.random.seed(seed)
    frame = prepare_satellite_dataset(
        archive_dir,
        satellite,
        Path(archive_dir) / "prepared" / f"{satellite.lower().replace('-', '')}.csv",
    )
    total = len(frame)
    train_end = int(total * 0.70)
    validation_end = int(total * 0.85)
    train_x, train_y, _ = _make_windows(frame, 0, train_end, window_stride)
    validation_x, validation_y, _ = _make_windows(
        frame, train_end, validation_end, window_stride
    )
    test_x, test_y, _ = _make_windows(
        frame, validation_end, total, window_stride
    )
    if not len(train_x) or not len(validation_x) or not len(test_x):
        raise ValueError(
            "Insufficient complete 5-minute windows for chronological 70/15/15 "
            f"training/validation/test partitions: {len(train_x)}/{len(validation_x)}/{len(test_x)}."
        )

    mean, std = _fit_input_scaler(train_x)
    train_x = _normalize(train_x, mean, std)
    validation_x = _normalize(validation_x, mean, std)
    test_x = _normalize(test_x, mean, std)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_tft(
        num_features=len(FEATURE_COLUMNS),
        hidden_size=hidden_size,
        lstm_layers=1,
        num_quantiles=len(QUANTILES),
        dropout=0.1,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = PhysicsInformedPinballLoss(
        quantiles=list(QUANTILES), lambda_physics=0.05
    )
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    validation_loader = DataLoader(
        TensorDataset(torch.from_numpy(validation_x), torch.from_numpy(validation_y)),
        batch_size=batch_size,
        shuffle=False,
    )
    best_validation_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epoch_results: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_total = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions, _ = model(x_batch)
            loss = criterion(predictions, y_batch)
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite training loss at epoch {epoch}.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_total += float(loss.detach()) * len(x_batch)

        model.eval()
        validation_loss_total = 0.0
        with torch.inference_mode():
            for x_batch, y_batch in validation_loader:
                predictions, _ = model(x_batch.to(device))
                loss = criterion(predictions, y_batch.to(device))
                validation_loss_total += float(loss) * len(x_batch)
        epoch_metrics = {
            "epoch": epoch,
            "train_pinball_loss": train_loss_total / len(train_x),
            "validation_pinball_loss": validation_loss_total / len(validation_x),
        }
        epoch_results.append(epoch_metrics)
        if epoch_metrics["validation_pinball_loss"] < best_validation_loss:
            best_validation_loss = epoch_metrics["validation_pinball_loss"]
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training did not produce a validation checkpoint.")
    model.load_state_dict(best_state)
    model.to(device)
    test_metrics = _evaluate(model, test_x, test_y, device)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prefix = satellite.lower().replace("-", "")
    checkpoint_path = output / f"{prefix}_tft.pth"
    scaler_path = output / f"{prefix}_scaler.npz"
    torch.save(best_state, checkpoint_path)
    np.savez(scaler_path, feature_names=np.asarray(FEATURE_COLUMNS), mean=mean, std=std)
    manifest: dict[str, Any] = {
        "status": "research_evaluation_only",
        "satellite": satellite,
        "random_seed": seed,
        "target_source": GOES_DATASETS[satellite],
        "target_variable": "E2W_COR_FLUX",
        "target_definition": "GOES EPEAD E2 integral electron flux >2 MeV, corrected for background, contamination, and dead time",
        "target_units": "electrons/(cm^2 s sr)",
        "target_quality_rule": "Only E2W_DQF == 0, finite, positive observed flux is admitted. Missing target windows are omitted, never filled.",
        "predictor_source": "NASA SPDF CDAWeb OMNI_HRO_5MIN, time-shifted to the bow-shock nose; direct WIND SWE/MFI CDF files are archived alongside it.",
        "predictor_quality_policy": "CDF fill values are converted to missing; incomplete windows are excluded without interpolation.",
        "feature_columns": list(FEATURE_COLUMNS),
        "input_steps": INPUT_STEPS,
        "output_steps": OUTPUT_STEPS,
        "sample_interval_minutes": 5,
        "quantiles": list(QUANTILES),
        "architecture": {
            "num_features": len(FEATURE_COLUMNS),
            "hidden_size": hidden_size,
            "lstm_layers": 1,
            "num_quantiles": len(QUANTILES),
            "dropout": 0.1,
        },
        "chronological_split": {
            "train_end_row": train_end,
            "validation_end_row": validation_end,
            "total_rows": total,
            "first_timestamp_utc": frame.index[0].isoformat(),
            "last_timestamp_utc": frame.index[-1].isoformat(),
            "training_windows": int(len(train_x)),
            "validation_windows": int(len(validation_x)),
            "test_windows": int(len(test_x)),
            "window_stride_rows": window_stride,
        },
        "provenance": {
            "archive_manifest": archive_manifest_path.name,
            "archive_manifest_sha256": hashlib.sha256(
                archive_manifest_path.read_bytes()
            ).hexdigest(),
            "prepared_dataset": Path("prepared", f"{prefix}.csv").as_posix(),
            "prepared_dataset_sha256": hashlib.sha256(
                (Path(archive_dir) / "prepared" / f"{prefix}.csv").read_bytes()
            ).hexdigest(),
            "training_script_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        },
        "epochs": epoch_results,
        "held_out_test_metrics": test_metrics,
        "artifacts": {
            "checkpoint": checkpoint_path.name,
            "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            "scaler": scaler_path.name,
            "scaler_sha256": hashlib.sha256(scaler_path.read_bytes()).hexdigest(),
        },
        "device": str(device),
        "limitations": [
            "Hourly training windows overlap substantially; held-out metrics are not independent event samples.",
            "This model is satellite-specific and has not been validated against GSAT-19/GRASP measurements.",
            "Held-out historical performance is not a forecast-operations safety approval.",
            "GOES mission changes and source coverage must be assessed before deployment to another spacecraft.",
        ],
    }
    manifest_path = output / f"{prefix}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and chronologically evaluate a satellite-specific GOES TFT."
    )
    parser.add_argument("--archive", type=Path, default=Path("radshield/data/archive"))
    parser.add_argument("--satellite", choices=tuple(GOES_DATASETS), required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("radshield/weights/archive_tft")
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = train_archive_tft(
        archive_dir=args.archive,
        satellite=args.satellite,
        output_dir=args.output,
        epochs=args.epochs,
        window_stride=args.stride,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
