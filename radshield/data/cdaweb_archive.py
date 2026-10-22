"""Acquire quality-flagged GOES electron and WIND solar-wind CDF archives."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
import hashlib
import json
import os
import argparse
from pathlib import Path
import tempfile
from typing import Any
import xml.etree.ElementTree as ET

from cdflib import CDF, cdfepoch
import numpy as np
import pandas as pd
import requests


CDAS_DATA_URL = (
    "https://cdaweb.gsfc.nasa.gov/WS/cdasr/1/dataviews/"
    "sp_phys/datasets/{dataset}/data/{start},{end}/{variables}?format=cdf"
)
GOES_DATASETS = {
    "GOES-13": "GOES13_EPEAD-SCIENCE-ELECTRONS-E13EW_1MIN",
    "GOES-14": "GOES14_EPEAD-SCIENCE-ELECTRONS-E13EW_1MIN",
    "GOES-15": "GOES15_EPEAD-SCIENCE-ELECTRONS-E13EW_1MIN",
}
GOES_VARIABLES = "E2W_COR_FLUX,E2W_DQF"
WIND_PRODUCTS = {
    "wind_swe": {
        "dataset": "WI_K0_SWE",
        "variables": "V_GSM,Np,Pressure,QF_V,QF_Np",
    },
    "wind_mfi": {"dataset": "WI_H0_MFI", "variables": "BGSM"},
}
OMNI_PRODUCT = {
    "dataset": "OMNI_HRO_5MIN",
    "variables": (
        "BY_GSM,BZ_GSM,Pressure,flow_speed,proton_density,"
        "AE_INDEX,SYM_H,IMF,PLS,percent_interp"
    ),
}
USER_AGENT = "RadShield research archive client/1.0"


@dataclass(frozen=True)
class ArchiveRequest:
    product: str
    dataset: str
    variables: str
    start: date
    end: date
    satellite: str | None = None

    @property
    def stem(self) -> str:
        return f"{self.product}_{self.start:%Y%m%d}_{self.end:%Y%m%d}"


def _month_starts(start: date, end: date) -> list[date]:
    month = date(start.year, start.month, 1)
    months: list[date] = []
    while month < end:
        months.append(month)
        month = date(month.year + (month.month == 12), month.month % 12 + 1, 1)
    return months


def _request_url(request: ArchiveRequest) -> str:
    return CDAS_DATA_URL.format(
        dataset=request.dataset,
        start=f"{request.start:%Y%m%d}T000000Z",
        end=f"{request.end:%Y%m%d}T000000Z",
        variables=request.variables,
    )


def _cdf_download_url(response_content: bytes) -> str | None:
    root = ET.fromstring(response_content)
    file_description = root.find(".//{*}FileDescription")
    if file_description is None:
        return None
    return file_description.findtext("{*}Name")


def _write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as temporary:
        temporary.write(content)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def _archive_request(
    request: ArchiveRequest,
    output_dir: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    cdf_path = output_dir / "cdf" / request.product / f"{request.stem}.cdf"
    if cdf_path.is_file() and cdf_path.stat().st_size:
        payload = cdf_path.read_bytes()
        return {
            "request": request,
            "status": "cached",
            "cdf_path": cdf_path,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "source_url": _request_url(request),
            "error": None,
        }

    try:
        response = requests.get(
            _request_url(request),
            headers={"User-Agent": USER_AGENT, "Accept": "application/xml"},
            timeout=timeout_seconds,
        )
        if response.status_code == 404:
            return {
                "request": request,
                "status": "no_data",
                "cdf_path": None,
                "sha256": None,
                "source_url": _request_url(request),
                "error": "CDAWeb has no archived data for this spacecraft and interval.",
            }
        response.raise_for_status()
        download_url = _cdf_download_url(response.content)
        if not download_url:
            return {
                "request": request,
                "status": "no_data",
                "cdf_path": None,
                "sha256": None,
                "source_url": _request_url(request),
                "error": "CDAWeb returned no CDF file for this interval.",
            }

        cdf_response = requests.get(
            download_url,
            headers={"User-Agent": USER_AGENT},
            timeout=timeout_seconds,
        )
        cdf_response.raise_for_status()
        if not cdf_response.content.startswith(b"\xcd\xf3\x00\x01"):
            raise ValueError("CDAWeb response is not a valid CDF file.")
        _write_atomic(cdf_path, cdf_response.content)
        return {
            "request": request,
            "status": "downloaded",
            "cdf_path": cdf_path,
            "sha256": hashlib.sha256(cdf_response.content).hexdigest(),
            "source_url": download_url,
            "error": None,
        }
    except (requests.RequestException, ET.ParseError, OSError, ValueError) as exc:
        return {
            "request": request,
            "status": "error",
            "cdf_path": None,
            "sha256": None,
            "source_url": _request_url(request),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _cdf_values(cdf: CDF, name: str) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(cdf.varget(name)).astype(np.float64, copy=True)
    attributes = cdf.varattsget(name)
    fill_value = attributes.get("FILLVAL")
    values[~np.isfinite(values)] = np.nan
    if fill_value is not None:
        values[values == float(np.asarray(fill_value).reshape(-1)[0])] = np.nan
    return values, attributes


def _epoch_index(cdf: CDF) -> pd.DatetimeIndex:
    timestamps = pd.DatetimeIndex(cdfepoch.to_datetime(cdf.varget("Epoch")))
    if timestamps.tz is None:
        timestamps = timestamps.tz_localize("UTC")
    else:
        timestamps = timestamps.tz_convert("UTC")
    return timestamps


def parse_goes_cdf(path: str | Path, satellite: str) -> pd.DataFrame:
    """Return E2 >2 MeV measurements, retaining proton-contamination quality flags."""
    cdf = CDF(str(path))
    epoch = _epoch_index(cdf)
    flux, _ = _cdf_values(cdf, "E2W_COR_FLUX")
    quality, _ = _cdf_values(cdf, "E2W_DQF")
    if len(epoch) != len(flux) or len(epoch) != len(quality):
        raise ValueError(f"{path}: GOES timestamps, flux and quality arrays do not align.")
    valid_flux = (quality == 0) & np.isfinite(flux) & (flux > 0)
    return pd.DataFrame(
        {
            "timestamp_utc": epoch,
            "satellite": satellite,
            "electron_flux_pfu": np.where(valid_flux, flux, np.nan),
            "quality_flag": pd.array(quality, dtype="Int64"),
        }
    )


def parse_wind_swe_cdf(path: str | Path) -> pd.DataFrame:
    """Return WIND SWE plasma parameters; only documented good-quality records survive."""
    cdf = CDF(str(path))
    epoch = _epoch_index(cdf)
    velocity, _ = _cdf_values(cdf, "V_GSM")
    density, _ = _cdf_values(cdf, "Np")
    pressure, _ = _cdf_values(cdf, "Pressure")
    velocity_quality, _ = _cdf_values(cdf, "QF_V")
    density_quality, _ = _cdf_values(cdf, "QF_Np")
    sizes = {
        len(epoch),
        len(velocity),
        len(density),
        len(pressure),
        len(velocity_quality),
        len(density_quality),
    }
    if len(sizes) != 1 or velocity.ndim != 2 or velocity.shape[1] < 3:
        raise ValueError(f"{path}: WIND SWE variables have incompatible dimensions.")
    velocity_good = velocity_quality == 0
    density_good = density_quality == 0
    velocity[~velocity_good, :] = np.nan
    density[~density_good] = np.nan
    pressure[~(velocity_good & density_good)] = np.nan
    return pd.DataFrame(
        {
            "timestamp_utc": epoch,
            "Vx_GSM_km_s": velocity[:, 0],
            "Vy_GSM_km_s": velocity[:, 1],
            "Vz_GSM_km_s": velocity[:, 2],
            "Vsw_km_s": np.linalg.norm(velocity, axis=1),
            "Np_cm3": density,
            "Pdyn_nPa": pressure,
            "velocity_quality_flag": pd.array(velocity_quality, dtype="Int64"),
            "density_quality_flag": pd.array(density_quality, dtype="Int64"),
        }
    )


def parse_wind_mfi_cdf(path: str | Path) -> pd.DataFrame:
    """Return WIND MFI 1-minute GSM magnetic-field components."""
    cdf = CDF(str(path))
    epoch = _epoch_index(cdf)
    magnetic_field, _ = _cdf_values(cdf, "BGSM")
    if magnetic_field.ndim != 2 or magnetic_field.shape != (len(epoch), 3):
        raise ValueError(f"{path}: WIND MFI BGSM must have three aligned components.")
    return pd.DataFrame(
        {
            "timestamp_utc": epoch,
            "Bx_GSM_nT": magnetic_field[:, 0],
            "BY_GSM_nT": magnetic_field[:, 1],
            "BZ_GSM_nT": magnetic_field[:, 2],
        }
    )


def parse_omni_cdf(path: str | Path) -> pd.DataFrame:
    """Return near-Earth, time-shifted OMNI features with source codes preserved."""
    cdf = CDF(str(path))
    epoch = _epoch_index(cdf)
    values = {
        name: _cdf_values(cdf, name)[0]
        for name in (
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
    }
    if any(len(value) != len(epoch) for value in values.values()):
        raise ValueError(f"{path}: OMNI variables do not align with timestamps.")
    return pd.DataFrame({"timestamp_utc": epoch, **values})


def _requests_for_range(start: date, end: date) -> list[ArchiveRequest]:
    requests_to_make: list[ArchiveRequest] = []
    for month_start in _month_starts(start, end):
        next_month = date(
            month_start.year + (month_start.month == 12),
            month_start.month % 12 + 1,
            1,
        )
        month_end = min(next_month, end)
        interval_start = max(month_start, start)
        for satellite, dataset in GOES_DATASETS.items():
            requests_to_make.append(
                ArchiveRequest(
                    product=f"goes_{satellite.lower().replace('-', '')}",
                    dataset=dataset,
                    variables=GOES_VARIABLES,
                    start=interval_start,
                    end=month_end,
                    satellite=satellite,
                )
            )
        for product, config in WIND_PRODUCTS.items():
            requests_to_make.append(
                ArchiveRequest(
                    product=product,
                    dataset=config["dataset"],
                    variables=config["variables"],
                    start=interval_start,
                    end=month_end,
                )
            )
        requests_to_make.append(
            ArchiveRequest(
                product="omni_hro_5min",
                dataset=OMNI_PRODUCT["dataset"],
                variables=OMNI_PRODUCT["variables"],
                start=interval_start,
                end=month_end,
            )
        )
    return requests_to_make


def _parse_download(result: dict[str, Any], output_dir: Path) -> Path | None:
    request: ArchiveRequest = result["request"]
    cdf_path: Path | None = result["cdf_path"]
    if cdf_path is None:
        return None
    if request.product.startswith("goes_"):
        frame = parse_goes_cdf(cdf_path, request.satellite or "")
    elif request.product == "wind_swe":
        frame = parse_wind_swe_cdf(cdf_path)
    elif request.product == "omni_hro_5min":
        frame = parse_omni_cdf(cdf_path)
    else:
        frame = parse_wind_mfi_cdf(cdf_path)
    output_path = output_dir / "csv" / request.product / f"{request.stem}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path


def acquire_archive(
    start: date,
    end: date,
    output_dir: str | Path = "radshield/data/archive",
    workers: int = 4,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    """Download and normalize the requested monthly GOES and WIND CDAWeb archives."""
    if start >= end:
        raise ValueError("Archive start date must be earlier than the exclusive end date.")
    if workers < 1 or workers > 8:
        raise ValueError("workers must be between 1 and 8.")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    requests_to_make = _requests_for_range(start, end)
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_archive_request, item, root, timeout_seconds): item
            for item in requests_to_make
        }
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda item: (item["request"].start, item["request"].product))
    for result in results:
        if result["cdf_path"]:
            try:
                result["csv_path"] = _parse_download(result, root)
                result["parse_error"] = None
            except (OSError, ValueError, KeyError, TypeError) as exc:
                result["csv_path"] = None
                result["parse_error"] = f"{type(exc).__name__}: {exc}"
        else:
            result["csv_path"] = None
            result["parse_error"] = None

    manifest = {
        "source": "NASA SPDF CDAWeb REST data service",
        "source_url": "https://cdaweb.gsfc.nasa.gov/WebServices/REST/",
        "requested_start_utc": start.isoformat(),
        "requested_end_utc_exclusive": end.isoformat(),
        "goes_product": {
            "variables": GOES_VARIABLES,
            "target": "E2W_COR_FLUX: integral electrons >2 MeV, background/contamination/dead-time corrected",
            "units": "electrons/(cm^2 s sr)",
            "quality": "E2W_DQF == 0 only; invalid flux and nonpositive values preserved as missing",
            "satellites": GOES_DATASETS,
        },
        "wind_products": WIND_PRODUCTS,
        "omni_product": OMNI_PRODUCT,
        "results": [
            {
                "product": item["request"].product,
                "dataset": item["request"].dataset,
                "start": item["request"].start.isoformat(),
                "end_exclusive": item["request"].end.isoformat(),
                "status": item["status"],
                "cdf_path": str(item["cdf_path"]) if item["cdf_path"] else None,
                "csv_path": str(item["csv_path"]) if item["csv_path"] else None,
                "sha256": item["sha256"],
                "source_url": item["source_url"],
                "error": item["error"],
                "parse_error": item["parse_error"],
            }
            for item in results
        ],
    }
    manifest_path = root / "archive_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download monthly science-quality GOES E2 and WIND CDF archives."
    )
    parser.add_argument("--start", type=date.fromisoformat, default=date(2010, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2021, 1, 1))
    parser.add_argument("--output", type=Path, default=Path("radshield/data/archive"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    manifest = acquire_archive(
        args.start,
        args.end,
        output_dir=args.output,
        workers=args.workers,
    )
    counts = pd.Series([row["status"] for row in manifest["results"]]).value_counts()
    print(f"Archive manifest: {args.output / 'archive_manifest.json'}")
    print(counts.to_string())
    failures = [
        row
        for row in manifest["results"]
        if row["status"] == "error" or row["parse_error"] is not None
    ]
    if failures:
        raise SystemExit(
            f"{len(failures)} archive requests failed; inspect archive_manifest.json."
        )


if __name__ == "__main__":
    main()
