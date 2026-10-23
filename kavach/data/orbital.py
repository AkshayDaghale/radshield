"""CelesTrak TLE ingestion and time-tagged SGP4 orbital-position estimates."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Sequence

import plotly.graph_objects as go
import requests
from sgp4.api import Satrec, SGP4_ERRORS, WGS72, jday
from sgp4.propagation import gstime
from sgp4.io import verify_checksum

CELESTRAK_GEO_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=geo&FORMAT=tle"
REQUEST_TIMEOUT_SECONDS = (3.05, 10.0)
TLE_STALE_AFTER = timedelta(days=7)
TLE_CACHE_SCHEMA_VERSION = 1
_CACHE_ROOT = Path(
    os.environ.get("LOCALAPPDATA")
    or os.environ.get("XDG_CACHE_HOME")
    or (Path.home() / ".cache")
)
CELESTRAK_CACHE_PATH = _CACHE_ROOT / "RadShieldAI" / "geo_tle_cache.json"


@dataclass(frozen=True)
class SatelliteTLE:
    name: str
    norad_id: str
    line1: str
    line2: str
    tle_epoch: datetime
    fetched_at: datetime


@dataclass(frozen=True)
class CatalogSnapshot:
    satellites: tuple[SatelliteTLE, ...]
    fetched_at: datetime | None
    parse_errors: tuple[str, ...] = ()
    error: str | None = None
    checked_at: datetime | None = None
    cache_warning: str | None = None


@dataclass(frozen=True)
class SatellitePosition:
    source_tle: SatelliteTLE
    propagated_at: datetime
    teme_x_km: float
    teme_y_km: float
    teme_z_km: float
    ecef_x_km: float
    ecef_y_km: float
    ecef_z_km: float
    longitude_deg: float
    latitude_deg: float


@dataclass(frozen=True)
class PropagationFailure:
    name: str
    norad_id: str
    tle_epoch: datetime
    message: str


@dataclass(frozen=True)
class PropagationReport:
    positions: tuple[SatellitePosition, ...]
    failures: tuple[PropagationFailure, ...]


def _as_utc(value: datetime, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be an explicitly timezone-aware timestamp.")
    return value.astimezone(timezone.utc)


def _tle_epoch(line1: str) -> datetime:
    epoch_field = line1[18:32]
    try:
        year_short = int(epoch_field[:2])
        day_of_year = float(epoch_field[2:])
    except (ValueError, IndexError) as exc:
        raise ValueError("TLE epoch field is malformed.") from exc

    year = 1900 + year_short if year_short >= 57 else 2000 + year_short
    days_in_year = (datetime(year + 1, 1, 1) - datetime(year, 1, 1)).days
    if not math.isfinite(day_of_year) or not 1.0 <= day_of_year < days_in_year + 1:
        raise ValueError("TLE epoch day is outside its calendar year.")

    return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(
        days=day_of_year - 1.0
    )


def _validate_tle(name: str, line1: str, line2: str, fetched_at: datetime) -> SatelliteTLE:
    if not name:
        raise ValueError("Satellite name is empty.")
    if not line1.startswith("1 ") or not line2.startswith("2 "):
        raise ValueError("TLE record must contain one line 1 and one line 2.")
    if len(line1) < 69 or len(line2) < 69:
        raise ValueError("TLE lines are incomplete.")
    if not line1[68].isdigit() or not line2[68].isdigit():
        raise ValueError("TLE checksum digit is missing.")
    verify_checksum(line1, line2)
    norad_id = line1[2:7].strip()
    second_norad_id = line2[2:7].strip()
    if not norad_id or norad_id != second_norad_id:
        raise ValueError("TLE lines have missing or mismatched NORAD identifiers.")

    # Parsing through Satrec validates the fixed-width fields expected by SGP4.
    Satrec.twoline2rv(line1, line2, WGS72)
    return SatelliteTLE(
        name=name,
        norad_id=norad_id,
        line1=line1,
        line2=line2,
        tle_epoch=_tle_epoch(line1),
        fetched_at=fetched_at,
    )


def parse_tle_catalog(
    text: str, fetched_at: datetime | None = None
) -> CatalogSnapshot:
    """Parse CelesTrak's name/line-1/line-2 TLE format, retaining valid records."""
    timestamp = _as_utc(fetched_at, "Fetch timestamp") if fetched_at else datetime.now(
        timezone.utc
    )
    if not isinstance(text, str) or not text.strip():
        return CatalogSnapshot(
            satellites=(),
            fetched_at=timestamp,
            error="CelesTrak returned an empty TLE response.",
            checked_at=timestamp,
        )

    lines = [line.strip() for line in text.lstrip("\ufeff").splitlines() if line.strip()]
    satellites: list[SatelliteTLE] = []
    errors: list[str] = []
    seen_ids: set[str] = set()
    index = 0

    while index < len(lines):
        name_line = lines[index]
        if name_line.startswith("1 "):
            errors.append(f"Line {index + 1}: TLE record is missing its name and line 2.")
            index += 1
            continue

        name = name_line[2:].strip() if name_line.startswith("0 ") else name_line
        if index + 2 >= len(lines):
            errors.append(f"{name}: incomplete TLE record.")
            break

        line1, line2 = lines[index + 1], lines[index + 2]
        if not line1.startswith("1 ") or not line2.startswith("2 "):
            errors.append(f"{name}: expected a line 1 followed by a line 2.")
            index += 1
            continue

        try:
            record = _validate_tle(name, line1, line2, timestamp)
            if record.norad_id in seen_ids:
                raise ValueError(f"duplicate NORAD identifier {record.norad_id}.")
            satellites.append(record)
            seen_ids.add(record.norad_id)
        except (ValueError, IndexError) as exc:
            errors.append(f"{name}: {exc}")
        index += 3

    error = None
    if not satellites:
        error = "No valid TLE records were found in the CelesTrak response."
    return CatalogSnapshot(
        satellites=tuple(satellites),
        fetched_at=timestamp,
        parse_errors=tuple(errors),
        error=error,
        checked_at=timestamp,
    )


def _read_cached_catalog() -> tuple[CatalogSnapshot | None, str | None]:
    try:
        payload = json.loads(CELESTRAK_CACHE_PATH.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("cache root must be a JSON object")
        if payload.get("schema_version") != TLE_CACHE_SCHEMA_VERSION:
            raise ValueError("saved TLE cache uses an unsupported schema")
        if payload.get("source_url") != CELESTRAK_GEO_URL:
            raise ValueError("saved TLE cache does not match the configured GEO feed")
        text = payload["tle_text"]
        fetched_at = datetime.fromisoformat(payload["fetched_at"])
        if not isinstance(text, str):
            raise ValueError("cached TLE data must be text")
        snapshot = parse_tle_catalog(text, fetched_at=fetched_at)
        if snapshot.error:
            raise ValueError(snapshot.error)
        return snapshot, None
    except FileNotFoundError:
        return None, None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return None, f"Saved CelesTrak cache could not be read: {exc}"


def _write_cached_catalog(text: str, fetched_at: datetime) -> None:
    cache_path = CELESTRAK_CACHE_PATH
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=cache_path.parent,
            prefix=f".{cache_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as cache_file:
            temporary_path = Path(cache_file.name)
            json.dump(
                {
                    "schema_version": TLE_CACHE_SCHEMA_VERSION,
                    "source_url": CELESTRAK_GEO_URL,
                    "fetched_at": fetched_at.isoformat(),
                    "tle_text": text,
                },
                cache_file,
            )
            cache_file.flush()
            os.fsync(cache_file.fileno())
        temporary_path.replace(cache_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _catalog_after_fetch_failure(
    error: str, checked_at: datetime
) -> CatalogSnapshot:
    cached_snapshot, cache_error = _read_cached_catalog()
    if cached_snapshot is None:
        if cache_error:
            error = f"{error} {cache_error}"
        return CatalogSnapshot(
            satellites=(),
            fetched_at=None,
            error=error,
            checked_at=checked_at,
        )

    return replace(
        cached_snapshot,
        error=error,
        checked_at=checked_at,
    )


def fetch_celestrak_catalog(
    fetched_at: datetime | None = None,
) -> CatalogSnapshot:
    """Fetch and validate the configured CelesTrak GEO TLE feed."""
    request_started = (
        _as_utc(fetched_at, "Fetch timestamp")
        if fetched_at is not None
        else datetime.now(timezone.utc)
    )
    try:
        response = requests.get(
            CELESTRAK_GEO_URL,
            headers={"User-Agent": "RadShield-AI/1.0 (GEO TLE catalogue)"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.Timeout as exc:
        return _catalog_after_fetch_failure(
            f"CelesTrak request timed out: {exc}", request_started
        )
    except requests.HTTPError as exc:
        return _catalog_after_fetch_failure(
            f"CelesTrak returned an HTTP error: {exc}", request_started
        )
    except requests.RequestException as exc:
        return _catalog_after_fetch_failure(
            f"CelesTrak request failed: {exc}", request_started
        )

    timestamp = (
        request_started
        if fetched_at is not None
        else datetime.now(timezone.utc)
    )
    snapshot = parse_tle_catalog(response.text, fetched_at=timestamp)
    if not snapshot.satellites:
        return snapshot
    try:
        _write_cached_catalog(response.text, timestamp)
    except OSError as exc:
        return replace(
            snapshot,
            cache_warning=f"Latest TLEs could not be saved for offline use: {exc}",
        )
    return snapshot


def tle_epoch_status(
    tle_epoch: datetime,
    at: datetime,
    stale_after: timedelta = TLE_STALE_AFTER,
) -> str:
    """Return a transparent age status; the seven-day threshold is operational only."""
    epoch_utc = _as_utc(tle_epoch, "TLE epoch")
    propagation_utc = _as_utc(at, "Status timestamp")
    age = propagation_utc - epoch_utc
    if age < -timedelta(days=1):
        return "FUTURE EPOCH"
    if age > stale_after:
        return "STALE TLE"
    return "CURRENT TLE"


def teme_to_ecef_km(
    teme_xyz_km: Sequence[float], gmst_radians: float
) -> tuple[float, float, float]:
    """Rotate TEME km coordinates to pseudo-Earth-fixed km coordinates with GMST."""
    if len(teme_xyz_km) != 3 or not all(math.isfinite(value) for value in teme_xyz_km):
        raise ValueError("TEME position must contain three finite coordinates in km.")
    if not math.isfinite(gmst_radians):
        raise ValueError("GMST angle must be finite.")

    x_teme, y_teme, z_teme = (float(value) for value in teme_xyz_km)
    cosine, sine = math.cos(gmst_radians), math.sin(gmst_radians)
    return (
        x_teme * cosine + y_teme * sine,
        -x_teme * sine + y_teme * cosine,
        z_teme,
    )


def _position_from_teme(
    teme_xyz_km: Sequence[float], gmst_radians: float
) -> tuple[tuple[float, float, float], float, float]:
    ecef_xyz = teme_to_ecef_km(teme_xyz_km, gmst_radians)
    x_ecef, y_ecef, z_ecef = ecef_xyz
    longitude = (math.degrees(math.atan2(y_ecef, x_ecef)) + 180.0) % 360.0 - 180.0
    latitude = math.degrees(math.atan2(z_ecef, math.hypot(x_ecef, y_ecef)))
    if not all(math.isfinite(value) for value in (*ecef_xyz, longitude, latitude)):
        raise ValueError("Coordinate conversion produced non-finite values.")
    return ecef_xyz, longitude, latitude


def propagate_satellite(record: SatelliteTLE, at: datetime) -> SatellitePosition:
    """Propagate one validated TLE at an explicit UTC timestamp using SGP4."""
    propagation_time = _as_utc(at, "Propagation timestamp")
    validated_record = _validate_tle(
        record.name, record.line1, record.line2, record.fetched_at
    )
    if (
        record.norad_id != validated_record.norad_id
        or record.tle_epoch != validated_record.tle_epoch
    ):
        raise ValueError("Satellite metadata does not match its source TLE.")
    satellite = Satrec.twoline2rv(record.line1, record.line2, WGS72)
    julian_day, fraction = jday(
        propagation_time.year,
        propagation_time.month,
        propagation_time.day,
        propagation_time.hour,
        propagation_time.minute,
        propagation_time.second + propagation_time.microsecond / 1_000_000.0,
    )
    error_code, teme_xyz, _ = satellite.sgp4(julian_day, fraction)
    if error_code:
        message = SGP4_ERRORS.get(error_code, "Unknown SGP4 propagation error.")
        raise ValueError(f"SGP4 error {error_code}: {message}")
    if not all(math.isfinite(value) for value in teme_xyz):
        raise ValueError("SGP4 returned non-finite TEME coordinates.")

    ecef_xyz, longitude, latitude = _position_from_teme(
        teme_xyz, gstime(julian_day + fraction)
    )
    return SatellitePosition(
        source_tle=record,
        propagated_at=propagation_time,
        teme_x_km=float(teme_xyz[0]),
        teme_y_km=float(teme_xyz[1]),
        teme_z_km=float(teme_xyz[2]),
        ecef_x_km=ecef_xyz[0],
        ecef_y_km=ecef_xyz[1],
        ecef_z_km=ecef_xyz[2],
        longitude_deg=longitude,
        latitude_deg=latitude,
    )


def propagate_catalog(
    records: Sequence[SatelliteTLE], at: datetime
) -> PropagationReport:
    """Propagate each validated catalogue record, isolating individual SGP4 errors."""
    propagation_time = _as_utc(at, "Propagation timestamp")
    positions: list[SatellitePosition] = []
    failures: list[PropagationFailure] = []
    for record in records:
        try:
            positions.append(propagate_satellite(record, propagation_time))
        except ValueError as exc:
            failures.append(
                PropagationFailure(
                    name=record.name,
                    norad_id=record.norad_id,
                    tle_epoch=record.tle_epoch,
                    message=str(exc),
                )
            )
    return PropagationReport(tuple(positions), tuple(failures))


def build_geo_globe_figure(
    positions: Sequence[SatellitePosition],
    *,
    using_cached_after_failure: bool = False,
) -> go.Figure:
    """Create an orthographic Earth view with only real SGP4 position markers."""
    statuses = [
        tle_epoch_status(position.source_tle.tle_epoch, position.propagated_at)
        for position in positions
    ]
    if using_cached_after_failure:
        statuses = [f"{status}; last successful catalogue" for status in statuses]

    customdata = [
        [
            position.source_tle.norad_id,
            position.source_tle.tle_epoch.isoformat().replace("+00:00", "Z"),
            position.source_tle.fetched_at.isoformat().replace("+00:00", "Z"),
            position.propagated_at.isoformat().replace("+00:00", "Z"),
            status,
        ]
        for position, status in zip(positions, statuses)
    ]
    figure = go.Figure(
        go.Scattergeo(
            lon=[position.longitude_deg for position in positions],
            lat=[position.latitude_deg for position in positions],
            text=[position.source_tle.name for position in positions],
            customdata=customdata,
            mode="markers",
            textposition="top center",
            marker=dict(size=7, color="#1769AA", line=dict(width=1, color="#FFFFFF")),
            hovertemplate=(
                "<b>%{text}</b><br>NORAD ID: %{customdata[0]}"
                "<br>Longitude: %{lon:.3f}°"
                "<br>Geocentric latitude: %{lat:.3f}°"
                "<br>TLE epoch (UTC): %{customdata[1]}"
                "<br>TLE fetched (UTC): %{customdata[2]}"
                "<br>Propagated (UTC): %{customdata[3]}"
                "<br>Status: %{customdata[4]}<extra></extra>"
            ),
        )
    )
    title = (
        "CelesTrak GEO catalogue — SGP4 orbital positions"
        if positions
        else "Earth view — satellite positions unavailable"
    )
    figure.update_layout(
        title=title,
        height=560,
        margin=dict(l=0, r=0, t=55, b=0),
        paper_bgcolor="#FFFFFF",
        font=dict(color="#14263D"),
        geo=dict(
            projection=dict(type="orthographic", rotation=dict(lon=0, lat=15)),
            showland=True,
            landcolor="#E8EEF4",
            showocean=True,
            oceancolor="#DCEAF5",
            showlakes=True,
            lakecolor="#DCEAF5",
            showcountries=True,
            countrycolor="#AAB8C5",
            showcoastlines=True,
            coastlinecolor="#8798A8",
            showframe=True,
            bgcolor="#FFFFFF",
        ),
    )
    if not positions:
        figure.add_annotation(
            text="Earth view only — no validated satellite positions available",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color="#405873", size=14),
            bgcolor="rgba(255,255,255,0.85)",
        )
    return figure
