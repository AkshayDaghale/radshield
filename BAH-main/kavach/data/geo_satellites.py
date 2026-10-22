"""CelesTrak GEO catalogue parsing and SGP4 position propagation."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
from typing import Any

import requests
from sgp4.api import Satrec, jday
from sgp4.propagation import gstime


CELESTRAK_GEO_TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=geo&FORMAT=tle"
POSITION_STALE_AFTER = timedelta(hours=24)
MAX_ECCENTRICITY = 0.01
MAX_INCLINATION_DEGREES = 5.0
MIN_MEAN_MOTION_REV_PER_DAY = 0.99
MAX_MEAN_MOTION_REV_PER_DAY = 1.01


def _tle_checksum_is_valid(line: str) -> bool:
    if len(line) < 69 or not line[68].isdigit():
        return False
    checksum = sum(int(char) for char in line[:68] if char.isdigit())
    checksum += line[:68].count("-")
    return checksum % 10 == int(line[68])


def _parse_tle_epoch(line1: str) -> datetime:
    year_part = int(line1[18:20])
    year = 1900 + year_part if year_part >= 57 else 2000 + year_part
    day_of_year = float(line1[20:32])
    return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day_of_year - 1)


def _is_geostationary_orbit(satellite: Satrec) -> bool:
    return (
        satellite.ecco <= MAX_ECCENTRICITY
        and math.degrees(satellite.inclo) <= MAX_INCLINATION_DEGREES
        and MIN_MEAN_MOTION_REV_PER_DAY
        <= satellite.no_kozai * 1440.0 / (2.0 * math.pi)
        <= MAX_MEAN_MOTION_REV_PER_DAY
    )


def _teme_to_geodetic(
    position_km: tuple[float, float, float],
    julian_date: float,
) -> tuple[float, float, float]:
    x_teme, y_teme, z_teme = position_km
    theta = gstime(julian_date)
    x = math.cos(theta) * x_teme + math.sin(theta) * y_teme
    y = -math.sin(theta) * x_teme + math.cos(theta) * y_teme
    z = z_teme

    longitude = math.atan2(y, x)
    horizontal_radius = math.hypot(x, y)
    semi_major_axis_km = 6378.137
    eccentricity_squared = 6.69437999014e-3
    latitude = math.atan2(z, horizontal_radius)
    altitude_km = 0.0

    for _ in range(8):
        sin_latitude = math.sin(latitude)
        prime_vertical_radius = semi_major_axis_km / math.sqrt(
            1.0 - eccentricity_squared * sin_latitude**2
        )
        cos_latitude = math.cos(latitude)
        if abs(cos_latitude) < 1e-12:
            altitude_km = abs(z) - prime_vertical_radius * (1.0 - eccentricity_squared)
        else:
            altitude_km = horizontal_radius / cos_latitude - prime_vertical_radius
        latitude = math.atan2(
            z,
            horizontal_radius
            * (
                1.0
                - eccentricity_squared
                * prime_vertical_radius
                / (prime_vertical_radius + altitude_km)
            ),
        )

    return (
        math.degrees(latitude),
        (math.degrees(longitude) + 180.0) % 360.0 - 180.0,
        altitude_km,
    )


def parse_geo_tle_catalog(
    text: str,
    position_time: datetime | None = None,
) -> list[dict[str, Any]]:
    """Parse valid CelesTrak TLEs, retain GEO orbits, and propagate them to a UTC time."""
    if position_time is None:
        position_time = datetime.now(timezone.utc)
    elif position_time.tzinfo is None:
        raise ValueError("position_time must include a timezone")
    else:
        position_time = position_time.astimezone(timezone.utc)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    records: list[dict[str, Any]] = []
    index = 0
    while index < len(lines):
        if not lines[index].startswith("1 "):
            if index + 2 >= len(lines):
                break
            name, line1, line2 = lines[index:index + 3]
            index += 3
        else:
            if index + 1 >= len(lines):
                break
            name = f"NORAD {lines[index][2:7].strip()}"
            line1, line2 = lines[index:index + 2]
            index += 2

        if not line1.startswith("1 ") or not line2.startswith("2 "):
            continue
        if not _tle_checksum_is_valid(line1) or not _tle_checksum_is_valid(line2):
            continue
        if line1[2:7] != line2[2:7]:
            continue

        try:
            satellite = Satrec.twoline2rv(line1, line2)
            if not _is_geostationary_orbit(satellite):
                continue
            epoch = _parse_tle_epoch(line1)
            julian_date, fraction = jday(
                position_time.year,
                position_time.month,
                position_time.day,
                position_time.hour,
                position_time.minute,
                position_time.second + position_time.microsecond / 1_000_000,
            )
            error, position, _velocity = satellite.sgp4(julian_date, fraction)
            if error:
                continue
            latitude, longitude, altitude_km = _teme_to_geodetic(
                position, julian_date + fraction
            )
        except (IndexError, TypeError, ValueError, ZeroDivisionError):
            continue

        records.append(
            {
                "name": name.strip(),
                "norad_id": int(line1[2:7]),
                "orbital_longitude_deg": longitude,
                "latitude_deg": latitude,
                "altitude_km": altitude_km,
                "position_timestamp_utc": position_time.isoformat(),
                "tle_epoch_utc": epoch.isoformat(),
                "element_age_seconds": (position_time - epoch).total_seconds(),
                "source": CELESTRAK_GEO_TLE_URL,
                "position_type": "SGP4 propagated from TLE",
            }
        )

    return records


def fetch_geo_satellite_positions(
    position_time: datetime | None = None,
    timeout_seconds: int = 12,
) -> list[dict[str, Any]]:
    """Fetch the public CelesTrak GEO TLE catalogue and return propagated positions."""
    response = requests.get(
        CELESTRAK_GEO_TLE_URL,
        headers={"User-Agent": "RadShield-AI/1.0", "Accept": "text/plain"},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    positions = parse_geo_tle_catalog(response.text, position_time=position_time)
    if not positions:
        raise ValueError("CelesTrak returned no valid near-circular, near-equatorial GEO TLEs")
    return positions


def classify_catalogue_freshness(
    tle_epoch: datetime,
    now: datetime | None = None,
    stale_after: timedelta = POSITION_STALE_AFTER,
) -> tuple[bool, float]:
    """Return whether an element set is stale and its age in hours."""
    if tle_epoch.tzinfo is None:
        raise ValueError("tle_epoch must include a timezone")
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        raise ValueError("now must include a timezone")
    age_seconds = (now.astimezone(timezone.utc) - tle_epoch.astimezone(timezone.utc)).total_seconds()
    if age_seconds < 0:
        raise ValueError("tle_epoch cannot be in the future")
    return age_seconds > stale_after.total_seconds(), age_seconds / 3600.0
