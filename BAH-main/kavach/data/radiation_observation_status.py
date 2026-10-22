"""Presentation-safe status for direct NOAA integral-electron observations."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from kavach.data.noaa_ingest import (
    NOAA_ELECTRON_CONTAMINATION_WARNING,
    NOAA_ELECTRON_PRODUCT,
    NOAA_ELECTRON_UNITS,
    NOAA_GOES_URL,
)

STALE_AFTER_MINUTES = 15.0


def build_radiation_observation_status(
    frame: pd.DataFrame,
    feed_status: dict[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Summarize observed_flux only; never use filled model-ready flux."""
    feed = feed_status.get("goes_electron_flux", {})
    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None or current_time.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    current_time = current_time.astimezone(timezone.utc)

    observed = frame.get("observed_flux")
    if observed is None:
        values = pd.Series(dtype=float)
    else:
        values = pd.to_numeric(observed, errors="coerce")
        values = values.where(np.isfinite(values) & (values >= 0))
        timestamps = pd.to_datetime(frame.index, utc=True, errors="coerce")
        valid = values.notna().to_numpy() & ~timestamps.isna()
        values = values.iloc[np.flatnonzero(valid)]
        timestamps = timestamps[valid]
        values.index = timestamps
        values = values.sort_index()

    result: dict[str, Any] = {
        "source": feed.get("product", NOAA_ELECTRON_PRODUCT),
        "source_url": feed.get("url", NOAA_GOES_URL),
        "channel": feed.get("channel", ">=2 MeV"),
        "units": feed.get("units", NOAA_ELECTRON_UNITS),
        "satellite_ids": feed.get("satellite_ids", []),
        "instrument": feed.get("instrument"),
        "feed_connected": bool(feed.get("connected", False)),
        "fetch_timestamp_utc": feed.get("fetch_timestamp_utc"),
        "valid_observation_count": int(len(values)),
        "first_observation_timestamp_utc": (
            values.index.min().isoformat() if len(values) else None
        ),
        "last_observation_timestamp_utc": (
            values.index.max().isoformat() if len(values) else None
        ),
        "latest_value": float(values.iloc[-1]) if len(values) else None,
        "latest_observation_timestamp_utc": (
            values.index[-1].isoformat() if len(values) else None
        ),
        "age_minutes": None,
        "stale_after_minutes": STALE_AFTER_MINUTES,
        "data_quality_status": "unavailable",
        "feed_error": feed.get("error"),
        "quality_flag_available": bool(feed.get("quality_flag_available", False)),
        "quality_warning": feed.get(
            "quality_warning", NOAA_ELECTRON_CONTAMINATION_WARNING
        ),
        "history_limit": (
            "The source endpoint provides a rolling seven-day history only; it is "
            "not a long-term training archive."
        ),
    }

    if not result["feed_connected"]:
        result["data_quality_status"] = "feed unavailable"
        return result
    if not len(values):
        result["data_quality_status"] = "no valid measured observation"
        return result

    latest = values.index[-1].to_pydatetime()
    age_minutes = (current_time - latest).total_seconds() / 60.0
    result["age_minutes"] = age_minutes
    if age_minutes < 0:
        result["data_quality_status"] = "invalid future observation timestamp"
        result["feed_error"] = "Latest observation timestamp is later than current time."
    elif age_minutes > STALE_AFTER_MINUTES:
        result["data_quality_status"] = "stale"
    elif not result["quality_flag_available"]:
        result["data_quality_status"] = "fresh; event quality unknown"
    else:
        result["data_quality_status"] = "fresh"
    return result
