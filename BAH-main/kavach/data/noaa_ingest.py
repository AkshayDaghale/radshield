"""
NOAA SWPC telemetry ingestion for RadShield AI.
Fetches live space weather plasma, magnetometer, and GOES particle flux streams.
"""
from datetime import datetime, timezone

import requests
import numpy as np
import pandas as pd

NOAA_PLASMA_URL = "https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json"
NOAA_MAG_URL    = "https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json"
NOAA_GOES_URL   = "https://services.swpc.noaa.gov/json/goes/primary/integral-electrons-7-day.json"
NOAA_GOES_MAG_URL = "https://services.swpc.noaa.gov/json/goes/primary/magnetometers-7-day.json"
NOAA_KP_URL     = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
NOAA_ELECTRON_CHANNEL = ">=2 MeV"
NOAA_ELECTRON_UNITS = "electrons/(cm^2 s sr) (pfu)"
NOAA_ELECTRON_PRODUCT = "NOAA SWPC GOES integral electrons (7-day)"
NOAA_ELECTRON_CONTAMINATION_WARNING = (
    "NOAA warns that the >2 MeV electron channel can be contaminated by energetic "
    "protons and is not valid during significant proton events. This feed does not "
    "provide an event-quality flag, so validity during such events is unknown."
)


def _new_feed_status(name: str, url: str, required: bool) -> dict:
    return {
        "source": name,
        "url": url,
        "required": required,
        "connected": False,
        "first_observation_timestamp_utc": None,
        "last_observation_timestamp_utc": None,
        "valid_observations": 0,
        "fetch_timestamp_utc": None,
        "error": None,
    }


def _record_observation_status(status: dict, data, timestamp_key: str = "time_tag") -> None:
    if not isinstance(data, list):
        return

    timestamps = []
    for row in data:
        value = row.get(timestamp_key) if isinstance(row, dict) else None
        if value is not None:
            timestamps.append(value)

    if timestamps:
        parsed = pd.to_datetime(
            pd.Series(timestamps),
            utc=True,
            errors="coerce",
            format="mixed",
        ).dropna()
        status["valid_observations"] = int(parsed.size)
        if not parsed.empty:
            status["first_observation_timestamp_utc"] = parsed.min().isoformat()
            status["last_observation_timestamp_utc"] = parsed.max().isoformat()


def fetch_live_noaa_telemetry(timeout_sec: int = 5):
    """
    Fetch NOAA SWPC telemetry and return the existing model-ready feature frame.
    Unfilled source measurements and per-feed status are preserved separately for display.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*"
    }
    feeds = {
        "solar_wind": _new_feed_status("NOAA SWPC solar wind (1-minute)", NOAA_PLASMA_URL, True),
        "solar_wind_magnetic_field": _new_feed_status("NOAA SWPC solar-wind magnetic field (1-minute)", NOAA_MAG_URL, True),
        "goes_electron_flux": _new_feed_status("NOAA GOES integral electrons (7-day)", NOAA_GOES_URL, False),
        "goes_magnetometer": _new_feed_status("NOAA GOES magnetometers (7-day)", NOAA_GOES_MAG_URL, False),
        "planetary_kp": _new_feed_status("NOAA planetary Kp index", NOAA_KP_URL, False),
    }

    def fetch_json(feed_key: str, url: str):
        status = feeds[feed_key]
        try:
            response = requests.get(url, headers=headers, timeout=timeout_sec)
            if response.status_code != 200:
                raise ValueError(f"HTTP {response.status_code}")
            payload = response.json()
            if not isinstance(payload, (list, dict)):
                raise ValueError("Response JSON must be an array or object")
            status["connected"] = True
            status["fetch_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
            _record_observation_status(status, payload)
            return payload
        except Exception as exc:
            status["error"] = str(exc)
            return None

    r_plasma = fetch_json("solar_wind", NOAA_PLASMA_URL)
    r_mag = fetch_json("solar_wind_magnetic_field", NOAA_MAG_URL)
    if r_plasma is None or r_mag is None:
        return None, feeds
    for feed_key, payload in (
        ("solar_wind", r_plasma),
        ("solar_wind_magnetic_field", r_mag),
    ):
        if not isinstance(payload, list) or not payload or not all(
            isinstance(row, dict) and "time_tag" in row for row in payload
        ):
            feeds[feed_key]["connected"] = False
            feeds[feed_key]["error"] = "Response has no timestamped observation records"
            return None, feeds

    r_goes = fetch_json("goes_electron_flux", NOAA_GOES_URL)
    r_goes_mag = fetch_json("goes_magnetometer", NOAA_GOES_MAG_URL)
    r_kp = fetch_json("planetary_kp", NOAA_KP_URL)

    df_goes = None
    if isinstance(r_goes, list):
        matching_flux_rows = [
            row for row in r_goes
            if isinstance(row, dict) and row.get("energy") == NOAA_ELECTRON_CHANNEL
        ]
        valid_flux_rows = []
        invalid_flux_rows = 0
        for row in matching_flux_rows:
            timestamp = pd.to_datetime(row.get("time_tag"), utc=True, errors="coerce")
            flux_value = pd.to_numeric(row.get("flux"), errors="coerce")
            if pd.isna(timestamp) or pd.isna(flux_value) or not np.isfinite(flux_value) or flux_value < 0:
                invalid_flux_rows += 1
                continue
            valid_flux_rows.append(
                {**row, "time_tag": timestamp.isoformat(), "flux": float(flux_value)}
            )
        if invalid_flux_rows:
            feeds["goes_electron_flux"]["error"] = (
                f"Ignored {invalid_flux_rows} GOES rows with invalid timestamps or flux values"
            )
        feeds["goes_electron_flux"].update(
            {
                "product": NOAA_ELECTRON_PRODUCT,
                "channel": NOAA_ELECTRON_CHANNEL,
                "units": NOAA_ELECTRON_UNITS,
                "satellite_ids": sorted(
                    {
                        str(row["satellite"])
                        for row in valid_flux_rows
                        if row.get("satellite") is not None
                    }
                ),
                "instrument": None,
                "quality_flag_available": False,
                "quality_warning": NOAA_ELECTRON_CONTAMINATION_WARNING,
            }
        )
        if not matching_flux_rows:
            feeds["goes_electron_flux"]["error"] = "No >=2 MeV electron-flux observations in response"
        elif not valid_flux_rows:
            feeds["goes_electron_flux"]["error"] = (
                feeds["goes_electron_flux"]["error"]
                or "No GOES rows have a valid timestamp and non-negative numeric flux"
            )
        df_goes = pd.DataFrame(valid_flux_rows)
        _record_observation_status(feeds["goes_electron_flux"], valid_flux_rows)
    else:
        feeds["goes_electron_flux"]["error"] = (
            feeds["goes_electron_flux"]["error"]
            or "Expected an array of GOES flux observations"
        )

    df_goes_mag = pd.DataFrame(r_goes_mag) if isinstance(r_goes_mag, list) else None
    if df_goes_mag is None:
        feeds["goes_magnetometer"]["error"] = "Expected an array of GOES magnetometer observations"
    elif not df_goes_mag.empty and "time_tag" not in df_goes_mag:
        feeds["goes_magnetometer"]["connected"] = False
        feeds["goes_magnetometer"]["error"] = "GOES magnetometer rows are missing time_tag"
        df_goes_mag = None

    live_kp_val = None
    if isinstance(r_kp, list) and len(r_kp) > 1:
        kp_rows = [row for row in r_kp[1:] if isinstance(row, (list, tuple)) and len(row) > 1]
        kp_status = feeds["planetary_kp"]
        kp_status["valid_observations"] = len(kp_rows)
        if not kp_rows:
            kp_status["error"] = "Kp response has no observation rows"
        if kp_rows:
            parsed_kp_time = pd.to_datetime(
                [row[0] for row in kp_rows], utc=True, errors="coerce"
            )
            valid_kp_rows = []
            for row, timestamp in zip(kp_rows, parsed_kp_time):
                try:
                    kp_value = float(row[1])
                except (TypeError, ValueError):
                    continue
                if not pd.isna(timestamp) and np.isfinite(kp_value):
                    valid_kp_rows.append(row)
            if valid_kp_rows:
                valid_kp_times = pd.to_datetime(
                    [row[0] for row in valid_kp_rows], utc=True
                )
                kp_status["valid_observations"] = len(valid_kp_rows)
                kp_status["first_observation_timestamp_utc"] = valid_kp_times.min().isoformat()
                latest_row = max(
                    valid_kp_rows,
                    key=lambda row: pd.to_datetime(row[0], utc=True, errors="coerce")
                )
                kp_status["last_observation_timestamp_utc"] = pd.to_datetime(
                    latest_row[0], utc=True
                ).isoformat()
                try:
                    live_kp_val = float(latest_row[1])
                except (TypeError, ValueError):
                    kp_status["error"] = "Latest Kp observation is not numeric"
                    kp_status["connected"] = False
            else:
                kp_status["error"] = "Kp response has no parseable timestamped observations"
    else:
        feeds["planetary_kp"]["error"] = "Expected a header and timestamped Kp observations"
        
    try:
        df_plasma = pd.DataFrame(r_plasma)
        df_mag    = pd.DataFrame(r_mag)
        
        # Merge telemetry on time_tag and Resample to 5-minute to match AI Model expectations
        df_merged = pd.merge(df_plasma, df_mag, on="time_tag", how="inner")
        df_merged["time_tag"] = pd.to_datetime(df_merged["time_tag"], utc=True)
        df_merged.set_index("time_tag", inplace=True)
        df_merged = df_merged.resample("5min").mean(numeric_only=True)
        
        if df_goes is not None and not df_goes.empty:
            df_goes["time_tag"] = pd.to_datetime(df_goes["time_tag"], utc=True)
            df_goes.set_index("time_tag", inplace=True)
            df_goes = df_goes.resample("5min").mean(numeric_only=True)
            df_merged = df_merged.join(df_goes, how="outer")
            
        if df_goes_mag is not None and not df_goes_mag.empty:
            df_goes_mag["time_tag"] = pd.to_datetime(df_goes_mag["time_tag"], utc=True)
            df_goes_mag.set_index("time_tag", inplace=True)
            df_goes_mag = df_goes_mag.resample("5min").mean(numeric_only=True)
            # Prefix columns to avoid collisions
            df_goes_mag = df_goes_mag.add_prefix("goes_")
            df_merged = df_merged.join(df_goes_mag, how="outer")
        
        # Clean NOAA missing data fill values (handling both numeric and string API representations)
        missing_vals = [99999.9, 99999.0, 9999.99, 9999.0, 999.99, 999.9, 999.0, 99.99, 99.0]
        missing_vals_str = [str(v) for v in missing_vals] + [str(int(v)) for v in missing_vals if v.is_integer()]
        df_merged.replace(missing_vals + missing_vals_str, np.nan, inplace=True)

        for source_key, frame, measurement_names in (
            ("solar_wind", df_plasma, ("proton_speed", "speed", "proton_density", "density")),
            ("solar_wind_magnetic_field", df_mag, ("bx_gsm", "by_gsm", "bz_gsm")),
        ):
            source_times = pd.to_datetime(frame.get("time_tag"), utc=True, errors="coerce")
            measurement_columns = [name for name in measurement_names if name in frame]
            numeric_values = frame[measurement_columns].apply(pd.to_numeric, errors="coerce")
            numeric_values.replace(missing_vals, np.nan, inplace=True)
            valid_mask = source_times.notna() & numeric_values.notna().any(axis=1)
            feeds[source_key]["valid_observations"] = int(valid_mask.sum())
            if valid_mask.any():
                feeds[source_key]["first_observation_timestamp_utc"] = source_times[valid_mask].min().isoformat()
                feeds[source_key]["last_observation_timestamp_utc"] = source_times[valid_mask].max().isoformat()
            else:
                feeds[source_key]["error"] = "No valid numeric measurement records in response"

        for feed_key, source_frame in (
            ("goes_magnetometer", df_goes_mag),
        ):
            if source_frame is not None and not source_frame.empty and "time_tag" in source_frame:
                source_times = pd.to_datetime(source_frame["time_tag"], utc=True, errors="coerce")
                measurements = source_frame.drop(columns="time_tag").apply(pd.to_numeric, errors="coerce")
                measurements.replace(missing_vals, np.nan, inplace=True)
                valid_mask = source_times.notna() & measurements.notna().any(axis=1)
                feeds[feed_key]["valid_observations"] = int(valid_mask.sum())
                if valid_mask.any():
                    feeds[feed_key]["first_observation_timestamp_utc"] = source_times[valid_mask].min().isoformat()
                    feeds[feed_key]["last_observation_timestamp_utc"] = source_times[valid_mask].max().isoformat()
                else:
                    feeds[feed_key]["error"] = "No valid numeric magnetometer records in response"
        
        # Extract features (handling both RTSW and DSCOVR field names)
        df_out = pd.DataFrame(index=df_merged.index)
        
        # Use fallback series and fillna to handle the outer join (missing RTSW data in the older 4 days)
        missing_series = pd.Series(np.nan, index=df_merged.index)
        vsw_raw = df_merged.get("proton_speed", df_merged.get("speed", missing_series))
        bx_raw  = df_merged.get("bx_gsm", missing_series)
        bz_raw  = df_merged.get("bz_gsm", missing_series)
        by_raw  = df_merged.get("by_gsm", missing_series)
        np_raw  = df_merged.get("proton_density", df_merged.get("density", missing_series))
        
        observed_vsw = pd.to_numeric(vsw_raw, errors="coerce")
        observed_bx = pd.to_numeric(bx_raw, errors="coerce")
        observed_bz = pd.to_numeric(bz_raw, errors="coerce")
        observed_by = pd.to_numeric(by_raw, errors="coerce")
        observed_np = pd.to_numeric(np_raw, errors="coerce")
        vsw  = observed_vsw.ffill().bfill().fillna(400.0).values
        bx   = observed_bx.ffill().bfill().fillna(0.0).values
        bz   = observed_bz.ffill().bfill().fillna(0.0).values
        by   = observed_by.ffill().bfill().fillna(0.0).values
        np_d = observed_np.ffill().bfill().fillna(5.0).values
        
        n = len(vsw)
        dates = df_merged.index
        
        bt    = np.sqrt(bx**2 + by**2 + bz**2)  # Correct 3D total field magnitude
        theta = np.arctan2(by, bz)
        # Kan-Lee Electric Field (mV/m) instead of Newell Coupling
        ec    = np.clip(vsw * bt * (np.sin(theta/2)**2) * 1e-3, 0, None)
        pdyn  = np.clip(0.5*1.67e-27*(np_d*1e6)*((vsw*1e3)**2)*1e9, 0.1, 50.0)
        
        if live_kp_val is not None:
            kp = np.full(n, live_kp_val)
        else:
            kp = np.clip(2 + 0.01*(vsw-400) + 0.3*np.abs(bz), 0, 9)
            
        dst_raw = -10 - 15*(kp/3.0)**1.5  # Mathematical fallback (Live Kyoto API usually requires auth)
        dst     = pd.Series(dst_raw).ewm(span=6).mean().values  # Smooth transitions to prevent unphysical gradient step spikes
        ae      = 100 + 120*(kp/2.0)
        
        # Real-time ULF Power Proxy
        # If GOES Magnetometer (Hp) is available, use it. Otherwise, fallback to Solar Wind Bz.
        # Pc5 waves are 150-600s period. A 30-min (6 steps of 5min) rolling variance captures this energy.
        if "goes_Hp" in df_merged.columns and df_merged["goes_Hp"].notna().sum() > 5:
            # Use GOES Magnetometer (Geosynchronous orbit, perfect for radiation belt)
            mag_series = df_merged["goes_Hp"].ffill().bfill()
        else:
            # Fallback to OMNI Solar Wind Proxy
            mag_series = pd.Series(bz)
            
        mag_var = mag_series.rolling(window=6, min_periods=1).var().fillna(0.1)
        ulf = (np.log10(mag_var + 1e-4) - 3.5).values
        
        # Incorporate Real GOES Flux with fallback for missing values
        fallback_log_flux = pd.Series(2.3 + 0.005*(vsw-400) + 0.3*(kp-2)).ewm(span=18).mean().values
        fallback_flux = np.clip(10**fallback_log_flux, 0.1, None)
        
        observed_flux = pd.to_numeric(
            df_merged["flux"] if "flux" in df_merged.columns else pd.Series(np.nan, index=dates),
            errors="coerce"
        )
        observed_flux = observed_flux.where(
            np.isfinite(observed_flux) & (observed_flux >= 0)
        )
        if observed_flux.notna().any():
            flux_series = observed_flux
            # Forward-fill real data first to prevent massive discontinuity cliffs if live stream drops
            flux_series = flux_series.ffill().bfill()
            flux = flux_series.fillna(pd.Series(fallback_flux, index=dates)).values
        else:
            flux = fallback_flux
            
        log_flux = np.log10(np.maximum(flux, 1e-3))
        
        bz_neg     = pd.Series((bz < 0).astype(int))
        bz_neg_dur = bz_neg.groupby((bz_neg != bz_neg.shift()).cumsum()).cumcount().values * 5.0
        dDst       = np.clip(np.gradient(dst) / 5.0, -1.5, 1.5)  # Clip unphysical single-step gradient spikes (-1.5 to +1.5 nT/min)
        ae_1h      = pd.Series(ae).rolling(12, min_periods=1).mean().values
        
        regime = np.zeros(n)
        regime[kp >= 6] = 2
        regime[(kp >= 3) & (kp < 6)] = 1
        regime[(dDst > 0) & (dst < -50)] = 3  # Recovery Phase

        
        df = pd.DataFrame({
            "flux": flux, "log_flux": log_flux,
            "observed_flux": observed_flux,
            "observed_Vsw": observed_vsw,
            "observed_BX_GSM": observed_bx,
            "observed_BY_GSM": observed_by,
            "observed_BZ_GSM": observed_bz,
            "observed_Np": observed_np,
            "Vsw": vsw, "BX_GSM": bx, "BZ_GSM": bz, "BY_GSM": by, "BT": bt,
            "Np": np_d, "KP": kp, "DST": dst, "AE": ae, "ULF_power": ulf,
            "Ec": ec, "Pdyn": pdyn, "Bz_neg_dur": bz_neg_dur, "dDst_dt": dDst,
            "AE_1h": ae_1h, "regime": regime.astype(float), "F10.7_index": 150.0,
            "goes_Hp": df_merged.get("goes_Hp", pd.Series(100.0, index=dates))
        }, index=dates)
        
        for lag, lbl in [(12,"1h"),(36,"3h"),(72,"6h"),(144,"12h"),(288,"24h")]:
            df[f"flux_lag_{lbl}"] = df["log_flux"].shift(lag)
            
        baseline_flux = df["log_flux"].iloc[0] if len(df) > 0 else 2.0
        model_df = df.bfill().fillna(baseline_flux)
        for column in (
            "observed_flux", "observed_Vsw", "observed_BX_GSM",
            "observed_BY_GSM", "observed_BZ_GSM", "observed_Np",
        ):
            model_df[column] = df[column]

        feeds["goes_electron_flux"]["valid_observations"] = int(observed_flux.notna().sum())
        if observed_flux.notna().any():
            feeds["goes_electron_flux"]["first_observation_timestamp_utc"] = (
                observed_flux.dropna().index.min().isoformat()
            )
            feeds["goes_electron_flux"]["last_observation_timestamp_utc"] = (
                observed_flux.dropna().index.max().isoformat()
            )
        else:
            feeds["goes_electron_flux"]["first_observation_timestamp_utc"] = None
            feeds["goes_electron_flux"]["last_observation_timestamp_utc"] = None
            feeds["goes_electron_flux"]["error"] = (
                feeds["goes_electron_flux"]["error"]
                or "No numeric valid >=2 MeV flux values in response"
            )
        model_df.attrs["feed_status"] = feeds
        model_df.attrs["provenance"] = {
            "flux": "NOAA GOES direct observation when observed_flux is present; otherwise derived fallback",
            "model_inputs": "Existing filled model-ready values; observed_* columns retain unfilled NOAA source values",
            "derived_indices": ["KP (estimated when NOAA Kp unavailable)", "DST (estimated)", "AE (estimated)", "ULF_power (GOES Hp or Bz proxy)"],
        }
        return model_df, feeds
        
    except Exception as e:
        feeds["solar_wind"]["error"] = feeds["solar_wind"]["error"] or f"Telemetry processing failed: {e}"
        return None, feeds
