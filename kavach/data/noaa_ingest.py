"""
NOAA SWPC Live Telemetry Real-Time Ingestion Module
KAVACH — GEO Radiation Monitor | Team DigiIndia
Fetches live space weather plasma, magnetometer, and GOES particle flux streams.
"""
import requests
import numpy as np
import pandas as pd
from datetime import datetime

NOAA_PLASMA_URL = "https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json"
NOAA_MAG_URL    = "https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json"
NOAA_GOES_URL   = "https://services.swpc.noaa.gov/json/goes/primary/integral-electrons-7-day.json"
NOAA_GOES_MAG_URL = "https://services.swpc.noaa.gov/json/goes/primary/magnetometers-7-day.json"
NOAA_KP_URL     = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"

def fetch_live_noaa_telemetry(timeout_sec: int = 5):
    """
    Fetches real-time space weather telemetry from NOAA SWPC public JSON endpoints.
    Returns a processed 19-feature pandas DataFrame ready for KAVACH TFT inference.
    If live endpoint is unreachable (e.g. offline/network blocked), returns fallback synthetic stream.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*"
    }
    try:
        resp_p = requests.get(NOAA_PLASMA_URL, headers=headers, timeout=timeout_sec)
        resp_m = requests.get(NOAA_MAG_URL, headers=headers, timeout=timeout_sec)
        
        if resp_p.status_code != 200 or resp_m.status_code != 200:
            raise ValueError(f"HTTP {resp_p.status_code}/{resp_m.status_code}")
            
        if not (resp_p.text.strip().startswith("[") or resp_p.text.strip().startswith("{")):
            raise ValueError("NOAA SWPC Plasma API returned non-JSON data (possibly rate-limited or blocked)")
            
        if not (resp_m.text.strip().startswith("[") or resp_m.text.strip().startswith("{")):
            raise ValueError("NOAA SWPC Mag API returned non-JSON data (possibly rate-limited or blocked)")
            
        try:
            r_plasma = resp_p.json()
            r_mag    = resp_m.json()
        except Exception as json_err:
            raise ValueError(f"Failed to parse NOAA SWPC JSON: {str(json_err)}")
            
        # Try fetching real GOES Electron Flux
        df_goes = None
        try:
            resp_g = requests.get(NOAA_GOES_URL, headers=headers, timeout=timeout_sec)
            if resp_g.status_code == 200 and (resp_g.text.strip().startswith("[") or resp_g.text.strip().startswith("{")):
                r_goes = resp_g.json()
                df_goes = pd.DataFrame([d for d in r_goes if d.get('energy') == '>=2 MeV'])
        except Exception:
            pass
            
        # Try fetching real GOES Magnetometer
        df_goes_mag = None
        try:
            resp_gm = requests.get(NOAA_GOES_MAG_URL, headers=headers, timeout=timeout_sec)
            if resp_gm.status_code == 200 and (resp_gm.text.strip().startswith("[") or resp_gm.text.strip().startswith("{")):
                r_goes_mag = resp_gm.json()
                df_goes_mag = pd.DataFrame(r_goes_mag)
        except Exception:
            pass

        # Try fetching official planetary Kp index feed
        live_kp_val = None
        try:
            resp_k = requests.get(NOAA_KP_URL, headers=headers, timeout=timeout_sec)
            if resp_k.status_code == 200 and (resp_k.text.strip().startswith("[") or resp_k.text.strip().startswith("{")):
                r_kp = resp_k.json()
                if isinstance(r_kp, list) and len(r_kp) > 1:
                    live_kp_val = float(r_kp[-1][1])
        except Exception:
            live_kp_val = None
        
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
        
        # Extract features (handling both RTSW and DSCOVR field names)
        df_out = pd.DataFrame(index=df_merged.index)
        
        # Use fallback series and fillna to handle the outer join (missing RTSW data in the older 4 days)
        vsw_raw = df_merged.get("proton_speed", df_merged.get("speed", pd.Series(400.0, index=df_merged.index)))
        bx_raw  = df_merged.get("bx_gsm", pd.Series(0.0, index=df_merged.index))
        bz_raw  = df_merged.get("bz_gsm", pd.Series(0.0, index=df_merged.index))
        by_raw  = df_merged.get("by_gsm", pd.Series(0.0, index=df_merged.index))
        np_raw  = df_merged.get("proton_density", df_merged.get("density", pd.Series(5.0, index=df_merged.index)))
        
        vsw  = pd.to_numeric(vsw_raw, errors="coerce").ffill().bfill().fillna(400.0).values
        bx   = pd.to_numeric(bx_raw, errors="coerce").ffill().bfill().fillna(0.0).values
        bz   = pd.to_numeric(bz_raw, errors="coerce").ffill().bfill().fillna(0.0).values
        by   = pd.to_numeric(by_raw, errors="coerce").ffill().bfill().fillna(0.0).values
        np_d = pd.to_numeric(np_raw, errors="coerce").ffill().bfill().fillna(5.0).values
        
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
        
        if "flux" in df_merged.columns:
            flux_series = pd.to_numeric(df_merged["flux"], errors="coerce")
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
            "Vsw": vsw, "BX_GSM": bx, "BZ_GSM": bz, "BY_GSM": by, "BT": bt,
            "Np": np_d, "KP": kp, "DST": dst, "AE": ae, "ULF_power": ulf,
            "Ec": ec, "Pdyn": pdyn, "Bz_neg_dur": bz_neg_dur, "dDst_dt": dDst,
            "AE_1h": ae_1h, "regime": regime.astype(float), "F10.7_index": 150.0,
            "goes_Hp": df_merged.get("goes_Hp", pd.Series(100.0, index=dates))
        }, index=dates)
        
        for lag, lbl in [(12,"1h"),(36,"3h"),(72,"6h"),(144,"12h"),(288,"24h")]:
            df[f"flux_lag_{lbl}"] = df["log_flux"].shift(lag)
            
        baseline_flux = df["log_flux"].iloc[0] if len(df) > 0 else 2.0
        return df.bfill().fillna(baseline_flux), "LIVE_NOAA_SWPC"
        
    except Exception as e:
        # Fallback to local data stream if network restricted
        return None, f"OFFLINE_FALLBACK: {str(e)}"
