"""
INTERMAGNET Magnetometer Ingestion & ULF Pc5 Wave Power Module
KAVACH — GEO Radiation Monitor | Team DigiIndia
"""
import glob
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

def load_iaga2002(filepath: str) -> pd.DataFrame:
    """Parses INTERMAGNET IAGA-2002 .min ASCII file for H-component."""
    rows = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.startswith('DATE') or line.startswith(' '):
                    continue
                if line[0] in ' \t#' or len(line) < 25:
                    continue
                try:
                    parts = line.split()
                    date_str = parts[0]
                    time_str = parts[1]
                    # H component is usually column index 3 (after date, time, doy) or first numeric val
                    val_str = parts[3] if len(parts) > 3 else parts[2]
                    H = float(val_str)
                    if H > 99990.0 or H < -99990.0:
                        H = np.nan
                    rows.append({'datetime': f"{date_str} {time_str}", 'H': H})
                except Exception:
                    continue
        if not rows:
            return pd.DataFrame(columns=['H'])
        df = pd.DataFrame(rows)
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df = df.dropna(subset=['datetime']).set_index('datetime').sort_index()
        return df.resample('1min').mean()
    except Exception as e:
        print(f"Warning: Could not parse IAGA2002 file {filepath}: {e}")
        return pd.DataFrame(columns=['H'])

def compute_ulf_power(H_series: pd.Series, resample_to: str = '5min') -> pd.Series:
    """
    Band-pass filters H-component to Pc5 band (1.7–6.7 mHz) then computes rolling variance.
    Pc5 ULF wave power at Indian longitude (HYB station) is a key 30-min precursor signal.
    """
    if H_series.empty or H_series.dropna().shape[0] < 60:
        return pd.Series(dtype=float)

    # Pitfall #5: Fill small gaps (<30 min) via interpolation
    H_interp = H_series.interpolate(method='linear', limit=30).bfill().ffill()
    H_values = H_interp.values

    # 1-min cadence => fs = 1/60 Hz
    fs = 1.0 / 60.0
    lowcut = 1.7e-3  # 1.7 mHz
    highcut = 6.7e-3 # 6.7 mHz

    try:
        b, a = butter(4, [lowcut, highcut], btype='band', fs=fs)
        H_pc5 = filtfilt(b, a, np.nan_to_num(H_values))
    except Exception as e:
        # Fallback if signal is too short or problematic
        H_pc5 = H_values - np.mean(H_values)

    # Rolling 45-minute variance as wave power estimate
    power_series = pd.Series(H_pc5, index=H_series.index).rolling(45, min_periods=10).var()
    
    # Pitfall #10: Log10 transform ULF power with 1e-6 lower clip for numerical stability
    power_log = np.log10(power_series.clip(lower=1e-6))
    
    return power_log.resample(resample_to).mean()

def load_all_magnetometer(data_dir: str, station_priority: list = ['HYB', 'ABG', 'TIR']) -> pd.DataFrame:
    """Loads magnetometer data starting with highest priority station (HYB - Hyderabad)."""
    for station in station_priority:
        pattern = os.path.join(data_dir, f"**/*{station.lower()}*.min")
        files = sorted(glob.glob(pattern, recursive=True))
        if not files:
            pattern = os.path.join(data_dir, f"**/*{station.upper()}*.min")
            files = sorted(glob.glob(pattern, recursive=True))

        if files:
            frames = [load_iaga2002(f) for f in files]
            merged_H = pd.concat(frames).sort_index()
            merged_H = merged_H.loc[~merged_H.index.duplicated(keep='first')]
            ulf_power = compute_ulf_power(merged_H['H'])
            return pd.DataFrame({'ULF_power': ulf_power})

    return pd.DataFrame(columns=['ULF_power'])
