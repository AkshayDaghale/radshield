"""
OMNI Solar Wind Parameters Ingestion Module
KAVACH — GEO Radiation Monitor | Team DigiIndia
"""
import glob
import os
import numpy as np
import pandas as pd

OMNI_COLS = ['Vsw', 'BZ_GSM', 'BY_GSM', 'BT', 'Np', 'Pressure', 'KP', 'DST', 'AE']
FILL_VALUES = {col: 9999.9 for col in OMNI_COLS}

def load_omni_cdf(filepath: str) -> pd.DataFrame:
    """Reads OMNI HRO 5-minute CDF file and strips fill values."""
    try:
        import cdflib
        cdf = cdflib.CDF(filepath)
        epoch = cdflib.cdfepoch.to_datetime(cdf['Epoch'])
        data = {}
        
        info = cdf.cdf_info()
        all_vars = info.get('rVariables', []) + info.get('zVariables', [])

        for col in OMNI_COLS:
            matched_var = None
            for v in all_vars:
                if col.lower() == v.lower():
                    matched_var = v
                    break
            
            if matched_var:
                arr = cdf[matched_var][:].astype(float)
                if len(arr.shape) > 1:
                    arr = arr[:, 0]
                # Pitfall #1: Replace fill value 9999.9 with NaN immediately
                arr[arr >= 9998.0] = np.nan
                arr[arr <= -9998.0] = np.nan
                data[col] = arr
            else:
                data[col] = np.nan

        df = pd.DataFrame(data, index=pd.DatetimeIndex(epoch))
        df.index = df.index.tz_localize(None)
        return df.resample('5min').mean()
    except Exception as e:
        print(f"Warning: Could not parse OMNI CDF file {filepath}: {e}")
        return pd.DataFrame(columns=OMNI_COLS)

def load_omni_ascii(filepath: str) -> pd.DataFrame:
    """Reads OMNI ASCII formatted space data."""
    try:
        df = pd.read_csv(filepath, delim_whitespace=True, header=None)
        # Parse standard OMNI format columns if headerless
        df[df >= 9998.0] = np.nan
        df[df <= -9998.0] = np.nan
        return df
    except Exception as e:
        print(f"Warning: Could not parse OMNI ASCII file {filepath}: {e}")
        return pd.DataFrame(columns=OMNI_COLS)

def load_all_omni(data_dir: str) -> pd.DataFrame:
    """Loads and merges all OMNI files in data_dir."""
    frames = []
    files = sorted(glob.glob(os.path.join(data_dir, '**/*omni*.cdf'), recursive=True))
    for f in files:
        df = load_omni_cdf(f)
        if not df.empty:
            frames.append(df)
            
    if not frames:
        return pd.DataFrame(columns=OMNI_COLS)

    merged = pd.concat(frames).sort_index()
    return merged.loc[~merged.index.duplicated(keep='first')]
