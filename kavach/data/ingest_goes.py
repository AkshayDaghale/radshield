"""
GOES >2 MeV Electron Flux Data Ingestion Module
KAVACH — GEO Radiation Monitor | Team DigiIndia
"""
import glob
import os
import numpy as np
import pandas as pd

GOES_FLUX_VARS = {
    'goes13': 'E2ChanP',
    'goes15': 'E2ChanP',
    'goes16': 'AvgDiffElectronFlux',
    'goes17': 'AvgDiffElectronFlux',
    'goes18': 'AvgDiffElectronFlux',
}

def load_goes_cdf(filepath: str, satellite: str = 'goes16') -> pd.DataFrame:
    """Reads a GOES CDF file using cdflib and extracts >2 MeV electron flux."""
    try:
        import cdflib
        cdf = cdflib.CDF(filepath)
        epoch = cdflib.cdfepoch.to_datetime(cdf['Epoch'])
        flux_var = GOES_FLUX_VARS.get(satellite.lower(), 'AvgDiffElectronFlux')
        
        # Check available variables if requested variable is missing
        info = cdf.cdf_info()
        all_vars = info.get('rVariables', []) + info.get('zVariables', [])
        if flux_var not in all_vars:
            for candidate in ['AvgDiffElectronFlux', 'E2ChanP', 'E3', 'P5', 'P8', 'flux']:
                if candidate in all_vars:
                    flux_var = candidate
                    break

        flux = cdf[flux_var][:].astype(float)
        # Reshape if multi-dimensional
        if len(flux.shape) > 1:
            flux = flux[:, 0]

        # Pitfall #3: Remove fill/non-positive values
        flux = np.where(flux <= 0, np.nan, flux)

        df = pd.DataFrame({'flux': flux}, index=pd.DatetimeIndex(epoch))
        df.index = df.index.tz_localize(None)
        return df.resample('5min').mean()
    except Exception as e:
        print(f"Warning: Could not parse CDF file {filepath}: {e}")
        return pd.DataFrame()

def load_goes_csv(filepath: str) -> pd.DataFrame:
    """Fallback reader for NOAA NGDC CSV files."""
    df = pd.read_csv(filepath, comment='#', parse_dates=True)
    if 'datetime' in df.columns:
        df = df.set_index('datetime')
    flux_col = [c for c in df.columns if 'flux' in c.lower() or 'e2' in c.lower()]
    target_col = flux_col[0] if flux_col else df.columns[0]
    
    df['flux'] = pd.to_numeric(df[target_col], errors='coerce')
    df['flux'] = np.where(df['flux'] <= 0, np.nan, df['flux'])
    return df[['flux']].resample('5min').mean()

def load_all_goes(data_dir: str) -> pd.DataFrame:
    """Recursively loads and merges all GOES data in data_dir."""
    frames = []
    cdf_files = sorted(glob.glob(os.path.join(data_dir, '**/*.cdf'), recursive=True))
    for f in cdf_files:
        sat = 'goes16' if 'g16' in f.lower() else ('goes18' if 'g18' in f.lower() else 'goes15')
        df = load_goes_cdf(f, sat)
        if not df.empty:
            frames.append(df)
            
    csv_files = sorted(glob.glob(os.path.join(data_dir, '**/*.csv'), recursive=True))
    for f in csv_files:
        if 'grasp' not in f.lower() and 'omni' not in f.lower():
            df = load_goes_csv(f)
            if not df.empty:
                frames.append(df)

    if not frames:
        return pd.DataFrame(columns=['flux'])

    merged = pd.concat(frames).sort_index()
    return merged.loc[~merged.index.duplicated(keep='first')]
