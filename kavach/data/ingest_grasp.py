"""
GRASP (GSAT-19) Indian GEO Sector Payload Data Ingestion
KAVACH — GEO Radiation Monitor | Team DigiIndia
"""
import glob
import os
import numpy as np
import pandas as pd

def load_grasp(filepath: str) -> pd.DataFrame:
    """
    Parses GSAT-19 GRASP payload CSV files.
    GRASP is located at 48°E Indian longitude sector, used for fine-tuning & domain adaptation.
    """
    try:
        df = pd.read_csv(filepath, comment='#')
        # Check column names dynamically
        date_col = [c for c in df.columns if 'time' in c.lower() or 'date' in c.lower()]
        if date_col:
            df['datetime'] = pd.to_datetime(df[date_col[0]], errors='coerce')
            df = df.dropna(subset=['datetime']).set_index('datetime').sort_index()

        flux_col = [c for c in df.columns if 'flux' in c.lower() or '2mev' in c.lower() or 'grasp' in c.lower()]
        target_col = flux_col[0] if flux_col else df.columns[-1]

        df['flux'] = pd.to_numeric(df[target_col], errors='coerce')
        # Remove non-positive fill values
        df['flux'] = df['flux'].where(df['flux'] > 0)
        return df[['flux']].resample('5min').mean()
    except Exception as e:
        print(f"Warning: Could not parse GRASP file {filepath}: {e}")
        return pd.DataFrame(columns=['flux'])

def load_all_grasp(data_dir: str) -> pd.DataFrame:
    """Loads and merges all GRASP files in data_dir."""
    frames = []
    files = sorted(glob.glob(os.path.join(data_dir, '**/*grasp*.csv'), recursive=True))
    for f in files:
        df = load_grasp(f)
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame(columns=['flux'])

    merged = pd.concat(frames).sort_index()
    return merged.loc[~merged.index.duplicated(keep='first')]
