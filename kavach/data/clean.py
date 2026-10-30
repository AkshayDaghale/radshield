"""
Data Cleaning & Despiking Module
KAVACH — GEO Radiation Monitor | Team DigiIndia
"""
import numpy as np
import pandas as pd

def clean_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw merged time-series:
    - Despiking using median rolling filter
    - Log-transforming flux (clipped to 0.1 pfu to prevent -inf)
    - Linear gap filling for small missing windows (<30 min)
    """
    df = df.copy()

    # Pitfall #3: Log-transform flux with minimum clip at 0.1 pfu
    if 'flux' in df.columns:
        df['flux'] = pd.to_numeric(df['flux'], errors='coerce')
        df['flux'] = df['flux'].clip(lower=0.1)
        df['log_flux'] = np.log10(df['flux'])

    # Despike numeric columns using rolling 5-point median difference
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if col in ['regime']:
            continue
        rolling_med = df[col].rolling(5, center=True, min_periods=1).median()
        std_dev = df[col].std()
        if pd.notna(std_dev) and std_dev > 0:
            spike_mask = (df[col] - rolling_med).abs() > 4 * std_dev
            df.loc[spike_mask, col] = rolling_med[spike_mask]

        # Interpolate small gaps (< 6 steps = 30 minutes)
        df[col] = df[col].interpolate(method='linear', limit=6).bfill().ffill()

    return df
