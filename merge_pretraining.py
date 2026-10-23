import os
import glob
import pandas as pd
import xarray as xr
import numpy as np
import warnings

# Suppress xarray/pandas warnings for clean output
warnings.filterwarnings('ignore')

DATA_DIR = r"c:\Users\Dange\OneDrive\Desktop\ISRO\11yr_dataset"

def process_omni():
    print(">> Parsing OMNI Solar Wind (11 Years)...")
    omni_files = glob.glob(os.path.join(DATA_DIR, 'omni_5min*.asc'))
    dfs = []
    for f in sorted(omni_files):
        df = pd.read_csv(f, sep=r'\s+', header=None, on_bad_lines='skip')
        df.rename(columns={0: 'Year', 1: 'Day', 2: 'Hour', 3: 'Minute', 
                           18: 'Bz_GSM', 21: 'Flow_Speed', 25: 'Proton_Density', 
                           26: 'Temperature', 27: 'Flow_Pressure'}, inplace=True)
        df = df[(df['Year'] >= 2013) & (df['Year'] <= 2025) & (df['Day'] >= 1)]
        df['datetime'] = pd.to_datetime(df['Year'].astype(int).astype(str) + df['Day'].astype(int).astype(str).str.zfill(3), format='%Y%j') + \
                         pd.to_timedelta(df['Hour'], unit='h') + pd.to_timedelta(df['Minute'], unit='m')
        dfs.append(df)
        
    df_omni = pd.concat(dfs, ignore_index=True)
    # FIX 3: Expanded OMNI/NOAA missing fill value list — NASA OMNIWeb uses many different
    # placeholder values depending on the parameter (Bz=9999.99, V=99999.9, Pdyn=99.99, etc.)
    omni_fill_vals = [9999999., 99999.9, 99999.0, 9999.99, 9999.0, 999.99, 999.9, 999.0, 99.99, 99.0]
    num_cols_omni = df_omni.select_dtypes(include=[np.number]).columns
    df_omni[num_cols_omni] = df_omni[num_cols_omni].replace(omni_fill_vals, np.nan)
    
    # We only need key solar wind params for baseline
    keep_cols = ['datetime', 'Bz_GSM', 'Flow_Speed', 'Proton_Density', 'Temperature', 'Flow_Pressure']
    df_omni = df_omni[keep_cols]
    df_omni.set_index('datetime', inplace=True)
    return df_omni

def process_goes15_csv():
    print(">> Parsing GOES-15 EPEAD CSVs (5-min)...")
    g15_files = glob.glob(os.path.join(DATA_DIR, 'g15_epead*.csv'))
    dfs = []
    for f in sorted(g15_files):
        skip = 0
        with open(f, 'r') as file:
            for i, line in enumerate(file):
                if line.startswith('time_tag,'):
                    skip = i
                    break
        df = pd.read_csv(f, skiprows=skip)
        df['datetime'] = pd.to_datetime(df['time_tag'].str.replace('Z', ''))
        
        # Use Westward uncorrected flux as it closely mimics the standard >2MeV reading
        df = df[['datetime', 'E2W_UNCOR_FLUX']].rename(columns={'E2W_UNCOR_FLUX': 'electron_flux'})
        df.set_index('datetime', inplace=True)
        # Handle extreme fill values
        df.loc[df['electron_flux'] < -9000, 'electron_flux'] = np.nan
        dfs.append(df)
    return pd.concat(dfs) if dfs else pd.DataFrame()

def process_goes13_nc():
    print(">> Parsing GOES-13 EPEAD NetCDF (Downsampling 16-sec to 5-min)...")
    g13_files = glob.glob(os.path.join(DATA_DIR, 'GOES13_EPEAD/**/*.nc'), recursive=True)
    dfs = []
    for f in sorted(g13_files):
        try:
            ds = xr.open_dataset(f)
            df = ds.to_dataframe().reset_index()
            if 'E2W_UNCOR_FLUX' in df.columns:
                df = df[['time_tag', 'E2W_UNCOR_FLUX']].rename(columns={'time_tag': 'datetime', 'E2W_UNCOR_FLUX': 'electron_flux'})
                df.set_index('datetime', inplace=True)
                df.loc[df['electron_flux'] < -9000, 'electron_flux'] = np.nan
                # Mathematically downsample the 16-sec readings into exactly 5-minute averages!
                df = df.resample('5min').mean()
                dfs.append(df)
            ds.close()
        except Exception as e:
            pass
    return pd.concat(dfs) if dfs else pd.DataFrame()

from multiprocessing import Pool

def _process_single_g16(f):
    try:
        ds = xr.open_dataset(f)
        time = ds['time'].values
        flux = ds['AvgIntElectronFlux'].values[:, 0]
        df = pd.DataFrame({'datetime': time, 'electron_flux': flux})
        df.set_index('datetime', inplace=True)
        df.loc[df['electron_flux'] < -9000, 'electron_flux'] = np.nan
        df = df.resample('5min').mean()
        ds.close()
        return df
    except Exception:
        return None

def process_goes16_nc():
    print(">> Parsing GOES-16 SEISS NetCDF (Downsampling 1-min & 5-min)...")
    g16_files = glob.glob(os.path.join(DATA_DIR, '**/*.nc'), recursive=True)
    g16_files = [f for f in g16_files if 'sci_mpsh' in f]
    
    with Pool(processes=8) as pool:
        results = pool.map(_process_single_g16, sorted(g16_files))
    
    dfs = [df for df in results if df is not None]
    return pd.concat(dfs) if dfs else pd.DataFrame()

if __name__ == '__main__':
    print("=== STARTING 11-YEAR PRE-TRAINING MERGE ===")
    df_omni = process_omni()
    df_g15 = process_goes15_csv()
    df_g13 = process_goes13_nc()
    df_g16 = process_goes16_nc()
    
    print(">> Merging GOES satellites...")
    df_goes = pd.concat([df_g13, df_g15, df_g16])
    
    # Sort and remove duplicates. Keep the last entry (in case GOES-16 overlaps GOES-15, GOES-16 is preferred)
    df_goes.sort_index(inplace=True)
    df_goes = df_goes[~df_goes.index.duplicated(keep='last')]
    
    print(">> Merging GOES Electron Flux with OMNI Solar Wind...")
    # Merge on exactly matching 5-minute index
    df_final = df_goes.join(df_omni, how='inner')
    
    print(">> Calculating Temporal Lag Features...")
    df_final['flux_t-1h'] = df_final['electron_flux'].shift(12)  # 12 * 5min = 1h
    df_final['flux_t-3h'] = df_final['electron_flux'].shift(36)  # 36 * 5min = 3h
    df_final['flux_t-24h'] = df_final['electron_flux'].shift(288) # 24h
    
    # Drop rows that have NaN because of the lags (the first 24 hours of 2013)
    df_final.dropna(inplace=True)
    
    # Convert flux to log10 to stabilize training (TFT prefers normally distributed targets)
    df_final['log_electron_flux'] = np.log10(np.maximum(df_final['electron_flux'], 1e-5))
    df_final['log_flux_t-1h'] = np.log10(np.maximum(df_final['flux_t-1h'], 1e-5))
    df_final['log_flux_t-3h'] = np.log10(np.maximum(df_final['flux_t-3h'], 1e-5))
    df_final['log_flux_t-24h'] = np.log10(np.maximum(df_final['flux_t-24h'], 1e-5))
    
    output_path = r"c:\Users\Dange\OneDrive\Desktop\ISRO\Kaggle_PreTraining_Dataset.csv"
    df_final.to_csv(output_path)
    print(f"=== SUCCESS! ===")
    print(f"Total Rows: {len(df_final):,}")
    print(f"Saved to: {output_path}")
