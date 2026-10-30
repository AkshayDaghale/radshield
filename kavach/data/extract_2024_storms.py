import os
import glob
import pandas as pd
import xarray as xr
import numpy as np
import warnings

warnings.filterwarnings('ignore')

DATA_DIR = r"c:\Users\Dange\OneDrive\Desktop\ISRO\11yr_dataset"
OUT_DIR = r"c:\Users\Dange\OneDrive\Desktop\ISRO\kavach\data\historical"

os.makedirs(OUT_DIR, exist_ok=True)

def process_goes16_folder(folder_path):
    print(f">> Parsing GOES-16 NetCDF in {folder_path}...")
    nc_files = glob.glob(os.path.join(folder_path, '*.nc'))
    dfs = []
    for f in sorted(nc_files):
        try:
            ds = xr.open_dataset(f)
            time = ds['time'].values
            flux = ds['AvgIntElectronFlux'].values[:, 0]
            df = pd.DataFrame({'datetime': time, 'flux': flux})
            df.set_index('datetime', inplace=True)
            df.loc[df['flux'] < -9000, 'flux'] = np.nan
            df = df.resample('5min').mean()
            ds.close()
            dfs.append(df)
        except Exception as e:
            print(f"Error parsing {f}: {e}")
    if dfs:
        return pd.concat(dfs)
    return pd.DataFrame()

def process_omni():
    print(">> Parsing 2024 OMNI Solar Wind...")
    f = os.path.join(DATA_DIR, 'omni_5min2024.asc')
    df = pd.read_csv(f, sep=r'\s+', header=None, on_bad_lines='skip')
    df.rename(columns={0: 'Year', 1: 'Day', 2: 'Hour', 3: 'Minute', 
                       14: 'BX_GSM_gse', 17: 'BY_GSM', 18: 'BZ_GSM',
                       21: 'Vsw', 25: 'Np', 26: 'Temperature', 27: 'Flow_Pressure', 37: 'AE'}, inplace=True)
    df['BX_GSM'] = df['BX_GSM_gse'] # GSE Bx == GSM Bx
    df['datetime'] = pd.to_datetime(df['Year'].astype(int).astype(str) + df['Day'].astype(int).astype(str).str.zfill(3), format='%Y%j') + \
                     pd.to_timedelta(df['Hour'], unit='h') + pd.to_timedelta(df['Minute'], unit='m')
    
    df.replace([999.9, 9999.99, 99999.9, 9999999.], np.nan, inplace=True)
    keep_cols = ['datetime', 'BX_GSM', 'BY_GSM', 'BZ_GSM', 'Vsw', 'Np', 'Flow_Pressure', 'AE']
    df = df[keep_cols]
    df.set_index('datetime', inplace=True)
    return df

def build_benchmark(storm_name, start_date, end_date, goes_df, omni_df):
    print(f">> Building {storm_name} ({start_date} to {end_date})...")
    # Slice Data
    mask_goes = (goes_df.index >= start_date) & (goes_df.index <= end_date)
    mask_omni = (omni_df.index >= start_date) & (omni_df.index <= end_date)
    
    df_g = goes_df.loc[mask_goes] if not goes_df.empty else pd.DataFrame(columns=['flux'], index=pd.date_range(start_date, end_date, freq='5min'))
    df_o = omni_df.loc[mask_omni]
    
    # Merge on 5-min intervals
    dates = pd.date_range(start=start_date, end=end_date, freq='5min')
    df = pd.DataFrame(index=dates)
    df = df.join(df_o).join(df_g)
    
    # Interpolate small gaps, fill large ones with baseline
    df.ffill(limit=12, inplace=True) # Forward fill up to 1 hour
    
    # Baseline fallbacks (same as noaa_ingest.py)
    df['Vsw'].fillna(400.0, inplace=True)
    df['BX_GSM'].fillna(0.0, inplace=True)
    df['BY_GSM'].fillna(0.0, inplace=True)
    df['BZ_GSM'].fillna(0.0, inplace=True)
    df['Np'].fillna(5.0, inplace=True)
    df['flux'].fillna(100.0, inplace=True) # Safe baseline
    
    # Compute Derived Features exactly like noaa_ingest.py
    bz = df['BZ_GSM'].values
    vsw = df['Vsw'].values
    bx = df['BX_GSM'].values
    by = df['BY_GSM'].values
    np_d = df['Np'].values
    
    df['BT'] = np.sqrt(bx**2 + by**2 + bz**2)
    pdyn_fallback = 0.5 * 1.67e-27 * (np_d * 1e6) * (vsw * 1e3)**2 * 1e9
    df['Pdyn'] = df['Flow_Pressure'].fillna(pd.Series(pdyn_fallback, index=df.index))
    del df['Flow_Pressure']  # Drop to prevent duplicate columns in the dashboard
    
    theta = np.arctan2(by, bz)
    theta = np.where(theta < 0, theta + 2*np.pi, theta)
    df['Ec'] = vsw * df['BT'] * np.sin(theta/2.0)**2 * 1e-3
    
    # Fallback Kp if missing (OMNI doesn't have 5-min Kp, we use the heuristic)
    kp = np.clip(2 + 0.01*(vsw-400) + 0.3*np.abs(bz), 0, 9)
    df['KP'] = kp
    
    dst_raw = -10 - 15*(kp/3.0)**1.5
    df['DST'] = pd.Series(dst_raw).ewm(span=6).mean().values
    df['dDst_dt'] = np.clip(np.gradient(df['DST']) / 5.0, -1.5, 1.5)
    
    ae_fallback = 100 + 120*(kp/2.0)
    ae = df['AE'].fillna(pd.Series(ae_fallback, index=df.index))
    df['AE'] = ae
    df['AE_1h'] = pd.Series(ae).rolling(12, min_periods=1).mean().values
    
    # ULF Wave Power proxy
    df['ULF_power'] = np.clip(-3.0 + 0.05*(vsw-400) + 0.1*np.abs(bz), -4.0, 1.0)
    
    bz_neg = pd.Series((bz < 0).astype(int))
    df['Bz_neg_dur'] = bz_neg.groupby((bz_neg != bz_neg.shift()).cumsum()).cumcount().values * 5.0
    
    # Clip the raw flux at 1.0 pfu to prevent instrument noise from creating 
    # massive spikes on the log scale during severe flux dropouts!
    df['flux'] = np.clip(df['flux'], 1.0, None)
    # Apply a 1-hour rolling median to smooth out the noise-floor barcode completely
    df['flux'] = df['flux'].rolling(window=12, min_periods=1, center=True).median()
    df['log_flux'] = np.log10(df['flux'])
    
    regime = np.zeros(len(df))
    regime[kp >= 6] = 2
    regime[(kp >= 3) & (kp < 6)] = 1
    regime[(df['dDst_dt'] > 0) & (df['DST'] < -50)] = 3
    df['regime'] = regime.astype(float)
    df['F10.7_index'] = 150.0
    df['goes_Hp'] = 100.0
    
    for lag, lbl in [(12,"1h"),(36,"3h"),(72,"6h"),(144,"12h"),(288,"24h")]:
        df[f"flux_lag_{lbl}"] = df["log_flux"].shift(lag)
        
    df.bfill(inplace=True)
    df.fillna(2.0, inplace=True)
    df.index.name = 'datetime'
    
    save_path = os.path.join(OUT_DIR, f"{storm_name}.csv")
    df.to_csv(save_path)
    print(f"Saved {save_path} ({len(df)} rows)")

if __name__ == '__main__':
    omni_df = process_omni()
    
    goes_may = process_goes16_folder(os.path.join(DATA_DIR, 'GOES16_STORMS', 'MAY_2024'))
    build_benchmark('may_2024_benchmark', '2024-05-10', '2024-05-14', goes_may, omni_df)
    
    goes_oct = process_goes16_folder(os.path.join(DATA_DIR, 'GOES16_STORMS', 'OCT_2024'))
    build_benchmark('oct_2024_benchmark', '2024-10-09', '2024-10-13', goes_oct, omni_df)
    
    print("Done!")
