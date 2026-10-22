"""
Synthetic 11-Year Space-Weather Bootstrap
RADSHIELD — GEO Radiation Monitor

This module generates formula-based sample data with simulated solar-cycle patterns.
It does not download or reconstruct measured NASA/NOAA telemetry. Its output is
synthetic and must not be described or used as a measured historical archive.
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_synthetic_11yr_bootstrap(output_path: str = "radshield/data/synthetic_11yr_bootstrap.csv", years: int = 11, seed: int = 2026):
    """
    Generate simulated space-weather data for pipeline experiments, not model validation.
    """
    print(f"[RADSHIELD-DATA] ==========================================================")
    print(f"[RADSHIELD-DATA] GENERATING SYNTHETIC SPACE-WEATHER SAMPLE DATA")
    print(f"[RADSHIELD-DATA] Simulated Years : {years}")
    print(f"[RADSHIELD-DATA] Temporal Cadence: 5 minutes")
    print(f"[RADSHIELD-DATA] Output          : {output_path}")
    print(f"[RADSHIELD-DATA] ==========================================================")
    
    start_date = pd.Timestamp("2014-01-01 00:00:00", tz="UTC")
    n_steps = int(years * 365.25 * 288)
    dates = pd.date_range(start=start_date, periods=n_steps, freq="5min")
    
    print(f"[RADSHIELD-DATA] Step 1/4: Generating synthetic periodic solar-wind features...")
    t = np.linspace(0, years * 2 * np.pi, n_steps) # 11-year solar cycle phase
    t_rot = np.linspace(0, years * (365.25 / 27.27) * 2 * np.pi, n_steps) # 27-day Bartels solar rotation phase
    
    np.random.seed(seed)
    
    # Base solar wind speed modulated by 11-yr sunspot cycle and 27-day Coronal Hole High Speed Streams (CHHSS)
    vsw_base = 400 + 80 * np.sin(t)**2 + 120 * np.sin(t_rot)**4
    vsw_noise = np.random.normal(0, 18, n_steps)
    vsw = np.clip(vsw_base + vsw_noise, 280, 1100)
    
    print(f"[RADSHIELD-DATA] Step 2/4: Injecting historical geomagnetic storms & IMF variations...")
    # Interplanetary Magnetic Field (IMF By, Bz in GSM coordinates)
    bz_cycle = 3 * np.cos(t_rot * 1.5) + np.random.normal(0, 2.5, n_steps)
    by_cycle = 4 * np.sin(t_rot * 1.2) + np.random.normal(0, 2.5, n_steps)
    
    # Add illustrative simulated storm-like intervals.
    storm_indices = np.random.choice(n_steps, size=int(years * 12), replace=False) # ~12 storms per year
    for idx in storm_indices:
        dur = np.random.randint(288, 288 * 3) # 1 to 3 days duration
        end_idx = min(idx + dur, n_steps)
        vsw[idx:end_idx] += np.random.uniform(250, 500)
        bz_cycle[idx:end_idx] -= np.random.uniform(15, 35) # Southward IMF turn
        
    bz = np.clip(bz_cycle, -45, 30)
    by = np.clip(by_cycle, -30, 30)
    np_d = np.clip(5 + 4 * np.cos(t_rot) + np.random.exponential(2, n_steps), 0.5, 80)
    
    print(f"[RADSHIELD-DATA] Step 3/4: Computing empirical magnetospheric coupling & GEO radiation belts...")
    # Kp and Dst calculation
    kp = np.clip(2 + 0.008 * (vsw - 400) + 0.25 * np.abs(np.minimum(bz, 0)) + np.random.normal(0, 0.2, n_steps), 0, 9)
    dst = -10 - 18 * (kp / 3.0)**1.6 - 1.5 * np.maximum(vsw - 500, 0)**0.8 + np.random.normal(0, 3, n_steps)
    ae = np.clip(100 + 130 * (kp / 2.0) + np.random.exponential(40, n_steps), 50, 3000)
    ulf = np.clip(-3.5 + 0.45 * (kp / 3.0) + 0.001 * (vsw - 400) + np.random.normal(0, 0.15, n_steps), -5.0, -1.0)
    
    # Boynton-Balasis Solar Wind Coupling Functions
    bt = np.sqrt(by**2 + bz**2)
    theta = np.arctan2(by, bz)
    ec = np.clip((vsw**(4/3)) * ((bt * np.abs(np.sin(theta / 2)))**(8/3)), 0, None)
    pdyn = np.clip(0.5 * 1.67e-27 * (np_d * 1e6) * ((vsw * 1e3)**2) * 1e9, 0.1, 80.0)
    
    # GEO Electron Flux (>2 MeV) physical formulation with Fokker-Planck radial diffusion delay
    log_flux_raw = 2.2 + 0.006 * (vsw - 400) + 0.32 * (kp - 2) + 0.45 * (ulf + 3.5)
    log_flux = pd.Series(log_flux_raw).ewm(span=36).mean().values # 3-hour radial diffusion inertia
    flux = np.clip(10**log_flux, 0.1, 1e6)
    
    bz_neg = pd.Series((bz < 0).astype(int))
    bz_neg_dur = bz_neg.groupby((bz_neg != bz_neg.shift()).cumsum()).cumcount().values * 5.0
    dDst = np.gradient(dst) / 5.0
    ae_1h = pd.Series(ae).rolling(12, min_periods=1).mean().values
    regime = np.where(kp >= 6, 2, np.where(kp >= 3, 1, 0)).astype(float)
    
    print(f"[RADSHIELD-DATA] Step 4/4: Constructing 19-feature matrix and exporting to disk...")
    df = pd.DataFrame({
        "time_tag": dates,
        "data_origin": "synthetic_space_weather_bootstrap",
        "flux": flux,
        "log_flux": np.log10(np.maximum(flux, 1e-3)),
        "Vsw": vsw,
        "BZ_GSM": bz,
        "BY_GSM": by,
        "BT": bt,
        "Np": np_d,
        "KP": kp,
        "DST": dst,
        "AE": ae,
        "ULF_power": ulf,
        "Ec": ec,
        "Pdyn": pdyn,
        "Bz_neg_dur": bz_neg_dur,
        "dDst_dt": dDst,
        "AE_1h": ae_1h,
        "regime": regime
    })
    
    for lag, lbl in [(12, "1h"), (36, "3h"), (72, "6h"), (144, "12h"), (288, "24h")]:
        df[f"flux_lag_{lbl}"] = df["log_flux"].shift(lag).bfill()
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[RADSHIELD-DATA] ==========================================================")
    print(f"[RADSHIELD-DATA] SYNTHETIC SAMPLE DATA EXPORTED (NOT MEASURED TELEMETRY)")
    print(f"[RADSHIELD-DATA] Total Rows Exported : {len(df):,}")
    print(f"[RADSHIELD-DATA] File Size on Disk   : {file_size_mb:.2f} MB")
    print(f"[RADSHIELD-DATA] Synthetic Date Index: {dates[0]} to {dates[-1]}")
    print(f"[RADSHIELD-DATA] Do not treat this output as measured training or evaluation data.")
    print(f"[RADSHIELD-DATA] ==========================================================")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic space-weather sample data")
    parser.add_argument("--output", type=str, default="radshield/data/synthetic_11yr_bootstrap.csv", help="Output file path")
    parser.add_argument("--years", type=int, default=11, help="Number of years to simulate")
    args = parser.parse_args()
    
    generate_synthetic_11yr_bootstrap(args.output, args.years)
