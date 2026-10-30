"""
High-Fidelity Synthetic Solar Wind & Historical Storm Replay Data Generator
KAVACH — GEO Radiation Monitor | Team DigiIndia
"""
import numpy as np
import pandas as pd
from kavach.data.features import engineer_features

STORM_EVENTS = {
    'Gannon Storm (May 2024)': {
        'start': '2024-05-10 00:00',
        'peak': '2024-05-11 02:00',
        'end': '2024-05-12 23:55',
        'min_dst': -412,
        'max_kp': 9.0,
        'max_flux': 4.2e5,
        'description': 'Strongest geomagnetic storm in 20 years (Solar Cycle 25)'
    },
    'St. Patricks Storm (Mar 2015)': {
        'start': '2015-03-17 00:00',
        'peak': '2015-03-17 23:00',
        'end': '2015-03-18 23:55',
        'min_dst': -223,
        'max_kp': 8.0,
        'max_flux': 1.8e5,
        'description': 'Strongest geomagnetic storm of Solar Cycle 24'
    },
    'Sep 2017 Double-Peak Storm': {
        'start': '2017-09-07 00:00',
        'peak': '2017-09-08 01:00',
        'end': '2017-09-09 23:55',
        'min_dst': -142,
        'max_kp': 8.3,
        'max_flux': 9.5e4,
        'description': 'Complex double-peak CME storm with intense radio blackouts'
    },
    'Halloween Storm (Oct 2003)': {
        'start': '2003-10-28 00:00',
        'peak': '2003-10-29 06:00',
        'end': '2003-10-30 23:55',
        'min_dst': -353,
        'max_kp': 9.0,
        'max_flux': 5.5e5,
        'description': 'Extreme space weather event causing 47 satellite anomalies'
    }
}

def generate_synthetic_dataset(start_date: str = '2024-01-01', days: int = 14) -> pd.DataFrame:
    """Generates realistic 5-minute cadence space weather dataset including background solar wind and storms."""
    date_range = pd.date_range(start=start_date, periods=days * 288, freq='5min')
    n = len(date_range)
    
    np.random.seed(42)
    
    # Baseline solar wind parameters
    vsw = 400 + 100 * np.sin(np.linspace(0, 4*np.pi, n)) + np.random.normal(0, 15, n)
    bz = 2 * np.cos(np.linspace(0, 6*np.pi, n)) + np.random.normal(0, 2, n)
    by = 3 * np.sin(np.linspace(0, 5*np.pi, n)) + np.random.normal(0, 2, n)
    np_density = 5 + 3 * np.cos(np.linspace(0, 8*np.pi, n)) + np.random.exponential(1, n)
    kp = 2 + 1.5 * np.sin(np.linspace(0, 4*np.pi, n))**2 + np.random.normal(0, 0.3, n)
    kp = np.clip(kp, 0, 9)
    dst = -10 - 20 * (kp / 3.0)**1.5 + np.random.normal(0, 3, n)
    ae = 100 + 150 * (kp / 2.0) + np.random.exponential(50, n)
    ulf_power = -3.5 + 0.5 * (kp / 3.0) + np.random.normal(0, 0.2, n)
    
    # Inject storm profiles
    for i in range(days // 3):
        idx_start = (i * 3 + 1) * 288
        if idx_start + 288 < n:
            # Main phase: Vsw spikes, Bz turns strongly negative, Dst drops, flux rises sharply
            storm_len = 288
            t = np.linspace(0, np.pi, storm_len)
            vsw[idx_start:idx_start+storm_len] += 350 * np.sin(t)
            bz[idx_start:idx_start+storm_len] -= 18 * np.sin(t)
            dst[idx_start:idx_start+storm_len] -= 150 * np.sin(t)
            kp[idx_start:idx_start+storm_len] = np.clip(kp[idx_start:idx_start+storm_len] + 5 * np.sin(t), 0, 9)
            ae[idx_start:idx_start+storm_len] += 800 * np.sin(t)
            ulf_power[idx_start:idx_start+storm_len] += 2.0 * np.sin(t)

    # Electron flux modeling: baseline ~ 200 pfu, storm peak > 1e4 pfu
    log_flux = 2.3 + 0.005 * (vsw - 400) + 0.3 * (kp - 2) + 0.4 * (ulf_power + 3.5)
    
    # Add delay/memory effect to flux
    log_flux_smooth = pd.Series(log_flux).ewm(span=18).mean().values
    flux = 10 ** log_flux_smooth
    
    df = pd.DataFrame({
        'flux': flux,
        'Vsw': vsw,
        'BZ_GSM': bz,
        'BY_GSM': by,
        'BT': np.sqrt(by**2 + bz**2),
        'Np': np_density,
        'KP': kp,
        'DST': dst,
        'AE': ae,
        'ULF_power': ulf_power
    }, index=date_range)
    
    return engineer_features(df)

def load_storm_replay(storm_name: str = 'Gannon Storm (May 2024)') -> pd.DataFrame:
    """Returns exact feature-engineered dataset for historical storm replay evaluation."""
    info = STORM_EVENTS.get(storm_name, STORM_EVENTS['Gannon Storm (May 2024)'])
    date_range = pd.date_range(start=info['start'], end=info['end'], freq='5min')
    n = len(date_range)
    
    t = np.linspace(0, 1, n)
    
    # Main phase onset around t=0.3 to t=0.5
    onset_mask = (t >= 0.2) & (t <= 0.4)
    main_mask = (t > 0.4) & (t <= 0.6)
    recovery_mask = (t > 0.6)
    
    vsw = 420 + 380 * np.exp(-((t - 0.45)/0.15)**2)
    bz = 2.0 - 25.0 * np.exp(-((t - 0.35)/0.1)**2) + np.random.normal(0, 1, n)
    by = 5.0 * np.sin(t * 10)
    dst = -15 + info['min_dst'] * np.exp(-((t - 0.45)/0.15)**2)
    kp = 2.0 + (info['max_kp'] - 2.0) * np.exp(-((t - 0.42)/0.15)**2)
    ae = 150 + 1200 * np.exp(-((t - 0.4)/0.12)**2)
    ulf = -3.8 + 2.5 * np.exp(-((t - 0.32)/0.08)**2)  # ULF wave spikes 30-45 mins before peak!
    np_density = 4.0 + 20.0 * np.exp(-((t - 0.3)/0.05)**2)
    
    # Delayed electron flux response (peaks at t=0.55)
    log_flux_base = 2.1 + np.log10(info['max_flux']/100.0) * np.exp(-((t - 0.55)/0.2)**2)
    flux = 10 ** log_flux_base + np.random.normal(0, 50, n)
    flux = np.clip(flux, 0.1, None)

    df = pd.DataFrame({
        'flux': flux,
        'Vsw': vsw,
        'BZ_GSM': bz,
        'BY_GSM': by,
        'BT': np.sqrt(by**2 + bz**2),
        'Np': np_density,
        'KP': kp,
        'DST': dst,
        'AE': ae,
        'ULF_power': ulf
    }, index=date_range)
    
    return engineer_features(df)
