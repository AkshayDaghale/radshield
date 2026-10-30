"""
Physics-Informed Feature Engineering Module (19 Feature Vector)
KAVACH — GEO Radiation Monitor | Team DigiIndia
"""
import numpy as np
import pandas as pd

def classify_regime(row) -> int:
    """
    Magnetospheric Regime Classifier:
    0: Quiet
    1: Storm Onset
    2: Main Phase
    3: Recovery
    """
    Kp = row.get('KP', 0)
    Dst = row.get('DST', 0)
    dDst = row.get('dDst_dt', 0)

    if Kp > 5 and Dst < -50:
        return 2  # Main Phase
    elif dDst > 5:
        return 3  # Recovery Phase
    elif Kp >= 3 or Dst < -20:
        return 1  # Storm Onset
    else:
        return 0  # Quiet

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers the exact 19 feature vector specified in KAVACH context:
    1. log_flux (t)
    2. flux_lag_1h (12 steps)
    3. flux_lag_3h (36 steps)
    4. flux_lag_6h (72 steps)
    5. flux_lag_12h (144 steps)
    6. flux_lag_24h (288 steps)
    7. Vsw (solar wind speed)
    8. BZ_GSM (IMF Bz)
    9. BY_GSM (IMF By)
    10. Np (proton density)
    11. Pdyn (dynamic pressure nPa)
    12. Ec (Newell IMF coupling function)
    13. DST (ring current index)
    14. dDst_dt (Dst rate of change)
    15. KP (geomagnetic activity index)
    16. AE_1h (1-hour rolling mean auroral electrojet)
    17. ULF_power (INTERMAGNET Pc5 wave power)
    18. Bz_neg_dur (southward IMF duration in minutes)
    19. regime (magnetospheric state code)
    """
    df = df.copy()

    # 1. Ensure log_flux exists
    if 'log_flux' not in df.columns:
        if 'flux' in df.columns:
            df['log_flux'] = np.log10(df['flux'].clip(lower=0.1))
        else:
            df['log_flux'] = 2.0

    # 2. Lagged flux history (radiation belt memory)
    for lag_steps, label in [(12, '1h'), (36, '3h'), (72, '6h'), (144, '12h'), (288, '24h')]:
        df[f'flux_lag_{label}'] = df['log_flux'].shift(lag_steps)

    # Fill defaults for solar wind columns if absent
    for col in ['Vsw', 'BZ_GSM', 'BY_GSM', 'BT', 'Np', 'Pressure', 'KP', 'DST', 'AE', 'ULF_power']:
        if col not in df.columns:
            df[col] = 0.0

    # 3. Newell IMF Coupling Function (Ec)
    Vsw = df['Vsw'].clip(lower=200)
    By = df['BY_GSM']
    Bz = df['BZ_GSM']
    Bt = np.sqrt(By**2 + Bz**2)
    theta = np.arctan2(By, Bz)
    
    sin_half = np.sin(theta / 2.0)
    sin_half_abs = np.abs(sin_half)
    
    # Newell formula: Vsw^(4/3) * (Bt * sin(theta/2))^(8/3)
    df['Ec'] = (Vsw ** (4.0 / 3.0)) * ((Bt * sin_half_abs) ** (8.0 / 3.0))
    df['Ec'] = df['Ec'].clip(lower=0.0).fillna(0.0)

    # 4. Dynamic Pressure (Pdyn)
    # Pdyn = 0.5 * m_p * Np * Vsw^2 converted to nPa
    df['Pdyn'] = 0.5 * 1.67e-27 * (df['Np'] * 1e6) * ((df['Vsw'] * 1e3) ** 2) * 1e9
    df['Pdyn'] = df['Pdyn'].clip(lower=0.1, upper=50.0).fillna(1.5)

    # 6. Southward IMF Duration (Bz_neg_dur in minutes)
    df['Bz_neg'] = (df['BZ_GSM'] < 0).astype(int)
    groups = (df['Bz_neg'] != df['Bz_neg'].shift()).cumsum()
    df['Bz_neg_dur'] = df['Bz_neg'].groupby(groups).cumcount() * 5.0  # 5-min steps

    # 7. Dst Rate of Change (dDst_dt in nT/min)
    df['dDst_dt'] = df['DST'].diff() / 5.0

    # 8. AE 1-Hour Rolling Mean
    df['AE_1h'] = df['AE'].rolling(12, min_periods=1).mean()

    # 9. Regime Classification
    df['regime'] = df.apply(classify_regime, axis=1)

    # Drop early NaNs from lag calculation
    df = df.bfill().fillna(0)
    return df
