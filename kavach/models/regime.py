"""
Magnetospheric State Regime Classifier Module
KAVACH — GEO Radiation Monitor | Team DigiIndia
"""
REGIME_LABELS = {
    0: 'Quiet Phase (Nominal Background)',
    1: 'Storm Onset (Rapid Injection)',
    2: 'Main Phase (Peak Compression)',
    3: 'Recovery Phase (Radial Relaxation)'
}

def classify_regime(row: dict) -> int:
    """
    Classifies geomagnetic regime:
    0 = Quiet, 1 = Storm Onset, 2 = Main Phase, 3 = Recovery
    """
    Kp = row.get('KP', 0)
    Dst = row.get('DST', 0)
    dDst = row.get('dDst_dt', 0)

    if Kp > 5 and Dst < -50:
        return 2  # Main Phase
    elif dDst > 5:
        return 3  # Recovery
    elif Kp >= 3 or Dst < -20:
        return 1  # Storm Onset
    else:
        return 0  # Quiet
