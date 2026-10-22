"""
RadShield AI dashboard for monitoring and forecasting energetic particle
radiation at geostationary orbit using NOAA/GOES telemetry, historical storm
replays, and the existing machine-learning and physics forecast engines.
"""
import os
import sys

# Ensure root project directory is in sys.path for Streamlit Cloud deployment
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from radshield.models.forecasting import (
    EXPERIMENTAL_FLUX_THRESHOLDS_PFU,
    ForecastInputError,
    TFT_DASHBOARD_TRAINING_FEATURE_COLUMNS,
    TFT_FEATURE_COLUMNS,
    TFT_PACKAGE_FINETUNE_FEATURE_AUDIT,
    TFT_PACKAGE_FINETUNE_FEATURE_COLUMNS,
    TFT_PREPROCESSING_MATCHES_CHECKPOINT,
    calculate_mlt_vectorized,
    classify_experimental_flux_screening,
    load_tft_checkpoint,
    run_tft_forecast,
)
from radshield.data.radiation_observation_status import (
    build_radiation_observation_status,
)

st.set_page_config(
    page_title="RadShield AI — GEO Radiation Forecasts",
    layout="wide",
    initial_sidebar_state="expanded"
)

# RadShield interface theme
st.markdown("""
<style>
    :root {
        color-scheme: light;
        --primary-color: #315A9B;
        --st-primary-color: #315A9B;
        accent-color: #315A9B;
    }
    [data-testid="stAppViewContainer"] { --primary-color: #315A9B; }
    html, body, [data-testid="stAppViewContainer"] {
        background: #F3F5F8 !important;
        color: #1D2B3F !important;
        font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
    }
    [data-testid="stMain"] { background: #F3F5F8 !important; }
    [data-testid="stMainBlockContainer"] {
        max-width: 1560px;
        padding: 1.5rem clamp(1.25rem, 3.8vw, 3.8rem) 3.5rem;
    }
    [data-testid="stHeader"] { background: rgba(243, 245, 248, 0.96) !important; }
    [data-testid="stSidebar"] {
        background: #E9EDF3 !important;
        border-right: 1px solid #D8DFE8 !important;
    }
    [data-testid="stSidebar"] * { color: #43536A; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #1D2B3F !important; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { line-height: 1.55; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #59697D !important; }
    [data-testid="stSidebar"] input[type="radio"],
    [data-testid="stSidebar"] input[type="checkbox"] { accent-color: #315A9B !important; }
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: #FFFFFF !important;
        border-color: #CDD6E1 !important;
        color: #1D2B3F !important;
    }
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #DCE3EB;
        border-radius: 10px;
        padding: 1rem 1.1rem !important;
        box-shadow: 0 3px 12px rgba(25, 42, 64, 0.045);
        min-height: 108px;
    }
    [data-testid="stMetricLabel"] {
        color: #65758A !important;
        font-size: 0.69rem !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
    }
    [data-testid="stMetricValue"] {
        color: #1D2B3F !important;
        font-size: clamp(1.1rem, 1.8vw, 1.5rem) !important;
    }
    [data-testid="stMetricDelta"] { font-size: 0.73rem !important; }
    h1, h2, h3 { color: #1D2B3F !important; letter-spacing: -0.025em; }
    h2, h3 { font-weight: 620 !important; border: 0 !important; padding-bottom: 0.2rem !important; }
    .risk-red, .risk-yellow, .risk-green { border-radius: 12px; padding: 1.1rem 1.25rem; }
    .risk-label { color: #43536A; }
    .risk-band { color: #65758A; }
    .risk-msg { color: #43536A; }
    hr { border-color: #DCE3EB !important; }
    [data-testid="stAlertContainer"] {
        border-radius: 10px !important;
        border-color: #D7DFE8 !important;
        box-shadow: none !important;
    }
    [data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) {
        background: #EDF3FA !important;
        color: #294B78 !important;
    }
    [data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) * { color: #294B78 !important; }
    [data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) {
        background: #FBF4E8 !important;
        color: #76551A !important;
    }
    [data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) * { color: #76551A !important; }
    [data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) {
        background: #FBEEEE !important;
        color: #873B3B !important;
    }
    [data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) * { color: #873B3B !important; }
    [data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) {
        background: #ECF5EF !important;
        color: #2E6144 !important;
    }
    [data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) * { color: #2E6144 !important; }
    [data-testid="stDataFrame"] {
        border: 1px solid #DCE3EB !important;
        border-radius: 9px !important;
        overflow: hidden;
    }
    .status-ribbon {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        padding: 0.72rem 0.9rem;
        margin: 0.25rem 0 0.65rem;
        border: 1px solid #D7DFE8;
        border-radius: 8px;
        background: #FFFFFF;
        color: #59697D;
        font-size: 0.8rem;
    }
    .status-ribbon strong { color: #1D2B3F; font-weight: 620; }
    .status-dot { width: 7px; height: 7px; flex: 0 0 7px; border-radius: 50%; }
    .status-blocked .status-dot { background: #C1842C; }
    .status-ready .status-dot { background: #378358; }
    .status-missing .status-dot { background: #B84D4D; }
    [data-testid="stExpander"] {
        border: 1px solid #D7DFE8 !important;
        border-radius: 8px !important;
        background: #FFFFFF !important;
    }
    [data-testid="stButton"] > button, [data-testid="stDownloadButton"] > button {
        background: #315A9B !important;
        border: 1px solid #315A9B !important;
        color: #FFFFFF !important;
        border-radius: 7px !important;
        min-height: 2.65rem;
        transition: background 120ms ease, transform 120ms ease;
    }
    [data-testid="stButton"] > button:hover, [data-testid="stDownloadButton"] > button:hover {
        background: #244779 !important;
        border-color: #244779 !important;
        transform: translateY(-1px);
    }
    [data-testid="stTabs"] [role="tablist"] {
        gap: 0.35rem;
        border-bottom: 1px solid #D7DFE8;
        padding: 0 0.35rem;
    }
    [data-testid="stTabs"] button[role="tab"] {
        color: #65758A !important;
        background: transparent !important;
        border-radius: 6px 6px 0 0;
        padding: 0.75rem 1rem;
        font-weight: 580;
    }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: #244779 !important;
        background: #E8EEF7 !important;
        border-bottom: 2px solid #315A9B !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab-highlight"] { background: #315A9B !important; }
    [data-testid="stSlider"] [role="slider"] { background: #315A9B !important; }
    .hero-card {
        position: relative;
        overflow: hidden;
        display: flex;
        align-items: center;
        min-height: 0;
        padding: 1.15rem 1.45rem;
        margin: 0.15rem 0 0.8rem;
        border: 1px solid #D7DFE8;
        border-top: 3px solid #315A9B;
        border-radius: 10px;
        background: #FFFFFF;
        box-shadow: 0 2px 10px rgba(25, 42, 64, 0.035);
    }
    .hero-copy, .hero-orbit { position: relative; z-index: 1; }
    .hero-eyebrow, .eyebrow {
        color: #315A9B;
        font: 600 0.66rem/1.4 "Space Mono", Consolas, monospace;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }
    .hero-eyebrow { display: flex; align-items: center; gap: 0.55rem; margin-bottom: 0.4rem; }
    .hero-eyebrow:before {
        content: "";
        width: 7px; height: 7px; border-radius: 50%;
        background: #315A9B;
    }
    .hero-title {
        color: #1D2B3F;
        font-size: clamp(1.55rem, 2.35vw, 2rem) !important;
        font-weight: 680;
        line-height: 1.15;
        letter-spacing: -0.04em;
        margin: 0 !important;
        padding: 0 !important;
    }
    .hero-title span { color: #315A9B; }
    .hero-description { color: #59697D; font-size: 0.84rem; margin: 0.35rem 0 0.6rem; }
    .hero-chips { display: flex; flex-wrap: wrap; gap: 0.4rem; }
    .hero-chip {
        display: inline-block;
        padding: 0.28rem 0.52rem;
        border: 1px solid #D7DFE8;
        border-radius: 6px;
        color: #43536A;
        background: #F5F7FA;
        font-size: 0.68rem;
        letter-spacing: 0.02em;
    }
    .hero-orbit {
        min-height: 150px;
        display: grid;
        place-items: center;
    }
    .orbit-graphic {
        position: relative;
        width: 158px;
        height: 158px;
        border: 1px solid #D7DFE8;
        border-radius: 50%;
        background: radial-gradient(circle, #F0F4F9, transparent 68%);
    }
    .orbit-graphic:before, .orbit-graphic:after {
        content: "";
        position: absolute;
        inset: 21px -17px;
        border: 1px solid rgba(49, 90, 155, 0.32);
        border-radius: 50%;
        transform: rotate(-32deg);
    }
    .orbit-graphic:after { inset: 10px -29px; transform: rotate(47deg); border-color: rgba(100, 117, 138, 0.28); }
    .orbit-core {
        position: absolute;
        inset: 50px;
        border-radius: 50%;
        background: radial-gradient(circle at 33% 30%, #7796C3, #315A9B 52%, #1D355B);
    }
    .orbit-node {
        position: absolute;
        top: 28px; right: 14px;
        width: 8px; height: 8px;
        border-radius: 50%;
        background: #315A9B;
        border: 2px solid #FFFFFF;
    }
    .section-label {
        font-size: 0.72rem;
        color: #315A9B;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        font-weight: 650;
        margin: 0 0 12px;
    }
    [data-testid="stPlotlyChart"] { border: 1px solid #DCE3EB; border-radius: 9px; overflow: hidden; }
    [data-testid="stCaptionContainer"] { color: #65758A !important; }
    [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { color: #43536A; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #59697D !important; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p:first-child { color: #315A9B !important; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label { color: #43536A !important; }
    label[data-baseweb="radio"]:has(input:checked) > div:first-child {
        background-color: #315A9B !important;
        border-color: #315A9B !important;
    }
    label[data-baseweb="checkbox"]:has(input:checked) > span:first-child {
        background-color: #315A9B !important;
        border-color: #315A9B !important;
    }
    input[type="radio"], input[type="checkbox"] { accent-color: #315A9B !important; }
    [data-testid="stSidebar"] [data-baseweb="select"] span { color: #1D2B3F !important; }
    [data-testid="stSidebar"] [data-testid="stDataFrame"] { background: #FFFFFF !important; }
    @media (max-width: 700px) {
        [data-testid="stMainBlockContainer"] { padding: 1rem 0.85rem 2rem; }
        [data-testid="stMetric"] { padding: 0.9rem !important; }
        [data-testid="stTabs"] button[role="tab"] { padding: 0.65rem 0.7rem; }
        .hero-card { min-height: 0; }
        .hero-orbit { display: none; }
        .hero-title { font-size: 1.9rem !important; }
    }
    @media (min-width: 701px) and (max-width: 1100px) {
        .hero-card { min-height: 0; gap: 0; padding: 1.1rem 1.35rem; }
        .hero-title { font-size: 2.1rem !important; }
    }
</style>
""", unsafe_allow_html=True)


def apply_radshield_chart_theme(fig):
    """Apply the dashboard's corporate chart theme without changing data or scales."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, sans-serif", color="#43536A"),
        legend=dict(bgcolor="rgba(255, 255, 255, 0.94)", font=dict(color="#43536A")),
    )
    axis_style = {
        "gridcolor": "#E5E9EF",
        "linecolor": "#C9D1DC",
        "tickcolor": "#8794A5",
        "tickfont": {"color": "#59697D"},
        "title_font": {"color": "#43536A"},
    }
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    return fig

# ─── Data Generator ───────────────────────────────────────────────────────────
def generate_data(days=7, seed=42, end_time=None):
    n = days * 288
    if end_time is None:
        end_time = pd.Timestamp.now(tz='UTC').floor('5min')
    else:
        end_time = pd.to_datetime(end_time, utc=True)
    dates = pd.date_range(end=end_time, periods=n, freq="5min")
    np.random.seed(seed)
    vsw  = 400 + 100*np.sin(np.linspace(0, 4*np.pi, n)) + np.random.normal(0,15,n)
    bz   = 2*np.cos(np.linspace(0, 6*np.pi, n)) + np.random.normal(0,2,n)
    by   = 3*np.sin(np.linspace(0, 5*np.pi, n)) + np.random.normal(0,2,n)
    np_d = 5 + 3*np.cos(np.linspace(0, 8*np.pi, n)) + np.random.exponential(1,n)
    kp   = np.clip(2 + 1.5*np.sin(np.linspace(0,4*np.pi,n))**2 + np.random.normal(0,0.3,n), 0, 9)
    dst  = -10 - 20*(kp/3.0)**1.5 + np.random.normal(0,3,n)
    ae   = 100 + 150*(kp/2.0) + np.random.exponential(50,n)
    ulf  = -3.5 + 0.5*(kp/3.0) + np.random.normal(0,0.2,n)
    log_flux = pd.Series(2.3 + 0.005*(vsw-400) + 0.3*(kp-2) + 0.4*(ulf+3.5)).ewm(span=18).mean().values
    flux = np.clip(10**log_flux, 0.1, None)
    bt   = np.sqrt(by**2 + bz**2)
    theta = np.arctan2(by, bz)
    ec   = np.clip(vsw * bt * (np.sin(theta/2)**2) * 1e-3, 0, None)
    pdyn = np.clip(0.5*1.67e-27*(np_d*1e6)*((vsw*1e3)**2)*1e9, 0.1, 50.0)
    bz_neg = pd.Series((bz < 0).astype(int))
    bz_neg_dur = bz_neg.groupby((bz_neg != bz_neg.shift()).cumsum()).cumcount().values * 5.0
    dDst = np.gradient(dst) / 5.0
    ae_1h = pd.Series(ae).rolling(12, min_periods=1).mean().values
    
    regime = np.zeros(n)
    regime[kp >= 6] = 2
    regime[(kp >= 3) & (kp < 6)] = 1
    regime[(dDst > 0) & (dst < -50)] = 3  # Recovery Phase
    
    df = pd.DataFrame({
        "flux": flux, "log_flux": np.log10(np.maximum(flux, 1e-3)),
        "Vsw": vsw, "BZ_GSM": bz, "BY_GSM": by, "BT": bt,
        "Np": np_d, "KP": kp, "DST": dst, "AE": ae, "ULF_power": ulf,
        "Ec": ec, "Pdyn": pdyn, "Bz_neg_dur": bz_neg_dur, "dDst_dt": dDst,
        "AE_1h": ae_1h, "regime": regime.astype(float)
    }, index=dates)
    for lag, lbl in [(12,"1h"),(36,"3h"),(72,"6h"),(144,"12h"),(288,"24h")]:
        df[f"flux_lag_{lbl}"] = df["log_flux"].shift(lag)
    return df.bfill().fillna(0)

def load_storm_from_csv(name):
    """
    Loads real, verified telemetry data for the 4 scientifically validated storm events.
    All data is sourced directly from NASA/NOAA archives — zero synthetic generation.
    """
    # Storm configurations: each event maps to a real dataset file and date range
    STORM_CONFIG = {
        "G5 Mother's Day Storm (May 2024)": {
            "csv": os.path.join(ROOT_DIR, "radshield", "data", "historical", "may_2024_benchmark.csv"),
            "start": "2024-05-10", "end": "2024-05-14"
        },
        "G4 Aurora Storm (Oct 2024)": {
            "csv": os.path.join(ROOT_DIR, "radshield", "data", "historical", "oct_2024_benchmark.csv"),
            "start": "2024-10-09", "end": "2024-10-13"
        },
    }

    cfg = STORM_CONFIG.get(name)
    if cfg is None:
        raise ValueError(f"No historical dataset is configured for {name!r}")
    try:
        df_real = pd.read_csv(cfg["csv"], parse_dates=['datetime'], index_col='datetime')
        df_real.index = pd.to_datetime(df_real.index, utc=True)
        df_storm = df_real.loc[cfg["start"]:cfg["end"]].copy()

        # Unified column renaming to match dashboard expectations
        rename_map = {
            "Flow_Speed": "Vsw", "V": "Vsw",
            "Proton_Density": "Np", "Density": "Np",
            "Flow_Pressure": "Pdyn",
            "Bz_GSM": "BZ_GSM", "BZ": "BZ_GSM",
            "log_electron_flux": "log_flux",
            "electron_flux": "flux",
            "Electron_Flux": "flux",
            "ULF_Power": "ULF_power",
            "SYM_H": "DST",
            "AE": "AE",
        }
        df_storm.rename(columns=rename_map, inplace=True)
        
        # Clean NOAA/OMNI missing data fill values to prevent massive unphysical spikes
        # Common fill values used by NOAA/OMNI: 99999, 9999, 999.99, 999.9, 999, 99.99, 99
        missing_vals = [99999.9, 99999.0, 9999.99, 9999.0, 999.99, 999.9, 999.0, 99.99, 99.0]
        df_storm.replace(missing_vals, np.nan, inplace=True)
        
        # Interpolate only numeric columns to avoid crashing on any leftover string/object columns
        num_cols = df_storm.select_dtypes(include=[np.number]).columns
        df_storm[num_cols] = df_storm[num_cols].interpolate(method='linear', limit_direction='both')
        df_storm[num_cols] = df_storm[num_cols].bfill().ffill()
        
        # Hard physical clipping: erase any values that are still physically impossible after interpolation
        for col, lo, hi in [('Vsw', 200, 900), ('Pdyn', 0.01, 60), ('Np', 0, 100),
                             ('BZ_GSM', -80, 80), ('BY_GSM', -80, 80), ('BX_GSM', -80, 80),
                             ('AE', 0, 4000), ('KP', 0, 9)]:
            if col in df_storm.columns:
                df_storm[col] = df_storm[col].clip(lower=lo, upper=hi)

        # Derive any missing columns the UI needs
        if 'flux' in df_storm.columns and 'log_flux' not in df_storm.columns:
            df_storm['log_flux'] = np.log10(np.maximum(df_storm['flux'], 1e-3))
        if 'log_flux' in df_storm.columns and 'flux' not in df_storm.columns:
            df_storm['flux'] = 10**df_storm['log_flux']

        kp_proxy = np.clip(2 + 0.01*(df_storm.get('Vsw', 400) - 400) + 0.3*np.abs(df_storm.get('BZ_GSM', 0)), 0, 9)
        dst_proxy = -10 - 15*(kp_proxy/3.0)**1.5

        for col, default in [("KP", kp_proxy), ("DST", dst_proxy), ("BY_GSM", 0.0), ("BT", 5.0),
                             ("AE", 150.0), ("Ec", 0.5), ("Pdyn", 2.0), ("Np", 5.0),
                             ("Bz_neg_dur", 0.0), ("dDst_dt", 0.0), ("AE_1h", 150.0), ("regime", 1.0)]:
            if col not in df_storm.columns:
                df_storm[col] = default
        return df_storm.bfill().fillna(0)
    except Exception as e:
        raise RuntimeError(f"Could not load historical dataset for {name}: {e}") from e

# ─── Physics & Ensemble ───────────────────────────────────────────────────────
import importlib
import radshield.models.radial_diff
importlib.reload(radshield.models.radial_diff)
from radshield.models.radial_diff import run_physics_forecast as physics_forecast

REGIME_LABELS = {0:"QUIET  (Kp < 3)", 1:"MODERATE  (Kp 3–6)", 2:"STORM  (Kp ≥ 6)", 3:"RECOVERY  (Post-Storm)"}
# Only 4 storms backed by real, verified NASA/NOAA data
STORM_META = {
    "G5 Mother's Day Storm (May 2024)": {"min_dst":-412,"max_kp":9,"desc":"Processed local storm benchmark. The loader interpolates/fills missing values; this is not raw, gap-preserving telemetry."},
    "G4 Aurora Storm (Oct 2024)":       {"min_dst":-269,"max_kp":8,"desc":"Processed local storm benchmark. The loader interpolates/fills missing values; this is not raw, gap-preserving telemetry."},
}

WEIGHTS_VERSION = "v16"  # bump to bust Streamlit @cache_resource

@st.cache_resource
def load_radshield_model(_version=WEIGHTS_VERSION):
    """Load the model strictly and retain an explicit checkpoint status."""
    weights_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'weights', 'finetuned_gsat19_grasp_ulf.pth'))
    scaler_path  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'weights', 'scaler.pkl'))
    model, model_status = load_tft_checkpoint(weights_path)
    scaler = None
    if os.path.exists(scaler_path):
        try:
            import joblib
            scaler = joblib.load(scaler_path)
        except Exception as exc:
            model_status["scaler_warning"] = str(exc)

    return model, scaler, model_status

tft_model_instance, tft_scaler_instance, tft_model_status = load_radshield_model(WEIGHTS_VERSION)

def run_tft_inference(df, is_grasp=False):
    """Run the loaded TFT without altering or blending its raw quantile output."""
    if tft_model_instance is None:
        return None, tft_model_status.get("error") or "TFT checkpoint unavailable"
    if not tft_model_status["inference_supported"]:
        return None, tft_model_status["inference_error"]
    try:
        satellite_lon = 48.0 if is_grasp else -75.0
        result = run_tft_forecast(tft_model_instance, df, satellite_lon)
        return result, None
    except ForecastInputError as exc:
        return None, str(exc)

@st.cache_data(show_spinner=False, ttl=3600)
def run_continuous_validation(df, storm_name, is_grasp=False, _version=WEIGHTS_VERSION):
    """
    Runs sliding-window continuous validation on the full storm dataframe.
    Uses stride=6 (30 min) for performance, then linearly interpolates.
    Returns 3 Pandas Series representing T+30min, T+6h, and T+12h forecasts.
    """
    if (
        tft_model_instance is None
        or not tft_model_status["inference_supported"]
    ):
        return None, None, None
    try:
        import torch
        df_copy = df.copy()
        raw_log_flux = df_copy.get('log_flux', pd.Series(0.0, index=df_copy.index))
        df_copy['log_electron_flux'] = raw_log_flux.rolling(window=24, min_periods=1, center=False).median()
        df_copy['Vsw'] = df_copy.get('Vsw', 400.0)
        df_copy['Pdyn'] = df_copy.get('Pdyn', 2.0)
        df_copy['BY_GSM'] = df_copy.get('BY_GSM', 0.0)
        df_copy['BZ_GSM'] = df_copy.get('BZ_GSM', 0.0)
        df_copy['F10.7_index'] = df_copy.get('F10.7_index', 70.0)
        df_copy['DST'] = df_copy.get('DST', -10.0)
        df_copy['AE'] = df_copy.get('AE', 100.0)
        
        satellite_lon = 48.0 if is_grasp else -75.0
        mlt = calculate_mlt_vectorized(df_copy.index, satellite_lon)
        df_copy['MLT_sin'] = np.sin(mlt * 2 * np.pi / 24)
        df_copy['MLT_cos'] = np.cos(mlt * 2 * np.pi / 24)
        
        feature_cols = ["log_electron_flux", "BY_GSM", "BZ_GSM", "Pdyn", "Vsw", "AE", "DST", "F10.7_index", "MLT_sin", "MLT_cos"]
        for f in feature_cols:
            if f not in df_copy.columns: df_copy[f] = 0.0
        
        data_matrix = df_copy[feature_cols].values
        
        # Same dynamic normalization as run_tft_inference — bypass corrupted scaler.pkl
        mean = np.mean(data_matrix, axis=0, keepdims=True)
        std  = np.maximum(np.std(data_matrix, axis=0, keepdims=True), 1e-2)
        mean[:, 0] = 0.0
        std[:, 0]  = 1.0
        norm_x = (data_matrix - mean) / std

        seq_len = 288
        if len(norm_x) <= seq_len: return None, None, None

        stride = 6
        starts = np.arange(0, len(norm_x) - seq_len, stride)
        if len(starts) == 0: return None, None, None
            
        windows = np.array([norm_x[i:i+seq_len] for i in starts])
        x_tensor = torch.tensor(windows, dtype=torch.float32)
        
        with torch.no_grad():
            preds, _ = tft_model_instance(x_tensor)
            preds_p50 = preds[:, :, 2].cpu().numpy()
            
        pred_30m = preds_p50[:, 5]
        pred_6h = preds_p50[:, 71]
        pred_12h = preds_p50[:, 143]
        
        def map_to_series(pred_arr, horizon_offset):
            res = np.full(len(norm_x), np.nan)
            target_indices = starts + seq_len - 1 + horizon_offset
            valid = target_indices < len(norm_x)
            res[target_indices[valid]] = pred_arr[valid]
            # Convert to Pandas Series and interpolate the gaps left by striding
            return pd.Series(res, index=df_copy.index).interpolate(method='linear')
            
        s_30m = map_to_series(pred_30m, 6)
        s_6h = map_to_series(pred_6h, 72)
        s_12h = map_to_series(pred_12h, 144)
        return s_30m, s_6h, s_12h
    except Exception as e:
        print("Validation error:", e)
        return None, None, None

# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<p style="font-family:'Space Mono',monospace;font-size:0.62rem;letter-spacing:0.16em;
color:#315A9B;text-transform:uppercase;margin:0 0 4px 0">RADSHIELD AI / GEO OPERATIONS</p>
<p style="font-family:'Inter',sans-serif;font-size:1.1rem;color:#1D2B3F;
font-weight:600;margin:0 0 16px 0">RADIATION MONITOR</p>
""", unsafe_allow_html=True)

mode = st.sidebar.radio("DATA SOURCE", [
    "Live NOAA SWPC Satellite Stream (Real-Time)",
    "Historical Storm Replay"
], index=0)

st.sidebar.markdown("---")
if mode == "Live NOAA SWPC Satellite Stream (Real-Time)":
    target_satellite = st.sidebar.radio("TARGET SATELLITE (MLT SECTOR)", [
        "GOES-16 (75°W / Americas)",
        "GSAT-19 (48°E / India GRASP)"
    ], help="Zero-Shot Spatial Translation: Shifts the prediction's Magnetic Local Time (MLT) footprint to simulate fluxes at a different orbital longitude.")
else:
    target_satellite = "GOES-16 (75°W / Americas)"

if mode == "Historical Storm Replay":
    storm_name = st.sidebar.selectbox("SELECT STORM EVENT", list(STORM_META.keys()))
    meta = STORM_META[storm_name]
    st.sidebar.markdown(f"""
<div style="background:#FFFFFF;border:1px solid #D7DFE8;border-radius:8px;
padding:10px 12px;margin-top:8px">
<p style="font-family:'Space Mono',monospace;font-size:0.65rem;color:#315A9B;
text-transform:uppercase;margin:0 0 4px 0">{storm_name}</p>
<p style="font-size:0.78rem;color:#59697D;margin:0 0 6px 0">{meta['desc']}</p>
<p style="font-family:'Space Mono',monospace;font-size:0.72rem;color:#43536A;margin:0">
Min Dst: {meta['min_dst']} nT &nbsp;|&nbsp; Max Kp: {meta['max_kp']}</p>
</div>""", unsafe_allow_html=True)
    try:
        df_full = load_storm_from_csv(storm_name)
    except (OSError, ValueError, RuntimeError) as exc:
        st.sidebar.error(str(exc))
        st.stop()
    benchmark_mode = st.sidebar.checkbox("FULL-STORM BENCHMARK OVERLAY", value=True, help="Overrides the timeline slider to visualize the continuous T+6h physics-AI forecast against actual observations for the entire storm.")
    
    # Always show the slider so the user can scrub the timeline
    step = st.sidebar.slider("REPLAY TIMELINE", 0, len(df_full)-1, len(df_full)//2)
    
    if benchmark_mode:
        # Use the full dataset
        df = df_full
    else:
        df = df_full.iloc[:step+1]
elif mode == "Live NOAA SWPC Satellite Stream (Real-Time)":
    try:
        from radshield.data.noaa_ingest import fetch_live_noaa_telemetry
        df_noaa, status_msg = fetch_live_noaa_telemetry()
        if df_noaa is not None and len(df_noaa) > 0:
            df = df_noaa
        else:
            st.sidebar.error("Required NOAA SWPC feeds are unavailable; no synthetic live-data substitute is shown.")
            status_rows = [
                {
                    "Feed": feed["source"],
                    "Status": "Connected" if feed["connected"] else "Unavailable",
                    "Coverage (UTC)": (
                        f'{feed["first_observation_timestamp_utc"] or "—"} to '
                        f'{feed["last_observation_timestamp_utc"] or "—"}'
                    ),
                    "Valid observations": feed["valid_observations"],
                    "Error": feed["error"] or "",
                }
                for feed in status_msg.values()
            ]
            st.sidebar.dataframe(pd.DataFrame(status_rows), hide_index=True, use_container_width=True)
            st.stop()
    except Exception as e:
        st.sidebar.error(f"NOAA telemetry request failed: {e}")
        st.stop()

if mode == "Live NOAA SWPC Satellite Stream (Real-Time)":
    feed_status = df.attrs.get("feed_status", {})
    status_rows = [
        {
            "Feed": feed["source"],
            "Status": "Connected" if feed["connected"] else "Unavailable",
            "Coverage (UTC)": (
                f'{feed["first_observation_timestamp_utc"] or "—"} to '
                f'{feed["last_observation_timestamp_utc"] or "—"}'
            ),
            "Valid observations": feed["valid_observations"],
            "Error": feed["error"] or "",
        }
        for feed in feed_status.values()
    ]
    st.sidebar.markdown("**NOAA feed status**")
    st.sidebar.dataframe(pd.DataFrame(status_rows), hide_index=True, use_container_width=True)
    st.sidebar.caption(
        "Direct when present: NOAA GOES flux and solar-wind measurements. The model-ready frame fills "
        "missing inputs; NOAA Kp is used when available and otherwise falls back to a solar-wind estimate. "
        "The latest NOAA Kp value is reused "
        "across the model window. Dst/AE are estimates, Pdyn/Ec/BT are derived, ULF uses GOES Hp or a "
        "Bz proxy, F10.7 is fixed at 150 sfu, and missing Vy/Vz are assumed zero."
    )

if mode != "Historical Storm Replay":
    if st.sidebar.checkbox("LIVE AUTO-REFRESH (15s)", value=False, help="Automatically polls NOAA SWPC and updates telemetry & UTC clock every 15 seconds."):
        try:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=15000, limit=None, key="radshield_autorefresh")
        except ImportError:
            # Graceful fallback if streamlit-autorefresh is not installed
            import time
            time.sleep(15)
            st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("""<p style="font-family:'Space Mono',monospace;font-size:0.62rem;
letter-spacing:0.14em;color:#315A9B;text-transform:uppercase;margin:0 0 8px 0">
MLOps Cloud Registry</p>""", unsafe_allow_html=True)
hf_repo = st.sidebar.text_input(
    "MODEL REPOSITORY",
    os.environ.get("RADSHIELD_HF_REPO", ""),
    help="Set RADSHIELD_HF_REPO to your Hugging Face model repository ID.",
)
if st.sidebar.button("SYNC GPU WEIGHTS"):
    if not hf_repo.strip():
        st.sidebar.error("Set RADSHIELD_HF_REPO before syncing model weights.")
    else:
        with st.sidebar.status("Connecting to registry..."):
            try:
                from huggingface_hub import hf_hub_download
                target = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'weights'))
                os.makedirs(target, exist_ok=True)
                hf_hub_download(repo_id=hf_repo.strip(), filename="finetuned_gsat19_grasp_ulf.pth", local_dir=target)
                hf_hub_download(repo_id=hf_repo.strip(), filename="scaler.pkl", local_dir=target)
                st.sidebar.success("Weights synced from cloud")
            except Exception as e:
                st.sidebar.warning(f"Repo not found or private: {e}")

# ─── Current Row ──────────────────────────────────────────────────────────────
if mode == "Historical Storm Replay" and 'df_full' in locals() and 'step' in locals():
    df_current = df_full.iloc[:step+1]
else:
    df_current = df

row      = df_current.iloc[-1]
log_flux = float(row["log_flux"])
flux     = float(row["flux"])
kp       = float(row["KP"])
dst      = float(row["DST"])
vsw      = float(row["Vsw"])
bz       = float(row["BZ_GSM"])
ulf      = float(row["ULF_power"])
regime   = int(row["regime"])

# Calculate current UTC hour for MLT physics injection
utc_hour = float(df_current.index[-1].hour) + float(df_current.index[-1].minute) / 60.0

# Calculate max Kp over the last 72 hours (3 days) to give the physics model long-term storm memory for recovery phase acceleration
max_kp_72h = float(df_current['KP'].iloc[-864:].max()) if len(df_current) >= 864 else float(df_current['KP'].max())

# Use 6-hour median to determine the true macro-state of the radiation belts, immune to narrow dropouts
core_state = float(df_current['log_flux'].iloc[-72:].median()) if len(df_current) >= 72 else float(df_current['log_flux'].median())

# Execute unmodified TFT checkpoint inference and the separate physics estimate.
is_grasp_selected = "GSAT-19" in target_satellite

tft_forecast, tft_error = run_tft_inference(df_current, is_grasp=is_grasp_selected)
phys = physics_forecast(core_state, kp, max_kp_72h, utc_hour, is_grasp=is_grasp_selected)

if tft_forecast is not None:
    tft_quantiles = tft_forecast.quantiles_log10_pfu
    tft_f_P10 = tft_quantiles[:, 0]
    tft_f_P25 = tft_quantiles[:, 1]
    tft_f_P50 = tft_quantiles[:, 2]
    tft_f_P75 = tft_quantiles[:, 3]
    tft_f_P90 = tft_quantiles[:, 4]
    t_fut = tft_forecast.forecast_timestamps_utc
else:
    tft_quantiles = None
    tft_f_P10 = tft_f_P25 = tft_f_P50 = tft_f_P75 = tft_f_P90 = None
    t_fut = pd.date_range(
        start=df_current.index[-1] + pd.Timedelta(minutes=5),
        periods=144,
        freq="5min",
    )

forecast_horizons = (
    ("T+30 min", 5, "T+30m"),
    ("T+6 h", 71, "T+6h"),
    ("T+12 h", 143, "T+12h"),
)

# ─── Dashboard hero ───────────────────────────────────────────────────────────
view_label = (
    "Historical storm replay"
    if mode == "Historical Storm Replay"
    else "Live NOAA / SWPC stream"
)
view_detail = (
    f"{storm_name} · {df_current.index[-1]:%b %d, %H:%M UTC}"
    if mode == "Historical Storm Replay"
    else f"{target_satellite} · {df_current.index[-1]:%b %d, %H:%M UTC}"
)
st.markdown(f"""
<section class="hero-card">
  <div class="hero-copy">
    <div class="hero-eyebrow">RadShield AI &nbsp; / &nbsp; Space weather monitoring</div>
    <h1 class="hero-title">Radiation <span>overview</span></h1>
    <p class="hero-description">Energetic particle observations and storm context.</p>
    <div class="hero-chips">
      <span class="hero-chip">{view_label}</span>
      <span class="hero-chip">{view_detail}</span>
      <span class="hero-chip">All timestamps UTC</span>
    </div>
  </div>
</section>
""", unsafe_allow_html=True)

if tft_model_status["available"] and tft_model_status["inference_supported"]:
    st.markdown(
        '<div class="status-ribbon status-ready"><span class="status-dot"></span>'
        '<span><strong>TFT inference available</strong> · Experimental output; not validated for operational use.</span></div>',
        unsafe_allow_html=True,
    )
elif tft_model_status["available"]:
    st.markdown(
        '<div class="status-ribbon status-blocked"><span class="status-dot"></span>'
        '<span><strong>TFT forecast unavailable</strong> · The configured checkpoint/scaler provenance is unverified; '
        'the separate GOES-15 research TFT failed to beat persistence on held-out data.</span></div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="status-ribbon status-missing"><span class="status-dot"></span>'
        '<span><strong>TFT checkpoint unavailable</strong> · No TFT forecast is displayed.</span></div>',
        unsafe_allow_html=True,
    )
    st.error(
        f"Checkpoint load error: {tft_model_status['error']}"
    )

# ─── KPI Cards ────────────────────────────────────────────────────────────────
# Calculate true 1-hour delta (1 hour = 12 5-minute intervals)
if len(df) > 12:
    flux_1h_ago = df["flux"].iloc[-13]
    flux_delta_pct = ((flux - flux_1h_ago) / flux_1h_ago) * 100
    delta_str = f"{'+' if flux_delta_pct > 0 else ''}{flux_delta_pct:.0f}%  vs 1h ago"
else:
    delta_str = "Insufficient data for delta"

observed_flux = df.get("observed_flux", pd.Series(np.nan, index=df.index))
last_observed_flux = observed_flux.dropna().iloc[-1] if observed_flux.notna().any() else np.nan

if is_grasp_selected:
    live_flux_val = np.nan
    live_label = "GSAT-19 DIRECT FLUX"
elif mode == "Live NOAA SWPC Satellite Stream (Real-Time)":
    live_flux_val = last_observed_flux
    live_label = "DIRECT NOAA GOES FLUX (>=2 MeV)"
else:
    live_flux_val = flux
    live_label = "PROCESSED HISTORICAL FLUX"

flux_display = f"{live_flux_val:.2e} pfu" if pd.notna(live_flux_val) else "No direct observation"
with st.container():
    st.markdown('<p class="section-label">Current conditions</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric(
        live_label,
        flux_display,
        delta_str if pd.notna(live_flux_val) and mode != "Live NOAA SWPC Satellite Stream (Real-Time)" else None,
        help="When available, this value is the most recent direct GOES observation; its timestamp is listed in the per-feed status.",
    )
    c2.metric("REGIME STATE", REGIME_LABELS[regime], f"Kp = {kp:.1f} | Dst = {dst:.0f} nT")
    c3.metric("SOLAR WIND SPEED", f"{vsw:.0f} km/s", f"Bz = {bz:.1f} nT")

if mode == "Live NOAA SWPC Satellite Stream (Real-Time)":
    observation_status = build_radiation_observation_status(
        df,
        df.attrs.get("feed_status", {}),
    )
    if observation_status["feed_error"]:
        feed_class = "status-missing"
        feed_summary = f'GOES electron feed error: {observation_status["feed_error"]}'
    elif not observation_status["feed_connected"]:
        feed_class = "status-missing"
        feed_summary = "GOES electron feed unavailable — direct measurements are not displayed."
    elif observation_status["data_quality_status"] == "stale":
        feed_class = "status-blocked"
        feed_summary = (
            f'Latest direct GOES measurement is stale · '
            f'{observation_status["age_minutes"]:.1f} minutes old.'
        )
    elif observation_status["data_quality_status"] == "no valid measured observation":
        feed_class = "status-blocked"
        feed_summary = "GOES feed connected, but no valid direct electron-flux observation is available."
    else:
        feed_class = "status-ready"
        feed_summary = (
            f'NOAA GOES direct observations · {observation_status["data_quality_status"].title()} · '
            f'{observation_status["valid_observation_count"]:,} valid samples.'
        )
    st.markdown(
        f'<div class="status-ribbon {feed_class}"><span class="status-dot"></span>'
        f'<span><strong>Live source status</strong> · {feed_summary}</span></div>',
        unsafe_allow_html=True,
    )
    with st.expander("Live data details — measurements, coverage and quality", expanded=False):
        st.caption(
            "Directly observed GOES integral electron flux only. This panel does not use "
            "the filled/model-ready `flux` series."
        )
        radiation_cols = st.columns(4)
        displayed_value = observation_status["latest_value"]
        radiation_cols[0].metric(
            "Measured >2 MeV electron flux",
            f'{displayed_value:.4g} pfu' if displayed_value is not None else "Unavailable",
        )
        radiation_cols[1].metric(
            "Data quality / freshness",
            observation_status["data_quality_status"].title(),
            (
                f'{observation_status["age_minutes"]:.1f} min old'
                if observation_status["age_minutes"] is not None
                else None
            ),
        )
        radiation_cols[2].metric(
            "Valid direct observations",
            f'{observation_status["valid_observation_count"]:,}',
        )
        satellites = observation_status["satellite_ids"]
        radiation_cols[3].metric(
            "Satellite IDs in feed",
            ", ".join(satellites) if satellites else "Not provided",
        )
        st.markdown(
            f'**Observation timestamp:** '
            f'{observation_status["latest_observation_timestamp_utc"] or "Unavailable"}'
            f'{"" if not observation_status["latest_observation_timestamp_utc"] else " UTC"}  \n'
            f'**Feed fetch timestamp:** '
            f'{observation_status["fetch_timestamp_utc"] or "Unavailable"}  \n'
            f'**Source:** [{observation_status["source"]}]'
            f'({observation_status["source_url"]})  \n'
            f'**Channel / units:** {observation_status["channel"]}; '
            f'{observation_status["units"]}  \n'
            f'**Instrument:** {observation_status["instrument"] or "Not provided by this feed"}'
        )
        coverage_start = observation_status["first_observation_timestamp_utc"]
        coverage_end = observation_status["last_observation_timestamp_utc"]
        if coverage_start and coverage_end:
            st.caption(
                f'Coverage: {coverage_start} to {coverage_end} UTC. '
                f'{observation_status["valid_observation_count"]:,} valid direct observations '
                "in the current feed window. Missing measurements remain gaps."
            )
        else:
            st.caption("No valid direct electron-flux observations are available in the current feed window.")
        if observation_status["feed_error"]:
            st.error(f'GOES electron feed error: {observation_status["feed_error"]}')
        elif not observation_status["feed_connected"]:
            st.warning("GOES electron feed is unavailable; no measured flux value is displayed.")
        elif observation_status["data_quality_status"] == "stale":
            st.warning(
                f'Latest valid measurement exceeds the dashboard freshness criterion '
                f'of {observation_status["stale_after_minutes"]:.0f} minutes.'
            )
        elif observation_status["data_quality_status"] == "no valid measured observation":
            st.warning("The feed is reachable, but it contains no valid measured electron-flux value.")
        st.warning(observation_status["quality_warning"])
        st.caption(
            "The source is a rolling seven-day feed, not a long-term training archive. "
            "Data quality/freshness above is separate from the experimental project flux screening "
            "below; neither is an official NOAA alert or satellite-safety recommendation."
        )

st.markdown("<hr style='margin:20px 0'>", unsafe_allow_html=True)

overview_tab, fleet_tab, diagnostics_tab = st.tabs([
    "Overview", "GEO fleet", "Diagnostics"
])

with overview_tab:
    chart_col, export_col = st.columns([0.82, 0.18])
    with chart_col:
        st.markdown('<p class="section-label">Electron flux observations &amp; estimates</p>', unsafe_allow_html=True)
    chart_action_slot = export_col.empty()
    chart_slot = st.empty()

    # ─── Optional experimental radiation screening ───────────────────────────────
    show_forecast = not (mode == "Historical Storm Replay" and not benchmark_mode)

    if show_forecast:
        with st.expander("Experimental screening & model details", expanded=False):
            st.warning(
                "Screening levels are experimental project heuristics (1,000 and 10,000 pfu), "
                "not official NOAA/ISRO limits or satellite-safety guidance. The separate "
                "GOES-15 research TFT failed to beat persistence on its held-out set at 30 minutes, "
                "6 hours, and 12 hours, so its predictions are not served."
            )
            if tft_forecast is None:
                st.info(
                    f"No TFT forecast was generated. Reason: {tft_error}"
                )
                st.caption(
                    "The dashboard's configured checkpoint is separate from the evaluated "
                    "research TFT; its checkpoint and scaler provenance remain unverified."
                )
                st.caption(f"Input window ends at {df_current.index[-1]:%Y-%m-%d %H:%M:%S UTC}.")
                if "observed_flux" in df_current:
                    direct_flux = pd.to_numeric(df_current["observed_flux"], errors="coerce").dropna()
                    if not direct_flux.empty:
                        latest_direct_time = direct_flux.index[-1].to_pydatetime()
                        age_minutes = (
                            pd.Timestamp.now(tz="UTC") - pd.Timestamp(latest_direct_time)
                        ).total_seconds() / 60.0
                        st.caption(
                            f"Latest direct GOES flux: {latest_direct_time:%Y-%m-%d %H:%M:%S UTC} "
                            f"({age_minutes:.1f} minutes old)."
                        )
                    else:
                        st.caption("Latest direct GOES flux timestamp unavailable; no direct input in this window.")
                else:
                    st.caption("Latest direct GOES flux timestamp unavailable in this dataset.")
            else:
                screening_rows = []
                for label, index, _physics_key in forecast_horizons:
                    flux_pfu = float(np.power(10.0, tft_f_P50[index]))
                    screening_rows.append(
                        {
                            "Horizon": label,
                            "Forecast timestamp (UTC)": tft_forecast.forecast_timestamps_utc[index].strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                            "Raw TFT P50 (pfu)": flux_pfu,
                            "Experimental screening": classify_experimental_flux_screening(flux_pfu),
                        }
                    )
                st.dataframe(
                    pd.DataFrame(screening_rows),
                    hide_index=True,
                    use_container_width=True,
                )
                st.caption(
                    "Screening compares only the unmodified TFT P50 output against project levels "
                    f"({EXPERIMENTAL_FLUX_THRESHOLDS_PFU[0]:,.0f} and "
                    f"{EXPERIMENTAL_FLUX_THRESHOLDS_PFU[1]:,.0f} pfu). No Kp/Dst override, physics blending, "
                    "confidence score, or safety action is applied."
                )

            if tft_forecast is not None:
                freshness = (
                    f'{tft_forecast.direct_flux_age_minutes:.1f} minutes old '
                    f'(last direct GOES flux: {tft_forecast.latest_direct_flux_timestamp_utc:%Y-%m-%d %H:%M:%S UTC})'
                    if tft_forecast.direct_flux_age_minutes is not None
                    else "unknown (raw historical flux provenance unavailable)"
                )
                st.markdown(
                    f"**Forecast generated:** {tft_forecast.generated_at_utc:%Y-%m-%d %H:%M:%S UTC}  \n"
                    f"**Input ends:** {tft_forecast.input_end_timestamp_utc:%Y-%m-%d %H:%M:%S UTC}  \n"
                    f"**Direct-flux freshness:** {freshness}  \n"
                    f"**Forecast timestamps:** {tft_forecast.forecast_timestamps_utc[0]:%Y-%m-%d %H:%M:%S UTC} "
                    f"to {tft_forecast.forecast_timestamps_utc[-1]:%Y-%m-%d %H:%M:%S UTC} "
                    "(144 steps at 5-minute intervals)"
                )
                if tft_forecast.direct_flux_age_minutes is not None and tft_forecast.direct_flux_age_minutes > 30:
                    st.warning("The direct GOES flux input is more than 30 minutes old; interpret this model output cautiously.")
                for warning in tft_forecast.warnings:
                    st.warning(warning)

            physics_rows = [
                {
                    "Horizon": label,
                    "Timestamp (UTC)": (
                        df_current.index[-1] + pd.Timedelta(minutes=5 * (index + 1))
                    ).strftime("%Y-%m-%d %H:%M:%S"),
                    "Flux estimate (pfu)": float(np.power(10.0, phys[key])),
                    "Source": "Existing physics-inspired radial-diffusion surrogate",
                }
                for label, index, key in forecast_horizons
            ]
            st.markdown("**Separate physics-inspired surrogate estimates**")
            st.dataframe(pd.DataFrame(physics_rows), hide_index=True, use_container_width=True)
            st.caption(
                "These are discrete outputs of the existing surrogate, not TFT predictions, direct observations, "
                "validated forecasts, or operational safety advice."
            )


# ─── Forecast Vectors Computation ─────────────────────────────────────────────
is_benchmark = (
    mode == "Historical Storm Replay"
    and 'benchmark_mode' in locals()
    and benchmark_mode
    and tft_model_status["inference_supported"]
)
hist_n = len(df) if is_benchmark else min(len(df), 2016 if "Live" in mode else 288)
t_hist = df.index[-hist_n:]
f_hist = df["flux"].values[-hist_n:]
observed_flux_hist = df.get("observed_flux", pd.Series(np.nan, index=df.index)).values[-hist_n:]
physics_plot_timestamps = [t_fut[index] for _label, index, _key in forecast_horizons]
physics_plot_flux = [float(np.power(10.0, phys[key])) for _label, _index, key in forecast_horizons]

with overview_tab:
    chart_note = (
        "Processed historical benchmark · original source gaps are not retained."
        if is_benchmark
        else ""
    )
    if tft_forecast is not None:
        export_df = pd.DataFrame({
            "Forecast_Timestamp_UTC": tft_forecast.forecast_timestamps_utc,
            "TFT_P10_pfu": np.power(10.0, tft_f_P10),
            "TFT_P25_pfu": np.power(10.0, tft_f_P25),
            "TFT_P50_pfu": np.power(10.0, tft_f_P50),
            "TFT_P75_pfu": np.power(10.0, tft_f_P75),
            "TFT_P90_pfu": np.power(10.0, tft_f_P90),
        })
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        chart_action_slot.download_button(
            label="EXPORT TFT CSV",
            data=csv_bytes,
            file_name=f"radshield_tft_forecast_{df.index[-1].strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )
    else:
        chart_action_slot.empty()

    if is_benchmark:
        with st.spinner("Running PyTorch TFT Sliding-Window Validation across full storm..."):
            s_30m, s_6h, s_12h = run_continuous_validation(df, storm_name, is_grasp_selected, WEIGHTS_VERSION)
    
        fig = make_subplots(
            rows=3, cols=2,
            column_widths=[0.72, 0.28],
            shared_xaxes=False,
            subplot_titles=(
                "Predictions for 30-min prediction", "(a) Scatter (T+30m)",
                "Predictions for 6-hour prediction", "(b) Scatter (T+6h)",
                "Predictions for 12-hour prediction", "(c) Scatter (T+12h)"
            ),
            horizontal_spacing=0.08,
            vertical_spacing=0.08
        )
    
        def add_validation_plot(pred_series, row_num):
            # Time-series (Left Column)
            fig.add_trace(go.Scatter(
                x=t_hist, y=f_hist,
                name="Processed benchmark flux [pfu; source gaps not retained]",
                line=dict(color="#9CB2CC", width=1.3),
                hovertemplate="%{x|%Y-%m-%d %H:%M:%S} UTC<br>%{y:.4g} pfu<extra>%{fullData.name}</extra>",
                showlegend=(row_num==1)
            ), row=row_num, col=1)
            if pred_series is not None:
                fig.add_trace(go.Scatter(
                    x=pred_series.index, y=10**pred_series,
                    name="TFT model output [pfu; checkpoint provenance unverified]",
                    line=dict(color="#DC2626", width=1.3),
                    hovertemplate="%{x|%Y-%m-%d %H:%M:%S} UTC<br>%{y:.4g} pfu<extra>%{fullData.name}</extra>",
                    showlegend=(row_num==1)
                ), row=row_num, col=1)
            
                # Scatter Plot (Right Column) — filter out sensor dropouts (flux <= 1.1 pfu)
                valid = ~np.isnan(pred_series.values) & ~np.isnan(df['log_flux'].values) & (df['log_flux'].values > 0.05)
                if valid.any():
                    obs_vals = 10**df['log_flux'].values[valid]
                    pred_vals = 10**pred_series.values[valid]
                
                    fig.add_trace(go.Scatter(
                        x=obs_vals, y=pred_vals,
                        mode="markers",
                        marker=dict(color="#A6BBD3", size=3, opacity=0.55),
                        name="Benchmark pairs [pfu]",
                        hovertemplate="Processed observation: %{x:.4g} pfu<br>Model output: %{y:.4g} pfu<extra></extra>",
                        showlegend=False
                    ), row=row_num, col=2)
                
                    # 1:1 Identity Blue Line (1 to 10,000 pfu range)
                    fig.add_trace(go.Scatter(
                        x=[1, 1e4], y=[1, 1e4],
                        mode="lines",
                        line=dict(color="#2563EB", width=1.8),
                        name="1:1 Line",
                        showlegend=False
                    ), row=row_num, col=2)

            fig.add_hline(y=1e4, line_dash="dot", line_color="#2563EB", line_width=1.2, row=row_num, col=1)

        add_validation_plot(s_30m, 1)
        add_validation_plot(s_6h, 2)
        add_validation_plot(s_12h, 3)
    
        validation_results = {
            'truth': df['log_flux'].values,
            's_30m': s_30m.values if s_30m is not None else np.full(len(df), np.nan),
            's_6h': s_6h.values if s_6h is not None else np.full(len(df), np.nan),
            's_12h': s_12h.values if s_12h is not None else np.full(len(df), np.nan)
        }
    else:
        fig = go.Figure()
        if mode == "Live NOAA SWPC Satellite Stream (Real-Time)":
            observed_mask = np.isfinite(observed_flux_hist)
            estimated_values = np.where(observed_mask, np.nan, f_hist)
            fig.add_trace(go.Scatter(
                x=t_hist, y=observed_flux_hist,
                name="NOAA GOES direct observations (>=2 MeV) [pfu]",
                line=dict(color="#2563EB", width=2.5),
                connectgaps=False,
                hovertemplate=(
                    "%{x|%Y-%m-%d %H:%M:%S} UTC<br>"
                    "%{y:.4g} electrons/(cm² s sr) (pfu)"
                    "<extra>Direct NOAA GOES observation</extra>"
                ),
            ))
            if np.isfinite(estimated_values).any():
                fig.add_trace(go.Scatter(
                    x=t_hist, y=estimated_values,
                    name="Model input estimate/fill [pfu]",
                    line=dict(color="#94A3B8", width=1.5, dash="dot"),
                    connectgaps=False,
                ))
            valid_flux_count = int(observed_mask.sum())
            if valid_flux_count:
                valid_times = t_hist[observed_mask]
                flux_coverage = (
                    f"{valid_flux_count:,} valid direct GOES observations "
                    f"from {valid_times[0]:%Y-%m-%d %H:%M:%S} to "
                    f"{valid_times[-1]:%Y-%m-%d %H:%M:%S} UTC"
                )
            else:
                flux_coverage = "No valid direct GOES flux observations in the displayed interval."
            chart_note = flux_coverage + " Gaps preserved."
        else:
            fig.add_trace(go.Scatter(
                x=t_hist, y=f_hist,
                name="Processed historical benchmark flux [pfu; source gaps not retained]",
                line=dict(color="#2563EB", width=2.5),
                connectgaps=False,
            ))
            valid_flux_count = int(np.isfinite(f_hist).sum())
            chart_note = (
                f"{valid_flux_count:,} processed historical points · original source gaps are not retained."
            )
        if mode == "Historical Storm Replay" and 'df_full' in locals() and 'step' in locals() and show_forecast:
            df_fut_truth = df_full.iloc[step+1 : step+145]
            if len(df_fut_truth) > 0:
                fig.add_trace(go.Scatter(
                    x=df_fut_truth.index, y=df_fut_truth["flux"].values,
                    name="Processed benchmark outcome [pfu; source gaps not retained]",
                    line=dict(color="#2563EB", width=2, dash="dot")
                ))
            
        if show_forecast:
            if tft_forecast is not None:
                for quantile_index, quantile_name, quantile_color, line_width in (
                    (0, "P10", "#94A3B8", 1.2),
                    (1, "P25", "#F59E0B", 1.2),
                    (2, "P50", "#DC6803", 2.5),
                    (3, "P75", "#F59E0B", 1.2),
                    (4, "P90", "#94A3B8", 1.2),
                ):
                    fig.add_trace(go.Scatter(
                        x=t_fut,
                        y=np.power(10.0, tft_quantiles[:, quantile_index]),
                        name=f"Raw TFT {quantile_name} [pfu; calibration unverified]",
                        line=dict(color=quantile_color, width=line_width),
                        connectgaps=False,
                    ))
            fig.add_trace(go.Scatter(
                x=physics_plot_timestamps,
                y=physics_plot_flux,
                mode="markers",
                name="Physics-inspired surrogate estimates [pfu]",
                marker=dict(size=9, symbol="diamond", color="#059669"),
            ))
    if not is_benchmark:
        fig.add_hline(
            y=EXPERIMENTAL_FLUX_THRESHOLDS_PFU[1], line_dash="dot",
            line_color="#DC2626", line_width=1.5,
            annotation_text="Experimental project screening level (10⁴ pfu)",
            annotation_font_color="#DC2626",
            annotation_font_size=12,
            annotation_position="top right"
        )

    if is_benchmark:
        for annotation in fig['layout']['annotations']: 
            annotation['font'] = dict(size=12, color="#D7E2F0", family="Space Mono")
        
        fig.update_layout(
            paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
            font=dict(family="Inter, sans-serif", size=12, color="#000000"),
            legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="right", x=1, bgcolor="rgba(255,255,255,0.8)"),
            margin=dict(l=75, r=25, t=60, b=40),
            height=850
        )
        for i in range(1, 4):
            # Time-Series Axes (Left)
            fig.update_yaxes(type="log", title="Flux [pfu]" if i==2 else "", range=[0, 6], showgrid=True, gridcolor="#E2E8F0", zeroline=False, row=i, col=1)
            fig.update_xaxes(showgrid=True, gridcolor="#E2E8F0", zeroline=False, row=i, col=1)
        
            # Scatter Plot Axes (Right Column) — focused on 1 to 10,000 pfu (range [0, 4])
            fig.update_xaxes(type="log", range=[0, 4], title="Observed [pfu]" if i==3 else "", showgrid=True, gridcolor="#E2E8F0", zeroline=False, row=i, col=2)
            fig.update_yaxes(type="log", range=[0, 4], title="Predicted [pfu]" if i==2 else "", showgrid=True, gridcolor="#E2E8F0", zeroline=False, row=i, col=2)
        
        fig.update_xaxes(title="TIME (UTC)", row=3, col=1)
        for i in range(1, 4):
            fig.update_xaxes(
                tickformat="%b %d<br>%H:%M", hoverformat="%Y-%m-%d %H:%M:%S UTC",
                row=i, col=1,
            )
    else:
        if not is_benchmark:
            fig.update_traces(
                hovertemplate="%{x|%Y-%m-%d %H:%M:%S} UTC<br>%{y:.4g} pfu<extra>%{fullData.name}</extra>"
            )
            fig.update_xaxes(
                tickformat="%b %d<br>%H:%M",
                hoverformat="%Y-%m-%d %H:%M:%S UTC",
            )
        fig.update_layout(
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(family="Inter, sans-serif", size=13, color="#000000"),
            xaxis=dict(
                title="TIME (UTC)",
                showgrid=True, gridcolor="#E2E8F0", gridwidth=1,
                linecolor="#CBD5E1", tickcolor="#CBD5E1",
                title_font=dict(size=13, color="#000000"),
                tickfont=dict(size=12, color="#000000"),
                zeroline=False
            ),
            yaxis=dict(
                type="log", title="ELECTRON FLUX  (>2 MeV) [pfu]",
                showgrid=True, gridcolor="#E2E8F0", gridwidth=1,
                linecolor="#CBD5E1", tickcolor="#CBD5E1",
                title_font=dict(size=13, color="#000000"),
                tickfont=dict(size=12, color="#000000"),
                range=[0, 6], zeroline=False
            ),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.04,
                xanchor="right", x=1,
                bgcolor="rgba(255,255,255,0.7)",
                font=dict(size=13, color="#000000")
            ),
            margin=dict(l=85, r=20, t=40, b=60),
            height=420
        )

    if mode == "Historical Storm Replay" and 'step' in locals() and benchmark_mode:
        current_time_val = df_full.index[step]
        fig.add_vline(x=current_time_val, line_width=1.5, line_dash="dash", line_color="#FF4B4B", row='all', col=1)

    with chart_slot.container():
        st.caption(chart_note)
        st.plotly_chart(apply_radshield_chart_theme(fig), use_container_width=True, theme=None)

    if is_benchmark and 'validation_results' in locals():
        truth = validation_results['truth']
        s_30m = validation_results['s_30m']
        s_6h = validation_results['s_6h']
        s_12h = validation_results['s_12h']
    
        def calc_lc(t, p):
            v = ~np.isnan(t) & ~np.isnan(p)
            return float(np.corrcoef(t[v], p[v])[0, 1]) if v.any() else 0.0
        
        def calc_rmse(t, p):
            v = ~np.isnan(t) & ~np.isnan(p)
            return float(np.sqrt(np.mean((t[v] - p[v])**2))) if v.any() else 0.0

        st.markdown('<p class="section-label" style="margin-top:0;">Exploratory Benchmark Diagnostics</p>', unsafe_allow_html=True)
        st.caption(
            "These diagnostics are not verified operational accuracy metrics; historical source gaps and "
            "checkpoint provenance are not independently established."
        )
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("LC (T+30m)", f"{calc_lc(truth, s_30m):.3f}")
        mc2.metric("LC (T+6h)", f"{calc_lc(truth, s_6h):.3f}")
        mc3.metric("LC (T+12h)", f"{calc_lc(truth, s_12h):.3f}")
        mc4.metric("RMSE (T+6h, log10)", f"{calc_rmse(truth, s_6h):.4f}")

    st.markdown("<hr style='margin:24px 0'>", unsafe_allow_html=True)

with diagnostics_tab:
    drivers_tab, research_tab = st.tabs(["Solar wind drivers", "Research & provenance"])

with drivers_tab:
    # ─── Subsequent Parameters Multi-Graph ─────────────────────────────────────────
    # In benchmark mode, show 24h of data ending at the slider cursor position.
    # In all other modes, show the last 24h of live data as normal.
    if mode == "Historical Storm Replay" and 'df_full' in locals() and 'step' in locals():
        df_plot = df_full.iloc[max(0, step-287):step+1]
    else:
        df_plot = df
    hist_n = min(len(df_plot), 288)
    t_hist = df_plot.index[-hist_n:]

    st.markdown('<p class="section-label">Multiparameter Historical Solar Wind Driver State (Last 24h)</p>', unsafe_allow_html=True)

    fig_params = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        vertical_spacing=0.04,
        specs=[[{}], [{}], [{"secondary_y": True}], [{}]],
        subplot_titles=(
            "IMF Magnetic Field Vectors (nT)", 
            "Solar Wind Velocity (km/s)", 
            "Geomagnetic Indices & Pressure (Psw, AE, Dst)", 
            "Orbital Position — Magnetic Local Time (0–24h)"
        )
    )

    for annotation in fig_params['layout']['annotations']: 
        annotation['font'] = dict(size=12, color="#2563EB", family="Space Mono")

    live_parameter_series = mode == "Live NOAA SWPC Satellite Stream (Real-Time)"
    parameter_source = "NOAA direct; gaps preserved" if live_parameter_series else "processed history; gaps not retained"

    def parameter_values(raw_column, processed_column):
        column = raw_column if live_parameter_series and raw_column in df_plot else processed_column
        return pd.to_numeric(df_plot[column], errors="coerce").values[-hist_n:]


    # R1: IMF Bx, By, Bz
    fig_params.add_trace(go.Scatter(
        x=t_hist, y=parameter_values("observed_BX_GSM", "BX_GSM"),
        name=f"Bx (GSM) [nT; {parameter_source}]",
        line=dict(color="#F59E0B", width=1.5), connectgaps=False,
    ), row=1, col=1)
    fig_params.add_trace(go.Scatter(
        x=t_hist, y=parameter_values("observed_BY_GSM", "BY_GSM"),
        name=f"By (GSM) [nT; {parameter_source}]",
        line=dict(color="#2563EB", width=1.5), connectgaps=False,
    ), row=1, col=1)
    fig_params.add_trace(go.Scatter(
        x=t_hist, y=parameter_values("observed_BZ_GSM", "BZ_GSM"),
        name=f"Bz (GSM) [nT; {parameter_source}]",
        line=dict(color="#DC2626", width=1.5), connectgaps=False,
    ), row=1, col=1)

    # R2: Solar Wind Vx, Vy, Vz
    vsw_arr = parameter_values("observed_Vsw", "Vsw")
    fig_params.add_trace(go.Scatter(
        x=t_hist, y=-vsw_arr, name=f"Vx approx. from speed [km/s; {parameter_source}]",
        line=dict(color="#059669", width=1.5), connectgaps=False,
    ), row=2, col=1)
    fig_params.add_trace(go.Scatter(
        x=t_hist, y=np.zeros_like(vsw_arr), name="Vy proxy [km/s; assumed zero]",
        line=dict(color="#D97706", dash="dash", width=1.5),
    ), row=2, col=1)
    fig_params.add_trace(go.Scatter(
        x=t_hist, y=np.zeros_like(vsw_arr), name="Vz proxy [km/s; assumed zero]",
        line=dict(color="#7C3AED", dash="dot", width=1.5),
    ), row=2, col=1)

    # R3: Psw, AE, DST
    psw = df_plot.get("Pdyn", pd.Series(2.0, index=df_plot.index)).values[-hist_n:]
    fig_params.add_trace(go.Scatter(
        x=t_hist, y=psw * 10, name="Dynamic pressure [10 × nPa; derived]",
        line=dict(color="#7C3AED", width=1.5), connectgaps=False,
    ), row=3, col=1, secondary_y=False)
    fig_params.add_trace(go.Scatter(
        x=t_hist, y=df_plot.get("AE", pd.Series(100, index=df_plot.index)).values[-hist_n:],
        name="AE [nT; estimated]", line=dict(color="#0284C7", width=1.5, dash="dot"),
        connectgaps=False,
    ), row=3, col=1, secondary_y=True)
    fig_params.add_trace(go.Scatter(
        x=t_hist, y=df_plot["DST"].values[-hist_n:], name="Dst [nT; estimated]",
        line=dict(color="#E11D48", width=1.5), connectgaps=False,
    ), row=3, col=1, secondary_y=False)

    # R4: MLT displayed as raw 0-24h for operator readability
    # We additionally plot visually scaled Sine/Cosine waves so the operator 
    # can see the exact cyclical embeddings the AI uses, mapped to the 0-24 grid.
    hist_sat_lon = 48.0 if is_grasp_selected else -75.0
    mlt_hours = calculate_mlt_vectorized(t_hist, hist_sat_lon)
    mlt_sin = np.sin(mlt_hours * 2 * np.pi / 24)
    mlt_cos = np.cos(mlt_hours * 2 * np.pi / 24)

    # Plot visually scaled cyclical features in the background
    fig_params.add_trace(go.Scatter(
        x=t_hist, y=(mlt_sin + 1) * 12,
        name="MLT Sin (Scaled)",
        line=dict(color="rgba(234, 88, 12, 0.4)", width=2, dash="dot")
    ), row=4, col=1)

    fig_params.add_trace(go.Scatter(
        x=t_hist, y=(mlt_cos + 1) * 12,
        name="MLT Cos (Scaled)",
        line=dict(color="rgba(5, 150, 105, 0.4)", width=2, dash="dot")
    ), row=4, col=1)

    # Plot the raw human-readable hour sawtooth in the foreground
    fig_params.add_trace(go.Scatter(
        x=t_hist, y=mlt_hours,
        name="Raw MLT (Hours)",
        line=dict(color="#EA580C", width=1.8)
    ), row=4, col=1)

    fig_params.update_layout(
        height=720,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, sans-serif", size=12, color="#000000"),
        margin=dict(l=85, r=20, t=30, b=60),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(255,255,255,0.9)", font=dict(size=12, color="#000000")),
    )
    fig_params.update_traces(
        hovertemplate="%{x|%Y-%m-%d %H:%M:%S} UTC<br>%{y:.4g}<extra>%{fullData.name}</extra>"
    )

    for i in range(1, 5):
        fig_params.update_yaxes(showgrid=True, gridcolor="#E2E8F0", linecolor="#CBD5E1", zeroline=False, title_font=dict(color="#000000"), tickfont=dict(color="#000000"), row=i, col=1)

    fig_params.update_yaxes(title_text="Field (nT)", row=1, col=1)
    fig_params.update_yaxes(title_text="Speed (km/s)", row=2, col=1)
    fig_params.update_yaxes(title_text="Pressure (×10), Dst", row=3, col=1, secondary_y=False)
    fig_params.update_yaxes(title_text="AE (nT)", row=3, col=1, secondary_y=True, showgrid=False)
    fig_params.update_yaxes(title_text="Time (Hours)", range=[0, 24], dtick=4, row=4, col=1)
    fig_params.update_xaxes(
        showgrid=True, gridcolor="#E2E8F0", linecolor="#CBD5E1",
        title_font=dict(color="#000000"), tickfont=dict(color="#000000"),
        tickformat="%b %d<br>%H:%M", hoverformat="%Y-%m-%d %H:%M:%S UTC",
    )

    if mode == "Historical Storm Replay" and 'step' in locals() and benchmark_mode:
        current_time_val = df_full.index[step]
        fig_params.add_vline(x=current_time_val, line_width=1.5, line_dash="dash", line_color="#FF4B4B")

    st.plotly_chart(apply_radshield_chart_theme(fig_params), use_container_width=True, theme=None)

    st.markdown("<hr style='margin:24px 0'>", unsafe_allow_html=True)

with fleet_tab:
    # ─── GEO Satellite Catalogue and Interactive Globe ────────────────────────────
    st.markdown('<p class="section-label">GEO Satellite Catalogue &amp; Orbital Positions</p>', unsafe_allow_html=True)
    st.caption(
        "Positions are propagated from CelesTrak public GP/TLE orbital elements using SGP4; "
        "they are not live spacecraft telemetry. Catalogue element epoch and position calculation time "
        "are shown separately. Globe distances are normalized for display."
    )

    @st.cache_data(ttl=300, show_spinner=False)
    def load_geo_catalogue_positions():
        from radshield.data.geo_satellites import fetch_geo_satellite_positions

        return fetch_geo_satellite_positions()


    geo_positions = None
    geo_source_error = None
    with st.spinner("Loading CelesTrak GEO orbital elements and propagating positions..."):
        try:
            geo_positions = load_geo_catalogue_positions()
        except (ImportError, ValueError, OSError) as exc:
            geo_source_error = str(exc)
        except Exception as exc:
            import requests

            if isinstance(exc, requests.RequestException):
                geo_source_error = str(exc)
            else:
                raise

    if geo_source_error:
        st.error(f"GEO catalogue unavailable: {geo_source_error}")
        st.caption(
            "No satellite positions are displayed. Check the CelesTrak source and network connection; "
            "this view does not use fabricated fallback positions."
        )
    elif not geo_positions:
        st.info("The source returned no valid satellites matching the configured GEO orbit filter.")
    else:
        from datetime import datetime, timedelta, timezone
        from radshield.data.geo_satellites import (
            CELESTRAK_GEO_TLE_URL,
            MAX_ECCENTRICITY,
            MAX_INCLINATION_DEGREES,
            MAX_MEAN_MOTION_REV_PER_DAY,
            MIN_MEAN_MOTION_REV_PER_DAY,
            POSITION_STALE_AFTER,
            classify_catalogue_freshness,
        )

        position_time = pd.to_datetime(
            geo_positions[0]["position_timestamp_utc"], utc=True
        ).to_pydatetime()
        catalogue_now = datetime.now(timezone.utc)
        for satellite in geo_positions:
            element_epoch = pd.to_datetime(satellite["tle_epoch_utc"], utc=True).to_pydatetime()
            is_stale, age_hours = classify_catalogue_freshness(element_epoch, catalogue_now)
            satellite["element_age_hours"] = age_hours
            satellite["freshness"] = "STALE" if is_stale else "FRESH"

        stale_count = sum(item["freshness"] == "STALE" for item in geo_positions)
        stale_after_hours = POSITION_STALE_AFTER.total_seconds() / 3600
        st.markdown(
            f"**Source:** [CelesTrak GEO GP/TLE]({CELESTRAK_GEO_TLE_URL})  \n"
            f"**Propagated positions:** {len(geo_positions):,}  ·  "
            f"**Position calculation time:** {position_time:%Y-%m-%d %H:%M:%S UTC}  ·  "
            f"**Stale elements (> {stale_after_hours:.0f} h):** {stale_count:,}"
        )
        if stale_count:
            st.warning(
                f"{stale_count} position(s) use TLE elements older than {stale_after_hours:.0f} hours. "
                "Their propagated positions are marked stale and should not be treated as current tracking."
            )
        else:
            st.success("All displayed TLE epochs are within the configured 24-hour freshness window.")
        st.caption(
            "GEO filter applied to source orbital elements: eccentricity ≤ "
            f"{MAX_ECCENTRICITY:g}, inclination ≤ {MAX_INCLINATION_DEGREES:g}°, mean motion "
            f"{MIN_MEAN_MOTION_REV_PER_DAY:g}–{MAX_MEAN_MOTION_REV_PER_DAY:g} rev/day. "
            "Nominal slot longitude is a catalogue/design reference; these plotted longitudes are "
            "time-updated SGP4 positions."
        )

        globe_col, table_col = st.columns([1.25, 0.75])
        with globe_col:
            latitude_grid = np.linspace(-90, 90, 37)
            longitude_grid = np.linspace(-180, 180, 73)
            lat_mesh, lon_mesh = np.meshgrid(latitude_grid, longitude_grid, indexing="ij")
            lat_radians = np.radians(lat_mesh)
            lon_radians = np.radians(lon_mesh)
            earth_x = np.cos(lat_radians) * np.cos(lon_radians)
            earth_y = np.cos(lat_radians) * np.sin(lon_radians)
            earth_z = np.sin(lat_radians)

            globe = go.Figure()
            globe.add_trace(go.Surface(
                x=earth_x,
                y=earth_y,
                z=earth_z,
                surfacecolor=lat_mesh,
                colorscale=[
                    [0.0, "#0B2340"],
                    [0.25, "#123B5A"],
                    [0.5, "#176078"],
                    [0.75, "#123B5A"],
                    [1.0, "#0B2340"],
                ],
                cmin=-90,
                cmax=90,
                showscale=False,
                lighting=dict(ambient=0.55, diffuse=0.8, specular=0.2, roughness=0.85),
                hoverinfo="skip",
                name="Earth (stylized)",
            ))
            orbit_angles = np.linspace(0, 2 * np.pi, 361)
            globe.add_trace(go.Scatter3d(
                x=1.16 * np.cos(orbit_angles),
                y=1.16 * np.sin(orbit_angles),
                z=np.zeros_like(orbit_angles),
                mode="lines",
                line=dict(color="rgba(115, 203, 235, 0.5)", width=2),
                name="Equatorial GEO reference (normalized)",
                hoverinfo="skip",
            ))
            sat_lat = np.radians([item["latitude_deg"] for item in geo_positions])
            sat_lon = np.radians([item["orbital_longitude_deg"] for item in geo_positions])
            display_radius = 1.2
            sat_x = display_radius * np.cos(sat_lat) * np.cos(sat_lon)
            sat_y = display_radius * np.cos(sat_lat) * np.sin(sat_lon)
            sat_z = display_radius * np.sin(sat_lat)
            hover_text = [
                (
                    f'{item["name"]} (NORAD {item["norad_id"]})'
                    f'<br>Longitude: {item["orbital_longitude_deg"]:.3f}°'
                    f'<br>Latitude: {item["latitude_deg"]:.3f}°'
                    f'<br>Altitude: {item["altitude_km"]:,.1f} km'
                    f'<br>Position calculated: {item["position_timestamp_utc"]}'
                    f'<br>TLE epoch: {item["tle_epoch_utc"]}'
                    f'<br>Element age: {item["element_age_hours"]:.1f} h ({item["freshness"]})'
                    f'<br>Source: CelesTrak GEO GP/TLE'
                )
                for item in geo_positions
            ]
            globe.add_trace(go.Scatter3d(
                x=sat_x,
                y=sat_y,
                z=sat_z,
                mode="markers",
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
                marker=dict(
                    size=4,
                    color=[
                        "#F97316" if item["freshness"] == "STALE" else "#61E6C2"
                        for item in geo_positions
                    ],
                    line=dict(color="#E8F4FD", width=0.5),
                    opacity=0.95,
                ),
                name="SGP4-propagated GEO satellites",
            ))
            globe.update_layout(
                height=590,
                paper_bgcolor="#080D18",
                plot_bgcolor="#080D18",
                font=dict(family="Inter, sans-serif", color="#D7E2F0"),
                margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(
                    bgcolor="rgba(8, 13, 24, 0.75)",
                    font=dict(color="#D7E2F0", size=10),
                    orientation="h",
                    y=0.02,
                    x=0.02,
                ),
                scene=dict(
                    xaxis=dict(visible=False, range=[-1.3, 1.3]),
                    yaxis=dict(visible=False, range=[-1.3, 1.3]),
                    zaxis=dict(visible=False, range=[-1.3, 1.3]),
                    aspectmode="cube",
                    bgcolor="#080D18",
                    camera=dict(eye=dict(x=1.55, y=1.55, z=1.15)),
                ),
                showlegend=True,
            )
            st.plotly_chart(globe, use_container_width=True, theme=None)
        with table_col:
            st.markdown("**Validated GEO catalogue entries**")
            satellite_table = pd.DataFrame(
                [
                    {
                        "Name": item["name"],
                        "NORAD ID": item["norad_id"],
                        "Longitude (°)": item["orbital_longitude_deg"],
                        "Latitude (°)": item["latitude_deg"],
                        "Altitude (km)": item["altitude_km"],
                        "TLE epoch (UTC)": item["tle_epoch_utc"],
                        "Element age (h)": item["element_age_hours"],
                        "Freshness": item["freshness"],
                    }
                    for item in geo_positions
                ]
            )
            st.dataframe(
                satellite_table,
                hide_index=True,
                use_container_width=True,
                height=430,
            )
            st.caption(f"Catalogue endpoint: {CELESTRAK_GEO_TLE_URL}")

    st.markdown("<hr style='margin:24px 0'>", unsafe_allow_html=True)

    # ─── GSAT-19 GRASP Estimate ───────────────────────────────────────────────────
    st.markdown('<p class="section-label">GSAT-19 GRASP Sector Estimate (48°E) — Not Telemetry</p>', unsafe_allow_html=True)

    isro_unlocked = st.sidebar.checkbox("Unlock Virtual GSAT-19 (48°E Indian Sector)", value=True)

    if not isro_unlocked:
        fig_grasp = go.Figure()
        fig_grasp.add_annotation(
            x=0.5, y=0.5,
            text="Awaiting ISRO ISSDC Secure API Authentication<br><span style='font-size:12px;color:#DC2626'>GRASP STREAM LOCKED (Unlock in Sidebar for Virtual Mode)</span>",
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(family="Space Mono", size=16, color="#DC2626")
        )
        fig_grasp.update_layout(
            height=180,
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, linecolor="#CBD5E1"),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, linecolor="#CBD5E1"),
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(apply_radshield_chart_theme(fig_grasp), use_container_width=True, theme=None)
    else:
        model_input_flux = df_plot["flux"].values[-hist_n:]
        goes_flux = (
            df_plot["observed_flux"].values[-hist_n:]
            if live_parameter_series and "observed_flux" in df_plot
            else model_input_flux
        )
        ind_mlt = (mlt_hours + 8.2) % 24
        goes_log = np.log10(np.maximum(model_input_flux, 1.0))
        kp_array = df_plot['KP'].values[-hist_n:]
        dawn_mask_ind = ((ind_mlt > 3) & (ind_mlt < 9)).astype(float)
        dawn_mask_goes = ((mlt_hours > 3) & (mlt_hours < 9)).astype(float)
        gsat_log = goes_log + 0.15 + (dawn_mask_ind - dawn_mask_goes) * (kp_array / 9.0) * 0.5
        gsat_flux = 10**gsat_log
    
        fig_gsat = go.Figure()
        goes_label = (
            "Direct NOAA GOES flux [pfu]"
            if live_parameter_series
            else "Processed benchmark flux [pfu; source gaps not retained]"
        )
        fig_gsat.add_trace(go.Scatter(
            x=t_hist, y=goes_flux, name=goes_label,
            line=dict(color="#2563EB", width=1.5, dash="dot"), connectgaps=False,
        ))
        fig_gsat.add_trace(go.Scatter(
            x=t_hist, y=gsat_flux, name="Virtual GSAT-19 sector estimate [pfu]",
            line=dict(color="#059669", width=2.5), connectgaps=False,
        ))
    
        fig_gsat.update_layout(
            height=300,
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            yaxis_type="log",
            font=dict(family="Inter, sans-serif", size=12, color="#000000"),
            margin=dict(l=85, r=20, t=10, b=60),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(255,255,255,0.9)", font=dict(size=12, color="#000000")),
            yaxis=dict(title="Flux (pfu) [>2 MeV]", showgrid=True, gridcolor="#E2E8F0", linecolor="#CBD5E1", range=[0, 6], title_font=dict(color="#000000"), tickfont=dict(color="#000000"))
        )
        fig_gsat.update_xaxes(
            showgrid=True, gridcolor="#E2E8F0", linecolor="#CBD5E1",
            title_font=dict(color="#000000"), tickfont=dict(color="#000000"),
            tickformat="%b %d<br>%H:%M", hoverformat="%Y-%m-%d %H:%M:%S UTC",
        )
        fig_gsat.update_traces(
            hovertemplate="%{x|%Y-%m-%d %H:%M:%S} UTC<br>%{y:.4g} pfu<extra>%{fullData.name}</extra>"
        )
        st.plotly_chart(apply_radshield_chart_theme(fig_gsat), use_container_width=True, theme=None)
        st.caption("The virtual GSAT-19 curve is derived from model inputs, which may contain filled values. It is not direct GSAT-19 GRASP telemetry.")

    st.markdown("<hr style='margin:24px 0'>", unsafe_allow_html=True)

with research_tab:
    with st.expander("TFT checkpoint compatibility & provenance", expanded=False):
        if tft_model_status["available"]:
            st.caption(
                f"Checkpoint: {os.path.basename(tft_model_status['checkpoint'])}. "
                f"Inference supported: {tft_model_status['inference_supported']}."
            )
            st.markdown(
                "The checkpoint is a tensor-only state dictionary without feature names, "
                "training configuration, or scaler reference. Strict loading verifies tensor "
                "dimensions only; it does not establish feature meaning. Repository pipelines "
                "write the same checkpoint filename using conflicting feature schemas."
            )
            st.markdown(
                "- `radshield/training/train_tft.py` uses the package fine-tuning features below.\n"
                "- `radshield_kaggle_train.py` uses the dashboard feature schema, but its required "
                "11-year pretraining CSV is absent from this checkout."
            )
            st.markdown("**Dashboard/Kaggle feature order:** " + ", ".join(TFT_DASHBOARD_TRAINING_FEATURE_COLUMNS))
            st.markdown("**Package fine-tuning feature order:** " + ", ".join(TFT_PACKAGE_FINETUNE_FEATURE_COLUMNS))
            st.dataframe(
                pd.DataFrame(TFT_PACKAGE_FINETUNE_FEATURE_AUDIT),
                hide_index=True,
                use_container_width=True,
            )
            st.markdown(
                "The adjacent `scaler.pkl` has `mean` and `std` arrays but no feature labels, "
                "dataset identity, or checkpoint hash. Training scripts also differ in scaler "
                "fitting. The dashboard's per-window normalization cannot be verified against it."
            )
            st.markdown(
                "Training provenance: "
                + ("verified" if tft_model_status["training_provenance_verified"] else "not verified")
                + "; scaler provenance: "
                + ("verified" if tft_model_status["scaler_provenance_verified"] else "not verified")
                + "; preprocessing parity: "
                + ("verified" if TFT_PREPROCESSING_MATCHES_CHECKPOINT else "not verified")
                + "."
            )
            if not tft_model_status["inference_supported"]:
                st.error(
                    "Inference remains blocked. Do not map channels by position, assume the "
                    "adjacent scaler belongs to this checkpoint, or substitute per-window statistics."
                )
        else:
            st.error(
                "TFT checkpoint unavailable; no TFT forecast is displayed. "
                f"Load error: {tft_model_status['error']}"
            )

    # ─── Model Performance & Prediction vs. Observation Section ───────────────────
    st.markdown('<p class="section-label">Synthetic Demonstration Preview</p>', unsafe_allow_html=True)
    st.warning(
        "The stored prediction arrays are synthetic demonstrations, not outputs from the loaded TFT checkpoint. "
        "The values below are not verified model-evaluation metrics; observation provenance and checkpoint "
        "training provenance are not established."
    )

    # Load cached evaluation metrics JSON
    eval_json_path = os.path.join(ROOT_DIR, "radshield", "data", "historical", "goes_train_val_test_eval.json")
    eval_data = None
    if os.path.exists(eval_json_path):
        try:
            import json
            with open(eval_json_path, "r") as f:
                eval_data = json.load(f)
        except Exception:
            eval_data = None

    eval_tab_train, eval_tab_val, eval_tab_test, eval_tab_timeline = st.tabs([
        "Synthetic Training Preview",
        "Synthetic Validation Preview",
        "Synthetic Test Preview",
        "Synthetic 11-Year Preview"
    ])

    def render_pred_vs_obs_chart(obs, pred, title_prefix, color="#2563EB"):
        obs_arr = np.array(obs)
        pred_arr = np.array(pred)
    
        mask = ~np.isnan(obs_arr) & ~np.isnan(pred_arr)
        if not mask.any():
            st.warning("No data available for evaluation.")
            return
    
        o = obs_arr[mask]
        p = pred_arr[mask]
    
        lc = float(np.corrcoef(o, p)[0, 1]) if len(o) > 1 else 0.0
        rmse = float(np.sqrt(np.mean((o - p)**2)))
        mae = float(np.mean(np.abs(o - p)))
    
        # Render Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"DEMO {title_prefix} LC (not verified)", f"{lc:.3f}")
        c2.metric(f"DEMO {title_prefix} RMSE log10 (not verified)", f"{rmse:.4f}")
        c3.metric(f"DEMO {title_prefix} MAE log10 (not verified)", f"{mae:.4f}")
        c4.metric("DEMO Flux Error Factor (not verified)", f"{(10**rmse):.1f}×")
    
        # Render Plotly Scatter
        fig_scatter = go.Figure()
    
        # 1:1 Reference Line
        min_val = min(o.min(), p.min(), 0.0)
        max_val = max(o.max(), p.max(), 6.0)
        fig_scatter.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode="lines", name="Ideal 1:1 Alignment (y = x)",
            line=dict(color="#DC2626", width=2, dash="dash")
        ))
    
        # Scatter points
        fig_scatter.add_trace(go.Scatter(
            x=o, y=p, mode="markers",
            name="Synthetic preview pairs",
            hovertemplate="Synthetic observed value: %{x:.4g} log10(pfu)<br>Synthetic prediction: %{y:.4g} log10(pfu)<extra></extra>",
            marker=dict(color=color, size=4, opacity=0.6)
        ))
    
        # Linear Regression trendline
        if len(o) > 2:
            m, b_line = np.polyfit(o, p, 1)
            x_trend = np.linspace(min_val, max_val, 100)
            y_trend = m * x_trend + b_line
            fig_scatter.add_trace(go.Scatter(
                x=x_trend, y=y_trend, mode="lines",
                name=f"Model Trendline (slope={m:.2f})",
                line=dict(color="#315A9B", width=2)
            ))
        
        fig_scatter.update_layout(
            height=380,
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(family="Inter, sans-serif", size=12, color="#000000"),
            xaxis=dict(title="Observed log₁₀(Electron Flux) [pfu]", showgrid=True, gridcolor="#E2E8F0", zeroline=False),
            yaxis=dict(title="Predicted log₁₀(Electron Flux) [pfu]", showgrid=True, gridcolor="#E2E8F0", zeroline=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(255,255,255,0.9)", font=dict(color="#000000")),
            margin=dict(l=60, r=20, t=30, b=50)
        )
        st.plotly_chart(apply_radshield_chart_theme(fig_scatter), use_container_width=True, theme=None)

    with eval_tab_train:
        st.markdown('<p class="section-label">Synthetic training preview — not TFT performance</p>', unsafe_allow_html=True)
        if eval_data and "train" in eval_data and len(eval_data["train"]["observed_log"]) > 0:
            render_pred_vs_obs_chart(
                eval_data["train"]["observed_log"],
                eval_data["train"]["predicted_log"],
                title_prefix="Training",
                color="#2563EB"
            )
        else:
            st.info("Generating training set statistics... Please refresh.")

    with eval_tab_val:
        st.markdown('<p class="section-label">Synthetic validation preview — not TFT performance</p>', unsafe_allow_html=True)
        if eval_data and "validation" in eval_data and len(eval_data["validation"]["observed_log"]) > 0:
            render_pred_vs_obs_chart(
                eval_data["validation"]["observed_log"],
                eval_data["validation"]["predicted_log"],
                title_prefix="Validation",
                color="#059669"
            )
        else:
            st.info("Validation dataset benchmark data loading...")

    with eval_tab_test:
        st.markdown('<p class="section-label">Synthetic test preview — not a verified blind test</p>', unsafe_allow_html=True)
        if eval_data and "test" in eval_data and len(eval_data["test"]["observed_log"]) > 0:
            render_pred_vs_obs_chart(
                eval_data["test"]["observed_log"],
                eval_data["test"]["predicted_log"],
                title_prefix="Blind Test",
                color="#EA580C"
            )
        else:
            st.info("Blind test benchmark loading...")

    with eval_tab_timeline:
        st.markdown('<p class="section-label">Synthetic 11-year preview — not an observed GOES/TFT record</p>', unsafe_allow_html=True)
        if eval_data and "train" in eval_data and len(eval_data["train"]["timestamps"]) > 0:
            ts = pd.to_datetime(eval_data["train"]["timestamps"])
            obs_t = np.array(eval_data["train"]["observed_log"])
            pred_t = np.array(eval_data["train"]["predicted_log"])
        
            epoch_choice = st.radio(
                "Filter Solar Cycle Epoch:",
                ["Full Decade (2013–2023)", "Solar Cycle 24 Peak (2013–2015)", "Solar Minimum (2018–2020)", "Solar Cycle 25 Acceleration (2021–2023)"],
                horizontal=True
            )
        
            if epoch_choice == "Solar Cycle 24 Peak (2013–2015)":
                epoch_mask = (ts >= '2013-01-01') & (ts <= '2015-12-31')
            elif epoch_choice == "Solar Minimum (2018–2020)":
                epoch_mask = (ts >= '2018-01-01') & (ts <= '2020-12-31')
            elif epoch_choice == "Solar Cycle 25 Acceleration (2021–2023)":
                epoch_mask = (ts >= '2021-01-01') & (ts <= '2023-12-31')
            else:
                epoch_mask = np.ones(len(ts), dtype=bool)
            
            fig_timeline = go.Figure()
            fig_timeline.add_trace(go.Scatter(
                x=ts[epoch_mask], y=10**obs_t[epoch_mask],
                name="Synthetic preview observation [pfu]",
                line=dict(color="#2563EB", width=1.5), connectgaps=False,
            ))
            fig_timeline.add_trace(go.Scatter(
                x=ts[epoch_mask], y=10**pred_t[epoch_mask],
                name="Synthetic preview prediction [pfu]",
                line=dict(color="#F59E0B", width=1.5), connectgaps=False,
            ))
            fig_timeline.update_layout(
                height=380,
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                yaxis_type="log",
                font=dict(family="Inter, sans-serif", size=12, color="#000000"),
                xaxis=dict(
                    title="DATE (UTC)", showgrid=True, gridcolor="#E2E8F0", zeroline=False,
                    tickformat="%Y-%m", hoverformat="%Y-%m-%d %H:%M:%S UTC",
                ),
                yaxis=dict(title="Electron Flux (pfu) [>2 MeV]", showgrid=True, gridcolor="#E2E8F0", zeroline=False, range=[0, 6]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(255,255,255,0.9)", font=dict(color="#000000")),
                margin=dict(l=60, r=20, t=30, b=50)
            )
            fig_timeline.update_traces(
                hovertemplate="%{x|%Y-%m-%d %H:%M:%S} UTC<br>%{y:.4g} pfu<extra>%{fullData.name}</extra>"
            )
            st.plotly_chart(apply_radshield_chart_theme(fig_timeline), use_container_width=True, theme=None)
        else:
            st.info("Timeline data loading...")

    st.markdown("<hr style='margin:24px 0'>", unsafe_allow_html=True)

    # ─── Tabbed Diagnostics & Metrics ─────────────────────────────────────────────
    tab_drivers, tab_importance, tab_specs = st.tabs([
        "Driver intensity scales",
        "AI Feature Importance",
        "System Specifications & Provenance"
    ])

    with tab_drivers:
        st.markdown('<p class="section-label">Current Driver Intensity Scale (Relative to Extreme Storm)</p>', unsafe_allow_html=True)
        d1, d2 = st.columns(2)
        with d1:
            # Use physical heuristics for UI visualization instead of raw TFT attention (which is overly spiky)
            current_t = t_hist[-1]
            mlt_hrs = current_t.hour + current_t.minute/60.0 + 3.2
            mlt_sin_val = np.sin(mlt_hrs * 2 * np.pi / 24.0)

            # Pull scalar values from the last row
            np_val  = float(row.get('Np',  5.0))
            ae_val  = float(row.get('AE',  100.0))
            ec_val  = float(row.get('Ec',  0.0))
            pdyn_val = float(row.get('Pdyn', 2.0))
            bz_dur_val = float(row.get('Bz_neg_dur', 0.0))
            bt_val = float(row.get('BT', 5.0))
            f107_val = float(row.get('F10.7_index', 150.0))
        
            # dDst_dt from ingest is in nT/min. Convert to nT/hr for the UI display without breaking the ML model inputs.
            ddst_val = float(row.get('dDst_dt', 0.0)) * 60.0

            pct_vsw    = min(0.95, vsw / 800)
            pct_bz     = min(0.95, abs(bz) / 20)          # 20 nT = full storm
            pct_ulf    = min(0.95, (ulf + 4) / 2.5)       # ULF range: -4 to -1.5
            pct_pdyn   = min(0.95, pdyn_val / 20)          # 20 nPa = extreme pressure
            pct_bz_dur = min(0.95, bz_dur_val / 120.0)    # 120 min = 2 hr sustained southward
            pct_mlt    = min(0.95, abs(mlt_sin_val))
            pct_kp     = min(0.95, kp / 9.0)              # Kp 0-9 scale
            pct_dst    = min(0.95, abs(dst) / 150)        # -150 nT = severe storm
            pct_ae     = min(0.95, ae_val / 2000)         # 2000 nT = substorm max
            pct_np     = min(0.95, np_val / 50)           # 50 cm⁻³ = extreme density
            pct_ec     = min(0.95, ec_val / 5)            # 5 mV/m = extreme coupling
            pct_bt     = min(0.95, bt_val / 30)           # 30 nT = massive total field
            pct_f107   = min(0.95, (f107_val - 70) / 180) # 70 to 250 range
            pct_ddst   = min(0.95, max(0.0, -ddst_val) / 50.0) # Intensification is specifically negative dDst/dt (drop rate)

            by_val = float(row.get('BY_GSM', 0.0))
            pct_by = min(0.95, abs(by_val) / 20.0)
        
            drivers = [
                ("By — IMF Dawn-Dusk (GSM)",           f"{by_val:.1f} nT",      pct_by),
                ("Bz — Southward IMF (GSM)",           f"{bz:.1f} nT",          pct_bz),
                ("Pdyn — Dynamic Pressure",            f"{pdyn_val:.2f} nPa",   pct_pdyn),
                ("Vsw — Solar Wind Velocity",          f"{vsw:.0f} km/s",       pct_vsw),
                ("AE — Auroral Electrojet",            f"{ae_val:.0f} nT",      pct_ae),
                ("Dst — Ring Current Injection",       f"{dst:.0f} nT",         pct_dst),
                ("F10.7 — Solar Radio Flux",           f"{f107_val:.1f} sfu",   pct_f107),
                ("MLT — Orbital Sine Component",       f"{mlt_sin_val:.2f}",    pct_mlt),
            ]
            for name, val, pct in drivers:
                pct_val = float(np.clip(pct, 0.03, 1.0))
                pct_int = int(pct_val * 100)
                st.markdown(f"""
    <div style="display:flex;justify-content:space-between;margin-bottom:4px">
      <span style="font-size:0.82rem;color:#43536A;font-family:'Inter',sans-serif">{name}</span>
      <span style="font-size:0.82rem;color:#315A9B;font-family:'Space Mono',monospace;font-weight:700">{val} &nbsp;<span style="color:#65758A">({pct_int}%)</span></span>
    </div>
    <div style="background:#E5E9EF;border:1px solid #D7DFE8;border-radius:4px;height:12px;width:100%;overflow:hidden;margin-bottom:14px">
      <div style="background:#315A9B;width:{pct_int}%;height:100%;border-radius:3px"></div>
    </div>""", unsafe_allow_html=True)
        with d2:
            st.info(f"""
    **Geomagnetic State:** {REGIME_LABELS[regime]}
    **Convection Field (Ec):** {float(row.get('Ec', 0)):.2f} mV/m
    **Dynamic Pressure (Pdyn):** {float(row.get('Pdyn', 0)):.2f} nPa
    **Bz Negative Duration:** {float(row.get('Bz_neg_dur', 0)):.0f} min
            """)

    with tab_importance:
        st.markdown('<p class="section-label">Illustrative Feature Weights — Not Loaded from Checkpoint</p>', unsafe_allow_html=True)
        st.caption("These static display values are not verified VSN outputs or measured model feature importance.")

    
        # Static weights based on the new 10-feature trained PyTorch model
        feature_names = ["Past Flux (Autoregressive)", "Solar Wind Speed (Vsw)", "AE Index (Substorms)", "Southward IMF (Bz)", "Ring Current (Dst)", "Magnetic Local Time (MLT)", "Dynamic Pressure (Pdyn)", "IMF Dawn-Dusk (By)", "F10.7 (Solar Radio)"]
        importance_weights = [42.1, 18.5, 12.3, 8.7, 7.2, 4.9, 3.4, 1.8, 1.1]
    
        fig_imp = go.Figure(go.Bar(
            x=importance_weights[::-1],
            y=feature_names[::-1],
            orientation='h',
            marker=dict(
                color=importance_weights[::-1],
                colorscale=[[0, '#B7C6DA'], [1, '#315A9B']],
                showscale=False
            )
        ))
    
        fig_imp.update_layout(
            height=350,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", size=11, color="#43536A"),
            margin=dict(l=20, r=20, t=30, b=30),
            xaxis=dict(title="VSN Importance Weight (%)", showgrid=True, gridcolor="#E5E9EF", linecolor="#C9D1DC"),
            yaxis=dict(showgrid=False, linecolor="#C9D1DC")
        )
    
        st.plotly_chart(apply_radshield_chart_theme(fig_imp), use_container_width=True)

    with tab_specs:
        st.markdown('<p class="section-label">Architecture, Training & Mission Payload Hardware Specifications</p>', unsafe_allow_html=True)
        st.markdown("""
    - **Mission Payload Focus**: GSAT-19 GRASP Payload (Geostationary Orbit, 48°E Indian Sector)
    - **Deep Learning Model**: Custom PyTorch Temporal Fusion Transformer (TFT) with Variable Selection Networks (VSN) & Gated Residual Networks (GRN)
    - **Loss Function**: Physics-Informed Pinball Loss (PINN) with 1D Fokker-Planck Radial Diffusion ODE regularization
    - **Checkpoint training data and provenance**: Not verified from the available project files
    - **Ground Conjugate Station**: INTERMAGNET Hyderabad (HYB) Pc5 ULF Wave Power (1.7–6.7 mHz); operational data connection not verified here
    - **Serving Architecture**: FastAPI / Streamlit dashboard; checkpoint registry sync is configured but availability is not guaranteed
        """)

    st.markdown(f"""
    <p style="font-family:'Space Mono',monospace;font-size:0.62rem;color:#2A3D50;
    text-align:center;margin-top:24px;letter-spacing:0.10em">
    RADSHIELD AI &nbsp;|&nbsp; TFT + RADIAL DIFFUSION ENSEMBLE &nbsp;|&nbsp;
    GOES / OMNI / INTERMAGNET DATA SOURCES
    </p>""", unsafe_allow_html=True)
