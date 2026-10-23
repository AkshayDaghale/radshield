"""
==============================================================================
KAVACH — GEO Radiation Monitor | Kaggle Training Pipeline
Bharatiya Antariksh Hackathon 2026 | Team DigiIndia | PS-14 ISRO

3-Stage Training:
  Stage 1: PRE-TRAIN  on 11-year OMNI dataset (846K rows)
  Stage 2: FINE-TUNE  on 2017/2018 GOES-16 + ATHA ULF dataset (123K rows)
  Stage 3: VALIDATE   on Mar 2015 held-out storms (7.5K rows)
  Output:  Push weights to Hugging Face Hub
==============================================================================
INSTRUCTIONS FOR KAGGLE:
1. Upload this script as a new Kaggle notebook (Python)
2. Add your 3 datasets as Kaggle Dataset inputs:
   - Kaggle_PreTraining_Dataset.csv
   - Kaggle_FineTuning_Dataset.csv
   - Kaggle_Validation_March2015.csv
3. Add your Hugging Face token as a Kaggle Secret named: HF_TOKEN
4. Enable GPU (P100) accelerator
5. Run All
==============================================================================
"""

# ─── Cell 1: Install & Imports ───────────────────────────────────────────────
import subprocess
subprocess.run(["pip", "install", "huggingface_hub", "joblib", "-q"])

import os, sys, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from torch.optim.lr_scheduler import CosineAnnealingLR
from huggingface_hub import HfApi, upload_file

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {DEVICE}")

# ─── Cell 2: Model Architecture (identical to kavach/models/tft_model.py) ────
class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.layer_norm = nn.LayerNorm(output_dim)
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = self.dropout(self.fc2(h))
        g = self.sigmoid(self.gate(x))
        return self.layer_norm(self.skip(x) + g * h)

class KAVACH_TFT(nn.Module):
    """10-feature TFT (Mentor Architecture): seq_len=288 (24h) → 144-step (12h) quantile forecast."""
    def __init__(self, num_features=10, hidden_size=128, lstm_layers=2, num_quantiles=5, dropout=0.1):
        super().__init__()
        self.num_features = num_features
        self.num_quantiles = num_quantiles
        self.quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
        self.vsn = nn.ModuleList([GatedResidualNetwork(1, hidden_size, hidden_size, dropout) for _ in range(num_features)])
        self.vsn_weights = nn.Linear(num_features, num_features)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=lstm_layers, batch_first=True,
                            dropout=dropout if lstm_layers > 1 else 0.0)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4, batch_first=True)
        self.grn_dec = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)
        self.quantile_head = nn.Linear(hidden_size, 144 * num_quantiles)

    def forward(self, x):
        b, seq_len, n_feat = x.shape
        vsn_outputs = [self.vsn[i](x[:, :, i:i+1]) for i in range(min(n_feat, self.num_features))]
        vsn_stack = torch.stack(vsn_outputs, dim=-1)
        attn_scores = torch.softmax(self.vsn_weights(x.mean(dim=1)), dim=-1).unsqueeze(1).unsqueeze(2)
        vsn_fused = (vsn_stack * attn_scores).sum(dim=-1)
        lstm_out, _ = self.lstm(vsn_fused)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        dec_out = self.grn_dec(attn_out[:, -1, :])
        q_out = self.quantile_head(dec_out).view(b, 144, self.num_quantiles)
        return q_out, attn_scores

class PhysicsInformedPinballLoss(nn.Module):
    def __init__(self, quantiles=[0.10, 0.25, 0.50, 0.75, 0.90], lambda_physics=0.05):
        super().__init__()
        self.quantiles = quantiles
        self.lambda_physics = lambda_physics

    def forward(self, pred, target):
        loss = sum(torch.max((q-1)*(target - pred[:,:,i]), q*(target - pred[:,:,i])).mean()
                   for i, q in enumerate(self.quantiles))
        d1 = pred[:, 1:, 2] - pred[:, :-1, 2]
        d2 = d1[:, 1:] - d1[:, :-1]
        return loss + self.lambda_physics * torch.mean(d2 ** 2)

# ─── Cell 3: Data Paths ───────────────────────────────────────────────────────
# Kaggle input paths — adjust if dataset names differ
PRETRAIN_PATH  = "/kaggle/input/kavach-isro-datasets/Kaggle_PreTraining_Dataset.csv"
FINETUNE_PATH  = "/kaggle/input/kavach-isro-datasets/Kaggle_FineTuning_Dataset.csv"
VAL_MAY_PATH   = "/kaggle/input/kavach-isro-datasets/may_2024_benchmark.csv"
VAL_OCT_PATH   = "/kaggle/input/kavach-isro-datasets/oct_2024_benchmark.csv"

# Fallback: check current directory, kaggle inputs, and cloned repo
search_paths = [
    "/kaggle/input",
    "/kaggle/working/BAH",
    "."
]

for attr, name in [("PRETRAIN_PATH", "Kaggle_PreTraining_Dataset.csv"),
                   ("FINETUNE_PATH", "Kaggle_FineTuning_Dataset.csv"),
                   ("VAL_MAY_PATH", "may_2024_benchmark.csv"),
                   ("VAL_OCT_PATH", "oct_2024_benchmark.csv")]:
    if not os.path.exists(globals()[attr]):
        found = False
        for search_dir in search_paths:
            if os.path.exists(search_dir):
                for root, dirs, files in os.walk(search_dir):
                    if name in files:
                        globals()[attr] = os.path.join(root, name)
                        print(f"Found {name} at {globals()[attr]}")
                        found = True
                        break
            if found: break

print(f"\nPre-train   : {PRETRAIN_PATH} — exists: {os.path.exists(PRETRAIN_PATH)}")
print(f"Fine-tune   : {FINETUNE_PATH} — exists: {os.path.exists(FINETUNE_PATH)}")
print(f"Val (May)   : {VAL_MAY_PATH}  — exists: {os.path.exists(VAL_MAY_PATH)}")
print(f"Val (Oct)   : {VAL_OCT_PATH}  — exists: {os.path.exists(VAL_OCT_PATH)}")

# ─── Cell 4: Feature Engineering ─────────────────────────────────────────────
# Mentor-approved strict 10-feature architecture
FEATURE_COLS = [
    "log_electron_flux", "BY_GSM", "BZ_GSM", "Pdyn", "Vsw", 
    "AE", "DST", "F10.7_index", "MLT_sin", "MLT_cos"
]
SEQ_LEN, PRED_LEN = 288, 144

# Shared NOAA/OMNI missing data fill values (used across all prepare functions)
NOAA_FILL_VALS = [9999999.0, 99999.9, 99999.0, 9999.99, 9999.0, 999.99, 999.9, 999.0, 99.99, 99.0]

def calculate_mlt_vectorized(dt_index: pd.DatetimeIndex, satellite_lon: float) -> np.ndarray:
    """Vectorized MLT calculation using Equation of Time (EoT) and Subsolar Longitude."""
    ut_hours = dt_index.hour + dt_index.minute / 60.0 + dt_index.second / 3600.0
    doy = dt_index.dayofyear
    year_fraction = (2 * np.pi / 365.0) * (doy - 1 + (ut_hours - 12.0) / 24.0)
    eot = 229.18 * (0.000075 + 0.001868 * np.cos(year_fraction) 
                    - 0.032077 * np.sin(year_fraction) 
                    - 0.014615 * np.cos(2 * year_fraction) 
                    - 0.040849 * np.sin(2 * year_fraction))
    subsolar_lon = -15.0 * (ut_hours - 12.0 + eot / 60.0)
    mlt = ut_hours + (satellite_lon - subsolar_lon) / 15.0
    return np.mod(mlt, 24.0).values

def prepare_pretrain(path):
    """Prepares the 11-year OMNI pre-training dataset."""
    print(f"\n[STAGE 1] Loading pre-training data from {path}...")
    df = pd.read_csv(path, parse_dates=['datetime'], index_col='datetime')
    df.sort_index(inplace=True)

    # Exclude March 2015 — it is now in the validation benchmark (Oct/May 2024 storms)
    # but we keep this exclusion to avoid any future leakage if datasets overlap.
    df = df[~((df.index.year == 2015) & (df.index.month == 3))]
    print(f"[STAGE 1] Excluded March 2015 from pre-training to prevent validation leakage.")

    # Replace NOAA/OMNI missing value placeholders before they pollute the model
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace(NOAA_FILL_VALS, np.nan)
    df[num_cols] = df[num_cols].interpolate(method='linear', limit_direction='both')
    df[num_cols] = df[num_cols].bfill().ffill()

    # ── Correct column name mapping (actual CSV columns vs 10-feature architecture) ──
    # Pre-training CSV uses: 'Bz_GSM', 'Flow_Speed', 'Flow_Pressure'
    # Architecture expects:  'BZ_GSM', 'Vsw',        'Pdyn'
    df['BZ_GSM'] = df.get('BZ_GSM', df.get('Bz_GSM', df.get('BZ', pd.Series(0.0, index=df.index))))
    df['Vsw']    = df.get('Vsw',    df.get('Flow_Speed',  pd.Series(400.0, index=df.index)))
    df['Pdyn']   = df.get('Pdyn',   df.get('Flow_Pressure', pd.Series(2.0, index=df.index)))

    # ── Physics-based proxies for features absent from pre-training CSV ──
    # BY_GSM: no data available — use 0 (genuinely unavailable, model will learn it
    #         matters only in fine-tuning where real values exist)
    df['BY_GSM'] = df.get('BY_GSM', pd.Series(0.0, index=df.index))

    # AE index proxy: Burton et al. (1975) — AE correlates with |Bz|*Vsw coupling
    # AE ≈ 300 * |Bz| * Vsw / 1000  (rough but physically motivated)
    bz_neg = np.minimum(df['BZ_GSM'].values, 0.0)          # only southward component drives AE
    ae_proxy = 300.0 * np.abs(bz_neg) * df['Vsw'].values / 1000.0
    df['AE'] = np.clip(ae_proxy, 0, 3000)

    # Dst proxy: O'Brien & McPherron (2000) simplified injection term
    # Dst ≈ -31 * sqrt(Pdyn) - 7.26 * Ey   where Ey = -Bz*Vsw/1000 (mV/m)
    ey = -df['BZ_GSM'].values * df['Vsw'].values / 1000.0
    dst_proxy = -31.0 * np.sqrt(np.maximum(df['Pdyn'].values, 0.1)) - 7.26 * np.maximum(ey, 0)
    df['DST'] = np.clip(dst_proxy, -500, 50)

    # F10.7 index proxy: varies with Solar Cycle 24/25.
    # We use year-based interpolation (SC24 min ~65 in 2019, SC25 rising to ~150 by 2024)
    year_frac = df.index.year + (df.index.dayofyear / 365.0)
    # SC24 peaked ~2014 (F10.7~150), dipped ~2019 (F10.7~70), SC25 rising ~2024
    sc24_f107 = 70 + 80 * np.abs(np.sin(np.pi * (year_frac - 2019) / 11.0))
    df['F10.7_index'] = np.clip(sc24_f107, 65, 200)

    # Ensure all 10 features exist safely
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0

    # Calculate MLT using Equation of Time (GOES-16 baseline longitude of -75 degrees)
    mlt = calculate_mlt_vectorized(df.index, satellite_lon=-75.0)
    df['MLT_sin'] = np.sin(mlt * 2 * np.pi / 24)
    df['MLT_cos'] = np.cos(mlt * 2 * np.pi / 24)

    df.dropna(subset=FEATURE_COLS, inplace=True)
    print(f"[STAGE 1] Rows after cleaning: {len(df):,}")

    # Sanity check — print per-feature std to confirm no flat columns remain
    feature_stds = df[FEATURE_COLS].std().round(3)
    print(f"[STAGE 1] Per-feature std check: {feature_stds.to_dict()}")

    return df[FEATURE_COLS].values.astype(np.float32)

def prepare_finetune(path):
    """Prepares the 2017/2018 GOES-16+ATHA ULF fine-tuning dataset."""
    print(f"\n[STAGE 2] Loading fine-tuning data from {path}...")
    df = pd.read_csv(path, parse_dates=['datetime'], index_col='datetime')
    df.sort_index(inplace=True)

    # FIX 2: Replace NOAA missing value placeholders before they pollute the model
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace(NOAA_FILL_VALS, np.nan)
    df[num_cols] = df[num_cols].interpolate(method='linear', limit_direction='both')
    df[num_cols] = df[num_cols].bfill().ffill()

    df['log_electron_flux'] = np.log10(np.maximum(df.get('Electron_Flux', df.get('electron_flux', 1e-5)), 1e-5))

    # Map standard feature names for the strict 10-feature architecture
    df['BY_GSM'] = df.get('BY_GSM', df.get('BY', df.get('By_GSM', 0.0)))
    df['BZ_GSM'] = df.get('BZ_GSM', df.get('BZ', df.get('Bz_GSM', 0.0)))
    df['Vsw'] = df.get('Vsw', df.get('V', df.get('Flow_Speed', 400.0)))

    # Calculate Pdyn if missing (Density is in cm^-3, V is in km/s)
    default_pdyn = 0.5 * 1.67e-27 * (df.get('Density', 5.0)*1e6) * (df.get('V', 400.0)*1e3)**2 * 1e9
    df['Pdyn'] = df.get('Pdyn', df.get('Flow_Pressure', default_pdyn))

    df['F10.7_index'] = df.get('F10.7_index', df.get('F10.7', 70.0))
    df['AE'] = df.get('AE', 100.0)
    df['DST'] = df.get('DST', df.get('Dst', -10.0))

    # Ensure all 10 features exist safely
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0

    # Calculate MLT using Equation of Time (assuming GOES-16 baseline longitude of -75 degrees)
    mlt = calculate_mlt_vectorized(df.index, satellite_lon=-75.0)
    df['MLT_sin'] = np.sin(mlt * 2 * np.pi / 24)
    df['MLT_cos'] = np.cos(mlt * 2 * np.pi / 24)

    df.dropna(subset=FEATURE_COLS, inplace=True)
    print(f"[STAGE 2] Rows after cleaning: {len(df):,}")
    return df[FEATURE_COLS].values.astype(np.float32)

def prepare_benchmark(path):
    """Prepares modern benchmark datasets (May/Oct 2024) for true out-of-sample validation."""
    print(f"\n[STAGE 3] Loading validation data from {path}...")
    df = pd.read_csv(path, parse_dates=['datetime'], index_col='datetime')
    df.sort_index(inplace=True)

    # Replace NOAA missing value placeholders before they pollute the validation metrics
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace(NOAA_FILL_VALS, np.nan)
    df[num_cols] = df[num_cols].interpolate(method='linear', limit_direction='both')
    df[num_cols] = df[num_cols].bfill().ffill()

    # Standardize column names for the 10-feature architecture
    if 'log_electron_flux' not in df.columns:
        if 'log_flux' in df.columns:
            df['log_electron_flux'] = df['log_flux']
        elif 'electron_flux' in df.columns:
            df['log_electron_flux'] = np.log10(np.maximum(df['electron_flux'], 1e-5))
        elif 'flux' in df.columns:
            df['log_electron_flux'] = np.log10(np.maximum(df['flux'], 1e-5))

    df['BY_GSM'] = df.get('BY_GSM', df.get('By_gsm', df.get('BY', 0.0)))
    df['BZ_GSM'] = df.get('BZ_GSM', df.get('Bz_gsm', df.get('BZ', 0.0)))
    df['Vsw'] = df.get('Vsw', df.get('Flow_Speed', df.get('V', 400.0)))
    df['Pdyn'] = df.get('Pdyn', df.get('Flow_Pressure', 2.0))
    df['F10.7_index'] = df.get('F10.7_index', df.get('F10.7', 70.0))
    df['AE'] = df.get('AE', 100.0)
    df['DST'] = df.get('DST', df.get('Dst', -10.0))

    # Ensure all 10 features exist safely
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0

    # Calculate MLT using Equation of Time (assuming GOES-16 baseline longitude of -75 degrees)
    mlt = calculate_mlt_vectorized(df.index, satellite_lon=-75.0)
    df['MLT_sin'] = np.sin(mlt * 2 * np.pi / 24)
    df['MLT_cos'] = np.cos(mlt * 2 * np.pi / 24)

    df.dropna(subset=FEATURE_COLS, inplace=True)
    print(f"Loaded {len(df):,} benchmark rows.")
    return df[FEATURE_COLS].values.astype(np.float32)

# ─── Cell 5: Normalization (Global — fit on combined pre-train + fine-tune) ───
def compute_global_scaler(matrices):
    """Compute global mean/std from all training data combined for consistent normalization."""
    combined = np.vstack(matrices)
    mean = np.mean(combined, axis=0, keepdims=True)
    std  = np.std(combined,  axis=0, keepdims=True) + 1e-7
    # Keep log_flux (feature 0) in natural scale for loss computation
    mean[:, 0] = 0.0
    std[:, 0]  = 1.0
    print(f"\nGlobal scaler computed from {len(combined):,} total rows")
    print(f"Feature means: {mean.flatten().round(2)}")
    print(f"Feature stds:  {std.flatten().round(2)}")
    return mean, std

def normalize(data, mean, std):
    return (data - mean) / std

def make_sequences(data_norm, data_raw, seq_len=288, pred_len=144, stride=36):
    """Build (X, y) sequence pairs with stride-based sampling."""
    X, y = [], []
    for i in range(0, len(data_norm) - seq_len - pred_len, stride):
        X.append(data_norm[i : i + seq_len])
        y.append(data_raw[i + seq_len : i + seq_len + pred_len, 0])  # raw log_flux target
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# ─── Cell 6: Training Function ────────────────────────────────────────────────
def train_stage(model, X_tensor, y_tensor, epochs, lr, batch_size=32, label=""):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    criterion = PhysicsInformedPinballLoss()
    dataset   = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader    = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    print(f"\n[{label}] Starting training: {len(X_tensor):,} sequences, {epochs} epochs, lr={lr}")
    best_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        epoch_loss, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred, _ = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
            n += len(xb)
        scheduler.step()
        avg_loss = epoch_loss / n
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.5f} | LR: {scheduler.get_last_lr()[0]:.2e}")

    # Restore best checkpoint
    model.load_state_dict(best_state)
    print(f"[{label}] Best loss: {best_loss:.5f}")
    return model

def evaluate(model, X_tensor, y_tensor, label="VALIDATION"):
    """Compute RMSE and Linear Correlation on log-scale."""
    model.eval()
    with torch.no_grad():
        preds, _ = model(X_tensor.to(DEVICE))
        median   = preds[:, :, 2].cpu().numpy()  # P50
        truth    = y_tensor.numpy()
    # Flatten for global metric
    median_flat = median.flatten()
    truth_flat  = truth.flatten()
    rmse = np.sqrt(np.mean((median_flat - truth_flat) ** 2))
    # Linear Correlation coefficient
    corr = np.corrcoef(truth_flat, median_flat)[0, 1]
    print(f"\n[{label}] RMSE (log10 scale):   {rmse:.4f}")
    print(f"[{label}] Linear Correlation:     {corr:.4f}")
    print(f"[{label}] Approx flux error:      {(10**rmse):.1f}× factor")
    return rmse, corr

# ─── Cell 7: Main Pipeline ────────────────────────────────────────────────────
print("=" * 70)
print("KAVACH TFT — 3-Stage Training Pipeline (Strict Train/Val/Test Split)")
print("=" * 70)

# ──────────────────────────────────────────────────────────────────────────────
# STRICT 3-WAY DATA SPLIT — As per ISRO mentor directive:
#   TRAINING   : OMNI 11-year PreTraining + GSAT-19 2017/2018 Fine-Tuning
#   VALIDATION : May 2024 G5 Storm  ← used during training to monitor progress
#   TEST (BLIND): Oct 2024 G4 Storm  ← NEVER seen during training; final eval only
# ──────────────────────────────────────────────────────────────────────────────

# Load data
print("\nLoading datasets into memory...")
pretrain_raw = prepare_pretrain(PRETRAIN_PATH)
finetune_raw = prepare_finetune(FINETUNE_PATH)

# VALIDATION SET: May 2024 G5 Mother's Day Storm
# Used during Stage 1 & 2 to monitor when the model is learning correctly.
val_raw = prepare_benchmark(VAL_MAY_PATH)
print(f"[VALIDATION SET] May 2024 G5 Storm rows: {len(val_raw):,}")

# BLIND TEST SET: October 2024 G4 Aurora Storm
# Completely locked away. Zero interaction during training.
# Only evaluated ONCE after training is fully complete.
test_raw = prepare_benchmark(VAL_OCT_PATH)
print(f"[BLIND TEST SET] Oct 2024 G4 Storm rows: {len(test_raw):,}")
print("[BLIND TEST SET] Oct 2024 data is LOCKED — will not be seen until Stage 3 final evaluation.")

# Compute global normalizer ONLY from training data
# (Scaler must never see validation or test data)
mean, std     = compute_global_scaler([pretrain_raw, finetune_raw])
pretrain_norm = normalize(pretrain_raw, mean, std)
finetune_norm = normalize(finetune_raw, mean, std)
val_norm      = normalize(val_raw,  mean, std)   # Apply same scaler to val
test_norm     = normalize(test_raw, mean, std)   # Apply same scaler to test

# Build sequences
print("\nBuilding training sequences...")
X_pre,  y_pre  = make_sequences(pretrain_norm, pretrain_raw, stride=72)   # stride=6h
X_fine, y_fine = make_sequences(finetune_norm, finetune_raw, stride=36)   # stride=3h
X_val,  y_val  = make_sequences(val_norm,      val_raw,      stride=36)   # Validation (May 2024)
X_test, y_test = make_sequences(test_norm,     test_raw,     stride=36)   # Blind Test (Oct 2024)

print(f"Pre-train sequences:  {len(X_pre):,}")
print(f"Fine-tune sequences:  {len(X_fine):,}")
print(f"Validation sequences: {len(X_val):,}  (May 2024 — seen during training)")
print(f"Blind Test sequences: {len(X_test):,}  (Oct 2024 — LOCKED until Stage 3)")

X_pre_t  = torch.tensor(X_pre,  dtype=torch.float32)
y_pre_t  = torch.tensor(y_pre,  dtype=torch.float32)
X_fine_t = torch.tensor(X_fine, dtype=torch.float32)
y_fine_t = torch.tensor(y_fine, dtype=torch.float32)
X_val_t  = torch.tensor(X_val,  dtype=torch.float32)
y_val_t  = torch.tensor(y_val,  dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

# ─── STAGE 1: Pre-train on 11-year OMNI ──────────────────────────────────────
print("\n" + "="*70)
print("STAGE 1: PRE-TRAINING on 11-Year OMNI Dataset")
print("="*70)
model = KAVACH_TFT(num_features=10, hidden_size=128, lstm_layers=2, num_quantiles=5).to(DEVICE)
model = train_stage(model, X_pre_t, y_pre_t, epochs=15, lr=3e-4, batch_size=64, label="STAGE-1 PRE-TRAIN")
# Monitor progress on MAY 2024 validation set
evaluate(model, X_val_t, y_val_t, label="STAGE-1 VAL (May 2024)")

# ─── STAGE 2: Fine-tune on 2017/2018 GSAT-19 data (transfer learning) ─────────
print("\n" + "="*70)
print("STAGE 2: FINE-TUNING on 2017/2018 GSAT-19 GRASP Dataset")
print("="*70)
# Lower LR to preserve pre-trained physics knowledge
model = train_stage(model, X_fine_t, y_fine_t, epochs=20, lr=5e-5, batch_size=32, label="STAGE-2 FINE-TUNE")
# Monitor progress on MAY 2024 validation set
evaluate(model, X_val_t, y_val_t, label="STAGE-2 VAL (May 2024)")

# ─── STAGE 3: FINAL BLIND TEST — October 2024 G4 Aurora Storm ────────────────
# This is the first and ONLY time the model sees the October 2024 storm.
# These metrics represent true out-of-sample operational performance.
print("\n" + "="*70)
print("STAGE 3: FINAL BLIND TEST — October 2024 G4 Aurora Storm")
print("(Model has NEVER seen this data — true zero-shot evaluation)")
print("="*70)
evaluate(model, X_test_t, y_test_t, label="STAGE-3 BLIND TEST (Oct 2024)")

# ─── Cell 8: Save Weights & Scaler ───────────────────────────────────────────
os.makedirs("/kaggle/working/weights", exist_ok=True)
WEIGHTS_PATH = "/kaggle/working/weights/finetuned_gsat19_grasp_ulf.pth"
SCALER_PATH  = "/kaggle/working/weights/scaler.pkl"

torch.save(model.state_dict(), WEIGHTS_PATH)
joblib.dump({'mean': mean, 'std': std}, SCALER_PATH)
print(f"\nWeights saved: {WEIGHTS_PATH}")
print(f"Scaler saved : {SCALER_PATH}")

# ─── Cell 9: Upload to Hugging Face ──────────────────────────────────────────
HF_TOKEN   = os.environ.get("HF_TOKEN", "")
HF_REPO    = "Supriyo760/kavach-weights"

if HF_TOKEN:
    print(f"\n[HF UPLOAD] Pushing to {HF_REPO}...")
    api = HfApi()
    for local_path, repo_path in [
        (WEIGHTS_PATH, "finetuned_gsat19_grasp_ulf.pth"),
        (SCALER_PATH,  "scaler.pkl"),
    ]:
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=HF_REPO,
            repo_type="model",
            token=HF_TOKEN,
        )
        print(f"  Uploaded: {repo_path}")
    print("[HF UPLOAD] Done! Your Streamlit dashboard will auto-load the new weights.")
else:
    print("\n[HF UPLOAD] HF_TOKEN not set — weights saved locally only.")
    print("           Add your HF token as a Kaggle Secret named 'HF_TOKEN' and re-run.")

print("\n" + "="*70)
print("KAVACH TRAINING PIPELINE COMPLETE!")
print("="*70)
