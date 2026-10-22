"""
11-Year Deep Learning Training Engine for Temporal Fusion Transformer (TFT)
RADSHIELD — GEO Radiation Monitor

Loads a caller-supplied dataset, constructs multi-horizon sliding windows, and trains
the PyTorch TFT model. It does not establish the input data's source or provenance.
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from radshield.models.tft_model import build_tft, PhysicsInformedPinballLoss
from radshield.models.radial_diff import run_physics_forecast

class ElevenYearSolarCycleDataset(Dataset):
    """
    Sliding-window dataset for a caller-supplied time series.
    Input sequence : 7 days (2016 timesteps @ 5-min cadence)
    Target sequence: 12 hours (144 timesteps @ 5-min cadence)
    """
    def __init__(self, df: pd.DataFrame, seq_len: int = 2016, pred_len: int = 144, stride: int = 288):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.stride = stride
        
        # Mentor-approved strict 10-feature architecture
        feature_cols = [
            "log_electron_flux", "BY_GSM", "BZ_GSM", "Pdyn", "Vsw", 
            "AE", "DST", "F10.7_index", "MLT_sin", "MLT_cos"
        ]
        
        # Ensure MLT columns exist in the dataset (Calculate if missing)
        if "MLT_sin" not in df.columns:
            # Assume GOES-16 baseline (-75W) for the 11-year archive
            lon_offset = -75 / 15.0
            mlt = (df.index.hour + df.index.minute / 60.0 + lon_offset) % 24
            df['MLT_sin'] = np.sin(mlt * 2 * np.pi / 24)
            df['MLT_cos'] = np.cos(mlt * 2 * np.pi / 24)
            
        available_cols = [c for c in feature_cols if c in df.columns]
        data_matrix = df[available_cols].values.astype(np.float32)
        
        # Standardize features (mean=0, std=1) except log_flux (target index 0)
        self.mean = np.mean(data_matrix, axis=0, keepdims=True)
        self.std = np.std(data_matrix, axis=0, keepdims=True) + 1e-7
        self.mean[:, 0] = 0.0 # Keep log_flux in natural log10 scale
        self.std[:, 0] = 1.0
        
        self.norm_data = (data_matrix - self.mean) / self.std
        self.n_samples = max(0, (len(self.norm_data) - self.seq_len - self.pred_len) // self.stride + 1)
        
    def __len__(self):
        return self.n_samples
        
    def __getitem__(self, idx):
        start = idx * self.stride
        mid = start + self.seq_len
        end = mid + self.pred_len
        
        x = self.norm_data[start:mid]
        y = self.norm_data[mid:end, 0] # target is future log_flux
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def train_11yr_model(data_path: str = "radshield/data/archive_11yr_goes_grasp.csv", epochs: int = 3, batch_size: int = 64, lr: float = 1e-3, weights_out: str = "radshield/weights/tft_model_11yr.pth"):
    """
    Train on an explicitly supplied dataset; no data or checkpoint is synthesized.
    """
    print(f"[RADSHIELD-TRAIN] ==========================================================")
    print(f"[RADSHIELD-TRAIN] INITIATING 11-YEAR PyTorch TFT MODEL TRAINING ENGINE")
    print(f"[RADSHIELD-TRAIN] Input Dataset : {data_path}")
    print(f"[RADSHIELD-TRAIN] Epochs        : {epochs} | Batch Size: {batch_size} | LR: {lr}")
    print(f"[RADSHIELD-TRAIN] Output Target : {weights_out}")
    print(f"[RADSHIELD-TRAIN] ==========================================================")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Training input not found: {data_path}. Supply a provenance-documented "
            "measured dataset; synthetic bootstrap data is not used as a fallback."
        )

    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required to train a model.")

    print(f"[RADSHIELD-TRAIN] Loading 11-year dataset into memory...")
    t0 = time.time()
    df = pd.read_csv(data_path)
    if "data_origin" in df and df["data_origin"].eq("synthetic_space_weather_bootstrap").any():
        raise ValueError("Synthetic bootstrap data is not accepted as a measured training dataset.")
    
    # Clean NOAA/OMNI missing data fill values before training
    missing_vals = [99999.9, 99999.0, 9999.99, 9999.0, 999.99, 999.9, 999.0, 999, 99.99, 99.0, 99]
    df.replace(missing_vals, np.nan, inplace=True)
    # Only interpolate numeric columns to avoid crash on datetime/string columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].interpolate(method='linear', limit_direction='both')
    df[num_cols] = df[num_cols].bfill().ffill()
    
    print(f"[RADSHIELD-TRAIN] Loaded and cleaned {len(df):,} observations in {time.time() - t0:.2f} seconds.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[RADSHIELD-TRAIN] Compute Device: {device.type.upper()}")
    
    # Train / Val split (80% train = ~8.8 years, 20% val = ~2.2 years out-of-sample)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    val_df = df.iloc[split_idx:].reset_index(drop=True)
    
    train_dataset = ElevenYearSolarCycleDataset(train_df, seq_len=288*7, pred_len=144, stride=288)
    val_dataset = ElevenYearSolarCycleDataset(val_df, seq_len=288*7, pred_len=144, stride=288)
    
    print(f"[RADSHIELD-TRAIN] Train Windows : {len(train_dataset):,} sequences (~8.8 Solar Years)")
    print(f"[RADSHIELD-TRAIN] Val Windows   : {len(val_dataset):,} sequences (~2.2 Solar Years)")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    num_dynamic_features = train_dataset.norm_data.shape[1]
    model = build_tft(num_features=num_dynamic_features, num_quantiles=5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = PhysicsInformedPinballLoss(quantiles=[0.1, 0.25, 0.5, 0.75, 0.9], lambda_physics=0.15)
    
    print(f"[RADSHIELD-TRAIN] Starting Multi-Epoch Optimization Loop...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds, _ = model(x)
            
            loss = criterion(preds, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            
        train_loss /= len(train_dataset)
        scheduler.step()
        
        # Validation evaluation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:  # FIX: was broken as enumerate(val_loader)
                x_val, y_val = x_val.to(device), y_val.to(device)
                preds, _ = model(x_val)
                loss = criterion(preds, y_val)
                val_loss += loss.item() * x_val.size(0)
        val_loss /= max(1, len(val_dataset))
        
        print(f"[RADSHIELD-TRAIN] Epoch [{epoch:02d}/{epochs:02d}] | Train PINN Loss: {train_loss:.4f} | Val PINN Loss: {val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        
    os.makedirs(os.path.dirname(weights_out), exist_ok=True)
    torch.save(model.state_dict(), weights_out)
    print(f"[RADSHIELD-TRAIN] ==========================================================")
    print(f"[RADSHIELD-TRAIN] 11-YEAR TRAINING COMPLETE — WEIGHTS SAVED")
    print(f"[RADSHIELD-TRAIN] Final Model Checkpoint : {weights_out}")
    print(f"[RADSHIELD-TRAIN] Out-of-Sample Val Loss : {val_loss:.4f}")
    print(f"[RADSHIELD-TRAIN] ==========================================================")
    return weights_out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RADSHIELD TFT on 11-Year Archive")
    parser.add_argument("--data", type=str, default="radshield/data/archive_11yr_goes_grasp.csv", help="Input dataset path")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    args = parser.parse_args()
    
    train_11yr_model(args.data, args.epochs, args.batch_size)
