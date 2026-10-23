"""
Transfer Learning & Fine-Tuning Script
KAVACH — GEO Radiation Monitor | Team DigiIndia
"""
import os
import torch
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from kavach.models.tft_model import build_tft, PhysicsInformedPinballLoss

import argparse

def train_finetune(data_path=r"DataSets\Kaggle_FineTuning_Dataset.csv"):
    print("[KAVACH-FINETUNE] Initiating Transfer Learning Pipeline...")
    if not os.path.exists(data_path):
        print(f"Error: Dataset {data_path} not found.")
        return
        
    df = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
    df.sort_index(inplace=True)
    
    # 1. Engineer exact features to match the 11-year base + ULF_Power
    print("[KAVACH-FINETUNE] Feature Engineering...")
    df['log_electron_flux'] = np.log10(np.maximum(df['Electron_Flux'], 1e-5))
    df['Flow_Speed'] = df['V']
    df['Bz_GSM'] = df['BZ']
    df['Proton_Density'] = df['Density']
    df['Temperature'] = 100000.0 # Placeholder, missing in GRASP dataset
    df['Flow_Pressure'] = 0.5 * 1.67e-27 * (df['Density']*1e6) * (df['V']*1e3)**2 * 1e9
    
    # df['ULF_Power'] = -3.5 # (Proxy removed, data now contains true ULF power)

    
    df['log_flux_t-1h'] = df['log_electron_flux'].shift(12)
    df['log_flux_t-3h'] = df['log_electron_flux'].shift(36)
    df['log_flux_t-24h'] = df['log_electron_flux'].shift(288)
    
    df.dropna(inplace=True)
    
    feature_cols = [
        "log_electron_flux", "Flow_Speed", "Bz_GSM", "Proton_Density", "Temperature", "Flow_Pressure",
        "log_flux_t-1h", "log_flux_t-3h", "log_flux_t-24h", "ULF_Power"
    ]
    
    data_matrix = df[feature_cols].values.astype(np.float32)
    
    # 2. Scale features (excluding target at index 0)
    mean = np.mean(data_matrix, axis=0, keepdims=True)
    std = np.std(data_matrix, axis=0, keepdims=True) + 1e-7
    mean[:, 0] = 0.0
    std[:, 0] = 1.0
    norm_data = (data_matrix - mean) / std
    
    seq_len, pred_len = 288, 144
    X_samples, y_samples = [], []
    for i in range(0, len(norm_data) - seq_len - pred_len, 36): # Stride = 3 hours
        X_samples.append(norm_data[i : i + seq_len])
        y_samples.append(data_matrix[i + seq_len : i + seq_len + pred_len, 0])
        
    X_tensor = torch.tensor(np.array(X_samples), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y_samples), dtype=torch.float32)
    
    # 3. Build 10-feature model and inject 9-feature 11-year weights
    print("[KAVACH-FINETUNE] Instantiating 10-feature TFT model...")
    model = build_tft(num_features=10, num_quantiles=5)
    
    base_weights_path = r"kavach\weights\tft_model_11yr.pth"
    if os.path.exists(base_weights_path):
        print("[KAVACH-FINETUNE] Intercepting 11-year weights for Transfer Learning...")
        state_dict = torch.load(base_weights_path, map_location='cpu')
        
        # Filter out vsn_weights due to shape mismatch (9x9 vs 10x10)
        state_dict = {k: v for k, v in state_dict.items() if 'vsn_weights' not in k}
        
        # Load the compatible layers (strict=False ignores the missing vsn.9 and vsn_weights)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[KAVACH-FINETUNE] Loaded shared weights. Randomly initializing: {missing}")
    else:
        print("[KAVACH-FINETUNE] Warning: Base 11-year weights not found. Training from scratch.")

    # 4. Rapid Fine-Tuning Pass
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = PhysicsInformedPinballLoss(quantiles=[0.1, 0.25, 0.5, 0.75, 0.9], lambda_physics=0.15)
    
    epochs, batch_size = 3, 16
    n_samples = len(X_tensor)
    model.train()
    
    print(f"[KAVACH-FINETUNE] Commencing fine-tuning over {n_samples} sequences...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        indices = torch.randperm(n_samples)
        for b in range(0, n_samples, batch_size):
            batch_idx = indices[b : b + batch_size]
            optimizer.zero_grad()
            q_preds, _ = model(X_tensor[batch_idx])
            loss = criterion(q_preds, y_tensor[batch_idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(batch_idx)
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss / n_samples:.4f}")
        
    os.makedirs("kavach/weights", exist_ok=True)
    save_path = "kavach/weights/finetuned_gsat19_grasp_ulf.pth"
    torch.save(model.state_dict(), save_path)
    joblib.dump({'mean': mean, 'std': std}, "kavach/weights/scaler.pkl")
    print(f"[KAVACH-FINETUNE] Successfully deployed fine-tuned weights to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune KAVACH TFT on 2017 Dataset")
    parser.add_argument("--data", type=str, default=r"DataSets\Kaggle_FineTuning_Dataset.csv", help="Input dataset path")
    args = parser.parse_args()
    
    train_finetune(args.data)
