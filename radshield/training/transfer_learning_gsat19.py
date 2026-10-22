"""
Transfer Learning & Cross-Satellite Domain Adaptation Engine (NOAA GOES -> GSAT-19 GRASP)
RADSHIELD — GEO Radiation Monitor

This module implements a 2-stage transfer learning pipeline:
  Stage 1: Load a caller-supplied base checkpoint.
  Stage 2: Fine-tune on a caller-supplied GSAT-19 GRASP dataset.
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
from radshield.training.train_11yr_model import ElevenYearSolarCycleDataset

def run_transfer_learning_pipeline(base_weights_path: str, grasp_2yr_path: str, epochs_finetune: int = 5, lr_finetune: float = 1e-4, weights_out: str = "radshield/weights/finetuned_gsat19_grasp.pth"):
    """
    Executes 2-stage transfer learning:
    1. Loads a caller-supplied base checkpoint.
    2. Freezes lower feature-selection layers.
    3. Fine-tunes attention & output heads on a caller-supplied dataset.
    """
    print(f"[RADSHIELD-TRANSFER] ==========================================================")
    print(f"[RADSHIELD-TRANSFER] INITIATING CROSS-SATELLITE TRANSFER LEARNING PIPELINE")
    print(f"[RADSHIELD-TRANSFER] Base Checkpoint : {base_weights_path}")
    print(f"[RADSHIELD-TRANSFER] Fine-Tune Input : {grasp_2yr_path}")
    print(f"[RADSHIELD-TRANSFER] Output Fine-Tuned Model : {weights_out}")
    print(f"[RADSHIELD-TRANSFER] ==========================================================")
    
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required to run transfer learning.")

    if not os.path.isfile(base_weights_path):
        raise FileNotFoundError(f"Base checkpoint not found: {base_weights_path}")
    if not os.path.isfile(grasp_2yr_path):
        raise FileNotFoundError(f"Fine-tuning dataset not found: {grasp_2yr_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[RADSHIELD-TRANSFER] Compute Device: {device.type.upper()}")
    
    # Instantiate Base TFT Model
    model = build_tft(num_features=15, num_quantiles=5).to(device)
    print(f"[RADSHIELD-TRANSFER] Loading base checkpoint from {base_weights_path}...")
    model.load_state_dict(torch.load(base_weights_path, map_location=device))
            
    # Freeze lower feature extractor (Variable Selection Network) to preserve 11-year solar physics knowledge
    print(f"[RADSHIELD-TRANSFER] Freezing VSN feature extraction layers for domain adaptation...")
    for param in model.vsn.parameters():
        param.requires_grad = False
        
    # Load 1-2 Year GSAT-19 GRASP Dataset
    df_grasp = pd.read_csv(grasp_2yr_path)
    if "data_origin" in df_grasp and df_grasp["data_origin"].eq("synthetic_space_weather_bootstrap").any():
        raise ValueError("Synthetic bootstrap data is not accepted as GRASP training data.")
    print(f"[RADSHIELD-TRANSFER] Loaded supplied fine-tuning data: {len(df_grasp):,} rows.")
    
    grasp_dataset = ElevenYearSolarCycleDataset(df_grasp, seq_len=288*7, pred_len=144, stride=288)
    grasp_loader = DataLoader(grasp_dataset, batch_size=64, shuffle=True)
    
    # Fine-tuning optimizer with lower learning rate
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_finetune, weight_decay=1e-4)
    criterion = PhysicsInformedPinballLoss(quantiles=[0.1, 0.25, 0.5, 0.75, 0.9], lambda_physics=0.15)
    
    print(f"[RADSHIELD-TRANSFER] Starting Stage-2 Fine-Tuning Loop on GSAT-19 GRASP Data...")
    for epoch in range(1, epochs_finetune + 1):
        model.train()
        epoch_loss = 0.0
        for batch_idx, (x, y) in enumerate(grasp_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds, _ = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)
            
        epoch_loss /= len(grasp_dataset)
        print(f"[RADSHIELD-TRANSFER] Fine-Tune Epoch [{epoch:02d}/{epochs_finetune:02d}] | GRASP PINN Loss: {epoch_loss:.4f}")
        
    os.makedirs(os.path.dirname(weights_out), exist_ok=True)
    torch.save(model.state_dict(), weights_out)
    print(f"[RADSHIELD-TRANSFER] ==========================================================")
    print(f"[RADSHIELD-TRANSFER] CROSS-SATELLITE TRANSFER LEARNING COMPLETE!")
    print(f"[RADSHIELD-TRANSFER] Fine-tuning completed for the supplied dataset.")
    print(f"[RADSHIELD-TRANSFER] Output Checkpoint Saved: {weights_out}")
    print(f"[RADSHIELD-TRANSFER] ==========================================================")
    return weights_out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a supplied RadShield checkpoint and dataset")
    parser.add_argument("--base-weights", type=str, default="radshield/weights/tft_model_11yr.pth", help="Path to base model checkpoint")
    parser.add_argument("--grasp", type=str, default="radshield/data/grasp_2yr.csv", help="Path to a verified GRASP dataset")
    parser.add_argument("--epochs", type=int, default=3, help="Fine-tuning epochs")
    args = parser.parse_args()
    
    run_transfer_learning_pipeline(args.base_weights, args.grasp, epochs_finetune=args.epochs)
