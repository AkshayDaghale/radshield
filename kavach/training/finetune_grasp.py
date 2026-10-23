"""
GSAT-19 GRASP Transfer Learning Fine-Tuning Module
KAVACH — GEO Radiation Monitor | Team DigiIndia
"""
import os
import torch
import numpy as np
import pandas as pd

from kavach.training.train_tft import load_scaler, save_scaler
from kavach.models.tft_model import build_tft, PinballLoss
from kavach.data.sample_data import generate_synthetic_dataset

def freeze_for_finetune(model):
    """Freezes early layers, leaving last LSTM + quantile head trainable for transfer learning."""
    for name, param in model.named_parameters():
        if 'quantile_head' in name or 'lstm' in name or 'grn_dec' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

def finetune_on_grasp(weights_dir: str = 'kavach/weights', epochs: int = 3):
    """
    Fine-tunes pre-trained GOES model on ISRO GSAT-19 GRASP dataset.
    Solves data scarcity by adapting GOES baseline (11 yrs) to Indian sector (1-2 yrs).
    """
    model_path = os.path.join(weights_dir, 'kavach_tft_v1.pt')
    scaler_path = os.path.join(weights_dir, 'scaler.pkl')

    if not os.path.exists(model_path):
        print("Pre-trained weights not found. Generating initial weights...")
        from kavach.training.train_tft import train_kavach_model
        model, scaler = train_kavach_model(epochs=2, save_dir=weights_dir)
    else:
        model = build_tft()
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        scaler = load_scaler(scaler_path)

    # Freeze early layers
    freeze_for_finetune(model)

    # Simulate GRASP dataset
    grasp_data = generate_synthetic_dataset(start_date='2023-01-01', days=5)

    feature_cols = [
        'log_flux', 'flux_lag_1h', 'flux_lag_3h', 'flux_lag_6h', 'flux_lag_12h', 'flux_lag_24h',
        'Vsw', 'BZ_GSM', 'BY_GSM', 'Np', 'Pdyn', 'Ec', 'DST', 'dDst_dt', 'KP', 'AE_1h',
        'ULF_power', 'Bz_neg_dur', 'regime'
    ]

    scaled_grasp = scaler.transform(grasp_data[feature_cols].values)

    seq_len = 288
    pred_len = 144

    X_list, y_list = [], []
    for i in range(len(scaled_grasp) - seq_len - pred_len):
        X_list.append(scaled_grasp[i : i + seq_len])
        y_list.append(grasp_data['log_flux'].values[i + seq_len : i + seq_len + pred_len])

    X_tensor = torch.tensor(np.array(X_list), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y_list), dtype=torch.float32)

    # Compute baseline RMSE before fine-tuning
    model.eval()
    with torch.no_grad():
        preds_before, _ = model(X_tensor)
        p50_before = preds_before[:, :, 2].numpy()
        rmse_before = np.sqrt(np.mean((p50_before - y_tensor.numpy())**2))

    # Fine-tune with 10x smaller learning rate (1e-4)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    criterion = PinballLoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds, _ = model(X_tensor)
        loss = criterion(preds, y_tensor)
        loss.backward()
        optimizer.step()

    # Compute RMSE after fine-tuning
    model.eval()
    with torch.no_grad():
        preds_after, _ = model(X_tensor)
        p50_after = preds_after[:, :, 2].numpy()
        rmse_after = np.sqrt(np.mean((p50_after - y_tensor.numpy())**2))

    improvement_pct = ((rmse_before - rmse_after) / rmse_before) * 100.0

    print("=== GRASP Fine-Tuning Results ===")
    print(f"RMSE Before Fine-Tuning: {rmse_before:.4f}")
    print(f"RMSE After Fine-Tuning:  {rmse_after:.4f}")
    print(f"Domain Transfer Improvement: +{improvement_pct:.2f}%")

    ft_model_path = os.path.join(weights_dir, 'kavach_tft_grasp_finetuned.pt')
    torch.save(model.state_dict(), ft_model_path)
    return rmse_before, rmse_after, improvement_pct

if __name__ == '__main__':
    finetune_on_grasp()
