"""
Synthetic Fine-Tuning Demonstration
RADSHIELD — GEO Radiation Monitor

Uses generated sample data only. This is not a GSAT-19 GRASP training pipeline and
its checkpoint must not be treated as a measured-data model.
"""
import os
import torch
import numpy as np
import pandas as pd

from radshield.training.train_tft import load_scaler, save_scaler
from radshield.models.tft_model import build_tft, PinballLoss
from radshield.data.sample_data import generate_synthetic_dataset

def freeze_for_finetune(model):
    """Freezes early layers, leaving last LSTM + quantile head trainable for transfer learning."""
    for name, param in model.named_parameters():
        if 'quantile_head' in name or 'lstm' in name or 'grn_dec' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

def run_synthetic_finetuning_demo(
    weights_dir: str = 'artifacts/synthetic_finetune_demo',
    epochs: int = 3,
):
    """
    Exercise the fine-tuning code using generated data for demonstration only.
    """
    model_path = os.path.join(weights_dir, 'radshield_tft_v1.pt')
    scaler_path = os.path.join(weights_dir, 'scaler.pkl')

    if not os.path.exists(model_path):
        print("Demo baseline not found. Generating synthetic demo weights...")
        from radshield.training.train_tft import train_radshield_model
        model, scaler = train_radshield_model(epochs=2, save_dir=weights_dir)
    else:
        model = build_tft()
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        scaler = load_scaler(scaler_path)

    # Freeze early layers
    freeze_for_finetune(model)

    # Generated samples are only for exercising the training code.
    demo_data = generate_synthetic_dataset(start_date='2023-01-01', days=5)

    feature_cols = [
        'log_flux', 'flux_lag_1h', 'flux_lag_3h', 'flux_lag_6h', 'flux_lag_12h', 'flux_lag_24h',
        'Vsw', 'BZ_GSM', 'BY_GSM', 'Np', 'Pdyn', 'Ec', 'DST', 'dDst_dt', 'KP', 'AE_1h',
        'ULF_power', 'Bz_neg_dur', 'regime'
    ]

    scaled_demo = scaler.transform(demo_data[feature_cols].values)

    seq_len = 288
    pred_len = 144

    X_list, y_list = [], []
    for i in range(len(scaled_demo) - seq_len - pred_len):
        X_list.append(scaled_demo[i : i + seq_len])
        y_list.append(demo_data['log_flux'].values[i + seq_len : i + seq_len + pred_len])

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

    print("=== Synthetic Fine-Tuning Demo Results ===")
    print(f"RMSE Before Fine-Tuning: {rmse_before:.4f}")
    print(f"RMSE After Fine-Tuning:  {rmse_after:.4f}")
    print(f"Domain Transfer Improvement: +{improvement_pct:.2f}%")

    ft_model_path = os.path.join(weights_dir, 'radshield_synthetic_demo.pt')
    torch.save(model.state_dict(), ft_model_path)
    return rmse_before, rmse_after, improvement_pct

if __name__ == '__main__':
    run_synthetic_finetuning_demo()
