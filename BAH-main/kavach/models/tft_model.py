"""
Temporal Fusion Transformer (TFT) Deep Learning Forecasting Engine
KAVACH — GEO Radiation Monitor | Team DigiIndia
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN) component of Temporal Fusion Transformer."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
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
    """
    Custom PyTorch Temporal Fusion Transformer for multi-horizon quantile electron flux forecasting.
    Input: 19 physical features over 288 steps (24 hours).
    Output: Quantile predictions [0.10, 0.25, 0.50, 0.75, 0.90] over 144 steps (12 hours ahead).
    """
    def __init__(self, num_features: int = 19, hidden_size: int = 128, lstm_layers: int = 2, num_quantiles: int = 5, dropout: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_quantiles = num_quantiles
        self.quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]

        # Feature selection / Variable Selection Network (VSN)
        self.vsn = nn.ModuleList([GatedResidualNetwork(1, hidden_size, hidden_size, dropout) for _ in range(num_features)])
        self.vsn_weights = nn.Linear(num_features, num_features)

        # LSTM Encoder
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=lstm_layers, batch_first=True, dropout=dropout if lstm_layers > 1 else 0.0)

        # Multi-Head Attention Mechanism
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4, batch_first=True)

        # Decoder & Output Quantile Heads
        self.grn_dec = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)
        self.quantile_head = nn.Linear(hidden_size, 144 * num_quantiles)

    def forward(self, x):
        # x shape: [batch_size, seq_len=288, num_features=19]
        b, seq_len, n_feat = x.shape
        
        # Variable Selection
        vsn_outputs = []
        for i in range(min(n_feat, self.num_features)):
            feat_i = x[:, :, i:i+1]  # [b, seq_len, 1]
            vsn_outputs.append(self.vsn[i](feat_i)) # [b, seq_len, hidden_size]

        vsn_stack = torch.stack(vsn_outputs, dim=-1) # [b, seq_len, hidden_size, n_feat]
        attn_scores = torch.softmax(self.vsn_weights(x.mean(dim=1)), dim=-1).unsqueeze(1).unsqueeze(2)
        vsn_fused = (vsn_stack * attn_scores).sum(dim=-1) # [b, seq_len, hidden_size]

        # LSTM Encoding
        lstm_out, _ = self.lstm(vsn_fused)

        # Multi-Head Self Attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Gated Post-Processing
        dec_out = self.grn_dec(attn_out[:, -1, :]) # Take last encoder step

        # Quantile Output Projection -> [b, 144, 5]
        q_out = self.quantile_head(dec_out).view(b, 144, self.num_quantiles)
        
        return q_out, attn_weights

class PhysicsInformedPinballLoss(nn.Module):
    """
    Physics-Informed Quantile Regression Pinball Loss (PINN).
    Combines Pinball Loss across [0.10, 0.25, 0.50, 0.75, 0.90] quantiles with 
    Fokker-Planck 1D Radial Diffusion temporal continuity regularization.
    """
    def __init__(self, quantiles=[0.10, 0.25, 0.50, 0.75, 0.90], lambda_physics: float = 0.05):
        super().__init__()
        self.quantiles = quantiles
        self.lambda_physics = lambda_physics

    def forward(self, pred, target):
        # 1. Standard Pinball Loss
        pinball_loss = 0.0
        for i, q in enumerate(self.quantiles):
            errors = target - pred[:, :, i]
            pinball_loss += torch.max((q - 1) * errors, q * errors).mean()
        
        # 2. Physics Regularization: Penalize 1st and 2nd temporal derivative discontinuities (Fokker-Planck ODE constraint)
        median_pred = pred[:, :, 2] # P50 prediction
        d1 = median_pred[:, 1:] - median_pred[:, :-1]
        d2 = d1[:, 1:] - d1[:, :-1]
        physics_loss = torch.mean(d2 ** 2)
        
        return pinball_loss + self.lambda_physics * physics_loss

# Alias for backwards compatibility
PinballLoss = PhysicsInformedPinballLoss

def build_tft(num_features: int = 19, hidden_size: int = 128, lstm_layers: int = 2, num_quantiles: int = 5, dropout: float = 0.1):
    """Factory function for initializing TFT model."""
    return KAVACH_TFT(num_features=num_features, hidden_size=hidden_size, lstm_layers=lstm_layers, num_quantiles=num_quantiles, dropout=dropout)

