"""PyTorch models, trained from scratch (no pretrained weights anywhere).

Imported only by training / export scripts. Production uses the exported ONNX files with onnxruntime.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from app.services.forecast_core import N_FEATURES
from app.services.normalize import MAX_LEN, VOCAB_SIZE


class CharCNN(nn.Module):
    """Char-level CNN for receipt-line classification (guide 7.2).

    ids [B, 64] -> Embedding(vocab, 64) -> Conv1d k=3,4,5 (128 channels) + ReLU + global max-pool
    -> concat [B, 384] -> Dropout -> Linear(384, 256) -> ReLU -> Dropout -> Linear(256, n_classes)
    """

    def __init__(self, n_classes: int, vocab_size: int = VOCAB_SIZE, emb_dim: int = 64, channels: int = 128,
                 kernels=(3, 4, 5), hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, channels, k, padding=k // 2) for k in kernels])
        self.drop1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(channels * len(kernels), hidden)
        self.drop2 = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, n_classes)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        e = self.embedding(ids).transpose(1, 2)                      # [B, emb, L]
        pooled = [torch.relu(conv(e)).max(dim=2).values for conv in self.convs]
        h = self.drop1(torch.cat(pooled, dim=1))
        h = self.drop2(torch.relu(self.fc1(h)))
        return self.out(h)


class ForecastLSTM(nn.Module):
    """LSTM(12 -> 64, 2 layers, dropout 0.2) -> last step -> concat(remaining days / 31) -> 65 -> 32 -> 1.

    Output is log1p(remaining spending / scale), see app/services/forecast_core.py.
    """

    def __init__(self, n_features: int = N_FEATURES, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, dropout=dropout, batch_first=True)
        self.fc1 = nn.Linear(hidden + 1, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor, rem: torch.Tensor) -> torch.Tensor:
        seq, _ = self.lstm(x)                                        # [B, T, hidden]
        z = torch.cat([seq[:, -1, :], rem], dim=1)                   # [B, hidden + 1]
        return self.fc2(torch.relu(self.fc1(z)))                     # [B, 1]


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


__all__ = ["CharCNN", "ForecastLSTM", "count_params", "MAX_LEN"]
