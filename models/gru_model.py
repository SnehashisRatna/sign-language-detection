# models/gru_model.py
# models/gru_model.py

import torch
import torch.nn as nn


class GRUClassifier(nn.Module):
    def __init__(self, input_size=1662, hidden_size=256, num_classes=3, dropout=0.4):
        super().__init__()

        # ── Input projection ──────────────────────────────────────────────────
        # Compresses raw 1662-dim landmark vector into a dense learned
        # representation before feeding into the GRU.
        # This reduces noise and significantly helps with overfitting.
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )

        # ── GRU ───────────────────────────────────────────────────────────────
        # 2-layer GRU; dropout applied between layers.
        # GRU chosen over LSTM: fewer parameters, faster training,
        # comparable accuracy on short sequences (30 frames).
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )

        # ── Classifier head ───────────────────────────────────────────────────
        self.bn       = nn.BatchNorm1d(hidden_size)
        self.dropout  = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, x):
        # x: (batch, 30, 1662)
        batch, seq_len, feat = x.shape

        # Project each timestep independently
        x = x.view(batch * seq_len, feat)       # (batch*30, 1662)
        x = self.input_proj(x)                  # (batch*30, hidden)
        x = x.view(batch, seq_len, -1)          # (batch, 30, hidden)

        # GRU — use last hidden state
        _, h_n = self.gru(x)                    # h_n: (2, batch, hidden)
        h_n = h_n[-1]                           # last layer: (batch, hidden)

        # Normalize + classify
        out = self.bn(h_n)
        out = self.dropout(out)
        out = self.fc(out)                      # (batch, num_classes)
        return out