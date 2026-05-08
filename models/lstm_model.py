# models/lstm_model.py
"""
LSTM Classifier — mirrors GRU architecture exactly for fair comparison.
Only difference: nn.GRU → nn.LSTM (and hidden state handling).
"""

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, input_size=1662, hidden_size=256,
                 num_classes=76, dropout=0.4, num_layers=2):
        super(LSTMClassifier, self).__init__()

        # ── INPUT PROJECTION ─────────────────────────────
        # Same as GRU: compress 1662 → 256 before recurrent layers
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── LSTM LAYERS ──────────────────────────────────
        # Identical config to GRU: 2 layers, 256 hidden, inter-layer dropout
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # ── CLASSIFIER HEAD ──────────────────────────────
        # Same as GRU: BatchNorm → 256→128→num_classes
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (batch, 30, 1662)

        # Project each frame
        b, t, f = x.shape
        x = x.view(b * t, f)
        x = self.input_proj(x)
        x = x.view(b, t, -1)           # (batch, 30, 256)

        # LSTM — returns (output, (h_n, c_n))
        # LSTM has TWO hidden states unlike GRU (h_n + c_n)
        _, (h_n, _) = self.lstm(x)     # h_n: (num_layers, batch, 256)

        # Take last layer's hidden state
        out = h_n[-1]                   # (batch, 256)

        # Classify
        out = self.classifier(out)      # (batch, num_classes)
        return out


if __name__ == "__main__":
    model = LSTMClassifier(input_size=1662, hidden_size=256, num_classes=76)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LSTM Trainable Parameters: {total:,}")

    # Quick forward pass test
    x = torch.randn(4, 30, 1662)
    out = model(x)
    print(f"Input shape  : {x.shape}")
    print(f"Output shape : {out.shape}")