# models/gru_model.py
import torch
import torch.nn as nn

class GRUClassifier(nn.Module):
    def __init__(self, input_size=1662, hidden_size=128, num_classes=3):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, 30, 1662)
        _, h_n = self.gru(x)        # h_n: (1, batch, hidden)
        h_n = h_n.squeeze(0)        # (batch, hidden)
        out = self.fc(h_n)          # (batch, num_classes)
        return out
